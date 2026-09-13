"""Lossless existing-PICO research bank, excluding walk008 and unrelated dance."""

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from gear_sonic.envs.mjlab import sonic_true23_original_intent as intent
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_curriculum import write_curriculum_bundle
from gear_sonic.utils.g1_true23_original_intent_mix import assemble_mix
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
BASE = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")
HERE = Path(__file__).resolve().parent


def main():
    output = HERE / "bank_v1"
    if output.exists():
        raise FileExistsError("PICO research bank refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise ValueError("PICO bank source changed: " + str(path))
        inputs[str(path)] = actual
        return path

    def read(path, expected=None):
        with np.load(bind(path, expected), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    members = []
    for name in ("walk002", "walk003", "pico"):
        old = BASE / "released_core_comparison_v1/normal" / name / "report.json"
        source = BASE / name / "original_source_bundle_v1/original_reference.npz"
        if name == "pico":
            old = BASE / "normal_core_pico_v1/report.json"
            source = BASE / "pico_freedancing_v1/optical_reference_v2/original29.npz"
        report = json.loads(bind(old).read_text())
        source_expected = report["inputs"][str(source)]
        timeline = report["timeline"]
        members.append(
            dict(
                name=name,
                recording_id="existing_pico_derived_" + name,
                timeline=timeline,
                motion=read(timeline["timeline_path"], timeline["timeline_sha256"]),
                original_reference=read(source, source_expected),
            )
        )
    source_model = bind(ASSETS / "gear_sonic/data/robots/g1/g1_29dof.xml")
    native_model = bind(ASSETS / MODEL)
    config = bind(ROOT / PHYSICS)
    geometry = mujoco.MjModel.from_xml_path(str(source_model))
    _, native, _ = prepare_true23_model(native_model, config)
    raw, spans, original, lifecycles, contract = assemble_mix(
        members,
        ["existing_pico_derived_walk008"],
        source_model=geometry,
        native_model=native,
        simulation_config=config,
    )
    bind(__file__)
    bind(HERE / "EXPERIMENT.md")
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and str(path).endswith(".py"):
            bind(path)
    output.mkdir()
    for name, value in (("source.npz", raw), ("original_reference.npz", original)):
        with (output / name).open("xb") as stream:
            np.savez_compressed(stream, **value)
    motion_path, _, spans_path = write_curriculum_bundle(
        output / "curriculum",
        lifecycles,
        contract["lifecycle_spans"],
        contract,
    )
    spec = intent.make_spec(
        original_reference=output / "original_reference.npz",
        native_motion=motion_path,
        source_model=source_model,
        native_model=native_model,
    )
    intent.load_reference(spec)
    with (output / "original_intent.spec.json").open("x") as stream:
        json.dump(spec, stream, indent=2, allow_nan=False)
    for path, expected in inputs.items():
        assert sha256_file(Path(path)) == expected, path
    report = dict(
        **contract,
        inputs=inputs,
        pico_used_for_training=True,
        source_recording_modalities=dict(
            walk002="public_PICO_workflow_robot_motion",
            walk003="public_PICO_workflow_robot_motion",
            pico="paired_PICO_OptiTrack_optical_body_reference",
        ),
        sensor_only_tracking_claimed=False,
        files=dict(spec=str(output / "original_intent.spec.json"), spans=str(spans_path), motion=str(motion_path)),
        outputs={str(p): sha256_file(p) for p in output.rglob("*") if p.is_file()},
    )
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "training_recording_ids",
                    "source_frames",
                    "lifecycle_frames",
                    "saved_native_lifecycles_bit_exact",
                    "all_original29_tasks_bit_exact",
                    "pico_used_for_training",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
