"""Prepare a pinned, multi-recording original-intent research mix; no physics."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_original_intent_mix import KIND, assemble_mix
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source-model", "native-model", "sim-config", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("original-intent mix refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"mix input changed: {path}")
        inputs[str(path)] = digest
        return path

    def archive(path, expected):
        with np.load(bind(path, expected), allow_pickle=False) as values:
            return {key: values[key].copy() for key in values.files}

    manifest = json.loads(bind(args.manifest).read_text())
    if set(manifest) != {"kind", "members", "evaluation_only_recording_ids"} or manifest["kind"] != KIND:
        raise ValueError("invalid original-intent development manifest")
    members = []
    for entry in manifest["members"]:
        if set(entry) != {"name", "recording_id", "baseline_report", "original_reference"}:
            raise ValueError("invalid mix member manifest")
        for field in ("baseline_report", "original_reference"):
            if set(entry[field]) != {"path", "sha256"}:
                raise ValueError("mix files require an explicit SHA256 binding")
        report = json.loads(bind(entry["baseline_report"]["path"], entry["baseline_report"]["sha256"]).read_text())
        timeline = report["timeline"]
        original = entry["original_reference"]
        members.append(
            dict(
                name=entry["name"],
                recording_id=entry["recording_id"],
                timeline=timeline,
                motion=archive(timeline["timeline_path"], timeline["timeline_sha256"]),
                original_reference=archive(original["path"], original["sha256"]),
            )
        )
    source = mujoco.MjModel.from_xml_path(str(bind(args.source_model)))
    _, native, _ = prepare_true23_model(bind(args.native_model), bind(args.sim_config))
    raw, spans, original, derived, contract = assemble_mix(
        members,
        manifest["evaluation_only_recording_ids"],
        source_model=source,
        native_model=native,
        simulation_config=args.sim_config,
    )
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    bind(__file__)
    output.mkdir(parents=True, exist_ok=False)
    for name, arrays in (
        ("source.npz", raw),
        ("original_reference.npz", original),
        ("expected_lifecycles.npz", derived),
    ):
        with (output / name).open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    spans["source_motion_sha256"] = sha256_file(output / "source.npz")
    metadata = dict(
        schema="g1_true23_low_latency_recovery_motion_v1",
        output=dict(filename="source.npz", sha256=spans["source_motion_sha256"]),
        original_intent_mix=contract,
        deployment_ready=False,
        hardware_authorized=False,
    )
    for name, value in (("source.spans.json", spans), ("source.json", metadata)):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"mix input changed during execution: {path}")
    report = dict(
        **contract,
        inputs=inputs,
        outputs={
            name: dict(path=str(output / name), sha256=sha256_file(output / name))
            for name in (
                "source.npz",
                "original_reference.npz",
                "expected_lifecycles.npz",
                "source.spans.json",
                "source.json",
            )
        },
    )
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                key: contract[key]
                for key in (
                    "kind",
                    "training_recording_ids",
                    "source_frames",
                    "lifecycle_frames",
                    "saved_native_lifecycles_bit_exact",
                    "all_original29_tasks_bit_exact",
                    "deployment_ready",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
