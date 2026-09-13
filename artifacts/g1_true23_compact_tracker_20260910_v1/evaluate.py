"""Complete saved-PICO lifecycle matrix for a compact native23 milestone."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from gear_sonic.scripts.record_g1_true23_original_intent import task_metrics
from gear_sonic.utils.g1_true23_compact_replay import CompactAdapter, CompactPolicy
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS, run_reference_diagnostic
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic
from gear_sonic.utils.g1_true23_original29_reference import (
    build_original29_reference,
    verify_unmodified_native_pair,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
BASE = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")
FLAGS = dict(deployment_ready=False, hardware_authorized=False, simulator_qualified=False)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def metrics(trace, motion, phase):
    start = phase["control_start"]
    stop = min(len(trace["qpos"]) - 1, phase["control_stop"])
    if stop <= start:
        return dict(source_controls=0)
    delta = trace["qpos"][start + 1 : stop + 1, 7:] - motion["joint_pos"][start + 11 : stop + 11]
    return dict(
        source_controls=stop - start,
        source_seconds=(stop - start) * 0.02,
        leg_rmse_rad=float(np.sqrt(np.mean(delta[:, :12] ** 2))),
        arm_rmse_rad=float(np.sqrt(np.mean(delta[:, 13:] ** 2))),
        root_p95_m=float(np.percentile(trace["pelvis_error_m"][start:stop], 95)),
        relative_foot_p95_m=np.percentile(trace["relative_landmark_error_m"][start:stop, :2], 95, axis=0).tolist(),
        world_foot_p95_m=np.percentile(trace["landmark_error_m"][start:stop, :2], 95, axis=0).tolist(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("compact evaluation refuses overwrite or implicit rerun")
    torch.set_num_threads(1)
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("changed input " + str(path))
        inputs[str(path)] = digest
        return path

    def load(path):
        with np.load(bind(path), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    bind(__file__)
    bind(Path(__file__).parent / "EXPERIMENT.md")
    policy = CompactPolicy(bind(args.checkpoint))
    source = mujoco.MjModel.from_xml_path(str(bind(ASSETS / "gear_sonic/data/robots/g1/g1_29dof.xml")))
    model = mujoco.MjModel.from_xml_path(str(bind(ASSETS / MODEL)))
    bind(ROOT / PHYSICS)
    args.output.mkdir(parents=True)
    summaries = []
    for name in ("pico", "walk002", "walk003", "walk008"):
        directory = BASE / "released_core_comparison_v1/normal" / name
        original_path = BASE / name / "original_source_bundle_v1/original_reference.npz"
        if name == "pico":
            directory = BASE / "normal_core_pico_v1"
            original_path = BASE / "pico_freedancing_v1/optical_reference_v2/original29.npz"
        old = json.loads(bind(directory / "report.json").read_text())
        timeline = old["timeline"]
        path = bind(timeline["timeline_path"], timeline["timeline_sha256"])
        motion = load(path)
        original = load(bind(original_path, old["inputs"][str(original_path)]))
        reference = build_original29_reference(source, original["source_qpos29"])
        for key, value in reference.arrays().items():
            np.testing.assert_array_equal(value, original[key])
        pair = verify_unmodified_native_pair(reference, motion)
        output = args.output / name
        output.mkdir()
        write(
            output / "request.json",
            dict(name=name, identity=policy.identity(), inputs=inputs, full_lifecycle_requested=True, **FLAGS),
        )
        print(json.dumps(dict(starting=name, updates=policy.completed_updates)), flush=True)
        adapter = CompactAdapter(motion, reference.virtual_vr21, root=ROOT, assets=ASSETS)
        result, trace = run_reference_diagnostic(
            root=ROOT, asset_root=ASSETS, motion_path=path, policy=policy, runtime_adapter=adapter
        )
        for filename, value in (("trace.npz", trace), ("attempts.npz", adapter.arrays())):
            with (output / filename).open("xb") as stream:
                np.savez_compressed(stream, **value)
        phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
        source_metrics = metrics(trace, motion, phase)
        tasks = (
            task_metrics(model, source, reference, motion, trace, phase)
            if source_metrics["source_controls"]
            else None
        )
        report = dict(
            name=name,
            identity=policy.identity(),
            timeline=timeline,
            source_pair=pair,
            inputs=inputs,
            physics=result,
            source_metrics=source_metrics,
            original29_tasks=tasks,
            assessment=assess_lifecycle_diagnostic(timeline, result, trace),
            adapter=adapter.contract(),
            trace_sha256=sha256_file(output / "trace.npz"),
            **FLAGS,
        )
        write(output / "report.json", report)
        summary = dict(
            name=name,
            completed=result["completed_controls"],
            requested=result["requested_controls"],
            failure=result["failure"],
            source_metrics=source_metrics,
            report_sha256=sha256_file(output / "report.json"),
            **FLAGS,
        )
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write(args.output / "summary.json", dict(cases=summaries, identity=policy.identity(), **FLAGS))


if __name__ == "__main__":
    main()
