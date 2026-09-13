"""Full fixed PICO/walking lifecycle evaluation of a declared training milestone."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from gear_sonic.scripts.record_g1_true23_original_intent import task_metrics
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS, run_reference_diagnostic
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_ankle_horizon_preview import AnkleHorizonNative23RangePreview
from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic
from gear_sonic.utils.g1_true23_pico_foot_precision_replay import PicoFootPrecisionAdapter, PicoFootPrecisionPolicy
from gear_sonic.utils.g1_true23_original29_reference import (
    build_original29_reference,
    verify_unmodified_native_pair,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
BASE = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")
HERE = Path(__file__).resolve().parent
FLAGS = dict(deployment_ready=False, hardware_authorized=False, simulator_qualified=False)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def metrics(trace, motion, phase, stop):
    start, end = phase["control_start"], min(stop, phase["control_stop"])
    if end <= start:
        return dict(source_controls=0)
    q = trace["qpos"][start + 1 : end + 1]
    delta = q[:, 7:] - motion["joint_pos"][start + 11 : end + 11]
    return dict(
        source_controls=end - start,
        leg_rmse_rad=float(np.sqrt(np.mean(delta[:, :12] ** 2))),
        arm_rmse_rad=float(np.sqrt(np.mean(delta[:, 13:] ** 2))),
        root_p95_m=float(np.percentile(trace["pelvis_error_m"][start:end], 95)),
        relative_foot_p95_m=np.percentile(trace["relative_landmark_error_m"][start:end, :2], 95, axis=0).tolist(),
        world_foot_p95_m=np.percentile(trace["landmark_error_m"][start:end, :2], 95, axis=0).tolist(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("checkpoint evaluation refuses overwrite or rerun")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("evaluation input changed: " + str(path))
        inputs[str(path)] = digest
        return path

    def read(path, expected=None):
        with np.load(bind(path, expected), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    bind(__file__)
    bind(HERE / "EXPERIMENT.md")
    checkpoint = bind(args.checkpoint, "ac2f844894e3e53c364ae466966a64db556ee255c6ab72e736fe15b87adb19fe")
    torch.set_num_threads(1)
    policy = PicoFootPrecisionPolicy(
        checkpoint,
        warm_start_path=bind(ASSETS / "sonic_release/g1_23dof_rev_1_0_init.pt"),
        source_checkpoint_path=bind(ASSETS / "sonic_release/last.pt"),
    )
    resolved = policy.lineage["materials"]["resolved_config"]["payload"]
    if resolved.get("pico_used_for_training") is not True or policy.completed_updates != 1000:
        raise ValueError("ankle horizon evaluation requires fixed final1000 milestone")
    geometry = mujoco.MjModel.from_xml_path(str(bind(ASSETS / "gear_sonic/data/robots/g1/g1_29dof.xml")))
    model = mujoco.MjModel.from_xml_path(str(bind(ASSETS / MODEL)))
    bind(ROOT / PHYSICS)
    cases = {}
    for name in ("pico", "walk002", "walk003", "walk008"):
        directory = BASE / "released_core_comparison_v1/normal" / name
        original_path = BASE / name / "original_source_bundle_v1/original_reference.npz"
        if name == "pico":
            directory = BASE / "normal_core_pico_v1"
            original_path = BASE / "pico_freedancing_v1/optical_reference_v2/original29.npz"
        old = json.loads(bind(directory / "report.json").read_text())
        timeline = old["timeline"]
        path = bind(timeline["timeline_path"], timeline["timeline_sha256"])
        motion = read(path)
        original = read(original_path, old["inputs"][str(original_path)])
        reference = build_original29_reference(geometry, original["source_qpos29"])
        for key, value in reference.arrays().items():
            np.testing.assert_array_equal(value, original[key])
        pair = verify_unmodified_native_pair(reference, motion)
        baseline_path = directory / "trace.npz"
        baseline = read(baseline_path, old["inputs"][str(baseline_path)])
        cases[name] = dict(
            path=path, motion=motion, reference=reference, pair=pair, baseline=baseline, timeline=timeline
        )
        for label, family, milestone in (
            ("same1000_20ms", "g1_true23_pico_foot_precision_20260910_v1", "eval1000_v1"),
            ("parent500_20ms", "g1_true23_pico_training_20260910_v1", "eval500_v1"),
        ):
            earlier = ROOT / "artifacts" / family / milestone / name
            earlier_report = json.loads(bind(earlier / "report.json").read_text())
            assert earlier_report["timeline"] == timeline
            cases[name][label] = read(earlier / "trace.npz", earlier_report["trace_sha256"])
        assert cases[name]["same1000_20ms"]["qpos"].shape[1] == 30
        assert all(
            np.array_equal(trace["qpos"][0], baseline["qpos"][0])
            for trace in (cases[name]["same1000_20ms"], cases[name]["parent500_20ms"])
        )
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and str(path).endswith(".py"):
            bind(path)
    identity = policy.identity()
    args.output.mkdir(parents=True)
    write(
        args.output / "started.json", dict(identity=identity, inputs=inputs, expected_cases=list(cases), **FLAGS)
    )
    summaries = []
    for name, case in cases.items():
        output = args.output / name
        output.mkdir()
        write(
            output / "request.json",
            dict(name=name, identity=identity, inputs=inputs, full_lifecycle_requested=True, **FLAGS),
        )
        print(json.dumps(dict(starting=name, updates=policy.completed_updates)), flush=True)
        adapter = PicoFootPrecisionAdapter(
            case["motion"], case["reference"].virtual_vr21, root=ROOT, assets=ASSETS
        )
        adapter.preview = AnkleHorizonNative23RangePreview(model_path=ASSETS / MODEL, physics_path=ROOT / PHYSICS)
        result, trace = run_reference_diagnostic(
            root=ROOT, asset_root=ASSETS, motion_path=case["path"], policy=policy, runtime_adapter=adapter
        )
        write(output / "physics_result.json", result)
        for filename, value in (("trace.npz", trace), ("attempts.npz", adapter.arrays())):
            with (output / filename).open("xb") as stream:
                np.savez_compressed(stream, **value)
        with (output / "preview_predictions.npz").open("xb") as stream:
            np.savez_compressed(stream, qpos=np.asarray(adapter.preview.predictions))
        write(
            output / "preview_records.json",
            dict(accepted=adapter.preview.records, rejected=adapter.preview.failed_search),
        )
        lifecycle = assess_lifecycle_diagnostic(case["timeline"], result, trace)
        source = next(row for row in case["timeline"]["phases"] if row["name"] == "source_motion")
        np.testing.assert_array_equal(trace["qpos"][0], case["baseline"]["qpos"][0])
        np.testing.assert_array_equal(trace["qvel"][0], case["baseline"]["qvel"][0])
        common = min(result["completed_controls"], len(case["baseline"]["qpos"]) - 1)
        comparisons = dict(
            candidate=metrics(trace, case["motion"], source, common),
            baseline=metrics(case["baseline"], case["motion"], source, common),
        )
        for label in ("same1000_20ms", "parent500_20ms"):
            other = case[label]
            same_stop = min(result["completed_controls"], len(other["qpos"]) - 1)
            comparisons[label] = dict(
                candidate=metrics(trace, case["motion"], source, same_stop),
                baseline=metrics(other, case["motion"], source, same_stop),
            )
        entire = metrics(trace, case["motion"], source, result["completed_controls"])
        assert policy.identity() == identity
        for path, expected in inputs.items():
            assert sha256_file(Path(path)) == expected, path
        report = dict(
            name=name,
            result=result,
            lifecycle=lifecycle,
            timeline=case["timeline"],
            identity=identity,
            inputs=inputs,
            original_reference_pair=case["pair"],
            original_task_metrics=task_metrics(model, geometry, case["reference"], case["motion"], trace, source),
            matched_prefix=comparisons,
            entire_executed_source=entire,
            trace_sha256=sha256_file(output / "trace.npz"),
            attempts_sha256=sha256_file(output / "attempts.npz"),
            preview_predictions_sha256=sha256_file(output / "preview_predictions.npz"),
            preview_records_sha256=sha256_file(output / "preview_records.json"),
            optimizer_recording=name != "walk008",
            previously_seen_development_evaluation=True,
            **FLAGS,
        )
        write(output / "report.json", report)
        summary = dict(
            name=name,
            completed=result["completed_controls"],
            requested=result["requested_controls"],
            failure=result["failure"],
            source_tracking_passed=lifecycle["source_motion_tracking"][
                "provisional_reference_landmark_screen_passed"
            ],
            hard_range_excess=result["hard_joint_limit_excess_max_rad"],
            velocity_ratio=result["joint_velocity_ratio_max"],
            source_metrics=entire,
            matched_prefix=comparisons,
            preview=adapter.preview.contract(),
            report_sha256=sha256_file(output / "report.json"),
            **FLAGS,
        )
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write(args.output / "summary.json", dict(identity=identity, cases=summaries, **FLAGS))


if __name__ == "__main__":
    main()
