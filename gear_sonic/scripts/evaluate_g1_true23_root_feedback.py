"""CPU-only same-policy lifecycle and external-force root-feedback diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

from gear_sonic.scripts.evaluate_g1_true23_generalist_baselines import write_json
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_generalist_benchmark import FLAGS, MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic, build_lifecycle_timeline
from gear_sonic.utils.g1_true23_root_feedback_benchmark import (
    ForcePulse,
    load_root_feedback_pair,
    run_root_feedback_diagnostic,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

CASES = {
    "nominal": (),
    "standing_push_x": (ForcePulse(500, 50, (40.0, 0.0, 0.0)),),
    "standing_push_y": (ForcePulse(500, 50, (0.0, 40.0, 0.0)),),
}
MOTION_FIELDS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cases", choices=tuple(CASES), nargs="+", default=list(CASES))
    parser.add_argument(
        "--return-target", choices=("planned_endpoint", "configured_origin"), default="planned_endpoint"
    )
    parser.add_argument(
        "--maximum-controls", type=int, help="Prefix diagnostic only; can never qualify completion"
    )
    args = parser.parse_args(argv)
    if len(set(args.cases)) != len(args.cases):
        raise ValueError("diagnostic case names must be unique")
    if args.maximum_controls is not None and args.maximum_controls <= 0:
        raise ValueError("maximum-controls must be positive")
    root = Path(__file__).resolve().parents[2]
    assets, motion_path, manifest_path = (
        path.resolve(strict=True) for path in (args.asset_root, args.motion, args.candidate_manifest)
    )
    output = args.output_directory.resolve()
    if output.exists() or args.output_directory.is_symlink():
        raise ValueError("root-feedback campaign requires a new evidence directory")
    paths = (motion_path, manifest_path, assets / MODEL, root / PHYSICS)
    inputs = {str(path): sha256_file(path) for path in paths}
    with np.load(motion_path, allow_pickle=False) as archive:
        source = {key: archive[key].copy() for key in MOTION_FIELDS}
    _, model, _ = prepare_true23_model(assets / MODEL, root / PHYSICS)
    motion, timeline = build_lifecycle_timeline(
        source, model=model, simulation_config=root / PHYSICS, return_target=args.return_target
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    torch.set_num_threads(1)
    policy, identity = load_root_feedback_pair(manifest_path, session_options=options)
    for path in (Path(value) for value in identity["component_paths"].values()):
        inputs[str(path)] = sha256_file(path)
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            inputs[str(Path(path).resolve())] = sha256_file(Path(path))
    output.mkdir(parents=True, exist_ok=False)
    timeline_path = output / "lifecycle_reference.npz"
    with timeline_path.open("xb") as stream:
        np.savez_compressed(stream, **motion)
    timeline.update(
        source_motion_path=str(motion_path),
        source_motion_sha256=sha256_file(motion_path),
        timeline_path=str(timeline_path),
        timeline_sha256=sha256_file(timeline_path),
    )
    write_json(
        output / "started.json",
        dict(
            kind="g1_true23_root_feedback_lifecycle_campaign_started_v1",
            inputs=inputs,
            policy_identity=identity,
            cases=args.cases,
            maximum_controls=args.maximum_controls,
            timeline=timeline,
            **FLAGS,
        ),
    )
    rows = []
    for case in args.cases:
        result, arrays = run_root_feedback_diagnostic(
            root=root,
            asset_root=assets,
            motion_path=timeline_path,
            policy=policy,
            pulses=CASES[case],
            maximum_controls=args.maximum_controls,
        )
        lifecycle = assess_lifecycle_diagnostic(timeline, result, arrays)
        trace = output / (case + ".npz")
        with trace.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        row = dict(
            case=case,
            policy_identity=identity,
            result=result,
            lifecycle=lifecycle,
            trace_path=str(trace),
            trace_sha256=sha256_file(trace),
        )
        write_json(output / (case + ".json"), row)
        rows.append(row)
        print(
            json.dumps(
                dict(
                    case=case,
                    completed=result["completed_controls"],
                    requested=result["available_controls"],
                    failure=result["failure"],
                    root_response=result["root_response"],
                )
            ),
            flush=True,
        )
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"root-feedback campaign input changed: {path}")
    for key in (
        "initial_state_and_history_sha256",
        "compiled_model_sha256",
        "motion_sha256",
        "kp_hardware",
        "kd_hardware",
    ):
        if any(row["result"][key] != rows[0]["result"][key] for row in rows[1:]):
            raise ValueError(f"root-feedback paired cases do not share {key}")
    write_json(
        output / "report.json",
        dict(
            kind="g1_true23_root_feedback_single_policy_lifecycle_campaign_v1",
            records=rows,
            inputs=inputs,
            timeline=timeline,
            only_one_hash_bound_policy_per_case=True,
            no_postinitial_robot_pose_rewrites=True,
            no_fallback_controller=True,
            root_measurement_is_privileged_simulator_state=True,
            hardware_world_state_estimator_qualified=False,
            perturbation_force_cases_are_not_hardware_instructions=True,
            **FLAGS,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
