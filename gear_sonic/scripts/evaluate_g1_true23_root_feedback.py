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
from gear_sonic.utils.g1_true23_buffered_reference import BUFFERED_TIMING, CAUSAL_TIMING, REFERENCE_TIMINGS
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
    parser.add_argument(
        "--lifecycle-reference-repair",
        type=Path,
        help="Explicit bound offline shoulder-ramp reference; never changes policy or robot state",
    )
    parser.add_argument(
        "--contact-step-reference",
        type=Path,
        help="Explicit new stepping entry/return diagnostic; never normal continuation evidence",
    )
    parser.add_argument("--reference-timing", choices=REFERENCE_TIMINGS, default=CAUSAL_TIMING)
    parser.add_argument(
        "--training-model-counterfactual",
        type=Path,
        help="Explicit compiled training-model diagnostic; does not qualify original replay physics",
    )
    parser.add_argument(
        "--release-source-geometry",
        type=Path,
        help="Required explicit reference geometry for release-compatible trained actors",
    )
    parser.add_argument("--cases", choices=tuple(CASES), nargs="+", default=list(CASES))
    parser.add_argument(
        "--release-action-convention",
        choices=("released_bounded_linear", "released29_scale_bounded_linear_v2"),
        default="released_bounded_linear",
    )
    parser.add_argument(
        "--return-target", choices=("planned_endpoint", "configured_origin"), default="planned_endpoint"
    )
    parser.add_argument(
        "--maximum-controls", type=int, help="Prefix diagnostic only; can never qualify completion"
    )
    args = parser.parse_args(argv)
    if args.contact_step_reference is not None and (
        args.lifecycle_reference_repair is not None
        or args.training_model_counterfactual is not None
        or args.reference_timing != BUFFERED_TIMING
        or args.maximum_controls is not None
    ):
        parser.error("contact-step diagnostic requires full nominal buffered replay and no other reference repair")
    if args.reference_timing == BUFFERED_TIMING and (
        args.release_source_geometry is None
        or args.release_action_convention != "released29_scale_bounded_linear_v2"
        or args.training_model_counterfactual is not None
    ):
        raise ValueError("buffered source requires explicit source geometry/action units and nominal physics")
    if args.release_action_convention != "released_bounded_linear" and args.release_source_geometry is None:
        raise ValueError("source action convention requires explicit source geometry")
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
    if args.training_model_counterfactual is not None:
        if args.release_source_geometry is None:
            raise ValueError("training-model diagnostic currently requires explicit corrected reference semantics")
        path = args.training_model_counterfactual.resolve(strict=True)
        inputs[str(path)] = sha256_file(path)
    with np.load(motion_path, allow_pickle=False) as archive:
        source = {key: archive[key].copy() for key in MOTION_FIELDS}
    _, model, _ = prepare_true23_model(assets / MODEL, root / PHYSICS)
    motion, timeline = build_lifecycle_timeline(
        source, model=model, simulation_config=root / PHYSICS, return_target=args.return_target
    )
    if args.lifecycle_reference_repair is not None:
        if args.training_model_counterfactual is not None:
            raise ValueError("repaired reference requires unchanged nominal physical model")
        from gear_sonic.utils.g1_true23_repaired_lifecycle_reference import load_repaired_lifecycle_reference

        motion, timeline, repair_inputs = load_repaired_lifecycle_reference(
            args.lifecycle_reference_repair, motion, timeline, model, inputs[str(motion_path)]
        )
        inputs.update(repair_inputs)
    if args.contact_step_reference is not None:
        from gear_sonic.utils.g1_true23_contact_step_lifecycle_reference import load_contact_step_reference

        motion, timeline, step_inputs = load_contact_step_reference(
            args.contact_step_reference, motion, timeline, model, inputs[str(motion_path)]
        )
        inputs.update(step_inputs)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    torch.set_num_threads(1)
    compatibility = None
    virtual_vr = None
    if args.release_source_geometry is not None:
        from gear_sonic.utils.g1_true23_release_compatibility import release_compatibility_contract
        from gear_sonic.utils.g1_true23_virtual_source_reference import virtual_source_vr_terms

        geometry = args.release_source_geometry.resolve(strict=True)
        inputs[str(geometry)] = sha256_file(geometry)
        compatibility = release_compatibility_contract(
            inputs[str(geometry)], args.release_action_convention, args.reference_timing
        )
        virtual_vr = virtual_source_vr_terms(motion, geometry)
    if args.reference_timing == BUFFERED_TIMING:
        from gear_sonic.teleop.buffered_source_simulation import BufferedSourceSimulationAdapter
    policy, identity = load_root_feedback_pair(
        manifest_path, session_options=options, expected_release_compatibility=compatibility
    )
    if compatibility is not None:
        from gear_sonic.scripts.diagnose_g1_true23_release_semantics import ReferenceAblation
        from gear_sonic.utils.g1_true23_generalist_benchmark import run_reference_diagnostic
        from gear_sonic.utils.g1_true23_root_feedback_benchmark import summarize_root_perturbations
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
            contact_step_reference_diagnostic=args.contact_step_reference is not None,
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
        if compatibility is None:
            result, arrays = run_root_feedback_diagnostic(
                root=root,
                asset_root=assets,
                motion_path=timeline_path,
                policy=policy,
                pulses=CASES[case],
                maximum_controls=args.maximum_controls,
            )
        else:
            if args.reference_timing == BUFFERED_TIMING:
                adapter = BufferedSourceSimulationAdapter(motion, virtual_vr, CASES[case])
            else:
                adapter = ReferenceAblation(
                    motion,
                    "causal_past_virtual_source",
                    virtual_vr,
                    args.release_action_convention,
                    case,
                    root_feedback=True,
                )
            result, arrays = run_reference_diagnostic(
                root=root,
                asset_root=assets,
                motion_path=timeline_path,
                policy=policy,
                runtime_adapter=adapter,
                maximum_controls=args.maximum_controls,
                training_model_counterfactual=args.training_model_counterfactual,
            )
            arrays.update(adapter.root_adapter.arrays())
            arrays["actual_policy_encoder267"] = np.asarray(adapter.actual_encoder_inputs)
            arrays["released_model_raw23"] = np.asarray(adapter.model_outputs)
            arrays["bounded_linear_projection_delta_rad"] = np.asarray(adapter.projection_delta_rad).reshape(
                -1, 23
            )
            if args.reference_timing == BUFFERED_TIMING:
                arrays["source_emission_anchor_setpoint_timestamps_s"] = np.asarray(adapter.timestamps)
                # Preserve exact simulated-source input, including separately
                # disclosed generated standing samples after the scored timeline.
                with (output / (case + ".received_source.npz")).open("xb") as stream:
                    np.savez_compressed(stream, **adapter.source)
            result["root_response"] = summarize_root_perturbations(
                adapter.root_adapter,
                arrays,
                completed_controls=result["completed_controls"],
                failure=result["failure"],
            )
            result["release_compatibility"] = compatibility
        lifecycle = assess_lifecycle_diagnostic(timeline, result, arrays)
        trace = output / (case + ".npz")
        with trace.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        row = dict(
            case=case,
            contact_step_reference_diagnostic=args.contact_step_reference is not None,
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
            kind=(
                "g1_true23_contact_step_lifecycle_policy_diagnostic_v1"
                if args.contact_step_reference is not None
                else "g1_true23_root_feedback_single_policy_lifecycle_campaign_v1"
            ),
            contact_step_reference_diagnostic=args.contact_step_reference is not None,
            release_compatibility=compatibility,
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
