"""Re-execute an explicit cached full-horizon step without recomputing dynamics."""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.scripts.optimize_g1_true23_pd_trajectory import (
    assert_bindings,
    evaluate_trajectory,
    load_arrays,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_pd_box_step import box_backward_pass
from gear_sonic.utils.g1_true23_pd_shooting import PdShootingPlant
from gear_sonic.utils.g1_true23_pd_trajectory_optimizer import (
    CONTROL_WEIGHT,
    SLEW_WEIGHT,
    MotionObjective,
    backward_pass,
    feedback_trial,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-report", type=Path, required=True)
    parser.add_argument("--reproduction-report", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--regularization", type=float, required=True)
    parser.add_argument("--step-method", choices=("clipped", "box"), default="clipped")
    args = parser.parse_args(argv)
    if not 0 < args.alpha <= 1 or not np.isfinite(args.regularization) or args.regularization <= 0:
        parser.error("requires alpha in (0,1] and positive finite regularization")
    root = Path(__file__).resolve().parents[2]
    diagnosis = json.loads(args.diagnosis_report.read_text())
    reproduction = json.loads(args.reproduction_report.read_text())
    assert_bindings(diagnosis["inputs"])
    assert_bindings(reproduction["inputs"])
    if diagnosis["inputs"].get(str(args.reproduction_report.resolve())) != sha256_file(args.reproduction_report):
        raise ValueError("cached step cannot change the seed reproduction")
    cache_path = Path(diagnosis["linearization_path"])
    trace_path, parent_path = Path(reproduction["trace_path"]), Path(reproduction["parent_evaluation"])
    if sha256_file(cache_path) != diagnosis["linearization_sha256"]:
        raise ValueError("cached derivatives changed")
    if sha256_file(trace_path) != reproduction["trace_sha256"]:
        raise ValueError("cached step seed changed")
    parent = json.loads(parent_path.read_text())
    row = next(item for item in parent["records"] if item["case"] == "nominal")
    reference_path = Path(row["result"]["motion_path"])
    if sha256_file(reference_path) != row["result"]["motion_sha256"]:
        raise ValueError("cached step original requested motion changed")
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    plant = PdShootingPlant(model, NativeModelActuationProfile.from_sim_config(root / PHYSICS))
    if plant.model_sha != reproduction["reproduced_physics_sha256"]:
        raise ValueError("cached step cannot change original physics")
    seed, local = load_arrays(trace_path), load_arrays(cache_path)
    controls = seed["target23"]
    objective = MotionObjective(plant, load_arrays(reference_path), parent["timeline"])
    if len(local["a"]) != objective.count or diagnosis["complete_requested_controls"] != objective.count:
        raise ValueError("cached step requires the complete original lifecycle")
    paths = [
        args.diagnosis_report,
        args.reproduction_report,
        cache_path,
        trace_path,
        parent_path,
        reference_path,
        args.asset_root / MODEL,
        root / PHYSICS,
        *collect_local_source_closure(root, [Path(__file__)]).files,
    ]
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    solve_step = box_backward_pass if args.step_method == "box" else backward_pass
    increments, feedback = solve_step(plant, local, controls, controls, args.regularization)
    candidate, targets = feedback_trial(
        plant, seed["integration_state"][0], seed, controls, increments, feedback, args.alpha
    )
    baseline_cost = objective.cost(seed, controls, controls)
    cost = objective.cost(candidate, targets, controls)
    if not cost < baseline_cost - max(1e-9, 1e-6 * abs(baseline_cost)):
        raise ValueError("cached step does not improve the complete original objective")
    metrics, errors = evaluate_trajectory(plant, objective, parent["timeline"], candidate)
    baseline_metrics, _ = evaluate_trajectory(plant, objective, parent["timeline"], seed)
    assert_bindings(bindings)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    output_trace = args.output_directory / "candidate.npz"
    with output_trace.open("xb") as stream:
        np.savez_compressed(stream, **{**candidate, "target23": targets, "landmark_error_m": errors})
    report = dict(
        kind="g1_true23_offline_actual_pd_trajectory_cached_step_v1",
        inputs=bindings,
        reproduction_report_sha256=sha256_file(args.reproduction_report),
        original_requested_motion_path=str(reference_path.resolve()),
        original_requested_motion_sha256=sha256_file(reference_path),
        original_policy_identity=reproduction["original_policy_identity"],
        compiled_physics_sha256=plant.model_sha,
        complete_controls=objective.count,
        baseline_cost=baseline_cost,
        initial_cost=baseline_cost,
        final_cost=cost,
        baseline_metrics=baseline_metrics,
        iterations=[
            dict(
                iteration=1,
                accepted=True,
                cost=cost,
                metrics=metrics,
                trials=[
                    dict(
                        alpha=args.alpha,
                        regularization=args.regularization,
                        status="accepted_full_cost_reduction",
                        complete_controls=len(targets),
                    )
                ],
            )
        ],
        candidate_trace_path=str(output_trace.resolve()),
        candidate_trace_sha256=sha256_file(output_trace),
        objective_contract=dict(
            control_weight=CONTROL_WEIGHT,
            slew_weight=SLEW_WEIGHT,
            original_reference_unchanged=True,
            original_timing_unchanged=True,
        ),
        local_step_solver=args.step_method,
        full_future_reference_used_offline=True,
        not_a_sonic_policy_or_causal_teleop_controller=True,
        original_bounded_pd_command_codec_used=True,
        external_forces_or_feedforward_used=False,
        state_writes_after_each_independent_trial_initial_condition=0,
        admissible_training_parent=False,
        simulator_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (args.output_directory / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            dict(complete_controls=objective.count, baseline_cost=baseline_cost, final_cost=cost, metrics=metrics),
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
