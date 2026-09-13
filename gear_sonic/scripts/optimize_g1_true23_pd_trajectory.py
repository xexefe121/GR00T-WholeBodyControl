"""Offline full-lifecycle trajectory repair; no policy or hardware qualification."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS, task_points
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_pd_box_step import box_backward_pass
from gear_sonic.utils.g1_true23_pd_contact_step import contact_backward_pass, linearize_self_envelopes
from gear_sonic.utils.g1_true23_pd_entry_guidance import EntryGuidance
from gear_sonic.utils.g1_true23_pd_predictive_contact import make_predictive_contact_law
from gear_sonic.utils.g1_true23_pd_protected_objective import (
    ProtectedMotionObjective,
    ProtectedPdPlant,
    protected_tracking_acceptance,
)
from gear_sonic.utils.g1_true23_pd_reactive_contact import terminal_nonregression
from gear_sonic.utils.g1_true23_pd_shooting import PdShootingPlant
from gear_sonic.utils.g1_true23_pd_source_history import verify_historical_report
from gear_sonic.utils.g1_true23_pd_standing_guidance import StandingVelocityGuidance
from gear_sonic.utils.g1_true23_pd_trajectory_optimizer import (
    CONTROL_WEIGHT,
    SLEW_WEIGHT,
    MotionObjective,
    feedback_trial,
    linearize_trajectory,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def assert_bindings(bindings):
    for name, expected in bindings.items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("trajectory optimizer input or implementation changed: " + name)


def evaluate_trajectory(plant, objective, timeline, trajectory):
    """Use the original referee's post-control world-space metrics and phases."""
    count, data = objective.count, mujoco.MjData(plant.model)
    errors, pelvis, joints = [], [], []
    if len(trajectory["qpos"]) != count + 1 or len(trajectory["qvel"]) != count + 1:
        raise ValueError("cannot assess an incomplete trajectory as a full lifecycle")
    for i in range(count):
        data.qpos[:], data.qvel[:] = trajectory["qpos"][i + 1], trajectory["qvel"][i + 1]
        mujoco.mj_forward(plant.model, data)  # Independent metric probe, never the integrated plant.
        actual = task_points(data.xpos[1:], data.xquat[1:])
        errors.append(np.linalg.norm(actual - objective.points[i + 1], axis=1))
        pelvis.append(float(np.linalg.norm(data.qpos[:3] - objective.poses[i + 1, :3])))
        joints.append(float(np.sqrt(np.mean((data.qpos[7:] - objective.poses[i + 1, 7:]) ** 2))))
    arrays = {**trajectory, "landmark_error_m": np.asarray(errors)}
    lifecycle = assess_lifecycle_diagnostic(
        timeline, dict(available_controls=count, completed_controls=count, failure=None), arrays
    )
    lifecycle["full_physical_lifecycle_integrated"] = lifecycle.pop("single_policy_full_lifecycle_integrated")
    lifecycle["kind"] = "offline_pd_trajectory_original_reference_lifecycle_diagnostic_v1"
    lifecycle["not_a_learned_policy"] = True
    q, v = trajectory["physics_post_qpos"][:, 7:], trajectory["physics_post_qvel"][:, 6:]
    evidence = dict(
        lifecycle=lifecycle,
        pelvis_world_position_p95_m=float(np.percentile(pelvis, 95)),
        joint_tracking_rmse_p95_rad=float(np.percentile(joints, 95)),
        physical_joint_limit_excess_max_rad=float(
            max(0, np.max(plant.model.jnt_range[1:, 0] - q), np.max(q - plant.model.jnt_range[1:, 1]))
        ),
        physical_motor_velocity_ratio_max=float(np.max(np.abs(v) / np.asarray(plant.profile.velocity))),
        applied_motor_effort_ratio_max=float(
            np.max(np.abs(trajectory["applied_torque23"]) / np.asarray(plant.profile.effort))
        ),
        root_height_min_m=float(np.min(trajectory["physics_post_qpos"][:, 2])),
        simulator_qualified=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    return evidence, np.asarray(errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduction-directory", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--resume-report", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--objective", choices=("protected", "original"), default="protected")
    parser.add_argument("--cache-linearizations", action="store_true")
    parser.add_argument("--step-method", choices=("box", "contact_predictive"), default="box")
    parser.add_argument(
        "--entry-guidance", choices=("none", "com", "com_lift", "com_lift_placement"), default="none"
    )
    parser.add_argument("--standing-velocity-guidance", action="store_true")
    parser.add_argument("--standing-orientation-guidance", action="store_true")
    parser.add_argument("--planning-clearance-m", type=float, default=0.001)
    parser.add_argument("--regularizations", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    args = parser.parse_args(argv)
    if args.standing_orientation_guidance and not args.standing_velocity_guidance:
        parser.error("standing orientation guidance requires standing velocity guidance")
    if not 1 <= args.iterations <= 20:
        parser.error("iterations must be between 1 and 20")
    if not np.isfinite(args.planning_clearance_m) or not 0 <= args.planning_clearance_m <= 0.005:
        parser.error("offline planning clearance must be between zero and five millimetres")
    if any(not np.isfinite(value) or value <= 0 for value in args.regularizations):
        parser.error("regularizations must be finite and positive")
    if args.objective != "protected" and (
        args.step_method != "box" or args.entry_guidance != "none" or args.standing_velocity_guidance
    ):
        parser.error("predictive or guided local search requires the protected physical referee")
    if args.step_method == "box" and args.planning_clearance_m != 0.001:
        parser.error("custom planning clearance requires the predictive local step")
    root, started = Path(__file__).resolve().parents[2], time.monotonic()
    reproduction_path = (args.reproduction_directory / "report.json").resolve(strict=True)
    reproduction = json.loads(reproduction_path.read_text())
    assert_bindings(reproduction["inputs"])
    seed_path = Path(reproduction["trace_path"]).resolve(strict=True)
    parent_path = Path(reproduction["parent_evaluation"]).resolve(strict=True)
    parent = json.loads(parent_path.read_text())
    row = next(item for item in parent["records"] if item["case"] == "nominal")
    reference_path = Path(row["result"]["motion_path"]).resolve(strict=True)
    if sha256_file(seed_path) != reproduction["trace_sha256"]:
        raise ValueError("verified replay trajectory changed")
    if sha256_file(reference_path) != row["result"]["motion_sha256"]:
        raise ValueError("original requested motion changed")
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    plant_class = ProtectedPdPlant if args.objective == "protected" else PdShootingPlant
    plant = plant_class(model, NativeModelActuationProfile.from_sim_config(root / PHYSICS))
    if plant.model_sha != reproduction["reproduced_physics_sha256"]:
        raise ValueError("trajectory optimization cannot change original compiled physics")
    objective_class = ProtectedMotionObjective if args.objective == "protected" else MotionObjective
    objective = objective_class(plant, load_arrays(reference_path), parent["timeline"])
    seed = load_arrays(seed_path)
    seed_controls = seed["target23"].copy()
    controls, trajectory = seed_controls.copy(), seed
    paths = [
        reproduction_path,
        seed_path,
        parent_path,
        reference_path,
        args.asset_root / MODEL,
        root / PHYSICS,
        *collect_local_source_closure(root, [Path(__file__)]).files,
    ]
    previous_report, historical_evidence = None, None
    if args.resume_report is not None:
        previous_path = args.resume_report.resolve(strict=True)
        previous_report = json.loads(previous_path.read_text())
        historical_evidence = verify_historical_report(
            previous_path, repository_root=root, source_archive=args.source_archive
        )
        if previous_report["reproduction_report_sha256"] != sha256_file(reproduction_path):
            raise ValueError("resume cannot change the baseline or original requested motion")
        candidate_path = Path(previous_report["candidate_trace_path"]).resolve(strict=True)
        if sha256_file(candidate_path) != previous_report["candidate_trace_sha256"]:
            raise ValueError("resume candidate changed")
        trajectory = load_arrays(candidate_path)
        controls = trajectory["target23"].copy()
        np.testing.assert_array_equal(trajectory["integration_state"][0], seed["integration_state"][0])
        paths.extend((previous_path, candidate_path))
        if historical_evidence["archive_path"] is not None:
            paths.append(Path(historical_evidence["archive_path"]))
        paths.extend(Path(row["archived_path"]) for row in historical_evidence["archived_python_sources_used"])
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    assert_bindings(bindings)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    events_path = args.output_directory / "events.jsonl"

    def progress(event):
        event = {"elapsed_s": round(time.monotonic() - started, 3), **event}
        message = json.dumps(event, allow_nan=False)
        with events_path.open("a") as stream:
            stream.write(message + "\n")
        print(message, flush=True)

    reproduced = plant.rollout(seed["integration_state"][0], controls, physics_records=True)
    for key, values in reproduced.items():
        if (
            values.shape != trajectory[key].shape
            or values.dtype != trajectory[key].dtype
            or values.tobytes() != trajectory[key].tobytes()
        ):
            raise ValueError("imported initialization did not reproduce actual physics exactly: " + key)
    del reproduced
    contact_envelope = None
    if args.objective == "protected":
        contact_envelope = plant.contact_observations()
        if min(plant.self_contact_depths.values(), default=0.0) < -0.005:
            raise ValueError("starting trajectory has severe self-penetration; it cannot seed protected repair")
        plant.set_research_contact_envelope(plant.self_contact_depths)
        plant.check_terminal_geometry(trajectory["qpos"][-1], trajectory["qvel"][-1])
    initial_metrics, _ = evaluate_trajectory(plant, objective, parent["timeline"], trajectory)
    progress(
        dict(
            stage="initialization_physics_reproduced", exact_physical_arrays=13, contact_envelope=contact_envelope
        )
    )
    baseline_metrics, baseline_errors = evaluate_trajectory(plant, objective, parent["timeline"], seed)
    original_trace = load_arrays(Path(reproduction["parent_trace"]))
    np.testing.assert_allclose(baseline_errors, original_trace["landmark_error_m"], atol=1e-12, rtol=0)
    del original_trace
    cost = objective.cost(trajectory, controls, seed_controls)
    baseline_cost = objective.cost(seed, seed_controls, seed_controls)
    initial_cost = cost
    resumed_objective_exact = False
    if previous_report is not None and previous_report["objective_contract"].get("kind") == (
        objective.contract()["kind"] if args.objective == "protected" else "original_squared_motion_v1"
    ):
        if cost != previous_report["final_cost"]:
            raise ValueError("resumed original full objective does not reproduce exactly")
        resumed_objective_exact = True
    guide = (
        EntryGuidance(
            model,
            load_arrays(reference_path),
            parent["timeline"],
            lift_weight=25000.0 if args.entry_guidance in ("com_lift", "com_lift_placement") else 0.0,
            foot_xy_weight=25000.0 if args.entry_guidance == "com_lift_placement" else 0.0,
        )
        if args.entry_guidance != "none"
        else None
    )
    standing = (
        StandingVelocityGuidance(
            model,
            load_arrays(reference_path),
            parent["timeline"],
            orientation_weight=1000.0 if args.standing_orientation_guidance else 0.0,
        )
        if args.standing_velocity_guidance
        else None
    )
    progress(
        dict(stage="initial", complete_controls=objective.count, cost=cost, original_baseline_cost=baseline_cost)
    )
    iterations, last_trace = [], seed_path if previous_report is None else candidate_path
    for iteration in range(1, args.iterations + 1):
        local = linearize_trajectory(
            plant, objective, trajectory, controls, lambda event: progress({"iteration": iteration, **event})
        )
        linearization_evidence = None
        if args.cache_linearizations:
            cache_path = (args.output_directory / f"linearization_{iteration:03d}.npz").resolve()
            with cache_path.open("xb") as stream:
                np.savez_compressed(stream, **local)
            bindings[str(cache_path)] = sha256_file(cache_path)
            linearization_evidence = dict(
                path=str(cache_path),
                sha256=bindings[str(cache_path)],
                nominal_trace_path=str(last_trace.resolve()),
                nominal_trace_sha256=sha256_file(last_trace),
            )
        guidance_contract = None
        if guide is not None:
            local, guide_arrays, guidance_contract = guide.augment(local, trajectory)
            guide_path = args.output_directory / f"local_transition_guidance_{iteration:03d}.npz"
            with guide_path.open("xb") as stream:
                np.savez_compressed(stream, **guide_arrays)
            bindings[str(guide_path.resolve())] = sha256_file(guide_path)
            progress(dict(stage="original_reference_entry_guidance", iteration=iteration, **guidance_contract))
        standing_contract = None
        if standing is not None:
            local, standing_arrays, standing_contract = standing.augment(local, trajectory)
            standing_path = args.output_directory / f"local_standing_velocity_guidance_{iteration:03d}.npz"
            with standing_path.open("xb") as stream:
                np.savez_compressed(stream, **standing_arrays)
            bindings[str(standing_path.resolve())] = sha256_file(standing_path)
            progress(
                dict(
                    stage="original_reference_standing_velocity_guidance", iteration=iteration, **standing_contract
                )
            )
        geometry, geometry_path = None, None
        if args.step_method == "contact_predictive":
            geometry = linearize_self_envelopes(
                model,
                trajectory,
                plant.self_contact_limits,
                lambda event: progress(dict(iteration=iteration, **event)),
            )
            geometry_path = args.output_directory / f"contact_constraints_{iteration:03d}.npz"
            with geometry_path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    control_index=np.concatenate(
                        [np.full(len(row["distance"]), i, dtype=np.int64) for i, row in enumerate(geometry)]
                    ),
                    jacobian=np.concatenate([row["jacobian"] for row in geometry]),
                    distance=np.concatenate([row["distance"] for row in geometry]),
                    floor=np.concatenate([row["floor"] for row in geometry]),
                    geoms=np.concatenate(
                        [np.asarray(row["geoms"], dtype=np.int64).reshape(-1, 2) for row in geometry]
                    ),
                )
            bindings[str(geometry_path.resolve())] = sha256_file(geometry_path)
        accepted, trials, solver_reports = False, [], []
        for regularization in args.regularizations:
            stage_models = []
            try:
                if geometry is None:
                    increments, feedback = box_backward_pass(plant, local, controls, seed_controls, regularization)
                else:
                    increments, feedback, rows = contact_backward_pass(
                        plant,
                        local,
                        controls,
                        seed_controls,
                        geometry,
                        regularization,
                        lambda event: progress(dict(iteration=iteration, **event)),
                        stage_models=stage_models,
                    )
                    solver_reports.append(dict(regularization=regularization, rows=rows))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                trial = dict(regularization=regularization, status="rejected_backward_pass", error=str(exc))
                trials.append(trial)
                progress(dict(stage="trial", iteration=iteration, **trial))
                continue
            for alpha in (2.0**-index for index in range(8)):
                trial = dict(regularization=regularization, alpha=alpha)
                state_law, reactive_stats = None, None
                if args.step_method == "contact_predictive":
                    state_law, reactive_stats = make_predictive_contact_law(
                        plant,
                        controls,
                        stage_models,
                        alpha,
                        plant.self_contact_limits,
                        clearance_m=args.planning_clearance_m,
                    )
                try:
                    candidate, targets = feedback_trial(
                        plant,
                        seed["integration_state"][0],
                        trajectory,
                        controls,
                        increments,
                        feedback,
                        alpha,
                        state_control_law=state_law,
                    )
                    if args.objective == "protected":
                        plant.check_terminal_geometry(candidate["qpos"][-1], candidate["qvel"][-1])
                    candidate_cost = objective.cost(candidate, targets, seed_controls)
                    if not np.isfinite(candidate_cost):
                        raise ValueError("nonfinite full-trajectory objective")
                    trial.update(cost=candidate_cost, complete_controls=len(targets))
                    accepted = candidate_cost < cost - max(1e-9, 1e-6 * abs(cost))
                    trial["status"] = "accepted_full_cost_reduction" if accepted else "rejected_no_cost_reduction"
                    if accepted and args.objective == "protected":
                        candidate_metrics, _ = evaluate_trajectory(plant, objective, parent["timeline"], candidate)
                        protection = protected_tracking_acceptance(initial_metrics, candidate_metrics)
                        trial["source_landmark_protection"] = protection
                        terminal_protection = terminal_nonregression(initial_metrics, candidate_metrics)
                        trial["terminal_protection"] = terminal_protection
                        if not protection["accepted"] or not terminal_protection["accepted"]:
                            accepted = False
                            trial["status"] = "rejected_source_or_terminal_nonregression"
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    accepted = False
                    trial.update(status="rejected_nonlinear_trial", error=str(exc))
                if reactive_stats is not None:
                    trial["reactive_constraint_solves"] = reactive_stats
                trials.append(trial)
                progress(dict(stage="trial", iteration=iteration, **trial))
                if accepted:
                    trajectory, controls, cost = candidate, targets, candidate_cost
                    break
            if accepted:
                break
        del local
        metrics, errors = evaluate_trajectory(plant, objective, parent["timeline"], trajectory)
        if accepted:
            last_trace = args.output_directory / f"iteration_{iteration:03d}.npz"
            with last_trace.open("xb") as stream:
                np.savez_compressed(stream, **{**trajectory, "target23": controls, "landmark_error_m": errors})
        step = dict(
            iteration=iteration,
            accepted=accepted,
            cost=cost,
            trials=trials,
            metrics=metrics,
            linearization_evidence=linearization_evidence,
            local_transition_guidance=guidance_contract,
            local_standing_velocity_guidance=standing_contract,
            constrained_solver_reports=solver_reports,
            contact_constraints_path=None if geometry_path is None else str(geometry_path.resolve()),
        )
        iterations.append(step)
        progress(
            dict(stage="iteration_complete", iteration=iteration, accepted=accepted, cost=cost, metrics=metrics)
        )
        if not accepted:
            break
    plant.assert_unchanged()
    assert_bindings(bindings)
    report = dict(
        kind="g1_true23_offline_actual_pd_trajectory_optimization_v2",
        inputs=bindings,
        reproduction_report_sha256=sha256_file(reproduction_path),
        original_requested_motion_path=str(reference_path),
        original_requested_motion_sha256=sha256_file(reference_path),
        original_policy_identity=reproduction["original_policy_identity"],
        compiled_physics_sha256=plant.model_sha,
        complete_controls=objective.count,
        baseline_cost=baseline_cost,
        initial_cost=initial_cost,
        initial_metrics=initial_metrics,
        historical_initialization_evidence=historical_evidence,
        initialization_physical_arrays_reproduced_exactly=13,
        resumed_original_objective_reproduced_exactly=resumed_objective_exact,
        initial_research_self_contact_envelope=contact_envelope,
        terminal_contact_geometry_checked=args.objective == "protected",
        terminal_nonregression_required=args.objective == "protected",
        final_cost=cost,
        baseline_metrics=baseline_metrics,
        iterations=iterations,
        candidate_trace_path=str(last_trace.resolve()),
        candidate_trace_sha256=sha256_file(last_trace),
        objective_contract=dict(
            **(objective.contract() if args.objective == "protected" else dict(kind="original_squared_motion_v1")),
            control_weight=CONTROL_WEIGHT,
            slew_weight=SLEW_WEIGHT,
            original_reference_unchanged=True,
            original_timing_unchanged=True,
        ),
        local_step_solver="box_active_set_with_coupled_free_variable_feedback"
        if args.step_method == "box"
        else "contact_backward_with_actual_state_substep_prediction",
        local_transition_guidance_mode=args.entry_guidance,
        local_standing_velocity_guidance=args.standing_velocity_guidance,
        local_standing_orientation_guidance=args.standing_orientation_guidance,
        forward_extra_planning_clearance_m=args.planning_clearance_m if args.step_method != "box" else None,
        actual_physical_contact_guard_and_tolerance_unchanged=True,
        local_step_regularizations=args.regularizations,
        line_search_alphas=[2.0**-index for index in range(8)],
        full_future_reference_used_offline=True,
        not_a_sonic_policy_or_causal_teleop_controller=True,
        original_bounded_pd_command_codec_used=True,
        offline_target_projection="nearest_original_float32_codec_lattice_with_exact_emission_witness",
        external_forces_or_feedforward_used=False,
        state_writes_after_each_independent_trial_initial_condition=0,
        admissible_training_parent=False,
        simulator_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (args.output_directory / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    progress(
        dict(
            stage="complete",
            initial_cost=initial_cost,
            final_cost=cost,
            candidate_trace_path=report["candidate_trace_path"],
            deployment_ready=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
