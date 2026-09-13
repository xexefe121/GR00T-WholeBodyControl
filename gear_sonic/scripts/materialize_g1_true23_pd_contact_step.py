"""Compare cached offline PD steps against the latest full replay, SIM only."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from gear_sonic.scripts.optimize_g1_true23_pd_trajectory import assert_bindings, evaluate_trajectory, load_arrays
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
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
from gear_sonic.utils.g1_true23_pd_reactive_contact import make_reactive_contact_law, terminal_nonregression
from gear_sonic.utils.g1_true23_pd_source_history import verify_historical_report
from gear_sonic.utils.g1_true23_pd_standing_guidance import StandingVelocityGuidance
from gear_sonic.utils.g1_true23_pd_trajectory_optimizer import feedback_trial
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def exact_replay(plant, state, saved):
    replay = plant.rollout(state, saved["target23"], physics_records=True)
    for key, value in replay.items():
        if (
            value.dtype != saved[key].dtype
            or value.shape != saved[key].shape
            or value.tobytes() != saved[key].tobytes()
        ):
            raise ValueError("cached contact-step input did not replay exactly: " + key)
    return replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-report", type=Path, required=True)
    parser.add_argument("--reproduction-report", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cached-iteration", type=int, default=2)
    parser.add_argument(
        "--step-method", choices=("box", "contact", "contact_reactive", "contact_predictive"), default="contact"
    )
    parser.add_argument("--regularizations", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    parser.add_argument(
        "--entry-guidance", choices=("none", "com", "com_lift", "com_lift_placement"), default="none"
    )
    parser.add_argument("--standing-velocity-guidance", action="store_true")
    parser.add_argument("--standing-orientation-guidance", action="store_true")
    parser.add_argument(
        "--planning-clearance-m",
        type=float,
        default=0.001,
        help="Extra offline planning clearance; actual physical contact guards remain unchanged.",
    )
    args = parser.parse_args(argv)
    if args.standing_orientation_guidance and not args.standing_velocity_guidance:
        parser.error("standing orientation guidance requires standing velocity guidance")
    if args.cached_iteration < 1 or any(not np.isfinite(r) or r <= 0 for r in args.regularizations):
        parser.error("positive cached iteration and finite positive regularizations required")
    if not np.isfinite(args.planning_clearance_m) or not 0 <= args.planning_clearance_m <= 0.005:
        parser.error("offline planning clearance must be between zero and five millimetres")
    if args.planning_clearance_m != 0.001 and args.step_method not in ("contact_reactive", "contact_predictive"):
        parser.error("custom planning clearance requires a reactive or predictive offline step")
    root, started = Path(__file__).resolve().parents[2], time.monotonic()
    source_report_path = args.optimizer_report.resolve(strict=True)
    source = json.loads(source_report_path.read_text())
    historical = verify_historical_report(
        source_report_path, repository_root=root, source_archive=args.source_archive
    )
    reproduction = json.loads(args.reproduction_report.read_text())
    assert_bindings(reproduction["inputs"])
    if source["reproduction_report_sha256"] != sha256_file(args.reproduction_report):
        raise ValueError("cached contact step cannot change original seed/reference")
    if source["objective_contract"]["kind"] != "g1_true23_original_motion_per_landmark_and_self_clearance_v1":
        raise ValueError("cached contact step requires the unchanged protected objective")
    step = next(row for row in source["iterations"] if row["iteration"] == args.cached_iteration)
    cache = step["linearization_evidence"]
    cache_path, nominal_path = Path(cache["path"]), Path(cache["nominal_trace_path"])
    seed_path, initial_path = Path(reproduction["trace_path"]), Path(source["candidate_trace_path"])
    reference_path, parent_path = (
        Path(source["original_requested_motion_path"]),
        Path(reproduction["parent_evaluation"]),
    )
    expected = {
        cache_path: cache["sha256"],
        nominal_path: cache["nominal_trace_sha256"],
        seed_path: reproduction["trace_sha256"],
        initial_path: source["candidate_trace_sha256"],
        reference_path: source["original_requested_motion_sha256"],
    }
    assert_bindings({str(path.resolve()): digest for path, digest in expected.items()})
    parent = json.loads(parent_path.read_text())
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    plant = ProtectedPdPlant(model, NativeModelActuationProfile.from_sim_config(root / PHYSICS))
    if plant.model_sha != source["compiled_physics_sha256"]:
        raise ValueError("cached contact step changed original physics")
    objective = ProtectedMotionObjective(plant, load_arrays(reference_path), parent["timeline"])
    paths = [
        source_report_path,
        args.reproduction_report,
        args.source_archive,
        parent_path,
        *expected,
        args.asset_root / MODEL,
        root / PHYSICS,
        *collect_local_source_closure(root, [Path(__file__)]).files,
        *(Path(row["archived_path"]) for row in historical["archived_python_sources_used"]),
    ]
    bindings = {str(path.resolve()): sha256_file(path) for path in paths}
    seed, initial, nominal, local = (
        load_arrays(seed_path),
        load_arrays(initial_path),
        load_arrays(nominal_path),
        load_arrays(cache_path),
    )
    args.output_directory.mkdir(parents=True, exist_ok=False)

    def progress(event):
        message = json.dumps(dict(elapsed_s=round(time.monotonic() - started, 3), **event), allow_nan=False)
        with (args.output_directory / "events.jsonl").open("a") as stream:
            stream.write(message + "\n")
        print(message, flush=True)

    initial_replay = exact_replay(plant, seed["integration_state"][0], initial)
    envelope = plant.contact_observations()
    limits = dict(plant.self_contact_depths)
    if min(limits.values(), default=0.0) < -0.005:
        raise ValueError("severe initial self-contact cannot seed cached repair")
    initial_metrics, _ = evaluate_trajectory(plant, objective, parent["timeline"], initial_replay)
    cost = objective.cost(initial_replay, initial["target23"], seed["target23"])
    if cost != source["final_cost"]:
        raise ValueError("cached contact-step objective differs from recorded original objective")
    del initial_replay
    # The older cached nominal is authenticated independently. It may have
    # deeper contacts than the latest candidate; it is NOT silently adopted as
    # the new research envelope or the default output.
    exact_replay(plant, seed["integration_state"][0], nominal)
    plant.set_research_contact_envelope(limits)
    plant.check_terminal_geometry(initial["qpos"][-1], initial["qvel"][-1])
    progress(
        dict(
            stage="initial_and_cached_nominal_reproduced",
            physical_arrays_each=13,
            initial_cost=cost,
            nominal_controls=len(nominal["target23"]),
            contact_envelope=envelope,
        )
    )
    guidance_contract = None
    if args.entry_guidance != "none":
        guide = EntryGuidance(
            model,
            load_arrays(reference_path),
            parent["timeline"],
            lift_weight=25000.0 if args.entry_guidance in ("com_lift", "com_lift_placement") else 0.0,
            foot_xy_weight=25000.0 if args.entry_guidance == "com_lift_placement" else 0.0,
        )
        local, guide_arrays, guidance_contract = guide.augment(local, nominal)
        guide_path = args.output_directory / "local_transition_guidance.npz"
        with guide_path.open("xb") as stream:
            np.savez_compressed(stream, **guide_arrays)
        bindings[str(guide_path.resolve())] = sha256_file(guide_path)
        progress(dict(stage="original_reference_entry_guidance", **guidance_contract))
    standing_contract = None
    if args.standing_velocity_guidance:
        standing = StandingVelocityGuidance(
            model,
            load_arrays(reference_path),
            parent["timeline"],
            orientation_weight=1000.0 if args.standing_orientation_guidance else 0.0,
        )
        local, standing_arrays, standing_contract = standing.augment(local, nominal)
        standing_path = args.output_directory / "local_standing_velocity_guidance.npz"
        with standing_path.open("xb") as stream:
            np.savez_compressed(stream, **standing_arrays)
        bindings[str(standing_path.resolve())] = sha256_file(standing_path)
        progress(dict(stage="original_reference_standing_velocity_guidance", **standing_contract))
    geometry = linearize_self_envelopes(model, nominal, limits, progress) if args.step_method != "box" else None
    if geometry is not None:
        geometry_path = args.output_directory / "contact_constraints.npz"
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
    trials, accepted, trajectory, targets, last_trace = [], False, initial, initial["target23"], initial_path
    initial_cost = cost
    solver_reports = []
    for regularization in args.regularizations:
        stage_models = []
        try:
            if geometry is not None:
                increments, gain, rows = contact_backward_pass(
                    plant,
                    local,
                    nominal["target23"],
                    seed["target23"],
                    geometry,
                    regularization,
                    progress,
                    stage_models=stage_models,
                )
                solver_reports.append(dict(regularization=regularization, rows=rows))
            else:
                increments, gain = box_backward_pass(
                    plant, local, nominal["target23"], seed["target23"], regularization
                )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            trial = dict(regularization=regularization, status="rejected_backward", error=str(exc))
            trials.append(trial)
            progress(dict(stage="trial", **trial))
            continue
        for alpha in (2.0**-index for index in range(8)):
            trial = dict(regularization=regularization, alpha=alpha)
            law, state_law, reactive_stats = None, None, None
            if args.step_method == "contact_reactive":
                law, reactive_stats = make_reactive_contact_law(
                    plant,
                    local,
                    nominal,
                    nominal["target23"],
                    stage_models,
                    alpha,
                    limits,
                    clearance_m=args.planning_clearance_m,
                )
            if args.step_method == "contact_predictive":
                state_law, reactive_stats = make_predictive_contact_law(
                    plant,
                    nominal["target23"],
                    stage_models,
                    alpha,
                    limits,
                    clearance_m=args.planning_clearance_m,
                )
            try:
                candidate, proposed = feedback_trial(
                    plant,
                    seed["integration_state"][0],
                    nominal,
                    nominal["target23"],
                    increments,
                    gain,
                    alpha,
                    control_law=law,
                    state_control_law=state_law,
                )
                plant.check_terminal_geometry(candidate["qpos"][-1], candidate["qvel"][-1])
                candidate_cost = objective.cost(candidate, proposed, seed["target23"])
                metrics, _ = evaluate_trajectory(plant, objective, parent["timeline"], candidate)
                protection = protected_tracking_acceptance(initial_metrics, metrics)
                terminal_protection = terminal_nonregression(initial_metrics, metrics)
                accepted = (
                    np.isfinite(candidate_cost)
                    and candidate_cost < cost - max(1e-9, 1e-6 * abs(cost))
                    and protection["accepted"]
                    and terminal_protection["accepted"]
                )
                trial.update(
                    cost=candidate_cost,
                    source_landmark_protection=protection,
                    terminal_protection=terminal_protection,
                    complete_controls=len(proposed),
                    status="accepted_complete_improvement"
                    if accepted
                    else "rejected_cost_source_or_terminal_nonregression",
                )
                if accepted:
                    trajectory, targets, cost = candidate, proposed, candidate_cost
            except (ValueError, RuntimeError) as exc:
                accepted = False
                trial.update(status="rejected_nonlinear", error=str(exc))
            if reactive_stats is not None:
                trial["reactive_constraint_solves"] = reactive_stats
            trials.append(trial)
            progress(dict(stage="trial", **trial))
            if accepted:
                break
        if accepted:
            break
    metrics, errors = evaluate_trajectory(plant, objective, parent["timeline"], trajectory)
    if accepted:
        last_trace = args.output_directory / "candidate.npz"
        with last_trace.open("xb") as stream:
            np.savez_compressed(stream, **{**trajectory, "target23": targets, "landmark_error_m": errors})
    assert_bindings(bindings)
    plant.assert_unchanged()
    report = dict(
        kind="g1_true23_cached_contact_constrained_pd_step_v1",
        inputs=bindings,
        reproduction_report_sha256=sha256_file(args.reproduction_report),
        original_requested_motion_path=str(reference_path),
        original_requested_motion_sha256=sha256_file(reference_path),
        original_policy_identity=source["original_policy_identity"],
        compiled_physics_sha256=plant.model_sha,
        complete_controls=objective.count,
        baseline_cost=source["baseline_cost"],
        baseline_metrics=source["baseline_metrics"],
        initial_cost=initial_cost,
        initial_metrics=initial_metrics,
        final_cost=cost,
        historical_initialization_evidence=historical,
        initialization_physical_arrays_reproduced_exactly=13,
        cached_nominal_physical_arrays_reproduced_exactly=13,
        cached_linearization=cache,
        initial_research_self_contact_envelope=envelope,
        terminal_contact_geometry_checked=True,
        terminal_nonregression_required=True,
        candidate_trace_path=str(last_trace.resolve()),
        candidate_trace_sha256=sha256_file(last_trace),
        objective_contract=source["objective_contract"],
        local_step_solver=args.step_method,
        local_transition_guidance=guidance_contract,
        local_standing_velocity_guidance=standing_contract,
        linear_contact_constraints_do_not_qualify_actual_physics=True,
        source_thresholds_and_timing_unchanged=True,
        contact_constraint_profile=dict(
            query_horizon_m=0.03,
            inequality_horizon_m=0.025,
            joint_difference_rad=1e-6,
            forward_extra_planning_clearance_m=args.planning_clearance_m,
            actual_physical_contact_guard_and_tolerance_unchanged=True,
        ),
        iterations=[dict(iteration=1, accepted=bool(accepted), cost=cost, trials=trials, metrics=metrics)],
        constrained_solver_reports=solver_reports,
        original_bounded_pd_command_codec_used=True,
        offline_target_projection="nearest_original_float32_codec_lattice_with_exact_emission_witness",
        not_a_sonic_policy_or_causal_teleop_controller=True,
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
            accepted=bool(accepted),
            initial_cost=initial_cost,
            final_cost=cost,
            metrics=metrics,
            deployment_ready=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
