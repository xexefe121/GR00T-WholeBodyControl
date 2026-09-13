"""Bounded offline native23 iLQR probes, with native manual-PD physical execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_ilqr_core import EPS, NoFeasiblePlan, ilqr
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy, preview_native_control
from gear_sonic.utils.g1_true23_mjbatch_mpc import (
    TRACKED,
    Native23Tracker,
    load_motion_override,
    load_native_bundle,
    motion_states,
    sha256,
)


def atomic_trace(path, trace, **extra):
    """Replace one self-contained archive; an interrupted write keeps its predecessor."""
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{name: np.asarray(value) for name, value in trace.items()}, **extra)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def finite_json(value):
    """Keep nonfinite failure evidence in NPZ while making reports valid JSON."""
    if isinstance(value, dict):
        return {name: finite_json(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def run(args):
    if min(args.horizon, args.commit, args.iterations, args.threads) <= 0 or args.commit > args.horizon:
        raise ValueError("invalid bounded MPC experiment")
    if args.feedback_clip < 0 or args.checkpoint_controls < 0:
        raise ValueError("feedback clip and checkpoint interval must be nonnegative")
    fresh_onnx = getattr(args, "fresh_bfm_seed_onnx", None)
    fresh_dependencies = getattr(args, "fresh_bfm_seed_dependencies", None)
    hard_feasibility = bool(getattr(args, "hard_feasibility", False))
    restoration_enabled = bool(getattr(args, "restoration", False))
    if restoration_enabled and (not hard_feasibility or args.horizon != 30):
        raise ValueError("restoration requires explicit hard feasibility and H30")
    if restoration_enabled:
        from gear_sonic.utils.g1_true23_mjbatch_restoration import restore_feasible_seed
    if fresh_dependencies is not None and fresh_onnx is None:
        raise ValueError("fresh BFM dependency directory requires the explicit seed option")
    args.output.mkdir(parents=True, exist_ok=False)
    native, contract, motion, timeline, manifest = load_native_bundle(args.bundle, args.clip)
    base_motion = motion
    base_states = motion_states(base_motion)
    motion_override = None
    if args.motion_override is not None:
        motion, motion_override = load_motion_override(
            args.motion_override, args.bundle, args.clip, native, contract, base_motion, timeline, manifest
        )
    kp, kd, effort, velocity_limits = [
        np.asarray(contract[name]) for name in ("kp", "kd", "native_effort", "native_velocity")
    ]
    servo = position_servo_copy(native, kp, kd, effort)
    planner = Native23Tracker(
        servo,
        contract,
        motion,
        horizon=args.horizon,
        threads=args.threads,
        ankle_limit_margin=args.ankle_limit_margin,
        ankle_limit_weight=args.ankle_limit_weight,
        all_joint_limit_margin=args.all_joint_limit_margin,
        all_joint_limit_weight=args.all_joint_limit_weight,
        relative_foot_weight=args.relative_foot_weight,
        fd_epsilon=args.fd_epsilon,
        hard_feasibility=hard_feasibility,
    )
    reference_states = motion_states(motion)
    phase = next(phase for phase in timeline["phases"] if phase["name"] == "source_motion")
    if args.probe == "standing":
        requested = int(round(args.standing_seconds * 50))
        initial = next(item for item in timeline["phases"] if item["name"] == "initial_standing")
        if requested > initial["control_stop"]:
            raise ValueError("standing probe cannot include the acquisition or source phases")
    elif args.probe == "full-lifecycle":
        requested = timeline["total_requested_controls"]
    else:
        requested = phase["control_start"] + int(round(args.source_seconds * 50))
    requested = min(requested, len(reference_states) - 11)
    if requested <= 0:
        raise ValueError("empty MPC probe")
    seed_targets = None
    seed_contract = None
    if args.target_seed is not None:
        seed_trace = args.target_seed / "trace.npz"
        seed_report = args.target_seed / "report.json"
        seed_summary = json.loads(seed_report.read_text())
        if seed_summary["clip"] != args.clip or seed_summary["failure"] is not None:
            raise ValueError("seed must be a successful replay of this clip")
        with np.load(seed_trace, allow_pickle=False) as seed:
            seed_targets = seed["target"].copy()
            if seed_targets.shape != (len(reference_states) - 11, 23) or not np.isfinite(seed_targets).all():
                raise ValueError("seed must cover the entire original lifecycle with finite native23 targets")
            np.testing.assert_allclose(seed["qpos"][0], base_states[10, :30], atol=1e-12, rtol=0)
            np.testing.assert_allclose(seed["qvel"][0], base_states[10, 30:], atol=1e-12, rtol=0)
            # Verify actual source clock and native joint order independently of
            # a filename: each saved error must use this exact post-control pose.
            np.testing.assert_allclose(
                seed["joint_error"], seed["qpos"][1:, 7:] - base_motion["joint_pos"][11:], atol=1e-12, rtol=0
            )
            np.testing.assert_allclose(
                seed["root_error"], seed["qpos"][1:, :3] - base_motion["body_pos_w"][11:, 0], atol=1e-12, rtol=0
            )
        np.testing.assert_array_equal(seed_targets, np.clip(seed_targets, planner.lo, planner.hi))
        seed_contract = dict(
            path=str(args.target_seed),
            trace_sha256=sha256(seed_trace),
            report_sha256=sha256(seed_report),
            controls=len(seed_targets),
            original_generator_preview_seconds=float(seed_summary["goal_buffer_ms"]) / 1000,
            selection="lower actual-rollout cost of recorded BFM targets and shifted previous MPC plan",
            physical_states_copied=False,
            seed_reference_differs_from_optimization_goal=motion_override is not None,
        )
    fresh_seed = None
    fresh_seed_contract = None
    if fresh_onnx is not None:
        from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import BFMSeedRolloutError, Native23BFMRolloutSeed

        fresh_seed = Native23BFMRolloutSeed(
            native, contract, base_motion, fresh_onnx, dependency_directory=fresh_dependencies, threads=1
        )
        fresh_seed_contract = fresh_seed.identity()
        fresh_seed_contract.update(
            original_reference_sha256=sha256(args.bundle / args.clip / "native_original.npz"),
            original_goal_differs_from_optimization_goal=motion_override is not None,
            selection="minimum actual-rollout cost among shifted, recorded, and fresh BFM candidates",
        )
        if seed_contract is not None:
            seed_contract["selection"] = fresh_seed_contract["selection"]
    seed_preview = max(
        seed_contract["original_generator_preview_seconds"] if seed_contract else 0.0,
        0.14 if fresh_seed is not None else 0.0,
    )
    inputs = [
        args.bundle / "manifest.json",
        args.bundle / "contract.json",
        args.bundle / args.clip / "native_original.npz",
        args.bundle / args.clip / "timeline.json",
        Path(__file__),
    ]
    utility = Path(__file__).resolve().parents[1] / "utils"
    inputs += [
        utility / name
        for name in (
            "g1_true23_mjbatch_ilqr_core.py",
            "g1_true23_mjbatch_model.py",
            "g1_true23_mjbatch_mpc.py",
            "g1_true23_relative_foot_cost.py",
        )
    ]
    if fresh_seed is not None:
        inputs += [utility / name for name in (
            "g1_true23_mjbatch_bfm_seed.py", "g1_true23_bfm_seed_observations.py"
        )]
    if restoration_enabled:
        inputs += [utility / name for name in (
            "g1_true23_mjbatch_restoration.py", "g1_true23_feasibility_referee.py"
        )]
    request = dict(
        kind="offline_native23_mjbatch_ilqr_probe",
        clip=args.clip,
        probe=args.probe,
        requested_controls=requested,
        source_requested_controls=phase["requested_controls"],
        source_phase=phase,
        horizon=args.horizon,
        commit=args.commit,
        iterations=args.iterations,
        finite_difference_epsilon=planner.fd_epsilon,
        finite_difference_scope=(
            "generalized tangent state and joint targets; dynamics and cost use the same instance value"
        ),
        batch_threads=args.threads,
        feedback_correction_clip_rad=args.feedback_clip,
        feedback_correction_clip_zero_means_disabled=True,
        checkpoint_controls=args.checkpoint_controls,
        graceful_stop_marker=str(args.output / "STOP_REQUESTED"),
        declared_future_preview_seconds=args.horizon * 0.02,
        conservative_effective_source_preview_seconds=args.horizon * 0.02 + seed_preview,
        effective_preview_field_scope="supplied reference packet fields",
        conservative_effective_raw_pose_support_seconds=args.horizon * 0.02 + seed_preview + 0.02,
        additional_raw_pose_support_scope="one sample for upstream central joint/linear derivatives",
        recorded_target_seed=seed_contract,
        fresh_bfm_seed=fresh_seed_contract,
        motion_override=motion_override,
        physical_initialization="optimization_reference_frame10",
        initial_qpos=reference_states[10, :30].tolist(),
        initial_qvel=reference_states[10, 30:].tolist(),
        exact_solver_reuse=dict(skip_unused_final_derivatives=True, reuse_selected_initial_rollout=True),
        source_clock_hz=50,
        physics_hz=500,
        source_reference_lift_m=0.0,
        source_reference_lift_scope="extra producer transform after the explicitly validated motion override",
        reference_floor_transform=motion_override.get("reference_floor_transform") if motion_override else None,
        source_frame_removal=0,
        input_hashes={str(path): sha256(path) for path in inputs},
        model_manifest=manifest,
        actual_execution="manual native clipped PD every2ms",
        physical_engine_guard="every2ms: finite state, expected cumulative clock, and zero warning counters",
        planning_actuators="algebraically equivalent affine PD, separately validated",
        costs=dict(
            tracked_bodies=TRACKED,
            body_position=planner.position_weights.tolist(),
            body_rotation=planner.rotation_weights.tolist(),
            joint=planner.joint_weight,
            generalized_velocity=planner.velocity_weights.tolist(),
            near_joint_limit=planner.limit_weight,
            joint_limit_margin=planner.limit_margin,
            ankle_limit_override=planner.ankle_limit_override,
            all_joint_limit_override=planner.all_joint_limit_override,
            root_relative_foot_position=planner.relative_foot_weight,
            root_relative_foot_position_frame="same world axes, ankle body origin minus pelvis body origin",
            target_regularization=planner.control_weight,
            native_joint_speed_margin_cost=planner.speed_weight,
        ),
        mujoco=mujoco.__version__,
        hardware_authorized=False,
        deployment_ready=False,
        received_stream_controller=False,
        feedback_gain_semantics="last_iLQR_backward_sweep_local_feedback_reused_about_accepted_trajectory",
    )
    if hard_feasibility:
        request.update(
            hard_feasibility=dict(
                enabled=True, **planner.feasibility.contract(),
                rollout_check="every2ms including initial state; infeasible line-search candidates rejected",
                seed_policy="require feasible incumbent; no_feasible_seed means abort, not recovery",
                imminent_execution_check="10 native manualPD substeps from complete private MjData copy",
                fallback_policy="none; rejection stops experiment before rejected control is applied",
                actual_expected_clock="independent expected += timestep per scheduled physics step",
                ideal_clock_roundoff="reported separately; step_count*timestep is not the strict clock comparator",
            ),
            feedback_gain_semantics="last accepted rollout's generating backward sweep; rejected gains discarded",
        )
    if restoration_enabled:
        request["restoration"] = dict(
            enabled=True, trigger="only when all standard seeds are infeasible",
            policy="guided first; one K0 rollout retry from same original shifted seed on certificate failure",
            maximum_iterations_per_attempt=10, backward_and_alpha_schedule_unchanged=True,
            acceptance="ordinary final proposal must pass strict nominal and full native-MjData 300-step oracle",
            merit="joint interior .02/.02; speed .9/.1cap; height .30/.05; tilt1.1/.1; target weight1e-4",
            main_tracking_objective_changed=False, full_recovery_qualified=False,
        )
        request["hard_feasibility"]["fallback_policy"] = request["restoration"]["policy"]
        request["hard_feasibility"]["seed_policy"] = "certify restoration if standard seeds fail; otherwise abort"
    (args.output / "request.json").write_text(json.dumps(request, indent=2))
    for path in (path for path in inputs if path.suffix == ".py"):
        (args.output / (path.stem + "_snapshot.py")).write_bytes(path.read_bytes())
    data = mujoco.MjData(native)
    data.qpos[:], data.qvel[:] = reference_states[10, :30], reference_states[10, 30:]
    mujoco.mj_forward(native, data)
    trace = {
        name: []
        for name in (
            "qpos",
            "qvel",
            "target",
            "planned_target",
            "planned_state",
            "feedback_gain",
            "feedback_correction_raw",
            "feedback_correction_applied",
            "source_frame",
            "joint_error",
            "root_error",
            "body_position_error",
            "physics_qpos",
            "physics_qvel",
            "physics_requested_torque",
            "physics_torque",
            "range_excess",
            "velocity_ratio",
            "effort_ratio",
            "physics_substeps",
        )
    }
    if fresh_seed is not None:
        trace.update(fresh_seed_previous_action=[], fresh_seed_measured_history=[])
    if hard_feasibility:
        trace.update(physics_actuator_force=[], physics_time=[float(data.time)],
                     physics_expected_time=[float(data.time)],
                     physics_warning_number=[data.warning.number.copy()],
                     physics_warning_lastinfo=[data.warning.lastinfo.copy()])
    for name, value in (
        ("qpos", data.qpos),
        ("qvel", data.qvel),
        ("physics_qpos", data.qpos),
        ("physics_qvel", data.qvel),
    ):
        trace[name].append(value.copy())
    plans = []
    restoration_events, restorer = [], None
    warm_targets = None
    completed = 0
    next_checkpoint = args.checkpoint_controls
    failure = None
    expected_physics_time = float(data.time)
    wall_started = time.perf_counter()
    while completed < requested:
        if (args.output / "STOP_REQUESTED").exists():
            failure = dict(kind="requested_stop", completed_controls=completed, time=float(data.time))
            break
        planner.window(completed + 10)
        if warm_targets is None:
            warm_targets = planner.target_reference(np.arange(args.horizon))
        else:
            warm_targets = np.concatenate(
                (
                    warm_targets[args.commit :],
                    planner.target_reference(np.arange(args.horizon - args.commit, args.horizon)),
                )
            )
        actual_state = np.r_[data.qpos, data.qvel]
        original_shifted_targets = warm_targets.copy() if restoration_enabled else None
        tick = time.perf_counter()
        seed_selection = None
        cached_rollout = None
        seed_feasibility = {}
        if seed_targets is not None:
            seed_indices = np.minimum(completed + np.arange(args.horizon), len(seed_targets) - 1)
            recorded_targets = seed_targets[seed_indices]
            recorded_states, _, recorded_costs = planner.rollout(actual_state, recorded_targets)
            if hard_feasibility:
                seed_feasibility["recorded_bfm"] = planner.last_rollout_feasibility
            warm_states, _, warm_costs = planner.rollout(actual_state, warm_targets)
            if hard_feasibility:
                seed_feasibility["shifted_mpc"] = planner.last_rollout_feasibility
            use_recorded = recorded_costs[0] < warm_costs[0]
            seed_selection = dict(
                selected="recorded_bfm" if use_recorded else "shifted_mpc",
                recorded_cost=float(recorded_costs[0]) if np.isfinite(recorded_costs[0]) else None,
                warm_cost=float(warm_costs[0]) if np.isfinite(warm_costs[0]) else None,
            )
            if use_recorded:
                warm_targets = recorded_targets.copy()
                cached_rollout = (recorded_states[:, 0], recorded_costs[0])
            else:
                cached_rollout = (warm_states[:, 0], warm_costs[0])
        if fresh_seed is not None:
            if cached_rollout is None:
                warm_states, _, warm_costs = planner.rollout(actual_state, warm_targets)
                if hard_feasibility:
                    seed_feasibility["shifted_mpc"] = planner.last_rollout_feasibility
                cached_rollout = (warm_states[:, 0], warm_costs[0])
                seed_selection = dict(
                    selected="shifted_mpc", recorded_cost=None,
                    warm_cost=float(warm_costs[0]) if np.isfinite(warm_costs[0]) else None,
                )
            try:
                fresh_targets, fresh_diagnostics = fresh_seed.propose(
                    completed, data.qpos, data.qvel, horizon=args.horizon
                )
            except BFMSeedRolloutError as error:
                seed_selection.update(
                    fresh_cost=None, fresh_rejected=True,
                    fresh_rejection=dict(message=str(error), diagnostics=error.diagnostics),
                )
            else:
                fresh_states, _, fresh_costs = planner.rollout(actual_state, fresh_targets)
                if hard_feasibility:
                    seed_feasibility["fresh_bfm"] = planner.last_rollout_feasibility
                seed_selection.update(
                    fresh_cost=float(fresh_costs[0]) if np.isfinite(fresh_costs[0]) else None,
                    fresh_diagnostics=fresh_diagnostics,
                )
                if np.isfinite(fresh_costs[0]) and (
                    not np.isfinite(cached_rollout[1]) or fresh_costs[0] < cached_rollout[1]
                ):
                    warm_targets = fresh_targets.copy()
                    cached_rollout = (fresh_states[:, 0], fresh_costs[0])
                    seed_selection["selected"] = "fresh_bfm"
        restoration_event = None
        if restoration_enabled:
            if cached_rollout is None:
                seed_states, _, seed_costs = planner.rollout(actual_state, warm_targets)
                cached_rollout = (seed_states[:, 0], seed_costs[0])
                seed_feasibility["shifted_mpc"] = planner.last_rollout_feasibility
            if not np.isfinite(cached_rollout[1]):
                def save_proposal(mode, targets):
                    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
                    integration = np.empty(mujoco.mj_stateSize(native, spec))
                    mujoco.mj_getState(native, data, integration, spec)
                    path = args.output / ("restoration_%05d_%s.npz" % (completed, mode))
                    history = {} if fresh_seed is None else {
                        "previous_action": fresh_seed.previous_action,
                        "recorded_controls": np.asarray(fresh_seed.recorded_controls),
                        "actual_action_max_abs": np.asarray(fresh_seed.actual_action_max_abs),
                        "actual_action_components_outside_five": np.asarray(
                            fresh_seed.actual_action_components_outside_five),
                        **{"history_" + key: value for key, value in fresh_seed.history.data.items()},
                    }
                    np.savez_compressed(
                        path, targets=targets, initial_integration=integration, integration_state_spec=spec,
                        original_warm_targets=original_shifted_targets, control=np.asarray(completed),
                        request_sha256=np.asarray(sha256(args.output / "request.json")), **history,
                    )
                    return dict(proposal_file=path.name, proposal_sha256=sha256(path))

                restored, restoration_event, restorer = restore_feasible_seed(
                    planner, native, data, contract, motion, original_shifted_targets, completed + 10,
                    restorer=restorer, threads=args.threads, retry_zero_feedback=True, save_proposal=save_proposal,
                )
                restoration_event["control"] = completed
                restoration_events.append(restoration_event)
                (args.output / "restoration_events.json").write_text(json.dumps(
                    finite_json(restoration_events), indent=2, allow_nan=False))
                if restored is not None:
                    cached_rollout, warm_targets = (restored[2], restored[0]), restored[3]
                    if seed_selection is None:
                        seed_selection = {}
                    seed_selection["selected"] = "restoration"
        try:
            planned_states, planned_targets, gains, cost = ilqr(
                planner, actual_state, warm_targets.copy(), iters=args.iterations, initial_rollout=cached_rollout
            )
        except NoFeasiblePlan as error:
            failure = dict(
                kind="no_feasible_seed", control=completed, executed_rejected_controls=0,
                seed_selection=seed_selection, seed_feasibility=seed_feasibility,
                solver_feasibility=error.diagnostics, recovery_success=False,
                restoration=restoration_event,
                solve_ms=(time.perf_counter() - tick) * 1000,
            )
            break
        solve_ms = (time.perf_counter() - tick) * 1000
        if not np.isfinite(cost) or not np.isfinite(planned_targets).all() or not np.isfinite(gains).all():
            failure = dict(kind="nonfinite_or_invalid_plan", control=completed)
            break
        count = min(args.commit, requested - completed)
        plans.append(
            dict(
                control=completed,
                solve_ms=solve_ms,
                cost=float(cost),
                source_first_state_frame=completed + 10,
                source_last_state_frame=min(completed + 10 + args.horizon, len(reference_states) - 1),
                controls_committed=count,
                seed_selection=seed_selection,
            )
        )
        if hard_feasibility:
            plans[-1].update(seed_feasibility=seed_feasibility,
                             solver_feasibility=planner.last_solve_feasibility,
                             imminent_control_checks=[], controls_executed=0)
        if restoration_event is not None:
            plans[-1]["restoration"] = restoration_event
        warm_targets = planned_targets.copy()
        for local in range(count):
            frame = completed + 11
            actual_state = np.r_[data.qpos, data.qvel]
            raw_correction = (
                gains[local] @ planner.difference(planned_states[local : local + 1], actual_state[None])[0]
            )
            correction = (
                np.clip(raw_correction, -args.feedback_clip, args.feedback_clip)
                if args.feedback_clip
                else raw_correction
            )
            target = np.clip(planned_targets[local] + correction, planner.lo, planner.hi)
            if hard_feasibility:
                witness, _, _ = preview_native_control(native, data, target, contract, planner.feasibility)
                plans[-1]["imminent_control_checks"].append(dict(control=completed, **witness))
                if not witness["feasible"]:
                    failure = dict(kind="imminent_control_infeasible", control=completed,
                                   executed_rejected_controls=0, recovery_success=False, witness=witness)
                    break
            if fresh_seed is not None:
                trace["fresh_seed_previous_action"].append(fresh_seed.previous_action.copy())
                trace["fresh_seed_measured_history"].append(np.concatenate([
                    fresh_seed.history.data[key].reshape(-1) for key in sorted(fresh_seed.history.data)
                ]))
                fresh_seed.record_control(completed, data.qpos, data.qvel, target)
            peak_range = peak_effort = peak_velocity = 0.0
            substeps = 0
            for _ in range(10):
                desired_torque = kp * (target - data.qpos[7:]) - kd * data.qvel[6:]
                data.ctrl[:] = np.clip(desired_torque, -effort, effort)
                mujoco.mj_step(native, data)
                if hard_feasibility:
                    expected_physics_time += native.opt.timestep
                substeps += 1
                for name, value in (
                    ("physics_qpos", data.qpos),
                    ("physics_qvel", data.qvel),
                    ("physics_requested_torque", desired_torque),
                    ("physics_torque", data.ctrl),
                ):
                    trace[name].append(value.copy())
                if hard_feasibility:
                    trace["physics_actuator_force"].append(data.qfrc_actuator[6:].copy())
                    trace["physics_time"].append(float(data.time))
                    trace["physics_expected_time"].append(expected_physics_time)
                    trace["physics_warning_number"].append(data.warning.number.copy())
                    trace["physics_warning_lastinfo"].append(data.warning.lastinfo.copy())
                peak_range = max(
                    peak_range, float(np.max(np.maximum(planner.lo - data.qpos[7:], data.qpos[7:] - planner.hi)))
                )
                peak_velocity = max(peak_velocity, float(np.max(np.abs(data.qvel[6:]) / velocity_limits)))
                peak_effort = max(peak_effort, float(np.max(np.abs(data.qfrc_actuator[6:]) / effort)))
                substep_tilt = float(np.arccos(np.clip(1 - 2 * np.sum(data.qpos[4:6] ** 2), -1, 1)))
                expected_time = (expected_physics_time if hard_feasibility
                                 else len(trace["physics_torque"]) * native.opt.timestep)
                if hard_feasibility:
                    invalid, reasons, metrics = planner.feasibility.assess(
                        data.qpos, data.qvel, force=data.qfrc_actuator[None, 6:],
                        warning=data.warning.number[None], time=np.array([data.time]),
                        expected_time=expected_time,
                    )
                    if invalid[0]:
                        failure = dict(
                            kind="hard_physical_feasibility", control=completed + 1, substep=substeps,
                            witness=planner.feasibility.witness(0, reasons, metrics),
                            expected_time=float(expected_time),
                            simulation_time=float(data.time) if np.isfinite(data.time) else None,
                        )
                        break
                if abs(float(data.time) - expected_time) > 1e-8 or np.any(data.warning.number):
                    failure = dict(
                        kind="engine_warning_or_clock_reset", control=completed + 1, substep=substeps,
                        simulation_time=float(data.time), expected_time=float(expected_time),
                        warning_counts=data.warning.number.tolist(),
                    )
                elif (
                    not np.isfinite(data.qpos).all()
                    or not np.isfinite(data.qvel).all()
                    or data.qpos[2] < 0.25
                    or substep_tilt > 1.2
                ):
                    failure = dict(
                        kind="fall_or_nonfinite",
                        control=completed + 1,
                        substep=substeps,
                        time=float(data.time),
                        height=float(data.qpos[2]),
                        tilt=substep_tilt,
                    )
                elif (
                    peak_range > (1e-6 if hard_feasibility else 0.01)
                    or peak_velocity > 1 or peak_effort > 1 + (1e-9 if hard_feasibility else 1e-8)
                ):
                    failure = dict(
                        kind="physical_limit",
                        control=completed + 1,
                        substep=substeps,
                        time=float(data.time),
                        range_excess=peak_range,
                        velocity_ratio=peak_velocity,
                        effort_ratio=peak_effort,
                    )
                if failure:
                    break
            mujoco.mj_kinematics(native, data)
            source_ids = np.asarray(planner.ids) - 1
            values = dict(
                qpos=data.qpos.copy(),
                qvel=data.qvel.copy(),
                target=target.copy(),
                planned_target=planned_targets[local].copy(),
                planned_state=planned_states[local].copy(),
                feedback_gain=gains[local].copy(),
                feedback_correction_raw=raw_correction.copy(),
                feedback_correction_applied=correction.copy(),
                source_frame=frame,
                joint_error=data.qpos[7:] - motion["joint_pos"][frame],
                root_error=data.qpos[:3] - motion["body_pos_w"][frame, 0],
                body_position_error=np.linalg.norm(
                    data.xpos[planner.ids] - motion["body_pos_w"][frame, source_ids], axis=-1
                ),
                range_excess=peak_range,
                velocity_ratio=peak_velocity,
                effort_ratio=peak_effort,
                physics_substeps=substeps,
            )
            for name, value in values.items():
                trace[name].append(value)
            completed += 1
            if hard_feasibility:
                plans[-1]["controls_executed"] += 1
            tilt = float(np.arccos(np.clip(data.xmat[1, 8], -1, 1)))
            if failure is None and (not np.isfinite(data.qpos).all() or data.qpos[2] < 0.25 or tilt > 1.2):
                failure = dict(kind="fall", control=completed, height=float(data.qpos[2]), tilt=tilt)
            elif failure is None and (peak_range > 0.01 or peak_velocity > 1):
                failure = dict(
                    kind="physical_limit", control=completed, range_excess=peak_range, velocity_ratio=peak_velocity
                )
            if failure:
                break
        print(
            json.dumps(
                dict(
                    controls=completed,
                    requested=requested,
                    solve_ms=solve_ms,
                    cost=float(cost),
                    height=float(data.qpos[2]),
                    source_frame=completed + 10,
                )
            ),
            flush=True,
        )
        if args.checkpoint_controls and completed and (completed >= next_checkpoint or failure):
            source_slice = slice(phase["control_start"], min(completed, phase["control_stop"]))
            substep_counts = np.asarray(trace["physics_substeps"])
            checkpoint = dict(
                kind="incomplete_native23_mpc_prefix_checkpoint",
                completed_controls=completed,
                completed_full_controls=int(np.count_nonzero(substep_counts == 10)),
                partial_controls=int(np.count_nonzero(substep_counts != 10)),
                requested_controls=requested,
                completed_source_controls=len(substep_counts[source_slice]),
                completed_source_full_controls=int(np.count_nonzero(substep_counts[source_slice] == 10)),
                requested_source_controls=phase["requested_controls"],
                reference_frame_first=int(trace["source_frame"][0]),
                reference_frame_last=int(trace["source_frame"][-1]),
                physics_steps=len(trace["physics_torque"]),
                simulation_time=float(data.time),
                engine_warning_counts=data.warning.number.tolist(),
                failure=failure,
                plans=plans,
                request_sha256=sha256(args.output / "request.json"),
                is_final_result=False,
                resume_implementation_qualified=False,
            )
            if hard_feasibility:
                checkpoint = finite_json(checkpoint)
            atomic_trace(
                args.output / "trace.partial.npz",
                trace,
                checkpoint_metadata=np.asarray(json.dumps(checkpoint)),
                checkpoint_warm_targets=warm_targets,
                checkpoint_qacc_warmstart=data.qacc_warmstart.copy(),
                checkpoint_ctrl=data.ctrl.copy(),
            )
            next_checkpoint = completed + args.checkpoint_controls
        if failure:
            break
    arrays = {name: np.asarray(values) for name, values in trace.items()}
    if hard_feasibility:
        trailing_shapes = dict(
            qpos=(30,), qvel=(29,), target=(23,), planned_target=(23,), planned_state=(59,),
            feedback_gain=(23, 58), feedback_correction_raw=(23,), feedback_correction_applied=(23,),
            joint_error=(23,), root_error=(3,), body_position_error=(len(TRACKED),),
            physics_qpos=(30,), physics_qvel=(29,), physics_requested_torque=(23,), physics_torque=(23,),
            physics_actuator_force=(23,), physics_warning_number=(8,), physics_warning_lastinfo=(8,),
            fresh_seed_previous_action=(23,), fresh_seed_measured_history=(300,),
        )
        for name, shape in trailing_shapes.items():
            if name in arrays and not arrays[name].size:
                arrays[name] = arrays[name].reshape((0, *shape))
    atomic_trace(args.output / "trace.npz", arrays)
    (args.output / "plans.json").write_text(json.dumps(plans, indent=2))
    source_start, source_stop = phase["control_start"], min(completed, phase["control_stop"])
    metric_start, metric_stop = (source_start, source_stop) if source_stop > source_start else (0, completed)
    metrics = None
    if metric_stop > metric_start:
        indices = slice(metric_start, metric_stop)
        metrics = dict(
            phase="source_motion" if source_stop > source_start else "standing_prefix",
            controls=metric_stop - metric_start,
            leg_rmse=float(np.sqrt(np.mean(arrays["joint_error"][indices, :12] ** 2))),
            arm_rmse=float(np.sqrt(np.mean(arrays["joint_error"][indices, 13:] ** 2))),
            root_p95=float(np.percentile(np.linalg.norm(arrays["root_error"][indices], axis=-1), 95)),
            tracked_body_position_p95=np.percentile(arrays["body_position_error"][indices], 95, axis=0).tolist(),
            tracked_body_position_frame="absolute world coordinates; this is not the root-relative foot gate",
        )
    result = dict(
        kind="offline_native23_mjbatch_ilqr_probe",
        clip=args.clip,
        probe=args.probe,
        completed_controls=completed,
        requested_controls=requested,
        probe_completed=completed == requested and failure is None,
        full_source_completed=completed >= phase["control_stop"] and failure is None,
        failure=failure,
        metrics=metrics,
        range_excess_max=float(arrays["range_excess"].max()) if completed else None,
        velocity_ratio_max=float(arrays["velocity_ratio"].max()) if completed else None,
        effort_ratio_max=float(arrays["effort_ratio"].max()) if completed else None,
        feedback_correction_clip_rad=args.feedback_clip,
        feedback_correction_clipped_controls=int(
            np.count_nonzero(
                np.any(arrays["feedback_correction_raw"] != arrays["feedback_correction_applied"], axis=1)
            )
        )
        if completed
        else 0,
        planning_ms_p50_p95_max=np.percentile([plan["solve_ms"] for plan in plans], [50, 95, 100]).tolist()
        if plans
        else None,
        planning_deadlines_missed=sum(plan["solve_ms"] > plan["controls_committed"] * 20 for plan in plans),
        elapsed_wall_seconds=time.perf_counter() - wall_started,
        simulated_seconds=float(data.time),
        engine_warning_counts=data.warning.number.tolist(),
        physics_steps=len(trace["physics_torque"]),
        declared_future_preview_seconds=args.horizon * 0.02,
        conservative_effective_source_preview_seconds=request["conservative_effective_source_preview_seconds"],
        effective_preview_field_scope=request["effective_preview_field_scope"],
        conservative_effective_raw_pose_support_seconds=request[
            "conservative_effective_raw_pose_support_seconds"
        ],
        recorded_target_seed=seed_contract,
        fresh_bfm_seed=fresh_seed_contract,
        fresh_bfm_seed_actual_history=(
            dict(recorded_controls=fresh_seed.recorded_controls,
                 normalized_action_max_abs=fresh_seed.actual_action_max_abs,
                 normalized_action_components_outside_five=fresh_seed.actual_action_components_outside_five)
            if fresh_seed is not None else None
        ),
        motion_override=motion_override,
        mujoco=mujoco.__version__,
        reference_floor_transform=motion_override.get("reference_floor_transform") if motion_override else None,
        controller="iLQR position targets with measured-state feedback and native manualPD",
        physical_state_rewrites_after_initialization=0,
        root_assistance_forces=0,
        timing_qualified=False,
        full_body_tracking_qualified=False,
        hardware_authorized=False,
        native323_replay_required=True,
        request_sha256=sha256(args.output / "request.json"),
        trace_sha256=sha256(args.output / "trace.npz"),
        plans_sha256=sha256(args.output / "plans.json"),
    )
    if hard_feasibility:
        result.update(hard_feasibility=request["hard_feasibility"],
                      feasibility_rejection_is_success=False, recovery_qualified=False,
                      ideal_clock_roundoff_seconds=expected_physics_time
                      - len(trace["physics_torque"]) * native.opt.timestep,
                      nonfinite_report_values="null; unmodified numerical evidence retained in trace.npz")
        result = finite_json(result)
    if restoration_enabled:
        result.update(restoration=request["restoration"], restoration_events=restoration_events,
                      restoration_triggers=len(restoration_events),
                      restoration_seeds_accepted=sum(event["accepted"] for event in restoration_events),
                      restoration_zero_feedback_retries=sum(
                          attempt["mode"] == "zero_feedback" for event in restoration_events
                          for attempt in event["attempts"]))
        result = finite_json(result)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--clip", choices=("walk002", "walk003", "walk008", "pico"), default="walk002")
    parser.add_argument("--probe", choices=("standing", "source-prefix", "full-lifecycle"), default="standing")
    parser.add_argument("--standing-seconds", type=float, default=1.0)
    parser.add_argument("--source-seconds", type=float, default=3.0)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--commit", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--hard-feasibility", action="store_true",
        help="Reject every2ms infeasible rollouts and privately guard each imminent applied control",
    )
    parser.add_argument("--restoration", action="store_true",
                        help="Opt-in H30 guided restoration with one zero-feedback retry and strict certificates")
    parser.add_argument(
        "--fd-epsilon",
        type=float,
        default=EPS,
        help="Per-instance finite-difference scale for both dynamics and cost",
    )
    parser.add_argument(
        "--ankle-limit-margin", type=float, help="Explicit ankle-roll-only predicted interior margin"
    )
    parser.add_argument("--ankle-limit-weight", type=float, help="Explicit ankle-roll-only predicted limit weight")
    parser.add_argument(
        "--all-joint-limit-margin", type=float, help="Explicit common predicted interior margin for all 23 joints"
    )
    parser.add_argument(
        "--all-joint-limit-weight", type=float, help="Explicit common predicted limit weight for all 23 joints"
    )
    parser.add_argument(
        "--relative-foot-weight", type=float, default=0.0, help="Optional world-axis root-relative foot cost"
    )
    parser.add_argument(
        "--feedback-clip",
        type=float,
        default=0.0,
        help="Per-joint feedback correction bound in radians; zero disables this bound",
    )
    parser.add_argument(
        "--checkpoint-controls",
        type=int,
        default=100,
        help="Atomically save an incomplete trace prefix every N controls; zero disables",
    )
    parser.add_argument("--target-seed", type=Path, help="Optional successful BFM replay directory; targets only")
    parser.add_argument(
        "--fresh-bfm-seed-onnx", type=Path,
        help="Enable measured-history BFM third seed using these validated graphs"
    )
    parser.add_argument(
        "--fresh-bfm-seed-dependencies", type=Path, help="Explicit isolated ONNX Runtime dependency directory"
    )
    parser.add_argument(
        "--motion-override", type=Path, help="Validated retarget NPZ with sibling portable_receipt.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
