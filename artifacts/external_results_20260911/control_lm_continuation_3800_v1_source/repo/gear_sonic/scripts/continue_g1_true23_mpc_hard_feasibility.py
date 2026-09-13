"""Bounded continuous hard-MPC diagnosis from the archived actual PICO state."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.scripts.evaluate_g1_true23_mjbatch_mpc import atomic_trace, finite_json
from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import BFMSeedRolloutError, Native23BFMRolloutSeed
from gear_sonic.utils.g1_true23_mjbatch_ilqr_core import NoFeasiblePlan, ilqr
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy, preview_native_control
from gear_sonic.utils.g1_true23_mjbatch_mpc import (
    Native23Tracker,
    load_motion_override,
    load_native_bundle,
    sha256,
)
from gear_sonic.utils.g1_true23_mjbatch_restoration_lm import restore_feasible_seed


def record_native_substep(native, data, target, contract, feasibility, trace,
                          expected_time, predicted_state, predicted_force):
    """Append raw actual evidence before reporting any physics/parity failure."""
    kp, kd, effort = [np.asarray(contract[name]) for name in ("kp", "kd", "native_effort")]
    data.ctrl[:] = np.clip(kp * (target - data.qpos[7:]) - kd * data.qvel[6:], -effort, effort)
    mujoco.mj_step(native, data)
    expected_time += native.opt.timestep
    for name, value in (("physics_qpos", data.qpos), ("physics_qvel", data.qvel),
                        ("physics_torque", data.ctrl), ("physics_actuator_force", data.qfrc_actuator[6:]),
                        ("physics_warning_number", data.warning.number),
                        ("physics_warning_lastinfo", data.warning.lastinfo)):
        trace[name].append(value.copy())
    trace["physics_time"].append(float(data.time))
    trace["physics_expected_time"].append(expected_time)
    invalid, reasons, metrics = feasibility.assess(
        data.qpos, data.qvel, force=data.qfrc_actuator[None, 6:], warning=data.warning.number[None],
        time=np.array([data.time]), expected_time=expected_time,
    )
    if invalid[0]:
        return expected_time, dict(kind="actual_physics_infeasible",
                                   witness=feasibility.witness(0, reasons, metrics))
    actual = np.r_[data.qpos, data.qvel]
    if not np.array_equal(actual, predicted_state) or not np.array_equal(data.qfrc_actuator[6:], predicted_force):
        return expected_time, dict(kind="private_forecast_mismatch",
                                   state_max_difference=float(np.max(np.abs(actual - predicted_state))),
                                   force_max_difference=float(np.max(
                                       np.abs(data.qfrc_actuator[6:] - predicted_force))))
    return expected_time, None


def run(args):
    if args.controls < 100:
        raise ValueError("continuous diagnosis requests at least two seconds")
    if args.restoration_zero_feedback_retry and not args.restoration:
        raise ValueError("zero-feedback retry requires the guided restoration experiment")
    if args.restoration_control_lm_retry and not args.restoration_zero_feedback_retry:
        raise ValueError("third control-LM retry requires both original restoration attempts")
    args.output.mkdir(parents=True, exist_ok=False)
    native, contract, original, timeline, manifest = load_native_bundle(args.bundle, "pico")
    motion, override = load_motion_override(
        args.motion_override, args.bundle, "pico", native, contract, original, timeline, manifest
    )
    kp, kd, effort = [np.asarray(contract[name]) for name in ("kp", "kd", "native_effort")]
    tracker = Native23Tracker(
        position_servo_copy(native, kp, kd, effort), contract, motion, horizon=30, threads=8,
        all_joint_limit_margin=.05, all_joint_limit_weight=2000, relative_foot_weight=400,
        fd_epsilon=1e-6, hard_feasibility=True,
    )
    fresh = Native23BFMRolloutSeed(native, contract, original, args.onnx,
                                  dependency_directory=args.dependencies, threads=1)
    with np.load(args.history, allow_pickle=False) as archive:
        for name in fresh.history.data:
            fresh.history.data[name][:] = archive["memory_" + name]
        fresh.previous_action = archive["previous_action"].copy()
        fresh.recorded_controls = int(archive["recorded_controls"])
        fresh.actual_action_max_abs = float(archive["actual_action_max_abs"])
        fresh.actual_action_components_outside_five = int(archive["actual_action_components_outside_five"])
    if fresh.recorded_controls != 3800:
        raise ValueError("bounded continuation fixture must start at actual control3800")
    with np.load(args.fixture, allow_pickle=False) as archive:
        initial_integration = archive["state_vector"].copy()
        state_spec = int(archive["state_spec"])
        warm = archive["targets"].copy()
    with np.load(args.recorded_seed / "trace.npz", allow_pickle=False) as archive:
        recorded = archive["target"].copy()
    recorded_report = json.loads((args.recorded_seed / "report.json").read_text())
    if recorded_report["clip"] != "pico" or recorded_report["failure"] is not None:
        raise ValueError("recorded seed is not the successful original PICO BFM rollout")
    if recorded.shape != (len(original["joint_pos"]) - 11, 23) or not np.isfinite(recorded).all():
        raise ValueError("recorded seed must preserve full original clock and native23 targets")
    np.testing.assert_array_equal(recorded, np.clip(recorded, tracker.lo, tracker.hi))
    data = mujoco.MjData(native)
    mujoco.mj_setState(native, data, initial_integration, state_spec)
    mujoco.mj_forward(native, data)
    mujoco.mj_setState(native, data, initial_integration, state_spec)
    initial_time = float(data.time)
    directory = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), directory / "scripts/evaluate_g1_true23_mjbatch_mpc.py"]
    sources += [directory / "utils" / (name + ".py") for name in (
        "g1_true23_mjbatch_model", "g1_true23_mjbatch_mpc", "g1_true23_mjbatch_ilqr_core",
        "g1_true23_relative_foot_cost", "g1_true23_mjbatch_bfm_seed", "g1_true23_bfm_seed_observations",
        "g1_true23_mjbatch_restoration_lm",
    )]
    if args.restoration:
        sources += [directory / "utils" / (name + ".py") for name in (
            "g1_true23_mjbatch_restoration", "g1_true23_feasibility_referee",
        )]
    inputs = sources + [args.fixture, args.history, args.fixture.parent / "receipt.json", args.motion_override,
                        args.recorded_seed / "trace.npz", args.recorded_seed / "report.json",
                        args.bundle / "manifest.json", args.bundle / "pico/native_original.npz"]
    inputs += [args.first_proposal_witnesses / "restoration_03800_guided.npz",
               args.first_proposal_witnesses / "restoration_03800_zero_feedback.npz",
               args.control_lm_witness, args.canonical_trace]
    with np.load(args.canonical_trace, allow_pickle=False) as archive:
        np.testing.assert_array_equal(data.qpos, archive["qpos"][-1])
        np.testing.assert_array_equal(data.qvel, archive["qvel"][-1])
    if args.parity_prefix is not None:
        inputs.append(args.parity_prefix)
    request = dict(
        kind="bounded_actual_state_continuation", actual_initial_control=3800,
        requested_controls=args.controls, requested_seconds=args.controls * .02,
        horizon=30, commit=5, iterations=5, threads=8, feedback_clip=.1, fd_epsilon=1e-6,
        all_joint_margin=.05, all_joint_weight=2000, relative_foot_weight=400,
        initial_warm_seed="exact archived canonical shifted MPC horizon at actual3800",
        controller_seeds=["shifted_mpc", "recorded_bfm", "fresh_bfm"],
        initial_integration_state_spec=state_spec, initial_simulation_time=initial_time,
        initial_history_controls=fresh.recorded_controls, fresh_seed=fresh.identity(), motion_override=override,
        hard_feasibility=tracker.feasibility.contract(), received_stream_controller=False,
        packet_preview_seconds=.74, raw_pose_preview_seconds=.76,
        input_hashes={str(path): sha256(path) for path in inputs},
        no_state_rewrites_after_initialization=True, full_source_replay=False, hardware_authorized=False,
        restoration_enabled=args.restoration,
        restoration_trigger="only when no standard shifted/recorded/fresh seed is feasible",
        restoration_maximum_iterations=10 if args.restoration else None,
        restoration_zero_feedback_retry=args.restoration_zero_feedback_retry,
        restoration_control_lm_retry=args.restoration_control_lm_retry,
        third_retry_policy="only after both original attempts fail; K0 control-LM from first guided final",
        third_retry_anchor="unchanged original shifted targets",
        restoration_retry_policy="once from same original shifted seed, only after guided certificate fails",
        restoration_certification="strict nominal300steps and full actual-MjData independent oracle300steps",
    )
    for source in sources:
        (args.output / (source.stem + "_snapshot.py")).write_bytes(source.read_bytes())
    (args.output / "request.json").write_text(json.dumps(request, indent=2))
    trace = {key: [] for key in (
        "qpos", "qvel", "target", "planned_state", "planned_target", "feedback_gain", "source_frame",
        "physics_qpos", "physics_qvel", "physics_torque", "physics_actuator_force", "physics_time",
        "physics_warning_number", "physics_warning_lastinfo", "physics_expected_time", "physics_substeps",
        "fresh_seed_previous_action", "fresh_seed_measured_history",
    )}
    for key, value in (("qpos", data.qpos), ("qvel", data.qvel),
                       ("physics_qpos", data.qpos), ("physics_qvel", data.qvel)):
        trace[key].append(value.copy())
    trace["physics_time"].append(initial_time)
    trace["physics_expected_time"].append(initial_time)
    trace["physics_warning_number"].append(data.warning.number.copy())
    trace["physics_warning_lastinfo"].append(data.warning.lastinfo.copy())
    plans, failure, done = [], None, 0
    restorer, parity = None, None
    expected_time = initial_time
    wall = time.perf_counter()
    while done < args.controls:
        control = 3800 + done
        if args.parity_prefix is not None and parity is None:
            with np.load(args.parity_prefix, allow_pickle=False) as archive:
                if done > len(archive["target"]):
                    raise AssertionError("passed the required old prefix without comparison")
                if done == len(archive["target"]):
                    compared = []
                    for name in trace:
                        if name in archive:
                            np.testing.assert_array_equal(np.asarray(trace[name]), archive[name], err_msg=name)
                            compared.append(name)
                    parity = dict(bit_exact=True, controls=done, fields=compared,
                                  source_sha256=sha256(args.parity_prefix))
        tracker.window(control + 10)
        if done:
            warm = np.concatenate((warm[5:], tracker.target_reference(np.arange(25, 30))))
        tick = time.perf_counter()
        candidates = [("shifted_mpc", warm),
                      ("recorded_bfm", recorded[np.minimum(control + np.arange(30), len(recorded) - 1)])]
        rejection = None
        try:
            proposed, diagnostics = fresh.propose(control, data.qpos, data.qvel, horizon=30)
            candidates.append(("fresh_bfm", proposed))
        except BFMSeedRolloutError as error:
            rejection = dict(message=str(error), diagnostics=error.diagnostics)
        scored = []
        best = None
        actual = np.r_[data.qpos, data.qvel]
        for name, targets in candidates:
            states, bounded_targets, costs = tracker.rollout(actual, targets)
            cost = float(costs[0])
            scored.append(dict(name=name, cost=cost if np.isfinite(cost) else None,
                               feasibility=tracker.last_rollout_feasibility))
            if np.isfinite(cost) and (best is None or cost < best[0]):
                best = (cost, name, states[:, 0].copy(), bounded_targets[:, 0].copy())
        plan = dict(control=control, seed_candidates=scored, fresh_proposal_rejection=rejection,
                    selected_seed=best[1] if best else None, controls_executed=0, imminent_checks=[])
        plans.append(plan)
        if best is None and args.restoration:
            initial_reproduction_failures = []
            def save_proposal(mode, targets):
                integration = np.empty_like(initial_integration)
                mujoco.mj_getState(native, data, integration, state_spec)
                path = args.output / ("restoration_%05d_%s.npz" % (control, mode))
                np.savez_compressed(
                    path, targets=targets, initial_integration=integration, integration_state_spec=state_spec,
                    original_warm_targets=warm, previous_action=fresh.previous_action,
                    recorded_controls=fresh.recorded_controls,
                    actual_action_max_abs=fresh.actual_action_max_abs,
                    actual_action_components_outside_five=fresh.actual_action_components_outside_five,
                    control=control, request_sha256=sha256(args.output / "request.json"),
                    **{"history_" + key: value for key, value in fresh.history.data.items()},
                )
                receipt = dict(proposal_file=path.name, proposal_sha256=sha256(path))
                if control == 3800:
                    expected = (args.control_lm_witness if mode == "control_lm_zero_feedback" else
                                args.first_proposal_witnesses / ("restoration_03800_%s.npz" % mode))
                    with np.load(expected, allow_pickle=False) as archive:
                        equal = (np.array_equal(targets, archive["targets"]) and
                                 np.array_equal(integration, archive["initial_integration"]))
                        target_delta = float(np.max(np.abs(targets-archive["targets"])))
                    receipt.update(initial_proposal_bitexact_witness=equal, witness_sha256=sha256(expected),
                                   initial_proposal_target_max_difference=target_delta)
                    if not equal:
                        initial_reproduction_failures.append(dict(mode=mode, receipt=receipt.copy()))
                return receipt

            def save_diagnostics(mode, arrays):
                path = args.output / ("restoration_search_%05d_%s.npz" % (control, mode))
                np.savez_compressed(path, **arrays, control=control,
                                    request_sha256=sha256(args.output / "request.json"))
                return dict(searches_file=path.name, searches_sha256=sha256(path))

            best, restoration, restorer = restore_feasible_seed(
                tracker, native, data, contract, motion, warm, control + 10, restorer=restorer, threads=8,
                retry_zero_feedback=args.restoration_zero_feedback_retry, save_proposal=save_proposal,
                retry_control_lm=args.restoration_control_lm_retry, save_diagnostics=save_diagnostics,
            )
            plan.update(restoration=restoration, selected_seed=best[1] if best else None)
            if initial_reproduction_failures:
                failure = dict(kind="initial_proposal_reproduction_mismatch", control=control,
                               witnesses=initial_reproduction_failures, recovery_success=False)
                break
        if best is None:
            failure = dict(kind="no_feasible_seed", control=control, recovery_success=False)
            break
        try:
            states, targets, gains, cost = ilqr(
                tracker, actual, best[3], iters=5, initial_rollout=(best[2], best[0])
            )
        except NoFeasiblePlan as error:
            failure = dict(kind="no_feasible_seed_on_revalidation", control=control,
                           diagnostics=error.diagnostics, recovery_success=False)
            break
        plan.update(cost=float(cost), solver=tracker.last_solve_feasibility,
                    solve_ms=(time.perf_counter() - tick) * 1000)
        warm = targets.copy()
        for local in range(min(5, args.controls - done)):
            actual = np.r_[data.qpos, data.qvel]
            correction = np.clip(
                gains[local] @ tracker.difference(states[local:local + 1], actual[None])[0], -.1, .1
            )
            target = np.clip(targets[local] + correction, tracker.lo, tracker.hi)
            witness, predicted, forces = preview_native_control(
                native, data, target, contract, tracker.feasibility
            )
            plan["imminent_checks"].append(witness)
            if not witness["feasible"]:
                failure = dict(kind="imminent_control_infeasible", control=3800 + done,
                               witness=witness, recovery_success=False)
                break
            trace["fresh_seed_previous_action"].append(fresh.previous_action.copy())
            trace["fresh_seed_measured_history"].append(np.concatenate([
                fresh.history.data[key].reshape(-1) for key in sorted(fresh.history.data)
            ]))
            fresh.record_control(3800 + done, data.qpos, data.qvel, target)
            substeps = 0
            for substep in range(10):
                expected_time, failure = record_native_substep(
                    native, data, target, contract, tracker.feasibility, trace, expected_time,
                    predicted[substep + 1], forces[substep],
                )
                substeps += 1
                if failure:
                    failure.update(control=3800 + done, substep=substeps)
                    break
            for key, value in (("qpos", data.qpos), ("qvel", data.qvel), ("target", target),
                               ("planned_state", states[local]), ("planned_target", targets[local]),
                               ("feedback_gain", gains[local])):
                trace[key].append(value.copy())
            trace["source_frame"].append(3800 + done + 11)
            trace["physics_substeps"].append(substeps)
            done += 1
            plan["controls_executed"] += 1
            if failure:
                break
        print(json.dumps(dict(control=3800 + done, cost=cost, failure=failure)), flush=True)
        atomic_trace(args.output / "trace.partial.npz", trace)
        if failure:
            break
    final_integration = np.empty_like(initial_integration)
    mujoco.mj_getState(native, data, final_integration, state_spec)
    atomic_trace(args.output / "trace.npz", trace, initial_integration=initial_integration,
                 final_integration=final_integration, integration_state_spec=np.asarray(state_spec),
                 final_warm_targets=warm, final_previous_action=fresh.previous_action,
                 final_recorded_controls=np.asarray(fresh.recorded_controls),
                 final_actual_action_max_abs=np.asarray(fresh.actual_action_max_abs),
                 final_actual_action_components_outside_five=np.asarray(fresh.actual_action_components_outside_five),
                 **{"final_history_" + k: v for k, v in fresh.history.data.items()})
    result = finite_json(dict(
        request=request, completed_controls=int(np.count_nonzero(np.asarray(trace["physics_substeps"]) == 10)),
        attempted_control_slots=done,
        partial_controls=int(np.count_nonzero(np.asarray(trace["physics_substeps"]) != 10)),
        requested_controls=args.controls, failure=failure,
        completed=done == args.controls and failure is None, physics_steps=len(trace["physics_torque"]),
        simulated_seconds=float(data.time) - initial_time, elapsed_seconds=time.perf_counter() - wall,
        ideal_clock_roundoff_seconds=expected_time - (initial_time + len(trace["physics_torque"]) * .002),
        final_control=3800 + int(np.count_nonzero(np.asarray(trace["physics_substeps"]) == 10)),
        final_attempted_slot_boundary=3800 + done,
        final_source_seconds=(3800 - 350) * .02 + len(trace["physics_torque"]) * .002,
        no_state_rewrites_after_initialization=True, deployment_ready=False, full_source_replay=False,
        recovery_qualified=False, native_warning_counts=data.warning.number.tolist(),
        plans=plans, trace_sha256=sha256(args.output / "trace.npz"),
        original_prefix_parity=parity,
        restoration_attempts=sum("restoration" in plan for plan in plans),
        restoration_seeds_accepted=sum(plan.get("restoration", {}).get("accepted", False) for plan in plans),
        restoration_zero_feedback_retries=sum(
            attempt["mode"] == "zero_feedback" for plan in plans
            for attempt in plan.get("restoration", {}).get("attempts", [])
        ),
        restoration_control_lm_retries=sum(
            attempt["mode"] == "control_lm_zero_feedback" for plan in plans
            for attempt in plan.get("restoration", {}).get("attempts", [])
        ),
        source_files_unchanged_during_run=all(
            sha256(path) == request["input_hashes"][str(path)] for path in sources
        ),
    ))
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: result[key] for key in (
        "completed", "completed_controls", "failure", "elapsed_seconds"
    )}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("bundle", "fixture", "history", "motion-override", "recorded-seed", "onnx",
                 "dependencies", "output", "first-proposal-witnesses", "control-lm-witness", "canonical-trace"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--controls", type=int, default=100)
    parser.add_argument("--restoration", action="store_true")
    parser.add_argument("--restoration-zero-feedback-retry", action="store_true")
    parser.add_argument("--restoration-control-lm-retry", action="store_true")
    parser.add_argument("--parity-prefix", type=Path)
    run(parser.parse_args())
