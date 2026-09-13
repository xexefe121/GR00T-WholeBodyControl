"""Bounded offline native23 iLQR probes, with native manual-PD physical execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_ilqr_core import ilqr
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import (
    Native23Tracker,
    TRACKED,
    load_native_bundle,
    motion_states,
    sha256,
)


def run(args):
    if min(args.horizon, args.commit, args.iterations, args.threads) <= 0 or args.commit > args.horizon:
        raise ValueError("invalid bounded MPC experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    native, contract, motion, timeline, manifest = load_native_bundle(args.bundle, args.clip)
    kp, kd, effort, velocity_limits = [
        np.asarray(contract[name]) for name in ("kp", "kd", "native_effort", "native_velocity")
    ]
    servo = position_servo_copy(native, kp, kd, effort)
    planner = Native23Tracker(servo, contract, motion, horizon=args.horizon, threads=args.threads)
    reference_states = motion_states(motion)
    phase = next(phase for phase in timeline["phases"] if phase["name"] == "source_motion")
    if args.probe == "standing":
        requested = int(round(args.standing_seconds * 50))
    else:
        requested = phase["control_start"] + int(round(args.source_seconds * 50))
    requested = min(requested, len(reference_states) - 11)
    if requested <= 0:
        raise ValueError("empty MPC probe")
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
        for name in ("g1_true23_mjbatch_ilqr_core.py", "g1_true23_mjbatch_model.py", "g1_true23_mjbatch_mpc.py")
    ]
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
        batch_threads=args.threads,
        declared_future_preview_seconds=args.horizon * 0.02,
        source_clock_hz=50,
        physics_hz=500,
        source_reference_lift_m=0.0,
        source_frame_removal=0,
        input_hashes={str(path): sha256(path) for path in inputs},
        model_manifest=manifest,
        actual_execution="manual native clipped PD every2ms",
        planning_actuators="algebraically equivalent affine PD, separately validated",
        costs=dict(
            tracked_bodies=TRACKED,
            body_position=planner.position_weights.tolist(),
            body_rotation=planner.rotation_weights.tolist(),
            joint=planner.joint_weight,
            generalized_velocity=planner.velocity_weights.tolist(),
            near_joint_limit=planner.limit_weight,
            joint_limit_margin=planner.limit_margin,
            target_regularization=planner.control_weight,
        ),
        mujoco=mujoco.__version__,
        hardware_authorized=False,
        deployment_ready=False,
        received_stream_controller=False,
    )
    (args.output / "request.json").write_text(json.dumps(request, indent=2))
    for path in inputs[-4:]:
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
        )
    }
    for name, value in (
        ("qpos", data.qpos),
        ("qvel", data.qvel),
        ("physics_qpos", data.qpos),
        ("physics_qvel", data.qvel),
    ):
        trace[name].append(value.copy())
    plans = []
    warm_targets = None
    completed = 0
    failure = None
    wall_started = time.perf_counter()
    while completed < requested:
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
        tick = time.perf_counter()
        planned_states, planned_targets, gains, cost = ilqr(
            planner, actual_state, warm_targets.copy(), iters=args.iterations
        )
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
            )
        )
        warm_targets = planned_targets.copy()
        for local in range(count):
            frame = completed + 11
            actual_state = np.r_[data.qpos, data.qvel]
            correction = (
                gains[local] @ planner.difference(planned_states[local : local + 1], actual_state[None])[0]
            )
            target = np.clip(planned_targets[local] + correction, planner.lo, planner.hi)
            peak_range = peak_effort = peak_velocity = 0.0
            for _ in range(10):
                desired_torque = kp * (target - data.qpos[7:]) - kd * data.qvel[6:]
                data.ctrl[:] = np.clip(desired_torque, -effort, effort)
                mujoco.mj_step(native, data)
                for name, value in (
                    ("physics_qpos", data.qpos),
                    ("physics_qvel", data.qvel),
                    ("physics_requested_torque", desired_torque),
                    ("physics_torque", data.ctrl),
                ):
                    trace[name].append(value.copy())
                peak_range = max(
                    peak_range, float(np.max(np.maximum(planner.lo - data.qpos[7:], data.qpos[7:] - planner.hi)))
                )
                peak_velocity = max(peak_velocity, float(np.max(np.abs(data.qvel[6:]) / velocity_limits)))
                peak_effort = max(peak_effort, float(np.max(np.abs(data.qfrc_actuator[6:]) / effort)))
            mujoco.mj_kinematics(native, data)
            source_ids = np.asarray(planner.ids) - 1
            values = dict(
                qpos=data.qpos.copy(),
                qvel=data.qvel.copy(),
                target=target.copy(),
                planned_target=planned_targets[local].copy(),
                planned_state=planned_states[local].copy(),
                feedback_gain=gains[local].copy(),
                source_frame=frame,
                joint_error=data.qpos[7:] - motion["joint_pos"][frame],
                root_error=data.qpos[:3] - motion["body_pos_w"][frame, 0],
                body_position_error=np.linalg.norm(
                    data.xpos[planner.ids] - motion["body_pos_w"][frame, source_ids], axis=-1
                ),
                range_excess=peak_range,
                velocity_ratio=peak_velocity,
                effort_ratio=peak_effort,
            )
            for name, value in values.items():
                trace[name].append(value)
            completed += 1
            tilt = float(np.arccos(np.clip(data.xmat[1, 8], -1, 1)))
            if not np.isfinite(data.qpos).all() or data.qpos[2] < 0.25 or tilt > 1.2:
                failure = dict(kind="fall", control=completed, height=float(data.qpos[2]), tilt=tilt)
            elif peak_range > 0.01 or peak_velocity > 1:
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
        if failure:
            break
    arrays = {name: np.asarray(values) for name, values in trace.items()}
    np.savez_compressed(args.output / "trace.npz", **arrays)
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
        planning_ms_p50_p95_max=np.percentile([plan["solve_ms"] for plan in plans], [50, 95, 100]).tolist()
        if plans
        else None,
        planning_deadlines_missed=sum(plan["solve_ms"] > plan["controls_committed"] * 20 for plan in plans),
        elapsed_wall_seconds=time.perf_counter() - wall_started,
        simulated_seconds=completed * 0.02,
        declared_future_preview_seconds=args.horizon * 0.02,
        mujoco=mujoco.__version__,
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
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--clip", choices=("walk002", "walk003", "walk008", "pico"), default="walk002")
    parser.add_argument("--probe", choices=("standing", "source-prefix"), default="standing")
    parser.add_argument("--standing-seconds", type=float, default=1.0)
    parser.add_argument("--source-seconds", type=float, default=3.0)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--commit", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
