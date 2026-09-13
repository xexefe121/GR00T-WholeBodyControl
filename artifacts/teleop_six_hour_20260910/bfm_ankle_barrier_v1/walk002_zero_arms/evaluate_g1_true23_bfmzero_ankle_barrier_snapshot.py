"""Full prerecorded native23 BFM replay with an explicit ankle torque experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import (
    MODEL,
    PACKAGE,
    PHYSICS,
    ROOT,
    corrected_goal,
    load_case_motion,
    load_motion,
)
from gear_sonic.utils.g1_true23_bfmzero_ankle_barrier import AnkleRollRepulsion
from gear_sonic.utils.g1_true23_bfmzero_inference import (
    BFMHistory,
    BFMZeroInference,
    load_contract,
    reference_features,
    state_and_terms,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import task_points
from gear_sonic.utils.g1_true23_step1b_mujoco import _quaternion_matrix, prepare_true23_model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    config = PACKAGE / "bfmzero_inspect_v1/config.yaml"
    weights = PACKAGE / "bfmzero_inference_v1/inference.safetensors"
    contract = load_contract(config)
    policy = BFMZeroInference(weights)
    model_path = ROOT.parent / "GR00T-WholeBodyControl" / MODEL
    _, model, physics = prepare_true23_model(model_path, ROOT / PHYSICS)
    assert (model.nq, model.nv, model.nu) == (30, 29, 23)
    limits = model.jnt_range[1:].copy()
    velocity_limits = np.asarray(
        json.loads((ROOT / PHYSICS).read_text())["physics"]["velocity_limit_hardware_radps"]
    )
    body_ids = [model.body(name).id for name in contract["body_names"]]
    joint_names = [model.joint(index).name for index in range(1, model.njnt)]
    barrier = AnkleRollRepulsion(
        joint_names, limits, margin=args.margin, stiffness=args.stiffness, damping=args.damping
    )
    assert np.array_equal(physics.effort[barrier.indices], [35.0, 35.0])
    motion_override = getattr(args, "motion_override", None)
    motion, timeline, path = load_case_motion(args.clip, motion_override)
    _, _, original_path = load_motion(args.clip)
    assert sha256(original_path) == timeline["timeline_sha256"]
    files = [
        Path(__file__),
        ROOT / "gear_sonic/utils/g1_true23_bfmzero_ankle_barrier.py",
        ROOT / "gear_sonic/utils/g1_true23_bfmzero_inference.py",
        ROOT / "gear_sonic/scripts/evaluate_g1_true23_bfmzero.py",
        ROOT / "gear_sonic/utils/g1_true23_step1b_mujoco.py",
        model_path,
        ROOT / PHYSICS,
        path,
        original_path,
        config,
        weights,
    ]
    request = {
        "clip": args.clip,
        "barrier": barrier.report(),
        "position_gain": 1.0,
        "yaw_gain": 2.0,
        "goal_horizon": 8,
        "arm_reference": args.arm_reference,
        "full_prerecorded_source_preview_ms": 140,
        "stream_lifecycle_qualification": False,
        "hardware_authorized": False,
        "source_timeline": timeline,
        "motion_override": None if motion_override is None else str(path),
        "reference_geometry_retargeted": motion_override is not None,
        "inputs": {str(file): sha256(file) for file in files},
    }
    (args.output / "request.json").write_text(json.dumps(request, indent=2))
    for file in files[:5]:
        (args.output / (file.stem + "_snapshot.py")).write_bytes(file.read_bytes())
    states, privileged = reference_features(motion, contract)
    data = mujoco.MjData(model)
    data.qpos[:] = np.r_[motion["body_pos_w"][10, 0], motion["body_quat_w"][10, 0], motion["joint_pos"][10]]
    data.qvel[:] = np.r_[
        motion["body_lin_vel_w"][10, 0],
        _quaternion_matrix(data.qpos[3:7]).T @ motion["body_ang_vel_w"][10, 0],
        motion["joint_vel"][10],
    ]
    mujoco.mj_forward(model, data)
    history = BFMHistory()
    action = np.zeros(23, np.float32)
    available = len(motion["joint_pos"]) - 11
    requested = available if args.max_controls is None else min(available, args.max_controls)
    trace = {
        key: []
        for key in (
            "qpos",
            "qvel",
            "action",
            "target",
            "unclipped_target",
            "state",
            "history",
            "goal",
            "joint_error",
            "root_error",
            "landmark_error",
            "relative_landmark_error",
            "range_excess",
            "velocity_ratio",
            "effort_ratio",
            "inference_ms",
            "physics_qpos",
            "physics_qvel",
            "physics_base_torque",
            "physics_barrier_torque",
            "physics_requested_torque",
            "physics_torque",
        )
    }
    for key, value in (
        ("qpos", data.qpos),
        ("qvel", data.qvel),
        ("physics_qpos", data.qpos),
        ("physics_qvel", data.qvel),
    ):
        trace[key].append(value.copy())
    failure = None
    started = time.perf_counter()
    for control in range(requested):
        frame = control + 11
        sensed, terms = state_and_terms(
            data.qpos[7:], data.qvel[6:], data.qpos[3:7], data.qvel[3:6], action, contract["default_q"]
        )
        hist = history.before_update(terms)
        tick = time.perf_counter()
        goal = corrected_goal(policy, states, privileged, motion, frame, data.qpos, 8, 1.0, 2.0)
        raw = policy.actor(
            torch.from_numpy(sensed[None]), torch.from_numpy(action[None]), torch.from_numpy(hist[None]), goal
        )[0].numpy()
        action = raw * 5.0
        target = contract["default_q"] + action * 0.25 * contract["training_effort"] / contract["kp"]
        if args.arm_reference:
            target[13:] = (
                motion["joint_pos"][frame, 13:]
                + contract["kd"][13:] / contract["kp"][13:] * motion["joint_vel"][frame, 13:]
            )
            action = (
                (target - contract["default_q"]) * contract["kp"] / (0.25 * contract["training_effort"])
            ).astype(np.float32)
        clipped = np.clip(target, limits[:, 0], limits[:, 1])
        inference_ms = (time.perf_counter() - tick) * 1000
        peak_range = peak_effort = peak_velocity = 0.0
        for _ in range(physics.decimation):
            base_torque = contract["kp"] * (clipped - data.qpos[7:]) - contract["kd"] * data.qvel[6:]
            barrier_torque = barrier.torque(data.qpos[7:], data.qvel[6:])
            requested_torque = base_torque + barrier_torque
            data.ctrl[:] = np.clip(requested_torque, -physics.effort, physics.effort)
            mujoco.mj_step(model, data)
            for key, value in (
                ("physics_qpos", data.qpos),
                ("physics_qvel", data.qvel),
                ("physics_base_torque", base_torque),
                ("physics_barrier_torque", barrier_torque),
                ("physics_requested_torque", requested_torque),
                ("physics_torque", data.ctrl),
            ):
                trace[key].append(value.copy())
            peak_effort = max(peak_effort, float(np.max(np.abs(data.qfrc_actuator[6:]) / physics.effort)))
            peak_range = max(
                peak_range, float(np.max(np.maximum(limits[:, 0] - data.qpos[7:], data.qpos[7:] - limits[:, 1])))
            )
            peak_velocity = max(peak_velocity, float(np.max(np.abs(data.qvel[6:]) / velocity_limits)))
        mujoco.mj_kinematics(model, data)
        actual = task_points(data.xpos[body_ids], data.xquat[body_ids])
        desired = task_points(motion["body_pos_w"][frame], motion["body_quat_w"][frame])
        root_error = data.qpos[:3] - motion["body_pos_w"][frame, 0]
        values = dict(
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            action=action.copy(),
            target=clipped,
            unclipped_target=target,
            state=sensed,
            history=hist,
            goal=goal[0].numpy().copy(),
            joint_error=data.qpos[7:] - motion["joint_pos"][frame],
            root_error=root_error,
            landmark_error=np.linalg.norm(actual - desired, axis=-1),
            relative_landmark_error=np.linalg.norm(actual - desired - root_error, axis=-1),
            range_excess=peak_range,
            velocity_ratio=peak_velocity,
            effort_ratio=peak_effort,
            inference_ms=inference_ms,
        )
        for key, value in values.items():
            trace[key].append(value)
        tilt = float(np.arccos(np.clip(_quaternion_matrix(data.qpos[3:7])[2, 2], -1, 1)))
        if not np.isfinite(data.qpos).all() or data.qpos[2] < 0.25 or tilt > 1.2:
            failure = dict(kind="fall", control=control, height=float(data.qpos[2]), tilt=tilt)
            break
        if peak_range > 0.01 or peak_velocity > 1.0:
            failure = dict(
                kind="physical_limit", control=control, range_excess=peak_range, velocity_ratio=peak_velocity
            )
            break
        if (control + 1) % 500 == 0:
            print(json.dumps(dict(clip=args.clip, controls=control + 1, height=float(data.qpos[2]))), flush=True)
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    np.savez_compressed(args.output / "trace.npz", **arrays)
    completed = len(arrays["action"])
    phase = next(phase for phase in timeline["phases"] if phase["name"] == "source_motion")
    start, stop = phase["control_start"], min(completed, phase["control_stop"])
    metrics = None
    if stop > start:
        error = arrays["joint_error"][start:stop]
        metrics = dict(
            source_controls=stop - start,
            source_requested=phase["requested_controls"],
            leg_rmse=float(np.sqrt(np.mean(error[:, :12] ** 2))),
            arm_rmse=float(np.sqrt(np.mean(error[:, 13:] ** 2))),
            root_p95=float(np.percentile(np.linalg.norm(arrays["root_error"][start:stop], axis=-1), 95)),
            landmark_p95=np.percentile(arrays["landmark_error"][start:stop], 95, axis=0).tolist(),
            relative_landmark_p95=np.percentile(
                arrays["relative_landmark_error"][start:stop], 95, axis=0
            ).tolist(),
        )
    result = dict(
        kind="bfmzero_explicit_ankle_repulsion_simulation",
        clip=args.clip,
        completed=completed,
        requested=requested,
        available=available,
        failure=failure,
        full_lifecycle_completed=completed == available and failure is None,
        source_metrics=metrics,
        range_excess_max=float(arrays["range_excess"].max()),
        effort_ratio_max=float(arrays["effort_ratio"].max()),
        velocity_ratio_max=float(arrays["velocity_ratio"].max()),
        elapsed_seconds=time.perf_counter() - started,
        barrier=barrier.report(),
        barrier_torque_max_Nm=float(np.abs(arrays["physics_barrier_torque"]).max()),
        barrier_active_physics_steps=int(np.count_nonzero(np.any(arrays["physics_barrier_torque"] != 0, axis=1))),
        arm_reference=args.arm_reference,
        goal_horizon=8,
        position_gain=1.0,
        yaw_gain=2.0,
        motion_override=None if motion_override is None else str(path),
        reference_geometry_retargeted=motion_override is not None,
        ground_truth_pose_feedback=True,
        source_tracking_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
        simulator_pose_writes_after_initialization=0,
        root_assistance_forces=0,
        request_sha256=sha256(args.output / "request.json"),
        trace_sha256=sha256(args.output / "trace.npz"),
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", choices=("pico", "walk002", "walk008"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--stiffness", type=float, default=150.0)
    parser.add_argument("--damping", type=float, default=2.0)
    parser.add_argument("--arm-reference", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-controls", type=int)
    parser.add_argument("--motion-override", type=Path)
    run(parser.parse_args())
