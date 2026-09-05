"""Robot-free standing transitions and one-time reference frame alignment.

The pinned Unitree actor is an unqualified 29-to-23 compatibility fallback.
It is never substituted for SONIC during the requested dance. No SDK, DDS,
mode RPC or hardware authorization exists in this module.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_ACTION_SCALE, MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import (
    _projected_gravity,
    _quaternion_matrix,
    _quaternion_multiply,
)


def align_reference_xy_yaw(motion: dict[str, np.ndarray], qpos: np.ndarray, index: int = 10):
    """One rigid horizontal transform; no per-frame recentering or height edit."""
    reference_q = motion["body_quat_w"][index, 0]
    actual_rotation = _quaternion_matrix(qpos[3:7])
    reference_rotation = _quaternion_matrix(reference_q)
    yaw = math.atan2(actual_rotation[1, 0], actual_rotation[0, 0]) - math.atan2(
        reference_rotation[1, 0], reference_rotation[0, 0]
    )
    delta_q = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    rotation = _quaternion_matrix(delta_q)
    offset = qpos[:3] - rotation @ motion["body_pos_w"][index, 0]
    offset[2] = 0.0
    aligned = {key: value.copy() for key, value in motion.items()}
    aligned["body_pos_w"] = np.asarray(motion["body_pos_w"], dtype=np.float64) @ rotation.T + offset
    for key in ("body_lin_vel_w", "body_ang_vel_w"):
        aligned[key] = np.asarray(motion[key], dtype=np.float64) @ rotation.T
    aligned["body_quat_w"] = np.asarray(
        [[_quaternion_multiply(delta_q, q) for q in row] for row in motion["body_quat_w"]]
    )
    return aligned, {
        "kind": "one_time_reference_xy_yaw_alignment",
        "yaw_delta_rad": yaw,
        "translation_m": offset.tolist(),
        "anchor_frame": index,
        "height_changed": False,
        "per_frame_recentering": False,
    }


def audit_reference_kinematics(controller, motion: dict[str, np.ndarray]) -> dict[str, Any]:
    """Check all source body positions against native23 forward kinematics."""
    probe = controller.module.MjData(controller.model)
    max_error = 0.0
    max_orientation_error = 0.0
    for index in range(motion["joint_pos"].shape[0]):
        probe.qpos[:3] = motion["body_pos_w"][index, 0]
        probe.qpos[3:7] = motion["body_quat_w"][index, 0]
        probe.qpos[7:] = motion["joint_pos"][index]
        controller.module.mj_forward(controller.model, probe)
        if probe.xpos[1:].shape != motion["body_pos_w"][index].shape:
            raise ValueError("reference body layout does not match native23 model")
        max_error = max(
            max_error, float(np.max(np.linalg.norm(probe.xpos[1:] - motion["body_pos_w"][index], axis=-1)))
        )
        reference_q = np.array(motion["body_quat_w"][index], dtype=np.float64, copy=True)
        reference_q /= np.linalg.norm(reference_q, axis=-1, keepdims=True)
        dot = np.clip(np.abs(np.sum(probe.xquat[1:] * reference_q, axis=-1)), 0.0, 1.0)
        max_orientation_error = max(max_orientation_error, float(np.max(2 * np.arccos(dot))))
    return {
        "frames_checked": len(motion["joint_pos"]),
        "maximum_body_position_fk_error_m": max_error,
        "position_fk_consistent": max_error <= 1e-5,
        "maximum_body_orientation_fk_error_rad": max_orientation_error,
        "orientation_fk_consistent": max_orientation_error <= 1e-5,
        "dynamics_feasibility_proven": False,
    }


def effort_feasible_target(requested, previous, q, dq, kp, kd, effort, *, dt, slew_rate):
    """Nearest target satisfying hard margin, slew and 95% of the existing guard.

    Empty intersections fail explicitly. This does not relax any effort limit
    or hide an infeasible braking demand through torque clipping.
    """
    if (
        any(
            np.shape(value) != (23,) or not np.isfinite(value).all()
            for value in (requested, previous, q, dq, kp, kd, effort)
        )
        or np.any(kp <= 0)
        or np.any(kd < 0)
        or np.any(effort <= 0)
    ):
        raise ValueError("target projection requires finite 23-joint state and positive gains/effort")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("target projection timestep must be finite and positive")
    if slew_rate is not None and (not np.isfinite(slew_rate) or slew_rate <= 0):
        raise ValueError("target projection slew must be finite and positive or None")
    cap = 0.95 * 0.25 * effort
    lower = np.maximum(np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + 0.05, q + (kd * dq - cap) / kp)
    upper = np.minimum(np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) - 0.05, q + (kd * dq + cap) / kp)
    if slew_rate is not None:
        lower = np.maximum(lower, previous - slew_rate * dt)
        upper = np.minimum(upper, previous + slew_rate * dt)
    if np.any(lower > upper):
        raise ValueError("empty effort/position/slew target intersection")
    return np.clip(requested, lower, upper)


def simulate_balance_transition(
    controller,
    policy,
    *,
    duration_s: float,
    slew_rate: float | None,
    previous_target: np.ndarray | None = None,
    project_effort: bool = False,
):
    """Simulate only a standing/acquisition interval, recording all guard breaches."""
    dt = controller.physics.timestep_s
    if (
        not np.isfinite(duration_s)
        or not 0 < duration_s <= 10
        or abs(round(duration_s / 0.02) * 0.02 - duration_s) > 1e-9
    ):
        raise ValueError("balance duration must be 0..10 seconds in exact 50 Hz steps")
    if slew_rate is not None and (not np.isfinite(slew_rate) or slew_rate <= 0):
        raise ValueError("balance slew must be positive or disabled for diagnostics")
    if not np.isfinite(dt) or dt <= 0 or controller.physics.decimation != 10 or abs(dt - 0.002) > 1e-12:
        raise ValueError("balance transition requires 500 Hz physics and 50 Hz inference")
    data, physics = controller.data, controller.physics
    start = data.qpos.copy()
    previous = start[7:].copy() if previous_target is None else previous_target.copy()
    if start.shape != (30,) or not np.isfinite(start).all() or not np.isfinite(data.qvel).all():
        raise ValueError("balance transition requires finite initial state")
    if previous.shape != (23,) or not np.isfinite(previous).all():
        raise ValueError("balance transition requires finite previous target")
    policy.reset()
    policy.activate(start[7:])
    requested_steps = round(duration_s / 0.02)
    states = [start.copy()]
    first_effort_guard = first_target_guard = None
    peak_effort_ratio = 0.0
    max_tilt = float(np.arccos(np.clip(-_projected_gravity(start[3:7])[2], -1, 1)))
    max_drift = 0.0
    min_height = float(start[2])
    failure = None
    projection_count = 0
    physics_steps = completed = 0
    low, high = np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE), np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE)
    for index in range(requested_steps):
        frame = controller._policy_frame()
        controller.history = [*controller.history[1:], frame.copy()]
        requested, kp, kd = policy.infer(
            joint_position_hardware=data.qpos[7:].copy(),
            joint_velocity_hardware=data.qvel[6:].copy(),
            base_angular_velocity_body=frame[:3],
            projected_gravity=frame[-3:],
        )
        if any(np.shape(value) != (23,) or not np.isfinite(value).all() for value in (requested, kp, kd)):
            raise ValueError("invalid balance policy target or gains")
        if (
            not project_effort
            and first_target_guard is None
            and np.any((requested < low + 0.05) | (requested > high - 0.05))
        ):
            first_target_guard = {"transition": index}
        pelvis_before = data.qpos[3:7].copy()
        for substep in range(physics.decimation):
            target = (
                requested
                if slew_rate is None
                else np.clip(requested, previous - slew_rate * dt, previous + slew_rate * dt)
            )
            if project_effort:
                try:
                    projected = effort_feasible_target(
                        requested,
                        previous,
                        data.qpos[7:].copy(),
                        data.qvel[6:].copy(),
                        kp,
                        kd,
                        physics.effort,
                        dt=dt,
                        slew_rate=slew_rate,
                    )
                except ValueError as error:
                    failure = str(error)
                    break
                projection_count += int(np.count_nonzero(np.abs(projected - target) > 1e-10))
                target = projected
            predicted = kp * (target - data.qpos[7:]) - kd * data.qvel[6:]
            ratio = float(np.max(np.abs(predicted) / (0.25 * physics.effort)))
            peak_effort_ratio = max(peak_effort_ratio, ratio)
            if first_effort_guard is None and ratio > 1:
                first_effort_guard = {"transition": index, "substep": substep, "ratio": ratio}
            data.ctrl[:] = np.clip(predicted, -physics.effort, physics.effort)
            controller.module.mj_step(controller.model, data)
            previous = target.copy()
            physics_steps += 1
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                failure = "nonfinite_state"
                break
            tilt = float(np.arccos(np.clip(-_projected_gravity(data.qpos[3:7])[2], -1, 1)))
            max_tilt = max(max_tilt, tilt)
            min_height = min(min_height, float(data.qpos[2]))
            max_drift = max(max_drift, float(np.linalg.norm(data.qpos[:2] - start[:2])))
            if data.qpos[2] < 0.45 or tilt > 1.0:
                failure = "standing_posture_lost"
                break
        # Equivalent previous executed target in SONIC's unscaled action units.
        action_hw = (previous - np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)) / np.asarray(HARDWARE_23_ACTION_SCALE)
        controller.previous_safe_native = action_hw[np.asarray(MUJOCO_TO_ISAACLAB_DOF)].astype(np.float32)
        controller.buffered_robot_pelvis_q9 = pelvis_before
        # Include partial-interval terminal state without crediting a full step.
        if physics_steps > completed * physics.decimation:
            states.append(data.qpos.copy())
        completed = physics_steps // physics.decimation
        if failure is not None:
            break
    standing = (
        completed == requested_steps
        and failure is None
        and min_height >= 0.45
        and max_tilt <= 0.5
        and max_drift <= 0.1
    )
    return (
        {
            "controller": "hash_pinned_unitree_29_to_23_zero_velocity_compatibility_actor",
            "requested_transitions": requested_steps,
            "completed_transitions": completed,
            "completed_physics_steps": physics_steps,
            "partial_transition_substeps": physics_steps % physics.decimation,
            "elapsed_simulation_s": physics_steps * dt,
            "standing_screen_passed": standing,
            "existing_guard_screen_passed": standing and first_effort_guard is None and first_target_guard is None,
            "failure": failure,
            "minimum_height_m": min_height,
            "maximum_tilt_rad": max_tilt,
            "maximum_horizontal_drift_m": max_drift,
            "first_quarter_effort_guard_violation": first_effort_guard,
            "first_target_margin_guard_violation": first_target_guard,
            "maximum_quarter_effort_ratio": peak_effort_ratio,
            "gain_kp_hardware": kp.tolist(),
            "gain_kd_hardware": kd.tolist(),
            "target_slew_rad_s": slew_rate,
            "effort_target_projection": project_effort,
            "projected_joint_substeps": projection_count,
            "threshold_status": "provisional_simulator_standing_screen",
            "hardware_authorized": False,
            "unitree_mode_transfer_simulated": False,
        },
        np.asarray(states),
        previous,
    )
