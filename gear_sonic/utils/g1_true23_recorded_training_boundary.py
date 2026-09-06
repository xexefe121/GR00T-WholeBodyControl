"""Execute training tensor/buffer code on recorded CPU evaluator states.

This is an offline mapping witness, not an MJLab rollout, a sensor simulation,
or hardware parity. No saved policy input is used to reconstruct another
input. Startup targets without an action trace are explicitly recovered from
the recorded applied PD effort, q/dq and the reported transition gains.
"""

from types import SimpleNamespace

import numpy as np
import torch

from gear_sonic.envs.mjlab import sonic_true23 as observation
from gear_sonic.envs.mjlab.sonic_true23_causal_history import (
    causal_history_lower_body,
    causal_motion_anchor_ori_b,
)
from gear_sonic.envs.mjlab.sonic_true23_stage_one_actuation import (
    native_support_pd_step,
    requested_stage_one_target,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_torch
from gear_sonic.utils.g1_true23_projected_controller_state import applied_target_native_torch


def finite(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite floating array with shape {shape}")
    return value


def comparison(left, right, tolerance):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim < 1 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("boundary comparison requires equal-shaped finite arrays")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("boundary tolerance must be finite and nonnegative")
    error = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return dict(
        shape=list(left.shape),
        maximum_absolute_difference=float(error.max(initial=0)),
        changed_elements=int(np.count_nonzero(error)),
        rows_outside_tolerance=np.flatnonzero((error > tolerance).reshape(len(error), -1).any(1)).tolist(),
        absolute_tolerance=tolerance,
        within_tolerance=bool(np.all(error <= tolerance)),
    )


def recorded_control_states(arrays, result):
    """Select actual 50 Hz call states, retaining terminal partial intervals.

    Both successful 2 ms rows and a terminal rejected request remain in the
    witness. Nothing after the first active failure is credited as motion.
    """
    count = len(arrays["policy_encoder267"])
    completed = result["completed_transitions"]
    if count < 1 or count not in (completed, completed + 1):
        raise ValueError("recording must contain completed and optional terminal inference calls")
    phase = arrays["physics_phase"]
    if phase.ndim != 1 or phase.dtype.kind not in "iu" or np.any(np.diff(phase) < 0):
        raise ValueError("physics phases must preserve startup/active/return order")
    if not np.isin(phase, [0, 1, 2]).all():
        raise ValueError("unknown physics phase")
    pre_q = finite(arrays["physics_pre_qpos"], (len(phase), 30), "physics qpos")
    pre_dq = finite(arrays["physics_pre_qvel"], (len(phase), 29), "physics qvel")
    active = np.flatnonzero(phase == 1)
    startup = np.flatnonzero(phase == 0)
    if (
        len(active) != result["completed_active_physics_steps"]
        or len(active) // 10 != completed
        or len(startup) % 10
        or not len(active)
    ):
        raise ValueError("recorded physics does not cover complete startup and exact active intervals")
    for key in ("q", "dq", "previous_target", "requested", "target", "effort"):
        finite(arrays["actuation_" + key], (len(active), 23), "actuation " + key)
    np.testing.assert_array_equal(pre_q[active, 7:], arrays["actuation_q"])
    np.testing.assert_array_equal(pre_dq[active, 6:], arrays["actuation_dq"])
    qpos, qvel, previous = [], [], []
    for control in range(count):
        step = control * 10
        if step < len(active):
            qpos.append(pre_q[active[step]])
            qvel.append(pre_dq[active[step]])
            previous.append(arrays["actuation_previous_target"][step])
        elif control == completed and count == completed + 1 and step == len(active):
            qpos.append(finite(arrays["terminal_active_qpos"], (30,), "terminal qpos"))
            qvel.append(finite(arrays["terminal_active_qvel"], (29,), "terminal qvel"))
            previous.append(finite(arrays["terminal_active_target"], (23,), "terminal target"))
        else:
            raise ValueError("inference has no actual recorded control-boundary state")
    qpos, qvel, previous = map(np.asarray, (qpos, qvel, previous))
    # The evaluator retains initial + completed control states independently.
    np.testing.assert_array_equal(qpos[: len(arrays["qpos"])], arrays["qpos"][: len(qpos)])
    warm_q, warm_dq, warm_target = [], [], []
    if len(startup):
        hold = result["startup_hold"]
        if hold["completed_physics_steps"] != len(startup) or not hold["standing_screen_passed"]:
            raise ValueError("active input witness requires successful recorded startup")
        kp = finite(hold["gain_kp_hardware"], (23,), "startup kp")
        kd = finite(hold["gain_kd_hardware"], (23,), "startup kd")
        if np.any(kp <= 0):
            raise ValueError("cannot reconstruct startup target with zero kp")
        # Only the nine startup observations still in the first active H10.
        # All have a preceding completed PD substep; no initial seed guessed.
        if len(startup) < 100:
            raise ValueError("startup witness needs at least ten complete control intervals")
        for step in startup[-90::10]:
            prior = step - 1
            target = pre_q[prior, 7:] + (arrays["physics_effort"][prior] + kd * pre_dq[prior, 6:]) / kp
            warm_q.append(pre_q[step])
            warm_dq.append(pre_dq[step])
            warm_target.append(target)
    return dict(
        qpos=qpos.copy(),
        qvel=qvel.copy(),
        previous_target=previous.copy(),
        warmup_qpos=np.asarray(warm_q).reshape(-1, 30),
        warmup_qvel=np.asarray(warm_dq).reshape(-1, 29),
        warmup_target=np.asarray(warm_target).reshape(-1, 23),
        startup_target_reconstruction_rows=len(warm_q),
        startup_last_quaternion=None if not len(startup) else pre_q[startup[-10], 3:7].copy(),
    )


def training_observations(motion, states, body_names, *, device="cpu"):
    """Use production observation functions, EntityData gravity and H10 buffer.

    Explicit state views supply q/dq, body-local free-joint angular velocity,
    final applied targets and native motion arrays. Sensor readout, automatic
    command stepping, noise, reset managers and physics are NOT emulated.
    """
    from mjlab.entity.data import EntityData
    from mjlab.utils.buffers import CircularBuffer

    def tensor(value):
        return torch.as_tensor(np.array(value, copy=True), dtype=torch.float32, device=device)

    qpos = np.concatenate((states["warmup_qpos"], states["qpos"]))
    qvel = np.concatenate((states["warmup_qvel"], states["qvel"]))
    previous = np.concatenate((states["warmup_target"], states["previous_target"]))
    warm, total, count = len(states["warmup_qpos"]), len(qpos), len(states["qpos"])
    entity_view = SimpleNamespace(
        root_link_quat_w=tensor(qpos[:, 3:7]), gravity_vec_w=tensor([[0, 0, -1]] * total)
    )
    gravity = EntityData.projected_gravity_b.fget(entity_view)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=tensor(qpos[:, 7:]), joint_vel=tensor(qvel[:, 6:]), projected_gravity_b=gravity
        )
    )
    action = SimpleNamespace(safe_native_action=applied_target_native_torch(tensor(previous)))
    env = SimpleNamespace(
        num_envs=total,
        device=device,
        scene={"robot": robot, "robot/imu_ang_vel": SimpleNamespace(data=tensor(qvel[:, 3:6]))},
        action_manager=SimpleNamespace(action=None, get_term=lambda name: action if name == "joint_pos" else None),
    )
    fields = (
        observation.base_ang_vel(env),
        observation.padded_joint_pos_rel(env),
        observation.padded_joint_vel(env),
        observation.padded_previous_action(env),
        observation.projected_gravity(env),
    )
    buffers = [CircularBuffer(10, 1, device) for _ in fields]
    histories = []
    for index in range(total):
        for buffer, field in zip(buffers, fields):
            buffer.append(field[index : index + 1])
        if index >= warm:
            histories.append(torch.cat([buffer.buffer.reshape(1, -1) for buffer in buffers], dim=1))
    history = torch.cat(histories)
    anchors = torch.arange(9, 9 + count, device=device)
    reference_names = tuple(observation._REQUIRED_REFERENCE_BODY_NAMES)
    if len(body_names) != motion["body_pos_w"].shape[1] or any(name not in body_names for name in reference_names):
        raise ValueError("motion must match the verified native model body layout")
    body_indices = [body_names.index(name) for name in reference_names]
    body_pos = tensor(motion["body_pos_w"])[anchors][:, body_indices]
    body_quat = tensor(motion["body_quat_w"])[anchors][:, body_indices]
    buffered = np.concatenate((np.asarray(motion["body_quat_w"][9, 0])[None], states["qpos"][:-1, 3:7]))
    if states["startup_last_quaternion"] is not None:
        buffered[0] = states["startup_last_quaternion"]
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=reference_names),
        time_steps=anchors,
        motion=SimpleNamespace(joint_pos=tensor(motion["joint_pos"]), time_step_total=len(motion["joint_pos"])),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        anchor_pos_w=body_pos[:, 0],
        anchor_quat_w=body_quat[:, 0],
        causal_robot_anchor_quat_w=tensor(buffered),
    )
    env.num_envs = count
    env.command_manager = SimpleNamespace(get_term=lambda name: command if name == "motion" else None)
    encoder = torch.cat(
        (
            causal_history_lower_body(env),
            observation.vr_3point_local_target(env, "motion"),
            observation.vr_3point_local_orn_target(env, "motion"),
            causal_motion_anchor_ori_b(env),
        ),
        dim=1,
    )
    return dict(
        encoder267=encoder.cpu().numpy(),
        history930=history.cpu().numpy(),
        current_frames93=torch.cat(fields, dim=1)[warm:].cpu().numpy(),
    )


def training_actuation(arrays, result, profile, *, device="cpu"):
    """Evaluate every successful active substep and each terminal rejection."""

    def tensor(value):
        return torch.as_tensor(np.array(value, copy=True), dtype=torch.float32, device=device)

    raw = tensor(arrays["policy_raw23"])
    _, full = safe_target_transform_torch(raw)
    requested = requested_stage_one_target(full, profile)
    n = len(arrays["actuation_q"])
    control = torch.arange(n, device=device) // 10
    target, effort, invalid = native_support_pd_step(
        requested[control],
        tensor(arrays["actuation_previous_target"]),
        tensor(arrays["actuation_q"]),
        tensor(arrays["actuation_dq"]),
        profile,
    )
    terminal = None
    if result["failure"] is not None and result["failure"].get("type") == "TargetIntersectionError":
        t_target, t_effort, t_invalid = native_support_pd_step(
            requested[-1:],
            tensor(arrays["terminal_active_target"])[None],
            tensor(arrays["terminal_active_qpos"][7:])[None],
            tensor(arrays["terminal_active_qvel"][6:])[None],
            profile,
        )
        terminal = dict(
            rejected_by_training=bool(t_invalid.item()),
            training_output_effort=t_effort.cpu().numpy()[0].tolist(),
            training_target=t_target.cpu().numpy()[0].tolist(),
            evaluator_physics_stopped_before_this_substep=True,
            training_zero_effort_until_50hz_reset_is_not_a_physical_return=True,
        )
    return dict(
        full_target=full.cpu().numpy(),
        requested_per_control=requested.cpu().numpy(),
        target=target.cpu().numpy(),
        effort=effort.cpu().numpy(),
        invalid=invalid.cpu().numpy(),
        terminal=terminal,
    )
