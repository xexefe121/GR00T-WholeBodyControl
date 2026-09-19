"""Explicit native23 controller features; no robot transport or state mutation."""

import numpy as np
import torch

from gear_sonic.utils.g1_23dof_contract import (
    ISAACLAB_TO_MUJOCO_DOF,
    NATIVE_IL23_TO_CANONICAL_IL29,
)

GOAL_DIM = 90
OBS_DIM = 165
HW_IN_PAD29 = tuple(np.asarray(NATIVE_IL23_TO_CANONICAL_IL29)[list(ISAACLAB_TO_MUJOCO_DOF)].tolist())


def quaternion_matrix(q):
    """Batched WXYZ rotation; inputs must already be normalized."""
    if q.shape[-1] != 4 or not torch.isfinite(q).all():
        raise ValueError("compact orientation requires finite WXYZ")
    if torch.any((q.square().sum(-1) - 1).abs() > 3e-4):
        raise ValueError("compact orientation requires normalized WXYZ")
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def make_goal(q_ref, dq_ref, root9, root_ref_quat, root_quat, vr21, feet_error_w, heights):
    """q1 received target; fixed-world error rotated into current pelvis frame."""
    prefix = q_ref.shape[:-1]
    for value, tail in (
        (q_ref, (23,)),
        (dq_ref, (23,)),
        (root9, (9,)),
        (root_ref_quat, (4,)),
        (root_quat, (4,)),
        (vr21, (21,)),
        (feet_error_w, (2, 3)),
        (heights, (2,)),
    ):
        if value.shape != (*prefix, *tail) or value.dtype != torch.float32:
            raise ValueError("compact goal shape/dtype mismatch")
        if value.device != q_ref.device or not torch.isfinite(value).all():
            raise ValueError("compact goal must be finite on one device")
    rotation = quaternion_matrix(root_quat)
    relative = rotation.transpose(-1, -2) @ quaternion_matrix(root_ref_quat)
    feet = feet_error_w @ rotation
    result = torch.cat(
        (q_ref, dq_ref, root9, relative[..., :, :2].flatten(-2), vr21, feet.flatten(-2), heights), -1
    )
    if result.shape[-1] != GOAL_DIM:
        raise AssertionError("compact goal width changed")
    return result


def pack_observation(history, goal):
    if history.shape[-1] != 930 or goal.shape != (*history.shape[:-1], GOAL_DIM):
        raise ValueError("compact observation requires matching history930 and goal90")
    if history.dtype != torch.float32 or goal.dtype != torch.float32 or history.device != goal.device:
        raise ValueError("compact observation requires float32 on one device")
    if not torch.isfinite(history).all() or not torch.isfinite(goal).all():
        raise ValueError("compact observation requires finite fields")
    indexes = torch.tensor(HW_IN_PAD29, device=history.device)
    latest = [history[..., 27:30]]
    for start in (30, 320, 610):
        value = history[..., start : start + 290].reshape(*history.shape[:-1], 10, 29)[..., -1, :]
        latest.append(value.index_select(-1, indexes))
    latest.append(history[..., 927:930])
    return torch.cat((*latest, goal), -1)


def contract():
    return dict(
        kind="native23_explicit_goal_residual165_v1",
        input_dim=OBS_DIM,
        goal_dim=GOAL_DIM,
        proprioception_dim=75,
        output_dim=23,
        current_proprioception="latest_frame_of_existing_term_major_history930",
        goal_order=[
            "hardware_q23",
            "hardware_dq23",
            "root9",
            "root_rotation6",
            "original29_vr21",
            "feet_error_current_pelvis6",
            "root_heights2",
        ],
        source_target="received_q1; adjacent_q1_minus_q0_velocity",
        source_timing_unchanged=True,
        source_motion_is_sonic_compatible=True,
        actor_is_frozen_sonic=False,
        ground_truth_root_is_sim_only=True,
        deployment_ready=False,
        hardware_authorized=False,
    )
