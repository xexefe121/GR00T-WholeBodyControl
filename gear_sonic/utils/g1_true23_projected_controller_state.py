"""Simulator-only controller state for random-reference training resets.

A reset target is a synthetic initial condition, not a reachable physical
history or an acquisition command. Live/measured transitions must carry the
target actually applied by their preceding controller instead of reseeding.
"""

from __future__ import annotations

import numpy as np
import torch

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_ACTION_SCALE, MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
)


def applied_target_native_torch(target: torch.Tensor) -> torch.Tensor:
    """Encode an applied hardware target; do not apply the tanh transform again."""
    if target.ndim < 1 or target.shape[-1] != 23:
        raise ValueError("applied target requires final dimension 23")
    hardware = (target - target.new_tensor(SAFE_TARGET_DEFAULT_Q_HARDWARE)) / target.new_tensor(
        HARDWARE_23_ACTION_SCALE
    )
    return hardware[..., list(MUJOCO_TO_ISAACLAB_DOF)]


def applied_target_native_numpy(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    if target.ndim < 1 or target.shape[-1] != 23 or not np.isfinite(target).all():
        raise ValueError("applied target requires finite final dimension 23")
    hardware = (target - np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float32)) / np.asarray(
        HARDWARE_23_ACTION_SCALE, dtype=np.float32
    )
    return hardware[..., list(MUJOCO_TO_ISAACLAB_DOF)]


def synthetic_reset_target_torch(q: torch.Tensor, dq: torch.Tensor, profile):
    """Closest-to-zero-effort PD seed within hard/effort bounds, plus bad rows.

    No previous target exists in this synthetic reset distribution, so no
    slew bound is imposed on initialization. Every subsequent substep has
    the unchanged slew bound. Infeasible rows remain explicitly invalid.
    """
    if q.ndim != 2 or q.shape[-1] != 23 or dq.shape != q.shape:
        raise ValueError("synthetic reset requires [N,23] q/dq")
    kp, kd, effort = (q.new_tensor(getattr(profile, key)) for key in ("kp", "kd", "effort"))
    cap = 0.95 * 0.25 * effort
    center = q + kd * dq / kp
    hard_low = q.new_tensor(SAFE_TARGET_HARD_LOWER_HARDWARE) + profile.target_margin_rad
    hard_high = q.new_tensor(SAFE_TARGET_HARD_UPPER_HARDWARE) - profile.target_margin_rad
    lower = torch.maximum(hard_low, q + (kd * dq - cap) / kp)
    upper = torch.minimum(hard_high, q + (kd * dq + cap) / kp)
    invalid = (lower > upper).any(-1) | (~torch.isfinite(q)).any(-1) | (~torch.isfinite(dq)).any(-1)
    target = torch.maximum(torch.minimum(center, upper), lower)
    fallback = torch.maximum(torch.minimum(torch.nan_to_num(q), hard_high), hard_low)
    return torch.where(invalid[:, None], fallback, target), invalid


def synthetic_reset_target_numpy(q, dq, kp, kd, effort, *, margin=0.05):
    """Scalar diagnostic equivalent; never use to initialize measured hardware."""
    values = tuple(np.asarray(item, dtype=np.float64) for item in (q, dq, kp, kd, effort))
    if any(item.shape != (23,) or not np.isfinite(item).all() for item in values):
        raise ValueError("synthetic reset requires finite 23-joint state and gains")
    q, dq, kp, kd, effort = values
    if np.any(kp <= 0) or np.any(kd <= 0) or np.any(effort <= 0) or not np.isfinite(margin) or margin <= 0:
        raise ValueError("synthetic reset requires positive gains, effort and margin")
    cap = 0.95 * 0.25 * effort
    lower = np.maximum(np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + margin, q + (kd * dq - cap) / kp)
    upper = np.minimum(np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) - margin, q + (kd * dq + cap) / kp)
    if np.any(lower > upper):
        raise ValueError("infeasible synthetic reset target")
    return np.clip(q + kd * dq / kp, lower, upper)
