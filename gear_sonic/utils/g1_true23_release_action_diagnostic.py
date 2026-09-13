"""Simulation-only bounded linear-action compatibility for released SONIC.

Released SONIC uses linear joint targets and feeds its normalized target back.
Native23 uses a nonlinear tanh target transform. This diagnostic precompensates
the tanh inside its existing reachable envelope, leaving that transform, raw
bound, joint margins and torque limits active. Out-of-envelope linear requests
are explicitly projected and counted. It is not an approved hardware adapter.
"""

import numpy as np

from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF, NATIVE_IL23_ACTION_SCALE
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_RAW_ACTION_CLIP,
)


def bounded_linear_precompensation(raw):
    """Return inverse-tanh native23 request and explicit projection difference."""
    raw = np.asarray(raw)
    if raw.shape != (23,) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError("requires finite float32 native23 action")
    if np.max(np.abs(raw)) >= SAFE_TARGET_RAW_ACTION_CLIP:
        raise ValueError("released raw request exceeds existing diagnostic action bound")
    indices = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
    scale = np.asarray(NATIVE_IL23_ACTION_SCALE, dtype=np.float64)
    positive = np.asarray(SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE)[indices]
    negative = np.asarray(SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE)[indices]
    delta = raw.astype(np.float64) * scale
    capacity = np.where(delta >= 0, positive, negative)
    # Remain strictly inside both the tanh domain and the existing raw<10 gate.
    raw_bound = SAFE_TARGET_RAW_ACTION_CLIP - 1e-3
    max_ratio = np.minimum(np.tanh(raw_bound * scale / capacity), 1 - 1e-6)
    requested_ratio = delta / capacity
    ratio = np.clip(requested_ratio, -max_ratio, max_ratio)
    inverse = (capacity * np.arctanh(ratio) / scale).astype(np.float32)
    projection_delta_rad = (ratio * capacity - delta).astype(np.float32)
    return inverse, projection_delta_rad


def bounded_linear_precompensation_torch(raw):
    """Batched training equivalent; reject invalid requests before physics."""
    import torch

    if raw.ndim < 1 or raw.shape[-1] != 23 or raw.dtype != torch.float32:
        raise ValueError("requires float32 native23 action batches")
    if not torch.isfinite(raw).all() or torch.any(raw.abs() >= SAFE_TARGET_RAW_ACTION_CLIP):
        raise ValueError("released raw request violates finite/raw action bound")
    indices = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
    scale = torch.as_tensor(NATIVE_IL23_ACTION_SCALE, dtype=torch.float64, device=raw.device)
    positive = torch.as_tensor(
        np.asarray(SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE)[indices], dtype=torch.float64, device=raw.device
    )
    negative = torch.as_tensor(
        np.asarray(SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE)[indices], dtype=torch.float64, device=raw.device
    )
    delta = raw.double() * scale
    capacity = torch.where(delta >= 0, positive, negative)
    maximum = torch.tanh((SAFE_TARGET_RAW_ACTION_CLIP - 1e-3) * scale / capacity).clamp(max=1 - 1e-6)
    ratio = torch.clamp(delta / capacity, -maximum, maximum)
    inverse = (capacity * torch.atanh(ratio) / scale).float()
    return inverse, (ratio * capacity - delta).float()
