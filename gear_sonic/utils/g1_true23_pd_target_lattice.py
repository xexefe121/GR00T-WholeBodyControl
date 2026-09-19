"""Nearest existing float32 SONIC-codec target for offline PD optimization.

This does not change the source decoder, native transform, or motor limits.
It searches the original monotone scalar codec and returns an actual emitted
target. In particular, encoding an already-emitted target preserves its bits.
"""

import numpy as np

from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_RAW_ACTION_CLIP,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_source_action_codec import (
    SOURCE_SCALE_NATIVE_IL23,
    source_scaled_precompensation,
)


def _ordered_float(bits):
    bits = np.asarray(bits, dtype=np.uint32)
    raw = np.where(bits & np.uint32(0x80000000), bits ^ np.uint32(0x80000000), ~bits)
    return raw.astype(np.uint32).view(np.float32)


def _ordered_bits(value):
    raw = np.asarray(value, dtype=np.float32).view(np.uint32)
    return np.where(raw & np.uint32(0x80000000), ~raw, raw ^ np.uint32(0x80000000)).astype(np.uint64)


def _emit(raw):
    inverse, _ = source_scaled_precompensation(raw)
    return safe_target_transform_numpy(inverse)[1]


def nearest_original_codec_target(target):
    """Project onto original emitted targets, with a reproducible raw witness.

    Binary search uses IEEE float ordering, not an approximate inverse followed
    by another rounding. All 23 channels remain inside the existing raw bound.
    The per-channel forward map is the unmodified production NumPy codec.
    """
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (23,) or not np.isfinite(target).all():
        raise ValueError("target lattice requires a finite hardware-order target23")
    indices = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
    limit = np.nextafter(np.float32(SAFE_TARGET_RAW_ACTION_CLIP), np.float32(0))
    low_raw, high_raw = np.full(23, -limit, np.float32), np.full(23, limit, np.float32)
    low_target, high_target = _emit(low_raw), _emit(high_raw)
    if np.any(target < low_target) or np.any(target > high_target):
        raise ValueError("target lattice cannot expand the original codec envelope")
    estimate = ((target - SAFE_TARGET_DEFAULT_Q_HARDWARE)[indices] / SOURCE_SCALE_NATIVE_IL23).astype(np.float32)
    estimate = np.clip(estimate, -limit, limit)
    emitted = _emit(estimate)
    if np.array_equal(emitted, target):
        return emitted, estimate
    goal = target[indices]
    low, high = _ordered_bits(low_raw), _ordered_bits(high_raw)
    for _ in range(32):
        active = high - low > 1
        if not np.any(active):
            break
        middle = low + (high - low) // 2
        value = _emit(_ordered_float(middle))[indices]
        low = np.where(active & (value <= goal), middle, low)
        high = np.where(active & (value > goal), middle, high)
    if np.any(high - low > 1):
        raise RuntimeError("original codec lattice search did not converge")
    low_raw, high_raw = _ordered_float(low), _ordered_float(high)
    low_error = np.abs(_emit(low_raw)[indices] - goal)
    high_error = np.abs(_emit(high_raw)[indices] - goal)
    witness = np.where(high_error < low_error, high_raw, low_raw).astype(np.float32)
    return _emit(witness), witness
