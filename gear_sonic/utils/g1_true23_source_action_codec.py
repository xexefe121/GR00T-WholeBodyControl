"""Simulation-only released29 action units inside unchanged native23 bounds.

Do not confuse the released policy's output units with native23 motor scales.
The retained hip-pitch outputs use the source 7520_22 scale, although native23
uses 7520_14 motors. This adapter changes target decoding and previous-action
units only. It neither changes native motor physics nor authorizes deployment.
Historical ``released_bounded_linear`` artifacts retain their v1 native scales.
"""

import numpy as np

from gear_sonic.utils.g1_23dof_contract import (
    MUJOCO_TO_ISAACLAB_DOF,
    NATIVE_IL23_ACTION_SCALE,
    NATIVE_IL23_JOINT_NAMES,
    NATIVE_IL23_TO_CANONICAL_IL29,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_RAW_ACTION_CLIP,
)
from gear_sonic.utils.g1_true23_generalist_corpus import canonical_digest

SOURCE_ACTION_CONVENTION = "released29_scale_bounded_linear_v2"
_OMEGA = 10 * 2.0 * 3.1415926535  # Original deployment header, not a native motor setting.
_MOTOR_SCALES = {
    "7520_22": 0.25 * 139.0 / (0.025101925 * _OMEGA * _OMEGA),
    "7520_14": 0.25 * 88.0 / (0.010177520 * _OMEGA * _OMEGA),
    "5020": 0.25 * 25.0 / (0.003609725 * _OMEGA * _OMEGA),
}


def _source_scale(name):
    if name.endswith(("hip_pitch_joint", "hip_roll_joint", "knee_joint")):
        return _MOTOR_SCALES["7520_22"]
    if name.endswith(("hip_yaw_joint", "waist_yaw_joint")):
        return _MOTOR_SCALES["7520_14"]
    return _MOTOR_SCALES["5020"]


SOURCE_SCALE_NATIVE_IL23 = tuple(_source_scale(name) for name in NATIVE_IL23_JOINT_NAMES)
_PREVIOUS_ACTION_INDICES = 610 + np.arange(10)[:, None] * 29 + np.asarray(NATIVE_IL23_TO_CANONICAL_IL29)[None, :]
_PREVIOUS_ACTION_RATIO = np.asarray(NATIVE_IL23_ACTION_SCALE) / np.asarray(SOURCE_SCALE_NATIVE_IL23)


def source_action_codec_contract():
    result = dict(
        kind=SOURCE_ACTION_CONVENTION,
        source_scale_native_il23=list(SOURCE_SCALE_NATIVE_IL23),
        joint_names=list(NATIVE_IL23_JOINT_NAMES),
        source_target="source_default_angle + retained_released_raw * source_action_scale",
        native_inverse="unchanged_native23_tanh_inverse_and_reachable_envelope",
        previous_action="projected_physical_target_minus_source_default_divided_by_source_scale",
        source_raw_history_exact_only_when_unprojected=True,
        native_physics_and_limits_unchanged=True,
        hardware_authorized=False,
        deployment_ready=False,
    )
    return {**result, "contract_sha256": canonical_digest(result)}


def source_scaled_precompensation(raw):
    """Return native inverse-tanh action plus source target projection in radians."""
    raw = np.asarray(raw)
    if raw.shape != (23,) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError("requires finite float32 native23 source action")
    if np.max(np.abs(raw)) >= SAFE_TARGET_RAW_ACTION_CLIP:
        raise ValueError("released source request exceeds existing raw action bound")
    indices = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
    native_scale = np.asarray(NATIVE_IL23_ACTION_SCALE, dtype=np.float64)
    delta = raw.astype(np.float64) * np.asarray(SOURCE_SCALE_NATIVE_IL23)
    capacity = np.where(
        delta >= 0,
        np.asarray(SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE)[indices],
        np.asarray(SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE)[indices],
    )
    maximum = np.minimum(np.tanh((SAFE_TARGET_RAW_ACTION_CLIP - 1e-3) * native_scale / capacity), 1 - 1e-6)
    ratio = np.clip(delta / capacity, -maximum, maximum)
    inverse = (capacity * np.arctanh(ratio) / native_scale).astype(np.float32)
    return inverse, (ratio * capacity - delta).astype(np.float32)


def source_action_history_numpy(history):
    """Convert all ten previous-target blocks, leaving measured state untouched."""
    if history.shape != (930,) or history.dtype != np.float32 or not np.isfinite(history).all():
        raise ValueError("requires finite float32 history930")
    result = history.copy()
    result[_PREVIOUS_ACTION_INDICES] = (
        result[_PREVIOUS_ACTION_INDICES].astype(np.float64) * _PREVIOUS_ACTION_RATIO
    ).astype(np.float32)
    return result


def source_action_history_torch(history):
    """Batched training counterpart for canonical padded29 term-major histories."""
    import torch

    if history.ndim < 1 or history.shape[-1] != 930 or history.dtype != torch.float32:
        raise ValueError("requires float32 history930 batches")
    if not torch.isfinite(history).all():
        raise ValueError("history930 must be finite")
    indices = torch.as_tensor(_PREVIOUS_ACTION_INDICES, device=history.device)
    ratios = torch.as_tensor(_PREVIOUS_ACTION_RATIO, device=history.device)
    result = history.clone()
    result[..., indices] = (result[..., indices].double() * ratios).float()
    return result


def source_normalize_native_action_torch(action):
    """Encode the applied native target in original units before history storage."""
    import torch

    if action.ndim < 1 or action.shape[-1] != 23 or action.dtype != torch.float32:
        raise ValueError("requires float32 safe native action23")
    if not torch.isfinite(action).all():
        raise ValueError("safe native action must be finite")
    return (action.double() * torch.as_tensor(_PREVIOUS_ACTION_RATIO, device=action.device)).float()


def source_scaled_precompensation_torch(raw):
    """Batched source target decoding; native inverse scales remain independent."""
    import torch

    if raw.ndim < 1 or raw.shape[-1] != 23 or raw.dtype != torch.float32:
        raise ValueError("requires float32 native23 source action batches")
    if not torch.isfinite(raw).all() or torch.any(raw.abs() >= SAFE_TARGET_RAW_ACTION_CLIP):
        raise ValueError("released source request violates finite/raw action bound")
    indices = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
    native_scale = torch.as_tensor(NATIVE_IL23_ACTION_SCALE, dtype=torch.float64, device=raw.device)
    source_scale = torch.as_tensor(SOURCE_SCALE_NATIVE_IL23, dtype=torch.float64, device=raw.device)
    positive = torch.as_tensor(
        np.asarray(SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE)[indices], dtype=torch.float64, device=raw.device
    )
    negative = torch.as_tensor(
        np.asarray(SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE)[indices], dtype=torch.float64, device=raw.device
    )
    delta = raw.double() * source_scale
    capacity = torch.where(delta >= 0, positive, negative)
    maximum = torch.tanh((SAFE_TARGET_RAW_ACTION_CLIP - 1e-3) * native_scale / capacity).clamp(max=1 - 1e-6)
    ratio = torch.clamp(delta / capacity, -maximum, maximum)
    return (capacity * torch.atanh(ratio) / native_scale).float(), (ratio * capacity - delta).float()
