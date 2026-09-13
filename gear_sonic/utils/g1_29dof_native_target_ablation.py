"""SIM-only original29 controlled native23 command/history ablation.

All29 measured joints and outputs remain physical29. The two independent
factors are retained23 target projection and retained23 previous-action
semantics. A rounding-only target control keeps the native float32 roundtrip
where feasible but restores the original C++ target on projected components.
Nothing here changes model, gains, raw stops, effort bounds or native23 guards.
"""

import numpy as np

from gear_sonic.utils.g1_23dof_contract import (
    ISAACLAB_TO_MUJOCO_DOF,
    MUJOCO_TO_ISAACLAB_DOF,
    NATIVE_IL23_ACTION_SCALE,
    NATIVE_IL23_TO_CANONICAL_IL29,
    SOURCE_MJ29_KEEP_INDICES,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_RAW_ACTION_CLIP,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_source_action_codec import (
    SOURCE_SCALE_NATIVE_IL23,
    source_scaled_precompensation,
)

TARGET_MODES = ("roundtrip_only", "bounded")
HISTORY_MODES = ("raw", "applied")
KEEP_IL = np.asarray(NATIVE_IL23_TO_CANONICAL_IL29)
KEEP_HW = np.asarray(SOURCE_MJ29_KEEP_INDICES)
TO_HW = np.asarray(ISAACLAB_TO_MUJOCO_DOF)
TO_IL = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
NATIVE_SCALE = np.asarray(NATIVE_IL23_ACTION_SCALE, dtype=np.float64)
SOURCE_SCALE = np.asarray(SOURCE_SCALE_NATIVE_IL23, dtype=np.float64)
ACTION_RATIO = NATIVE_SCALE / SOURCE_SCALE


def retained_pipeline(raw23):
    """Exact existing pipeline, explicit projection mask, and applied history."""
    inverse, projection = source_scaled_precompensation(raw23)
    native_safe, target23 = safe_target_transform_numpy(inverse)
    delta = raw23.astype(np.float64) * SOURCE_SCALE
    capacity = np.where(
        delta >= 0,
        np.asarray(SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE)[TO_IL],
        np.asarray(SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE)[TO_IL],
    )
    ratio = delta / capacity
    maximum = np.minimum(np.tanh((SAFE_TARGET_RAW_ACTION_CLIP - 1e-3) * NATIVE_SCALE / capacity), 1 - 1e-6)
    clamped = ratio != np.clip(ratio, -maximum, maximum)
    encoded = (native_safe.astype(np.float64) * ACTION_RATIO).astype(np.float32)
    return target23, encoded, clamped, projection


def previous_action29(raw29, *, target_mode):
    """Re-encode only real previous29 requests; never reference/robot state."""
    if target_mode not in TARGET_MODES:
        raise ValueError("unknown original29 retained-target mode")
    raw = np.asarray(raw29)
    if raw.shape != (29,) or raw.dtype != np.float32 or not np.isfinite(raw).all() or np.any(np.abs(raw) >= 10):
        raise ValueError("previous action requires bounded finite float32 raw29")
    _, encoded, clamped, _ = retained_pipeline(raw[KEEP_IL])
    if target_mode == "roundtrip_only":
        encoded = np.where(clamped, raw[KEEP_IL], encoded).astype(np.float32)
    result = raw.copy()
    result[KEEP_IL] = encoded
    return result


def transform_history930(history, *, target_mode, history_mode):
    if target_mode not in TARGET_MODES or history_mode not in HISTORY_MODES:
        raise ValueError("unknown original29 target/history ablation mode")
    value = np.asarray(history)
    if value.shape != (930,) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("history requires finite float32 term-major930")
    result = value.copy()
    if history_mode == "applied":
        for index in range(10):
            section = slice(610 + 29 * index, 610 + 29 * (index + 1))
            result[section] = previous_action29(value[section], target_mode=target_mode)
    return result


class Retained23TargetParameters:
    """Preserve original29 actuation; vary only the retained23 target boundary."""

    def __init__(self, parameters, *, target_mode):
        if target_mode not in TARGET_MODES:
            raise ValueError("unknown original29 retained-target mode")
        np.testing.assert_array_equal(parameters.default_angles[KEEP_HW], SAFE_TARGET_DEFAULT_Q_HARDWARE)
        np.testing.assert_array_equal(parameters.action_scale[KEEP_HW], SOURCE_SCALE[TO_HW])
        self.base = parameters
        self.target_mode = target_mode
        for name in ("default_angles", "action_scale", "kps", "kds", "effort"):
            setattr(self, name, getattr(parameters, name))

    def target(self, raw29):
        # Original C++ boundary still validates all29 outputs before projection.
        original = self.base.target(raw29)
        bounded, _, clamped, _ = retained_pipeline(raw29[KEEP_IL])
        if self.target_mode == "roundtrip_only":
            bounded = np.where(clamped[TO_HW], original[KEEP_HW], bounded)
        result = original.copy()
        result[KEEP_HW] = bounded
        return result

    def descriptor(self):
        return dict(
            kind="original29_native23_retained_target_ablation_v1",
            target_mode=self.target_mode,
            retained_source_hardware29_indices=KEEP_HW.tolist(),
            native_target_bounds_unchanged=True,
            original29_model_gains_effort_and_missing6_targets_unchanged=True,
            target_roundtrip_control_restores_original_on_projected_components=self.target_mode
            == "roundtrip_only",
            original_raw_finite_height_tilt_stops_unchanged=True,
            native23_guard_disabled=False,
            actual_joint_range_safety_qualified=False,
            hardware_authorized=False,
            deployment_ready=False,
        )
