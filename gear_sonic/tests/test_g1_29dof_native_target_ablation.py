"""Synthetic arithmetic/interface tests; actual trials provide dynamics evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.scripts import simulate_g1_sonic_library_motions as stock
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_INNER_LOWER_HARDWARE,
    SAFE_TARGET_INNER_UPPER_HARDWARE,
)
from gear_sonic.utils.g1_29dof_native_target_ablation import (
    KEEP_HW,
    KEEP_IL,
    SOURCE_SCALE,
    TO_HW,
    Retained23TargetParameters,
    previous_action29,
    retained_pipeline,
    transform_history930,
)


@pytest.fixture
def parameters():
    default = np.zeros(29, dtype=np.float64)
    default[KEEP_HW] = SAFE_TARGET_DEFAULT_Q_HARDWARE
    scale = np.ones(29, dtype=np.float64)
    scale[KEEP_HW] = SOURCE_SCALE[TO_HW]

    def target(raw):
        if raw.shape != (29,) or not np.isfinite(raw).all() or np.any(np.abs(raw) >= 10):
            raise ValueError("original raw bound")
        return (
            (default + raw.astype(np.float64)[stock.MUJOCO_TO_ISAAC_INDEX] * scale)
            .astype(np.float32)
            .astype(np.float64)
        )

    return SimpleNamespace(
        default_angles=default,
        action_scale=scale,
        kps=np.ones(29),
        kds=np.ones(29),
        effort=np.ones(29) * 25,
        target=target,
    )


def test_zero_actions_and_history_stay_zero(parameters):
    raw = np.zeros(29, np.float32)
    for mode in ("bounded", "roundtrip_only"):
        p = Retained23TargetParameters(parameters, target_mode=mode)
        np.testing.assert_array_equal(p.target(raw), parameters.target(raw))
        np.testing.assert_array_equal(previous_action29(raw, target_mode=mode), raw)
        np.testing.assert_array_equal(
            transform_history930(np.zeros(930, np.float32), target_mode=mode, history_mode="applied"),
            np.zeros(930, np.float32),
        )


def test_bounded_targets_preserve_missing6_and_all_actuation(parameters):
    raw = np.linspace(-5, 5, 29, dtype=np.float32)
    p = Retained23TargetParameters(parameters, target_mode="bounded")
    target = p.target(raw)
    missing = np.setdiff1d(np.arange(29), KEEP_HW)
    np.testing.assert_array_equal(target[missing], parameters.target(raw)[missing])
    assert np.all(target[KEEP_HW] >= np.asarray(SAFE_TARGET_INNER_LOWER_HARDWARE) - 2e-7)
    assert np.all(target[KEEP_HW] <= np.asarray(SAFE_TARGET_INNER_UPPER_HARDWARE) + 2e-7)
    for name in ("default_angles", "action_scale", "kps", "kds", "effort"):
        assert getattr(p, name) is getattr(parameters, name)


def test_roundtrip_control_differs_only_where_projection_active(parameters):
    raw = np.linspace(-5, 5, 29, dtype=np.float32)
    a = Retained23TargetParameters(parameters, target_mode="bounded").target(raw)
    b = Retained23TargetParameters(parameters, target_mode="roundtrip_only").target(raw)
    _, _, clipped, _ = retained_pipeline(raw[KEEP_IL])
    assert clipped.any() and (~clipped).any()
    np.testing.assert_array_equal(a[KEEP_HW][~clipped[TO_HW]], b[KEEP_HW][~clipped[TO_HW]])
    np.testing.assert_array_equal(b[KEEP_HW][clipped[TO_HW]], parameters.target(raw)[KEEP_HW][clipped[TO_HW]])


def test_applied_history_changes_only_retained_previous_actions():
    rng = np.random.default_rng(19)
    history = rng.uniform(-5, 5, 930).astype(np.float32)
    before = history.copy()
    after = transform_history930(history, target_mode="bounded", history_mode="applied")
    allowed = (610 + np.arange(10)[:, None] * 29 + KEEP_IL[None, :]).reshape(-1)
    other = np.setdiff1d(np.arange(930), allowed)
    np.testing.assert_array_equal(before, history)
    np.testing.assert_array_equal(after[other], before[other])
    assert not np.array_equal(after[allowed], before[allowed])
    np.testing.assert_array_equal(
        transform_history930(history, target_mode="bounded", history_mode="raw"), history
    )


@pytest.mark.parametrize("mode", ["bad", "native23"])
def test_unknown_modes_rejected(parameters, mode):
    with pytest.raises(ValueError):
        Retained23TargetParameters(parameters, target_mode=mode)
    with pytest.raises(ValueError):
        transform_history930(np.zeros(930, np.float32), target_mode="bounded", history_mode=mode)


@pytest.mark.parametrize("value", [np.zeros(930), np.zeros(929, np.float32), np.full(930, np.nan, np.float32)])
def test_invalid_history_rejected(value):
    with pytest.raises(ValueError):
        transform_history930(value, target_mode="bounded", history_mode="raw")


def test_source_bound_never_bypassed(parameters):
    p = Retained23TargetParameters(parameters, target_mode="bounded")
    raw = np.zeros(29, np.float32)
    raw[28] = 10
    with pytest.raises(ValueError, match="original raw bound"):
        p.target(raw)


def test_wrong_source_scaling_rejected(parameters):
    parameters.action_scale[KEEP_HW[0]] += 0.01
    with pytest.raises(AssertionError):
        Retained23TargetParameters(parameters, target_mode="bounded")
