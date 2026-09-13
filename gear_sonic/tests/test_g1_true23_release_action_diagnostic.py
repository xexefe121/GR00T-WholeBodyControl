import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_ACTION_SCALE, ISAACLAB_TO_MUJOCO_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_INNER_LOWER_HARDWARE,
    SAFE_TARGET_INNER_UPPER_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_release_action_diagnostic import bounded_linear_precompensation


def test_small_actions_recover_original_linear_targets_through_unchanged_transform():
    raw = np.linspace(-0.1, 0.1, 23, dtype=np.float32)
    compensated, projection = bounded_linear_precompensation(raw)
    safe, target = safe_target_transform_numpy(compensated)
    expected = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE) + raw[np.asarray(ISAACLAB_TO_MUJOCO_DOF)] * np.asarray(
        HARDWARE_23_ACTION_SCALE
    )
    np.testing.assert_allclose(target, expected, atol=1e-7, rtol=0)
    np.testing.assert_allclose(safe, raw, atol=2e-8, rtol=0)
    np.testing.assert_allclose(projection, 0, atol=1e-15, rtol=0)


@pytest.mark.parametrize("sign", [-1, 1])
def test_large_actions_still_obey_all_existing_target_and_raw_bounds(sign):
    raw = np.full(23, sign * 9.9, np.float32)
    compensated, projection = bounded_linear_precompensation(raw)
    safe, target = safe_target_transform_numpy(compensated)
    assert np.max(np.abs(compensated)) < 10
    assert np.all(target >= np.asarray(SAFE_TARGET_INNER_LOWER_HARDWARE) - 2e-7)
    assert np.all(target <= np.asarray(SAFE_TARGET_INNER_UPPER_HARDWARE) + 2e-7)
    assert np.any(np.abs(projection) > 0.1)
    assert np.isfinite(safe).all()


@pytest.mark.parametrize(
    "raw",
    [
        np.zeros(29, np.float32),
        np.zeros(23, np.float64),
        np.full(23, np.nan, np.float32),
        np.full(23, 10, np.float32),
    ],
)
def test_invalid_source_request_is_rejected(raw):
    with pytest.raises(ValueError):
        bounded_linear_precompensation(raw)
