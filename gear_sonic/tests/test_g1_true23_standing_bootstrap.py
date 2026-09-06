"""Teacher labels must obey the existing output boundary, with explicit loss."""

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_standing_bootstrap import representable_standing_request


def test_zero_request_remains_exact_native_zero():
    raw, target, report = representable_standing_request(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    np.testing.assert_array_equal(raw, np.zeros(23, dtype=np.float32))
    np.testing.assert_array_equal(target, np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float32))
    assert not report["unrepresentable_joint_indices"]
    assert not report["output_boundary_changed"]


def test_interior_requests_round_trip_in_correct_joint_order():
    rng = np.random.default_rng(60)
    for _ in range(100):
        raw = rng.uniform(-0.5, 0.5, 23).astype(np.float32)
        _, requested = safe_target_transform_numpy(raw)
        inverse, actual, report = representable_standing_request(requested)
        np.testing.assert_allclose(inverse, raw, atol=3e-7, rtol=0)
        np.testing.assert_allclose(actual, requested, atol=6e-8, rtol=0)
        assert not report["unrepresentable_joint_indices"]


@pytest.mark.parametrize("sign", [-1, 1])
def test_out_of_image_targets_require_explicit_projection_and_finite_endpoints(sign):
    q = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE).copy()
    q[[4, 5, 10, 11]] += sign * 100
    with pytest.raises(ValueError, match="not representable"):
        representable_standing_request(q)
    raw, target, report = representable_standing_request(q, allow_projection=True)
    expected_hw = np.zeros(23, dtype=np.float32)
    expected_hw[[4, 5, 10, 11]] = sign * 10
    expected = expected_hw[np.asarray(MUJOCO_TO_ISAACLAB_DOF)]
    np.testing.assert_array_equal(raw, expected)
    np.testing.assert_array_equal(target, safe_target_transform_numpy(raw)[1])
    assert report["unrepresentable_joint_indices"] == [4, 5, 10, 11]
    assert report["maximum_request_change_rad"] > 99
    assert report["raw_action_limit"] == 10


@pytest.mark.parametrize("bad", [np.zeros(29), np.full(23, np.nan), np.full(23, np.inf)])
def test_invalid_target_rejected(bad):
    with pytest.raises(ValueError, match="finite 23"):
        representable_standing_request(bad)


def test_projection_option_must_be_boolean():
    with pytest.raises(ValueError, match="explicit boolean"):
        representable_standing_request(SAFE_TARGET_DEFAULT_Q_HARDWARE, allow_projection=1)
