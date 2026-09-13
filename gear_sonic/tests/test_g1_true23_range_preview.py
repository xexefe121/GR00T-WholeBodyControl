import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_range_preview import (
    RANGE_RESERVE_RAD,
    choose_range_target,
    target_to_raw,
)


def safe_prediction():
    poses = np.zeros((10, 30))
    poses[:, 3] = 1
    poses[:, 7:] = SAFE_TARGET_DEFAULT_Q_HARDWARE
    return poses


def test_safe_prediction_preserves_exact_original_action():
    raw = np.linspace(-0.1, 0.1, 23, dtype=np.float32)
    before = raw.copy()
    row, details = choose_range_target(raw, lambda target: safe_prediction())
    assert not details["intervened"] and details["preview_calls"] == 1
    np.testing.assert_array_equal(row["raw"], before)
    np.testing.assert_array_equal(raw, before)


def test_brakes_only_predicted_violating_joint_and_checks_all_substeps():
    target = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=np.float32)
    target[5] = -0.2
    raw, _ = target_to_raw(target)
    checked = []

    def predict(candidate):
        poses = safe_prediction()
        poses[4, 12] = -0.31 + 0.3 * (candidate[5] + 0.2)
        checked.append(poses.copy())
        return poses

    result, details = choose_range_target(raw, predict)
    assert details["intervened"] and details["changed_joints"] == [5]
    assert 1 < details["preview_calls"] <= 16
    assert result["target"][5] > target[5]
    assert result["predicted"][4, 12] >= SAFE_TARGET_HARD_LOWER_HARDWARE[5] + RANGE_RESERVE_RAD
    np.testing.assert_allclose(safe_target_transform_numpy(result["raw"])[1], result["target"], atol=0, rtol=0)


def test_no_feasible_checked_target_never_returns_original_unsafe_request():
    def predict(target):
        poses = safe_prediction()
        poses[3, 12] = -0.28
        return poses

    with pytest.raises(ValueError, match="no verified next-control"):
        choose_range_target(np.zeros(23, dtype=np.float32), predict)


def test_coupled_new_violation_is_not_accepted():
    def predict(target):
        poses = safe_prediction()
        poses[1, 12] = -0.28 + target[5]
        poses[2, 18] = 0.28 + target[5]
        return poses

    with pytest.raises(ValueError, match="no verified next-control"):
        choose_range_target(np.zeros(23, dtype=np.float32), predict)


@pytest.mark.parametrize("bad", [np.full(23, np.nan, dtype=np.float32), np.zeros(22, dtype=np.float32)])
def test_invalid_input_is_rejected(bad):
    with pytest.raises(ValueError, match="finite float32"):
        choose_range_target(bad, lambda target: safe_prediction())


def test_invalid_prediction_and_out_of_envelope_target_rejected():
    with pytest.raises(ValueError, match="finite ten"):
        choose_range_target(np.zeros(23, dtype=np.float32), lambda target: np.zeros((1, 30)))
    with pytest.raises(ValueError, match="outside invertible"):
        target_to_raw(np.full(23, 100.0))


def test_target_round_trip_inside_actual_envelope():
    for factor in (-0.3, 0.0, 0.3):
        raw = np.full(23, factor, dtype=np.float32)
        target = safe_target_transform_numpy(raw)[1]
        inverse, decoded = target_to_raw(target)
        assert np.max(np.abs(inverse)) < 10
        np.testing.assert_allclose(decoded, target, atol=5e-7, rtol=0)
