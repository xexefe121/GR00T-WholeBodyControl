import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_ankle_horizon_preview import (
    FAILURE,
    MAX_PREDICTIONS,
    choose_ankle_horizon_target,
)
from gear_sonic.utils.g1_true23_range_preview import RANGE_RESERVE_RAD, choose_range_target


def poses():
    result = np.zeros((50, 30))
    result[:, 3] = 1
    result[:, 7:] = SAFE_TARGET_DEFAULT_Q_HARDWARE
    return result


def test_safe_nominal_is_exact_and_one_prediction():
    raw = np.linspace(-0.2, 0.2, 23, dtype=np.float32)
    before, calls = raw.copy(), []

    def predict(target):
        calls.append(target.copy())
        return poses()

    selected, details = choose_ankle_horizon_target(raw, predict)
    np.testing.assert_array_equal(selected["raw"], before)
    np.testing.assert_array_equal(raw, before)
    assert len(calls) == details["preview_calls"] == 1


def test_anticipation_intervenes_before_old_next_control_violation():
    raw = np.zeros(23, np.float32)
    _, nominal = safe_target_transform_numpy(raw)
    calls = []

    def predict(target):
        result = poses()
        result[40, 18] = SAFE_TARGET_HARD_UPPER_HARDWARE[11] + 0.01 + 0.2 * (target[11] - nominal[11])
        calls.append(target.copy())
        return result

    old, old_details = choose_range_target(raw, lambda target: predict(target)[:10])
    np.testing.assert_array_equal(old["raw"], raw)
    assert not old_details["intervened"]
    calls.clear()
    selected, details = choose_ankle_horizon_target(raw, predict)
    assert details["intervened"] and details["changed_joints"] == [11]
    assert details["nominal_next_control_violation_rad"] == 0
    assert len(calls) <= MAX_PREDICTIONS
    assert selected["predicted"][40, 18] <= SAFE_TARGET_HARD_UPPER_HARDWARE[11] - RANGE_RESERVE_RAD
    unchanged_il = np.asarray(MUJOCO_TO_ISAACLAB_DOF) != 11
    np.testing.assert_array_equal(selected["raw"][unchanged_il], raw[unchanged_il])
    assert len({target.tobytes() for target in calls}) == len(calls)


def test_saturated_target_does_not_weaken_braking_or_retry_duplicate():
    raw = np.zeros(23, np.float32)
    raw[list(MUJOCO_TO_ISAACLAB_DOF).index(11)] = -9.99
    calls = []

    def predict(target):
        calls.append(target.copy())
        result = poses()
        result[8, 18] = SAFE_TARGET_HARD_UPPER_HARDWARE[11] + 0.01
        return result

    with pytest.raises(ValueError, match=FAILURE) as caught:
        choose_ankle_horizon_target(raw, predict)
    assert len(calls) == 1
    assert caught.value.preview_details["reason"] == "no additional inward target authority"


def test_new_coupled_next_control_violation_cannot_pass():
    def predict(target):
        result = poses()
        result[40, 18] = SAFE_TARGET_HARD_UPPER_HARDWARE[11] + 0.01 + target[11]
        result[4, 12] = SAFE_TARGET_HARD_UPPER_HARDWARE[5] + 0.01 - target[11]
        return result

    with pytest.raises(ValueError, match=FAILURE):
        choose_ankle_horizon_target(np.zeros(23, np.float32), predict)


def test_both_ankle_bounds_rejected_with_one_call():
    def predict(target):
        result = poses()
        result[25, 18], result[40, 18] = -10, 10
        return result

    with pytest.raises(ValueError, match=FAILURE) as caught:
        choose_ankle_horizon_target(np.zeros(23, np.float32), predict)
    assert caught.value.preview_details["preview_calls"] == 1
    assert caught.value.preview_details["reason"] == "same joint crosses both checked bounds"


def test_extended_non_ankle_motion_is_not_a_new_extra_gate():
    def predict(target):
        result = poses()
        result[40, 7] = 10
        return result

    selected, details = choose_ankle_horizon_target(np.zeros(23, np.float32), predict)
    assert not details["intervened"]
    assert selected["predicted"][40, 7] == 10


@pytest.mark.parametrize(
    "bad", [np.zeros(23), np.zeros(22, np.float32), np.full(23, np.nan, np.float32), np.full(23, 10, np.float32)]
)
def test_invalid_action_never_reaches_predictor(bad):
    with pytest.raises(ValueError):
        choose_ankle_horizon_target(bad, lambda _: pytest.fail("invalid action reached predictor"))


@pytest.mark.parametrize("bad", [np.zeros((10, 30)), np.full((50, 30), np.nan)])
def test_invalid_prediction_rejected(bad):
    with pytest.raises(ValueError, match="finite fifty"):
        choose_ankle_horizon_target(np.zeros(23, np.float32), lambda _: bad)
