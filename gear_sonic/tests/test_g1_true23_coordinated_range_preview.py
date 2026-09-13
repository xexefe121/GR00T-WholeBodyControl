import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_coordinated_range_preview import MAX_PREDICTIONS, choose_coordinated_target
from gear_sonic.utils.g1_true23_range_preview import choose_range_target


def neutral_poses():
    q = np.zeros((10, 30))
    q[:, 3] = 1
    q[:, 7:] = (np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE)) / 2
    return q


def test_safe_action_retains_exact_bytes_and_one_call():
    raw = np.linspace(-0.1, 0.1, 23, dtype=np.float32)
    calls = []

    def predict(target):
        calls.append(target.copy())
        return neutral_poses()

    chosen, details = choose_coordinated_target(raw, predict)
    np.testing.assert_array_equal(chosen["raw"], raw)
    assert len(calls) == 1 and not details["intervened"]


def test_coupled_constraint_can_require_other_leg_joint_not_violating_ankle():
    raw = np.zeros(23, np.float32)
    _, requested = safe_target_transform_numpy(raw)

    def predict(target):
        q = neutral_poses()
        q[:, 18] = SAFE_TARGET_HARD_UPPER_HARDWARE[11] + 0.005 - 0.1 * (target[6] - requested[6])
        return q

    with pytest.raises(ValueError, match="no verified next-control"):
        choose_range_target(raw, predict)
    chosen, details = choose_coordinated_target(raw, predict)
    assert chosen["maximum_violation_rad"] == 0 and np.min(chosen["margins"]) >= 0
    assert 6 in details["changed_joints"]
    np.testing.assert_allclose(chosen["target"][13:], requested[13:], rtol=0, atol=5e-7)
    assert max(np.abs(chosen["raw"])) < 10
    assert details["coordinated_predictions"] <= MAX_PREDICTIONS


def test_unreachable_callback_rejects_with_bounded_predictions():
    calls = []

    def predict(target):
        calls.append(target.copy())
        q = neutral_poses()
        q[:, 18] = SAFE_TARGET_HARD_UPPER_HARDWARE[11] + 0.01
        return q

    with pytest.raises(ValueError, match="no verified coordinated") as caught:
        choose_coordinated_target(np.zeros(23, np.float32), predict)
    assert len(calls) <= MAX_PREDICTIONS
    assert caught.value.coordinated_details["best_unverified_violation_rad"] > 0


@pytest.mark.parametrize(
    "bad", [np.zeros(23), np.zeros(22, np.float32), np.full(23, np.nan, np.float32), np.full(23, 10, np.float32)]
)
def test_bad_source_action_never_reaches_predictor(bad):
    with pytest.raises(ValueError):
        choose_coordinated_target(bad, lambda _: pytest.fail("invalid action reached prediction"))


def test_bad_prediction_cannot_be_accepted():
    with pytest.raises(ValueError, match="finite ten"):
        choose_coordinated_target(np.zeros(23, np.float32), lambda _: np.full((10, 30), np.nan))
