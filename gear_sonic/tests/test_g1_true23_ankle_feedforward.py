from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
)
from gear_sonic.utils.g1_true23_ankle_feedforward import (
    FAILURE,
    choose_ankle_feedforward,
    native_pd_feedforward_numpy,
    validate_feedforward,
)
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile, native_model_pd_numpy

ROOT = Path(__file__).resolve().parents[2]


def profile():
    return NativeModelActuationProfile.from_sim_config(
        ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
    )


def prediction():
    q, v = np.zeros((10, 30)), np.zeros((10, 29))
    q[:, 3], q[:, 7:] = 1, SAFE_TARGET_DEFAULT_Q_HARDWARE
    return q, v


def choose(predict):
    return choose_ankle_feedforward(
        np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE),
        predict,
        np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE),
        np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE),
        np.asarray(profile().velocity),
    )


def test_zero_offset_preserves_original_pd_results_exactly():
    p = profile()
    q, target, v = (np.linspace(a, b, 23) for a, b in ((-0.2, 0.3), (-0.5, 0.1), (-2, 4)))
    for original, new in zip(
        native_model_pd_numpy(target, q, v, p),
        native_pd_feedforward_numpy(target, q, v, p, np.zeros(23)),
        strict=True,
    ):
        np.testing.assert_array_equal(original, new)


def test_total_effort_is_clipped_after_correction_not_before():
    p, ff = profile(), np.zeros(23)
    ff[11] = 10
    requested, applied, invalid, cost = native_pd_feedforward_numpy(
        np.full(23, 10.0), np.zeros(23), np.zeros(23), p, ff
    )
    assert not invalid and cost > 0
    assert requested[11] > 35
    assert applied[11] == 35
    assert np.max(np.abs(applied) / p.effort) == 1


def test_smallest_verified_candidate_only_and_target_unchanged():
    seen = []

    def predict(target, correction):
        seen.append((target.copy(), correction.copy()))
        q, v = prediction()
        q[4, 18] = SAFE_TARGET_HARD_UPPER_HARDWARE[11] + 0.005 + 0.01 * correction[11]
        return q, v

    selected, details = choose(predict)
    assert details["magnitude_nm"] == 2.5 and len(seen) == 2
    assert selected["feedforward23"][11] == -2.5
    assert np.count_nonzero(selected["feedforward23"]) == 1
    assert selected["range_excess_rad"] == 0
    for target, _ in seen:
        np.testing.assert_array_equal(target, SAFE_TARGET_DEFAULT_Q_HARDWARE)


def test_coupled_new_range_or_velocity_violation_is_never_accepted():
    for kind in ("range", "velocity"):

        def predict(target, correction):
            q, v = prediction()
            q[4, 18] = SAFE_TARGET_HARD_UPPER_HARDWARE[11] + 0.005 + 0.01 * correction[11]
            if kind == "range":
                q[6, 12] = SAFE_TARGET_HARD_UPPER_HARDWARE[5] - 0.01 * correction[11]
            else:
                v[6, 6] = profile().velocity[0] * 1.01
            return q, v

        with pytest.raises(ValueError, match=FAILURE):
            choose(predict)


def test_nonankle_violation_does_not_expand_torque_scope():
    calls = []

    def predict(target, correction):
        calls.append(correction)
        q, v = prediction()
        q[0, 7] = 10
        return q, v

    with pytest.raises(ValueError, match=FAILURE) as caught:
        choose(predict)
    assert len(calls) == 1
    assert caught.value.feedforward_details["reason"] == "nominal range violation includes a non-ankle joint"


@pytest.mark.parametrize("bad", [np.zeros(22), np.zeros(23, np.float32), np.full(23, np.nan), np.full(23, 11)])
def test_bad_feedforward_rejected(bad):
    with pytest.raises(ValueError):
        validate_feedforward(bad)


def test_arm_feedforward_forbidden():
    value = np.zeros(23)
    value[16] = 1
    with pytest.raises(ValueError, match="four ankle"):
        validate_feedforward(value)


def test_invalid_state_produces_no_actuator_output():
    correction = np.zeros(23)
    correction[11] = -2.5
    _, applied, invalid, _ = native_pd_feedforward_numpy(
        np.full(23, np.nan), np.zeros(23), np.zeros(23), profile(), correction
    )
    assert invalid
    np.testing.assert_array_equal(applied, np.zeros(23))
