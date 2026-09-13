"""Sign, damping direction, and joint isolation for the explicit torque law."""

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_bfmzero_ankle_barrier import AnkleRollRepulsion


def barrier(**kwargs):
    return AnkleRollRepulsion(list(HARDWARE_23_JOINT_NAMES), np.tile([-0.2618, 0.2618], (23, 1)), **kwargs)


def test_ankle_inward_spring_and_outward_damping_have_physical_sign():
    model = barrier()
    q, dq = np.zeros(23), np.zeros(23)
    q[5], q[11] = -0.2368, 0.2368
    dq[5], dq[11] = -1.0, 1.0
    expected = np.zeros(23)
    expected[5], expected[11] = 5.75, -5.75
    np.testing.assert_allclose(model.torque(q, dq), expected, atol=1e-12)
    dq *= -1
    expected[5], expected[11] = 3.75, -3.75
    np.testing.assert_allclose(model.torque(q, dq), expected, atol=1e-12)


def test_central_joints_and_non_ankle_joints_receive_no_torque():
    q = np.linspace(-1, 1, 23)
    dq = np.linspace(-100, 100, 23)
    q[[5, 11]] = 0
    np.testing.assert_array_equal(barrier().torque(q, dq), np.zeros(23))
    q[[5, 11]] = [-1, 1]
    torque = barrier().torque(q, dq)
    assert np.count_nonzero(torque) == 2
    assert torque[5] > 0 and torque[11] < 0


def test_zero_gains_are_identical_baseline_and_motor_clip_remains_external():
    q = np.zeros(23)
    dq = np.zeros(23)
    q[5], dq[5] = -0.28, -2
    np.testing.assert_array_equal(barrier(stiffness=0, damping=0).torque(q, dq), np.zeros(23))
    extra = barrier().torque(q, dq)
    assert extra[5] == pytest.approx(14.23)
    base = np.zeros(23)
    base[5] = 30
    assert (base + extra)[5] > 35
    assert np.clip(base + extra, -35, 35)[5] == 35


@pytest.mark.parametrize("parameters", [{"margin": 0.3}, {"stiffness": -1}, {"damping": float("nan")}])
def test_invalid_torque_law_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        barrier(**parameters)
