import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import encoder267_from_reference, motion_reference_terms
from gear_sonic.utils.g1_true23_step1b_mujoco import (
    _projected_gravity,
    _quaternion_matrix,
    world_angular_velocity_to_body,
)


@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize(
    "operation",
    [
        _quaternion_matrix,
        _projected_gravity,
        lambda q: world_angular_velocity_to_body(q, [1.0, 2.0, 3.0]),
    ],
)
def test_quaternion_observation_does_not_modify_input(operation, readonly):
    source = np.array([0.98, 0.02, -0.01, 0.2], dtype=np.float64)
    original = source.copy()
    source.setflags(write=not readonly)
    assert np.isfinite(operation(source)).all()
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("q", [[0.0] * 4, [1.0, 0.0], [float("nan"), 0.0, 0.0, 1.0]])
def test_invalid_rotation_rejected(q):
    with pytest.raises(ValueError, match="finite nonzero"):
        _quaternion_matrix(q)


def test_reference_encoder_does_not_mutate_readonly_robot_quaternion():
    motion = {
        "joint_pos": np.tile(SAFE_TARGET_DEFAULT_Q_HARDWARE, (12, 1)),
        "joint_vel": np.zeros((12, 23)),
        "body_pos_w": np.zeros((12, 24, 3)),
        "body_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (12, 24, 1)),
    }
    packet = motion_reference_terms(motion, 9)
    robot = np.array([0.98, 0.02, -0.01, 0.2])
    before = robot.copy()
    robot.setflags(write=False)
    result = encoder267_from_reference(packet, robot)
    assert result.shape == (267,) and np.isfinite(result).all()
    np.testing.assert_array_equal(robot, before)
