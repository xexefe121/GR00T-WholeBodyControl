import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_true23_collision_sampling import interpolate_original_poses
from gear_sonic.utils.g1_true23_original_time_constraints import sampled_task_variables


class Coordinates:
    def __init__(self, poses):
        self.source = poses
        self.source_rotation = Rotation.from_quat(poses[:, [4, 5, 6, 3]])

    def qpos(self, x):
        q = (Rotation.from_rotvec(x[:, 3:6]) * self.source_rotation).as_quat()[:, [3, 0, 1, 2]]
        return np.column_stack((self.source[:, :3] + x[:, :3], q, x[:, 6:]))

    def serialized_variables(self, motion):
        rotations = Rotation.from_quat(motion["body_quat_w"][:, 0, [1, 2, 3, 0]])
        return np.column_stack(
            (
                motion["body_pos_w"][:, 0] - self.source[:, :3],
                (rotations * self.source_rotation.inv()).as_rotvec(),
                motion["joint_pos"],
            )
        )


@pytest.mark.parametrize("angle_scale", [0.0, 0.05, 0.4])
def test_full_slerp_task_coordinate_chain_rule_matches_independent_finite_difference(angle_scale):
    rng = np.random.default_rng(23)
    source = np.zeros((3, 30))
    source[:, :3] = rng.normal(0, 0.1, (3, 3))
    source[:, 3:7] = Rotation.from_rotvec(rng.normal(0, angle_scale, (3, 3))).as_quat()[:, [3, 0, 1, 2]]
    times, requested = np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.2, 0.8, 1.25, 1.9, 2.0])
    original_source = interpolate_original_poses(source, times, requested)
    original_source[:, 3:7] = (
        Rotation.from_rotvec(rng.normal(0, 0.02, (6, 3))) * Rotation.from_quat(original_source[:, [4, 5, 6, 3]])
    ).as_quat()[:, [3, 0, 1, 2]]
    control, original = Coordinates(source), Coordinates(original_source)
    x = rng.normal(0, 0.04, (3, 29))
    values, jac = sampled_task_variables(control, original, x, times, requested)
    reconstructed = original.qpos(values)
    np.testing.assert_allclose(
        reconstructed, interpolate_original_poses(control.qpos(x), times, requested), atol=1e-14
    )
    for column in range(x.size):
        delta = np.zeros_like(x)
        delta.ravel()[column] = 1e-6
        plus, _ = sampled_task_variables(control, original, x + delta, times, requested, derivatives=False)
        minus, _ = sampled_task_variables(control, original, x - delta, times, requested, derivatives=False)
        np.testing.assert_allclose(
            jac[:, column].toarray().ravel(), ((plus - minus) / 2e-6).ravel(), atol=1e-8, rtol=0
        )


def test_sampling_without_derivatives_returns_identical_variables_and_no_matrix():
    source = np.zeros((3, 30))
    source[:, 3] = 1
    control = original = Coordinates(source)
    x = np.zeros((3, 29))
    a, matrix = sampled_task_variables(control, original, x, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    b, absent = sampled_task_variables(control, original, x, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], derivatives=False)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(matrix.toarray(), np.eye(87), atol=1e-15)
    assert absent is None
