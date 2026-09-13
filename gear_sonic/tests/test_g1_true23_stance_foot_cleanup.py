import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_true23_stance_foot_cleanup import bounded_stance_targets, clean_stance_feet


def test_small_stance_defect_flattens_with_same_horizontal_position_and_yaw():
    positions = np.array([[[0.1, 0.2, 0.043], [-0.1, -0.2, 0.028]]])
    rotations = Rotation.from_euler("zyx", [[0.4, 0.02, -0.01], [-0.3, -0.03, 0.01]]).as_matrix()[None]
    result, orientation, proof = bounded_stance_targets(
        positions, rotations, np.ones((1, 2), dtype=bool), [0.035, 0.035]
    )
    np.testing.assert_array_equal(result[:, :, :2], positions[:, :, :2])
    np.testing.assert_allclose(result[:, :, 2], 0.035)
    np.testing.assert_allclose(orientation[:, :, 2, :], np.array([[[0, 0, 1], [0, 0, 1]]]), atol=1e-15)
    np.testing.assert_allclose(
        np.arctan2(orientation[:, :, 1, 0], orientation[:, :, 0, 0]),
        np.arctan2(rotations[:, :, 1, 0], rotations[:, :, 0, 0]),
        atol=1e-15,
    )
    assert proof["translation_capped_frames_by_foot"] == [[], []]


def test_large_defect_is_partially_corrected_not_silently_declared_grounded():
    positions = np.array([[[0, 0.1, 0.1], [0, -0.1, 0.08]]])
    rotations = Rotation.from_euler("x", [0.3, 0.2]).as_matrix()[None]
    result, orientation, proof = bounded_stance_targets(
        positions, rotations, np.ones((1, 2), dtype=bool), [0.035, 0.035]
    )
    np.testing.assert_allclose(result[:, :, 2], [[0.08, 0.06]])
    change = Rotation.from_matrix((orientation @ rotations.transpose(0, 1, 3, 2)).reshape(-1, 3, 3)).magnitude()
    np.testing.assert_allclose(change, 0.15)
    assert proof["translation_capped_frames_by_foot"] == [[0], [0]]
    assert proof["rotation_capped_frames_by_foot"] == [[0], [0]]
    assert proof["partial_corrections_are_not_complete_grounding"]


def test_inactive_airborne_foot_is_bit_exact():
    positions = np.array([[[0.1, 0.2, 0.043], [-0.1, -0.2, 0.3]]])
    rotations = Rotation.from_euler("zyx", [[0.4, 0.02, -0.01], [-0.3, -0.03, 0.01]]).as_matrix()[None]
    result, orientation, _ = bounded_stance_targets(
        positions, rotations, np.array([[True, False]]), [0.035, 0.035]
    )
    np.testing.assert_array_equal(result[:, 1], positions[:, 1])
    np.testing.assert_array_equal(orientation[:, 1], rotations[:, 1])


@pytest.mark.parametrize("poses", [np.zeros((2, 30)), np.zeros((4, 29)), np.full((4, 30), np.nan)])
def test_incomplete_nonfinite_or_wrong_robot_rejected_before_physics(poses):
    with pytest.raises(ValueError, match="full finite native23"):
        clean_stance_feet(None, poses)
