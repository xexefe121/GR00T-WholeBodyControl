import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_true23_pico_optical import (
    PARENTS, global_rotations, register_initial_se2, resample_optical, validate_pair,
)


def coherent_pair():
    rng = np.random.default_rng(45)
    count = 61
    aa = rng.normal(0, 0.1, (count, 24, 3))
    matrices = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(count, 24, 3, 3)
    world = global_rotations(matrices)
    offsets = rng.normal(0, 0.1, (24, 3))
    points = np.zeros((count, 24, 3))
    points[:, 0, 0] = np.arange(count) / 60
    points[:, 0, 2] = 0.8
    for joint, parent in enumerate(PARENTS[1:], 1):
        points[:, joint] = points[:, parent] + world[:, parent] @ offsets[joint]
    gt = dict(betas=np.zeros((count, 10)), global_orient=aa[:, 0], body_pose=aa[:, 1:].reshape(count, 69),
              transl=points[:, 0].copy(), joints=points, gender="female")
    sensors = dict(sensor_coordinates=points[:, [15, 20, 21]].copy(),
                   sensor_orientation=np.tile(np.eye(3), (count, 5, 1, 1)),
                   sensor_acceleration=np.zeros((count, 5, 3)))
    return gt, sensors


def test_coherent_optical_and_separate_sensors():
    gt, sensors = coherent_pair()
    sensors["sensor_coordinates"] += 0.03
    _, report = validate_pair(gt, sensors)
    assert report["rest_offset_max_variation_m"] < 1e-12
    assert not report["sensor_only_full_body_reconstruction"]
    assert report["sensor_head_to_optical_head_distance_p95_m"] > 0.05


@pytest.mark.parametrize("corrupt", ["world_position", "reflection", "nonfinite", "shape", "changing_betas"])
def test_bad_pairs_fail_closed(corrupt):
    gt, sensors = coherent_pair()
    if corrupt == "world_position":
        gt["joints"][30, 10, 1] += 0.001
    elif corrupt == "reflection":
        sensors["sensor_orientation"][30, 0, 0, 0] = -1
    elif corrupt == "nonfinite":
        sensors["sensor_acceleration"][0, 0, 0] = np.nan
    elif corrupt == "shape":
        sensors["sensor_coordinates"] = sensors["sensor_coordinates"][:-1]
    else:
        gt["betas"][30, 0] = 0.02
    with pytest.raises(ValueError):
        validate_pair(gt, sensors)


def test_uniform_resampling_preserves_speed_and_whole_span():
    gt, sensors = coherent_pair()
    local, _ = validate_pair(gt, sensors)
    points, world, times, queries = resample_optical(gt["joints"], local)
    assert len(times) == 61 and len(queries) == 51
    np.testing.assert_allclose(points[:, 0, 0], queries, atol=1e-15)
    np.testing.assert_allclose(points[[0, -1]], gt["joints"][[0, -1]], atol=1e-15)
    np.testing.assert_allclose(world[[0, -1]], global_rotations(local)[[0, -1]], atol=1e-14)


def test_one_registration_preserves_trajectory_and_joint_coordinates():
    poses = np.zeros((3, 36))
    poses[:, :3] = [[4, 5, 0.8], [4, 6, 0.7], [4, 7, 0.9]]
    poses[:, 3:7] = Rotation.from_euler("z", [np.pi / 2] * 3).as_quat()[:, [3, 0, 1, 2]]
    poses[:, 7:] = np.arange(87).reshape(3, 29) / 50
    output, report = register_initial_se2(poses)
    np.testing.assert_allclose(output[:, :2], [[0, 0], [1, 0], [2, 0]], atol=1e-14)
    np.testing.assert_array_equal(output[:, 2], poses[:, 2])
    np.testing.assert_array_equal(output[:, 7:], poses[:, 7:])
    assert not report["per_frame_reanchoring"]
