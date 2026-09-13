import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_true23_start_registration import register_motion_start


def fixture(dtype=np.float64):
    rng = np.random.default_rng(17)
    p = rng.normal(size=(5, 24, 3))
    p[:, 0, 2] += 0.76
    q = Rotation.from_rotvec(rng.normal(scale=0.2, size=(120, 3))).as_quat(scalar_first=True).reshape(5, 24, 4)
    q[0, 0] = Rotation.from_euler("xyz", [0.1, -0.2, -1.5]).as_quat(scalar_first=True)
    return {
        "body_pos_w": p.astype(dtype),
        "body_quat_w": q.astype(dtype),
        "body_lin_vel_w": rng.normal(size=(5, 24, 3)).astype(dtype),
        "body_ang_vel_w": rng.normal(size=(5, 24, 3)).astype(dtype),
        "joint_pos": rng.normal(size=(5, 23)).astype(dtype),
        "joint_vel": rng.normal(size=(5, 23)).astype(dtype),
        "fps": np.array([50.0]),
        "joint_names": np.array([f"j{i}" for i in range(23)]),
    }


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_registration_is_one_fixed_se2_transform_not_changed_choreography(dtype):
    original = fixture(dtype)
    before = {k: v.copy() for k, v in original.items()}
    result, proof = register_motion_start(original, target_root_xy=(2.0, -3.0), target_yaw_rad=0.7)
    for key in original:
        np.testing.assert_array_equal(original[key], before[key])
        if key not in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            np.testing.assert_array_equal(result[key], original[key])
    np.testing.assert_allclose(result["body_pos_w"][0, 0, :2], [2.0, -3.0], atol=1e-6)
    ori = Rotation.from_quat(result["body_quat_w"][0, 0], scalar_first=True).as_euler("xyz")
    np.testing.assert_allclose(ori, [0.1, -0.2, 0.7], atol=1e-6)
    np.testing.assert_array_equal(result["body_pos_w"][..., 2], original["body_pos_w"][..., 2])
    for axis in (0, 1):
        before_distance = np.linalg.norm(np.diff(original["body_pos_w"], axis=axis), axis=-1)
        after_distance = np.linalg.norm(np.diff(result["body_pos_w"], axis=axis), axis=-1)
        np.testing.assert_allclose(after_distance, before_distance, atol=1e-6, rtol=1e-6)
    assert not proof["measured_robot_state_used"] and not proof["midrun_reregistration_or_pose_reset"]
    assert not proof["same_unregistered_benchmark_claimed"] and not proof["deployment_ready"]


def test_all_world_vectors_and_quaternions_receive_same_rotation():
    original = fixture()
    result, proof = register_motion_start(original)
    rotation = Rotation.from_euler("z", proof["fixed_world_yaw_rotation_rad"])
    for key in ("body_lin_vel_w", "body_ang_vel_w"):
        np.testing.assert_allclose(result[key], rotation.apply(original[key].reshape(-1, 3)).reshape(5, 24, 3))
    expected = rotation * Rotation.from_quat(original["body_quat_w"].reshape(-1, 4), scalar_first=True)
    actual = Rotation.from_quat(result["body_quat_w"].reshape(-1, 4), scalar_first=True)
    np.testing.assert_allclose((expected.inv() * actual).magnitude(), 0.0, atol=1e-12)


def test_start_calibration_uses_first_sample_not_later_source_state():
    original = fixture()
    first, proof = register_motion_start(original)
    original["body_pos_w"][1:] += 100
    original["body_quat_w"][1:] *= -1
    second, changed = register_motion_start(original)
    assert changed == proof
    np.testing.assert_array_equal(second["body_pos_w"][0], first["body_pos_w"][0])


@pytest.mark.parametrize("key", ["body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_vel"])
def test_nonfinite_source_rejected(key):
    original = fixture()
    original[key].flat[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        register_motion_start(original)


def test_bad_heading_or_nonunit_quaternion_rejected():
    with pytest.raises(ValueError, match="finite"):
        register_motion_start(fixture(), target_yaw_rad=np.inf)
    original = fixture()
    original["body_quat_w"][0, 0] *= 1.1
    with pytest.raises(ValueError, match="finite"):
        register_motion_start(original)
