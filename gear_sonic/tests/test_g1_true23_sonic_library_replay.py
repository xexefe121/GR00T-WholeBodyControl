from __future__ import annotations

import copy

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_sonic_library_replay import _quaternion_error_rad, validate_library_motion


def _motion() -> dict[str, np.ndarray]:
    # Schema-only fixture. Real hash-pinned physics is exercised separately by
    # test_g1_true23_deployment_envelope; this test needs no private motion files.
    quaternions = np.zeros((16, 24, 4))
    quaternions[..., 0] = 1.0
    return {
        "fps": np.array([50.0]),
        "joint_pos": np.zeros((16, 23)), "joint_vel": np.zeros((16, 23)),
        "body_pos_w": np.zeros((16, 24, 3)), "body_quat_w": quaternions,
        "body_lin_vel_w": np.zeros((16, 24, 3)),
        "body_ang_vel_w": np.zeros((16, 24, 3)),
    }


def test_true23_library_motion_schema() -> None:
    motion = _motion()
    assert validate_library_motion(motion) == 16
    assert motion["joint_pos"].shape[1] == 23
    assert motion["body_pos_w"].shape[1] == 24


def test_true23_library_motion_rejects_wrong_dof_and_nonfinite() -> None:
    motion = _motion()
    bad = copy.deepcopy(motion)
    bad["joint_pos"] = bad["joint_pos"][:, :22]
    with pytest.raises(ValueError, match="joint_pos shape"):
        validate_library_motion(bad)
    bad = copy.deepcopy(motion)
    bad["joint_pos"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_library_motion(bad)


def test_orientation_diagnostic_cannot_mutate_physics_or_reference():
    left = np.array([1.000001, 0., 0., 0.])
    right = np.array([0.999999, 0., 0., 0.])
    saved_left, saved_right = left.copy(), right.copy()
    assert _quaternion_error_rad(left, right) == pytest.approx(0.)
    np.testing.assert_array_equal(left, saved_left)
    np.testing.assert_array_equal(right, saved_right)
    left.flags.writeable = right.flags.writeable = False
    assert _quaternion_error_rad(left, right) == pytest.approx(0.)
