from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_ramp_clearance_path import (
    arm_path_columns,
    com_path_jacobian,
    ramp_linear_constraints,
    verify_arm_pose_replacement,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


def example():
    poses = np.zeros((80, 30))
    poses[:, 3] = 1
    timeline = {
        "phases": [
            {"name": "acquisition_ramp", "frame_start": 5, "frame_stop": 20},
            {"name": "source_motion", "frame_start": 20, "frame_stop": 60},
            {"name": "return_ramp", "frame_start": 60, "frame_stop": 75},
        ]
    }
    return poses, timeline


def test_only_bounded_generated_arm_changes_allowed():
    poses, timeline = example()
    after = poses.copy()
    after[8, 7 + arm_path_columns(HARDWARE_23_JOINT_NAMES)[0]] = 0.29
    assert verify_arm_pose_replacement(poses, after, timeline, HARDWARE_23_JOINT_NAMES) == 0.29


@pytest.mark.parametrize(
    "frame,column,value", [(22, 20, 0.1), (0, 20, 0.1), (8, 0, 0.1), (8, 7, 0.1), (8, 20, 0.31), (8, 20, np.nan)]
)
def test_source_standing_root_leg_unbounded_and_nonfinite_mutations_rejected(frame, column, value):
    poses, timeline = example()
    after = poses.copy()
    after[frame, column] = value
    with pytest.raises(ValueError):
        verify_arm_pose_replacement(poses, after, timeline, HARDWARE_23_JOINT_NAMES)


def test_ramp_scope_cannot_overlap_source():
    poses, timeline = example()
    timeline = deepcopy(timeline)
    timeline["phases"][0]["frame_stop"] = 21
    with pytest.raises(ValueError, match="source frames"):
        verify_arm_pose_replacement(poses, poses, timeline, HARDWARE_23_JOINT_NAMES)


def test_rate_rows_preserve_nonzero_incoming_velocity_without_inventing_zero_initial_rate():
    path = np.arange(6)[:, None] * np.array([[0.04, -0.06]])
    actual = ramp_linear_constraints(6, 2) @ path.ravel()
    np.testing.assert_array_equal(actual[:12], path.ravel())
    np.testing.assert_allclose(actual[12:22].reshape(5, 2), np.tile([0.04, -0.06], (5, 1)))
    np.testing.assert_allclose(actual[22:], 0, atol=1e-16)


def test_com_jacobian_matches_physical_finite_difference_and_does_not_edit_model():
    model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).resolve().parents[1] / "data/robots/g1/g1_23dof_rev_1_0.xml")
    )
    before = compiled_model_sha256(model)
    pose = model.qpos0.copy()
    columns = arm_path_columns(HARDWARE_23_JOINT_NAMES)
    pose[columns + 7] = np.linspace(-0.2, 0.2, 10)
    com, jacobian = com_path_jacobian(model, pose[None, :], columns + 6)
    numerical = np.zeros((3, 10))
    for j, column in enumerate(columns + 7):
        plus, minus = pose.copy(), pose.copy()
        plus[column] += 1e-6
        minus[column] -= 1e-6
        values, _ = com_path_jacobian(model, np.stack((plus, minus)), columns + 6)
        numerical[:, j] = (values[0] - values[1]) / 2e-6
    assert com.shape == (1, 3)
    np.testing.assert_allclose(jacobian.toarray(), numerical, atol=2e-10, rtol=0)
    assert compiled_model_sha256(model) == before
