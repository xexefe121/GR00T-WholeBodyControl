from copy import deepcopy

import numpy as np
import pytest

from gear_sonic.scripts.refine_g1_true23_lifecycle_ramps import shoulder_clearance_bump
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_repaired_lifecycle_reference import verify_shoulder_pose_replacement


def fixture():
    before = np.zeros((250, 30))
    before[:, 3] = 1
    after = before.copy()
    names = tuple(HARDWARE_23_JOINT_NAMES)
    phases = [
        dict(name="acquisition_ramp", frame_start=10, frame_stop=110),
        dict(name="return_ramp", frame_start=120, frame_stop=220),
    ]
    proof = {"phases": []}
    for phase in phases:
        start, stop = phase["frame_start"], phase["frame_stop"]
        after[start:stop, 7 + names.index("left_shoulder_roll_joint")] += 0.05 * shoulder_clearance_bump(100)
        after[start:stop, 7 + names.index("right_shoulder_roll_joint")] -= 0.1 * shoulder_clearance_bump(100)
        proof["phases"].append(
            {
                "phase": phase["name"],
                "selected_attempt": 0,
                "attempts": [{"left_outward_rad": 0.05, "right_outward_rad": 0.1, "accepted": True}],
            }
        )
    return before, after, {"phases": phases}, proof, names


def test_only_exact_declared_smooth_shoulder_samples_can_replace_generated_ramps():
    args = fixture()
    saved = deepcopy(args)
    verify_shoulder_pose_replacement(*args)
    np.testing.assert_array_equal(args[0], saved[0])
    np.testing.assert_array_equal(args[1], saved[1])


@pytest.mark.parametrize("frame,column", [(0, 7), (50, 0), (50, 7), (115, 21), (249, 25), (50, 21)])
def test_replacement_cannot_touch_standing_root_legs_source_or_undeclared_shoulder_value(frame, column):
    args = fixture()
    args[1][frame, column] += 0.001
    with pytest.raises(ValueError):
        verify_shoulder_pose_replacement(*args)


def test_rejected_offset_cannot_be_relabelled_as_a_reference():
    args = fixture()
    args[3]["phases"][0]["attempts"][0]["accepted"] = False
    with pytest.raises(ValueError):
        verify_shoulder_pose_replacement(*args)
