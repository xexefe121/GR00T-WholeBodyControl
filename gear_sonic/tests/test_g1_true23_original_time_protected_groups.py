"""Bounded multi-window selection; all final independent gates remain mandatory."""

import numpy as np
import pytest

from gear_sonic.scripts.refine_g1_true23_original_time_protected_groups import (
    coordinate_groups,
    grouped_repair_coordinates,
)
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES


def _audit(**categories):
    return {
        "failures": list(categories),
        "categories": {key: {"frame_indices": value} for key, value in categories.items()},
    }


def _select(audit):
    return grouped_repair_coordinates(
        audit, np.arange(563) / 120, np.linspace(0, 562 / 120, 469), tuple(HARDWARE_23_JOINT_NAMES)
    )


def test_separated_foot_and_com_repairs_are_bounded_and_do_not_edit_root():
    coords = _select(
        _audit(
            com_position_regression=[225],
            left_foot_orientation_regression=[307, 308, 309, 404, 540],
            right_foot_orientation_regression=[96, 97, 98, 102, 103, 104, 243, 244, 245, 516, 517],
        )
    )
    groups = coordinate_groups(coords)
    # Adjacent right-foot defects around frames 96 and 102 share one group.
    assert len(groups) == 7
    assert len(coords) <= 512
    assert all(len(group) <= 128 for group in groups)
    assert set(coords[:, 1]) == set(range(23))
    for a, b in zip(groups, groups[1:]):
        assert b[:, 0].min() - a[:, 0].max() > 2


def test_com_changes_only_four_bracketing_control_knots():
    coords = _select(_audit(com_position_regression=[225]))
    assert len(coords) == 4 * 23
    assert len(np.unique(coords[:, 0])) == 4


def test_foot_only_selection_does_not_add_other_joints():
    coords = _select(_audit(right_foot_orientation_regression=[3]))
    assert all(HARDWARE_23_JOINT_NAMES[i].startswith("right_") for i in coords[:, 1])
    assert len(set(coords[:, 1])) == 6


@pytest.mark.parametrize(
    "audit",
    [
        _audit(),
        _audit(root_offset=[3]),
        _audit(left_foot_position_error=[3]),
        _audit(com_position_regression=[-1]),
        _audit(com_position_regression=[563]),
        _audit(com_position_regression=[3.0]),
        _audit(com_position_regression=list(range(33))),
    ],
)
def test_unrelated_large_or_invalid_failure_set_is_rejected(audit):
    with pytest.raises(ValueError):
        _select(audit)


def test_derivative_coupled_windows_must_not_be_split_to_evade_variable_cap():
    with pytest.raises(ValueError, match="coupled"):
        coordinate_groups(np.array([(frame, joint) for frame in range(6) for joint in range(23)]))


@pytest.mark.parametrize(
    "coordinates",
    [
        np.empty((0, 2), dtype=int),
        np.array([[1, 2], [1, 2]]),
        np.array([(frame, 0) for frame in range(513)]),
    ],
)
def test_empty_duplicate_or_unbounded_coordinates_rejected(coordinates):
    with pytest.raises(ValueError):
        coordinate_groups(coordinates)


@pytest.mark.parametrize("mutation", ["reverse", "nonfinite", "uncovered", "bad_joints"])
def test_invalid_time_or_joint_identity_rejected(mutation):
    source = np.arange(9) / 120
    controls = np.linspace(0, 8 / 120, 7)
    joints = tuple(HARDWARE_23_JOINT_NAMES)
    if mutation == "reverse":
        controls = controls[::-1]
    elif mutation == "nonfinite":
        source[2] = np.nan
    elif mutation == "uncovered":
        source[-1] += 0.1
    else:
        joints = joints[:-1] + (joints[0],)
    with pytest.raises(ValueError):
        grouped_repair_coordinates(_audit(com_position_regression=[3]), source, controls, joints)
