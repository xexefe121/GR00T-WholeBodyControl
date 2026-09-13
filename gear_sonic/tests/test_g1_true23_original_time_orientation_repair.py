"""Synthetic knot-selection and real-FK tests; no licensed dataset fixture."""

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.scripts.refine_g1_true23_original_time_foot_orientation import (
    LEG_JOINTS,
    LocalOrientationFit,
    repair_coordinates,
    verified_repair_acceptance,
)
from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE


def _audit(frames=(3,), category="right_foot_orientation_regression"):
    return {"failures": [category], "categories": {category: {"frame_indices": list(frames)}}}


def test_knots_cover_both_original_sample_brackets_and_only_requested_leg():
    coordinates = repair_coordinates(
        _audit(), np.arange(9) / 120, np.linspace(0, 8 / 120, 7), tuple(HARDWARE_23_JOINT_NAMES)
    )
    assert set(coordinates[:, 0]) == {1, 2, 3, 4}
    assert {HARDWARE_23_JOINT_NAMES[i] for i in coordinates[:, 1]} == {
        f"right_{name}_joint" for name in LEG_JOINTS
    }
    assert len({tuple(row) for row in coordinates}) == len(coordinates)


@pytest.mark.parametrize("frame", [0, 8])
def test_endpoint_knots_never_wrap_or_exceed_control_grid(frame):
    coordinates = repair_coordinates(
        _audit((frame,)), np.arange(9) / 120, np.linspace(0, 8 / 120, 7), tuple(HARDWARE_23_JOINT_NAMES)
    )
    assert coordinates[:, 0].min() >= 0
    assert coordinates[:, 0].max() < 7
    assert (0 if frame == 0 else 6) in coordinates[:, 0]


@pytest.mark.parametrize(
    "failures", [[], ["com_position_regression"], ["right_foot_orientation_regression", "root_offset"]]
)
def test_does_not_relabel_other_failures_as_ankle_repair(failures):
    audit = _audit()
    audit["failures"] = failures
    with pytest.raises(ValueError, match="only original-time"):
        repair_coordinates(audit, np.arange(9), np.arange(7), tuple(HARDWARE_23_JOINT_NAMES))


def test_large_repair_is_not_implicitly_authorized():
    with pytest.raises(ValueError, match="1..32"):
        repair_coordinates(_audit(tuple(range(33))), np.arange(40), np.arange(40), tuple(HARDWARE_23_JOINT_NAMES))


def test_real_fk_solver_preserves_source_root_and_unselected_joints():
    root = Path(__file__).resolve().parents[2]
    source_model, target_model = ik.load_models(root / ik.DEFAULT_SOURCE_MODEL, root / ik.DEFAULT_TARGET_MODEL)
    source_names = ik._model_layout(source_model).joint_names
    target_names = ik._model_layout(target_model).joint_names
    defaults = dict(zip(HARDWARE_23_JOINT_NAMES, SAFE_TARGET_DEFAULT_Q_HARDWARE, strict=True))
    count = 7
    source = {
        "joint_pos": np.tile([defaults.get(name, 0) for name in source_names], (count, 1)),
        "root_pos_w": np.tile([0, 0, 0.8], (count, 1)),
        "root_quat_wxyz": np.tile([1, 0, 0, 0], (count, 1)),
        "timestamps_s": np.arange(count) / 50,
        "contact_flags": np.ones((count, 2), dtype=bool),
    }
    pose = (
        np.column_stack(
            (
                source["root_pos_w"],
                source["root_quat_wxyz"],
                source["joint_pos"][:, [source_names.index(name) for name in target_names]],
            )
        )
        .astype(np.float32)
        .astype(float)
    )
    ankle = target_names.index("right_ankle_pitch_joint")
    pose[3, 7 + ankle] += 0.006
    coordinates = np.array([[3, ankle]])
    source_before = {key: value.copy() for key, value in source.items()}
    pose_before = pose.copy()
    problem = LocalOrientationFit(
        source_model,
        target_model,
        source,
        source,
        pose,
        source["timestamps_s"],
        coordinates,
        ik.RetargetConfig(optimize_lower_body=True, allow_acceleration_constraint_relaxation=False),
    )
    result, report = problem.solve()
    assert report["optimizer_success"]
    assert report["serialized_selected_constraint_min"] >= -1e-7
    assert report["serialized_max_correction_rad"] <= 0.01
    assert abs(result[3, 7 + ankle] - pose_before[2, 7 + ankle]) < 0.001
    mask = np.ones(pose.shape, dtype=bool)
    mask[3, 7 + ankle] = False
    np.testing.assert_array_equal(result[mask], pose_before[mask])
    np.testing.assert_array_equal(pose, pose_before)
    for key, value in source_before.items():
        np.testing.assert_array_equal(source[key], value)


def test_independent_feasibility_does_not_claim_optimizer_convergence_or_optimality():
    before = np.zeros((3, 30))
    after = before.copy()
    after[1, 17] = 0.00018
    audit = verified_repair_acceptance(
        before, after, np.array([[1, 10]]), [], {"original_timestamp_fidelity_passed": True}
    )
    assert audit["passed"]
    assert not audit["optimizer_convergence_is_feasibility_evidence"]
    assert not audit["minimum_norm_optimality_proven"]


@pytest.mark.parametrize(
    "mutation", ["root", "other_joint", "too_large", "nonfinite", "control_failure", "original_failure"]
)
def test_serialized_scope_and_full_gates_cannot_be_waived(mutation):
    before = np.zeros((3, 30))
    after = before.copy()
    after[1, 17] = 0.00018
    control_failures, original_audit = [], {"original_timestamp_fidelity_passed": True}
    if mutation == "root":
        after[1, 0] = 1e-12
    elif mutation == "other_joint":
        after[1, 18] = 1e-12
    elif mutation == "too_large":
        after[1, 17] = 0.010001
    elif mutation == "nonfinite":
        after[1, 17] = np.nan
    elif mutation == "control_failure":
        control_failures = ["right_foot_orientation_regression"]
    else:
        original_audit["original_timestamp_fidelity_passed"] = False
    assert not verified_repair_acceptance(before, after, np.array([[1, 10]]), control_failures, original_audit)[
        "passed"
    ]
