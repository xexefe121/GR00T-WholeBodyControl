from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import clarabel
import numpy as np
import pytest
from scipy import sparse

from gear_sonic.scripts.refine_g1_true23_collision_clearance import (
    collision_repair_acceptance,
    interpolate_original_poses,
    joint_search_config,
)
from gear_sonic.utils.g1_true23_collision_path import (
    clearance_merit,
    clearance_step_targets,
    collision_line_search_fractions,
)
from gear_sonic.utils.g1_true23_generalist_protected_root import solve_box_soc


def test_tighter_solver_precision_does_not_relax_independent_acceptance():
    group = {"indices": np.array([0, 1]), "scales": np.ones(2), "radius": 1.0}
    solution, report = solve_box_soc(
        np.ones(2), [-2.0, 0.0], sparse.eye(2), [-10.0, -10.0], [10.0, 10.0], [group], solver_tolerance=1e-11
    )
    assert report["accepted"] and np.linalg.norm(solution) <= 1 + 1e-8
    assert report["requested_feasibility_tolerance"] == 1e-11
    assert report["original_row_audit_tolerance"] == 1e-8


@pytest.mark.parametrize("tolerance", [1e-8, 0.0, np.inf, np.nan, 1e-13])
def test_solver_cannot_use_looser_precision(tolerance):
    with pytest.raises(ValueError, match="only tighten"):
        solve_box_soc([1.0], [0.0], sparse.eye(1), [-1.0], [1.0], [], solver_tolerance=tolerance)


@pytest.mark.parametrize(
    "feasible, explicit, accepted", [(True, True, True), (True, False, False), (False, True, False)]
)
def test_almost_solved_requires_explicit_selection_and_exact_original_constraint_audit(
    monkeypatch, feasible, explicit, accepted
):
    answer = SimpleNamespace(
        status=clarabel.SolverStatus.AlmostSolved,
        x=np.array([0.99 if feasible else 1.01]),
        iterations=30,
        solve_time=0.001,
    )
    monkeypatch.setattr(clarabel, "DefaultSolver", lambda *args: SimpleNamespace(solve=lambda: answer))
    group = {"indices": np.array([0]), "scales": np.ones(1), "radius": 1.0}
    solution, report = solve_box_soc(
        [1.0], [0.0], sparse.eye(1), [-1.0], [1.0], [group], accept_independently_feasible_inaccurate=explicit
    )
    assert report["accepted"] is accepted
    assert (solution is not None) is accepted
    assert report["original_row_audit_tolerance"] == 1e-8
    if accepted:
        assert report["inaccurate_solution_independently_feasible"] and not report["qp_optimality_proven"]


def test_restoration_targets_improve_negative_distance_without_relabelling_it_clear():
    result = clearance_step_targets(np.array([-0.1, -0.003, 0.0, 0.02]))
    np.testing.assert_allclose(result, [-0.095, 0.001, 0.001, 0.001])
    assert result[0] < 0


def test_worst_first_does_not_require_simultaneous_repair_of_every_small_contact():
    distances = np.array([-0.1, -0.003, 0.0001, 0.02])
    result = clearance_step_targets(distances, strategy="worst_contact_first_v2")
    np.testing.assert_allclose(result, [-0.095, -0.003, 0.0001, 0.001])
    assert np.all(result[distances < 0] >= distances[distances < 0])
    assert np.all(result[distances >= 0] >= 0)


def test_worst_first_eventually_asks_for_positive_serialization_margin():
    result = clearance_step_targets(np.array([-0.003, -0.001, 0.03]), strategy="worst_contact_first_v2")
    np.testing.assert_allclose(result, [0.001, 0.001, 0.001])


def test_unknown_strategy_is_not_silently_accepted():
    with pytest.raises(ValueError, match="unknown"):
        clearance_step_targets(np.array([-0.1]), strategy="disable_collisions")


def test_full_safe_search_changes_only_seed_neighborhood_not_physical_or_task_bounds():
    low, high = np.full(23, -2.0), np.full(23, 2.5)
    old = joint_search_config("original_seed_neighborhood_v1", low, high, 48)
    new = joint_search_config("full_existing_safe_envelope_v2", low, high, 48)
    old_values, new_values = asdict(old), asdict(new)
    assert old_values.pop("maximum_joint_change_rad") == 0.6
    assert new_values.pop("maximum_joint_change_rad") == 4.5
    assert old_values == new_values
    for seed in (low, high, np.zeros(23)):
        np.testing.assert_array_equal(np.maximum(low, seed - new.maximum_joint_change_rad), low)
        np.testing.assert_array_equal(np.minimum(high, seed + new.maximum_joint_change_rad), high)


def test_joint_search_does_not_accept_invalid_or_reversed_safe_limits():
    with pytest.raises(ValueError, match="safe bounds"):
        joint_search_config("full_existing_safe_envelope_v2", np.ones(23), -np.ones(23), 48)
    with pytest.raises(ValueError, match="unknown"):
        joint_search_config("raise_motor_limits", -np.ones(23), np.ones(23), 48)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [[0]], [-np.inf]])
def test_invalid_distances_rejected(values):
    with pytest.raises(ValueError, match="finite vector"):
        clearance_step_targets(values)


def valid_evidence():
    return dict(
        control_failures=[],
        original_audit={"original_timestamp_fidelity_passed": True},
        serialized_path={"passed": True},
        control_contacts={"frames_with_robot_robot_penetration": 0},
        original_contacts={"frames_with_robot_robot_penetration": 0},
    )


def test_even_collision_clear_reference_is_not_dynamics_or_hardware_qualification():
    answer = collision_repair_acceptance(**valid_evidence())
    assert answer["passed"]
    for flag in (
        "dynamic_feasibility_proven",
        "continuous_between_sample_clearance_proven",
        "hardware_authorized",
        "deployment_ready",
    ):
        assert answer[flag] is False


@pytest.mark.parametrize("failure", ["control", "original", "path", "control_contact", "original_contact"])
def test_every_independent_gate_required(failure):
    evidence = deepcopy(valid_evidence())
    if failure == "control":
        evidence["control_failures"] = ["protected COM"]
    elif failure == "original":
        evidence["original_audit"] = None
    elif failure == "path":
        evidence["serialized_path"]["passed"] = False
    else:
        evidence["control_contacts" if failure == "control_contact" else "original_contacts"][
            "frames_with_robot_robot_penetration"
        ] = 1
    assert not collision_repair_acceptance(**evidence)["passed"]


def test_original_grid_interpolation_preserves_all_endpoints_and_pose_channels():
    pose = np.zeros((3, 30))
    pose[:, 3] = 1
    pose[:, 0] = [0.0, 1.0, 2.0]
    pose[:, 7:] = np.arange(3)[:, None]
    original = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    result = interpolate_original_poses(pose, np.arange(3), original)
    assert result.shape == (6, 30)
    np.testing.assert_array_equal(result[[0, -1]], pose[[0, -1]])
    np.testing.assert_allclose(result[:, 0], original)
    np.testing.assert_allclose(result[:, 7], original)


def test_cropped_original_grid_rejected():
    pose = np.zeros((3, 30))
    pose[:, 3] = 1
    with pytest.raises(ValueError, match="both source endpoints"):
        interpolate_original_poses(pose, np.arange(3), np.array([0.0, 1.0]))


def test_total_collision_merit_can_escape_a_worst_contact_trap_without_changing_final_acceptance():
    before, after = np.array([-0.09, -0.06]), np.array([-0.095, -0.02])
    assert clearance_merit(after) > clearance_merit(before)
    assert clearance_merit(after, "sum_squared_clearance_v2") < clearance_merit(before, "sum_squared_clearance_v2")
    evidence = valid_evidence()
    evidence["control_contacts"]["frames_with_robot_robot_penetration"] = 1
    assert not collision_repair_acceptance(**evidence)["passed"]


def test_additional_clear_nearby_pairs_do_not_change_total_collision_merit():
    assert clearance_merit([-0.01], "sum_squared_clearance_v2") == clearance_merit(
        [-0.01, 0.02, 0.01], "sum_squared_clearance_v2"
    )
    assert clearance_merit([], "sum_squared_clearance_v2") == 0
    with pytest.raises(ValueError, match="unknown"):
        clearance_merit([-0.01], "ignore_collisions")


def test_extended_line_search_adds_smaller_trials_without_changing_old_defaults():
    old = collision_line_search_fractions()
    new = collision_line_search_fractions("extended_bisection_v2")
    assert old == (1.0, 0.5, 0.25, 0.125)
    assert new[:4] == old and new[-1] == 1 / 128 and len(new) == 8
    assert all(0 < b < a <= 1 for a, b in zip(new, new[1:]))
    with pytest.raises(ValueError, match="unknown"):
        collision_line_search_fractions("relax_constraints")
