import numpy as np
import pytest
from scipy import sparse

from gear_sonic.utils.g1_true23_affine_task_soc import solve_affine_task_soc
from gear_sonic.utils.g1_true23_generalist_protected_root import solve_box_soc


@pytest.mark.parametrize("objective_scale", [1.0, 0.001])
def test_elimination_matches_original_augmented_objective_and_constraints(objective_scale):
    j = sparse.csc_matrix([[1.0, 0.1], [0.2, 1.0], [0.5, -0.3]])
    residual = np.array([-1.0, 0.7, 0.2])
    posture, displacement = np.array([0.1, 0.2]), np.array([0.05, -0.1])
    lower, upper = np.array([-0.2, -0.5]), np.array([0.3, 0.7])
    group = {"indices": np.arange(3), "scales": np.ones(3), "radius": 1.0}
    direct, report = solve_affine_task_soc(
        posture, displacement, residual, j, sparse.eye(2), lower, upper, [group], objective_scale=objective_scale
    )
    matrix = sparse.vstack(
        (sparse.hstack((sparse.eye(2), sparse.csc_matrix((2, 3)))), sparse.hstack((j, -sparse.eye(3)))),
        format="csc",
    )
    augmented, previous = solve_box_soc(
        objective_scale * np.r_[posture, np.ones(3)],
        objective_scale * np.r_[posture * displacement, np.zeros(3)],
        matrix,
        np.r_[lower, -residual],
        np.r_[upper, -residual],
        [{**group, "indices": np.arange(2, 5)}],
    )
    assert report["accepted"] and previous["accepted"]
    np.testing.assert_allclose(direct, augmented[:2], atol=1e-6)
    actual_residual = residual + j @ direct
    assert np.linalg.norm(actual_residual) <= 1 + 1e-8
    assert report["original_row_audit_tolerance"] == 1e-8
    assert not report["objective_and_constraints_changed"]


def test_contact_constraint_cannot_be_sacrificed_for_lower_task_error():
    group = {"indices": np.array([0]), "scales": np.ones(1), "radius": 0.1}
    answer, report = solve_affine_task_soc(
        [1.0], [0.0], [0.0], sparse.eye(1), sparse.eye(1), [2.0], [3.0], [group]
    )
    assert answer is None and not report["accepted"]


@pytest.mark.parametrize("posture", [[-1.0], [np.nan], [np.inf]])
def test_invalid_objective_does_not_reach_solver(posture):
    with pytest.raises(ValueError, match="finite"):
        solve_affine_task_soc(posture, [0.0], [0.0], sparse.eye(1), sparse.eye(1), [-1.0], [1.0], [])


def test_minimum_change_projects_current_pose_without_unrelated_task_improvements():
    answer, report = solve_affine_task_soc(
        [1.0, 2.0],
        [4.0, -5.0],
        [0.0, 100.0],
        sparse.eye(2),
        sparse.eye(2),
        [0.1, -2.0],
        [2.0, 2.0],
        [],
        objective_profile="minimum_change_v2",
    )
    assert report["accepted"]
    np.testing.assert_allclose(answer, [0.1, 0.0], atol=1e-6)
    assert report["objective_changed_from_task_lsq"] and not report["hard_constraints_changed"]


def test_minimum_change_keeps_protected_norms_hard():
    group = {"indices": np.array([0]), "scales": np.ones(1), "radius": 0.1}
    answer, report = solve_affine_task_soc(
        [1.0],
        [0.0],
        [0.0],
        sparse.eye(1),
        sparse.eye(1),
        [0.2],
        [1.0],
        [group],
        objective_profile="minimum_change_v2",
    )
    assert answer is None and not report["accepted"]


@pytest.mark.parametrize("profile,posture", [("unknown", [1.0]), ("minimum_change_v2", [0.0])])
def test_projection_requires_declared_objective_and_strictly_positive_metric(profile, posture):
    with pytest.raises(ValueError):
        solve_affine_task_soc(
            posture,
            [0.0],
            [0.0],
            sparse.eye(1),
            sparse.eye(1),
            [-1.0],
            [1.0],
            [],
            objective_profile=profile,
        )


def test_pose_guidance_changes_only_objective_and_cannot_override_hard_task_bounds():
    group = {"indices": np.array([0]), "scales": np.ones(1), "radius": 0.2}
    answer, report = solve_affine_task_soc(
        [1.0],
        [-0.7],
        [0.0],
        sparse.eye(1),
        sparse.eye(1),
        [0.1],
        [1.0],
        [group],
        objective_profile="pose_escape_guided_v3",
    )
    assert report["accepted"] and report["objective_changed_from_task_lsq"]
    assert not report["hard_constraints_changed"]
    np.testing.assert_allclose(answer, [0.2], atol=1e-7)
