import numpy as np
import pytest
from scipy import sparse

from gear_sonic.utils.g1_true23_elastic_collision_step import solve_elastic_collision_step


def solve(*, lower=-0.2, upper=0.2, group_radius=None, guide=None, distances=(-0.1, -0.01)):
    groups = (
        [] if group_radius is None else [{"indices": np.array([0]), "scales": np.ones(1), "radius": group_radius}]
    )
    return solve_elastic_collision_step(
        [1.0],
        [0.0 if guide is None else -guide],
        [0.0],
        sparse.eye(1),
        sparse.eye(1),
        [lower],
        [upper],
        groups,
        sparse.csc_matrix([[1.0], [-1.0]]),
        distances,
        objective_profile="minimum_change_v2" if guide is None else "pose_escape_guided_v3",
    )


def test_competing_contacts_can_trade_intermediate_violation_not_final_acceptance():
    delta, report = solve()
    # Old non-worsening inequalities require delta >= 0 and delta <= 0.
    # Elastic least squares instead balances the two opposed penetrations.
    np.testing.assert_allclose(delta, [0.09 * 10000 / 20001], atol=1e-6)
    assert report["accepted"]
    assert report["predicted_contacts_worsened"] == 1
    assert report["predicted_squared_clearance_violation_m2"] < 0.101**2 + 0.011**2
    assert report["clearance_slack_max_m"] > 0.05  # Still physically unacceptable.
    assert not report["numerical_step_is_accepted_reference"]
    assert not report["final_collision_acceptance_changed"]
    assert not report["physical_collision_clearance_proven"]


@pytest.mark.parametrize("radius", [0.01, 0.02])
def test_task_norm_cannot_be_relaxed_by_contact_slacks(radius):
    delta, report = solve(group_radius=radius)
    assert report["accepted"]
    np.testing.assert_allclose(delta, [radius], atol=1e-7)
    assert not report["original_noncontact_linear_and_task_constraints_changed"]


def test_original_linear_bound_stays_hard():
    delta, report = solve(lower=-0.2, upper=0.005)
    assert report["accepted"]
    np.testing.assert_allclose(delta, [0.005], atol=1e-7)
    assert report["contact_slack_row_violation"] <= 1e-8
    assert report["slack_nonnegativity_violation"] <= 1e-8


def test_incompatible_noncontact_constraints_stay_infeasible():
    delta, report = solve(lower=0.1, upper=0.2, group_radius=0.01)
    assert delta is None and not report["accepted"]


def test_no_violating_contact_needs_no_unrelated_motion():
    delta, report = solve(distances=(0.01, 0.01))
    assert report["accepted"]
    np.testing.assert_allclose(delta, [0.0], atol=1e-7)
    assert report["predicted_squared_clearance_violation_m2"] == 0


def test_guidance_cannot_override_hard_task_norms():
    delta, report = solve(guide=1.0, group_radius=0.01)
    assert report["accepted"]
    np.testing.assert_allclose(delta, [0.01], atol=1e-7)


def test_actual_path_step_keeps_temporal_safe_root_and_task_bounds():
    from types import SimpleNamespace

    from gear_sonic.utils.g1_true23_collision_path import collision_step
    from gear_sonic.utils.g1_true23_original_task_trajectory import OriginalTaskConfig

    current = np.zeros((3, 29))
    problem = SimpleNamespace(
        config=OriginalTaskConfig(),
        lower=np.full_like(current, -0.5),
        upper=np.full_like(current, 0.5),
        velocity=np.full(29, 5.0),
        acceleration=np.full(29, 80.0),
        initial_velocity=np.zeros(29),
        initial=current.copy(),
        posture_weight=np.ones(current.size),
    )
    contact = sparse.csc_matrix(([1.0, -1.0], ([0, 1], [6, 6])), shape=(2, current.size))
    task = sparse.csc_matrix(([1.0], ([0], [6])), shape=(1, current.size))
    group = {"indices": np.array([0]), "scales": np.ones(1), "radius": 0.01}
    result, report = collision_step(
        problem,
        current,
        np.zeros(1),
        task,
        [group],
        contact,
        np.array([-0.1, -0.01]),
        0.005,
        strategy="elastic_squared_clearance_v3",
        solver_form="eliminated_residual_v2",
        objective_profile="minimum_change_v2",
    )
    assert report["accepted"] and report["independent_absolute_noncontact_row_violation"] <= 1e-8
    assert result.shape == current.shape
    assert abs(result[0, 6]) <= 0.01 * problem.config.serialization_margin_fraction + 1e-8
    assert np.max(abs(np.diff(result, axis=0))) <= 5 * 0.02 + 1e-8
    np.testing.assert_allclose(result[:, :6], 0.0, atol=1e-6)
    assert not report["final_collision_acceptance_changed"]


def test_elastic_full_path_requires_explicit_squared_merit_before_any_geometry():
    from gear_sonic.utils.g1_true23_collision_path import fit_collision_clearance_path

    with pytest.raises(ValueError, match="explicit"):
        fit_collision_clearance_path(
            None,
            None,
            None,
            None,
            strategy="elastic_squared_clearance_v3",
            solver_form="eliminated_residual_v2",
            objective_profile="minimum_change_v2",
        )


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("clearance_m", 0.0),
        ("clearance_m", float("nan")),
        ("slack_weight", 0.0),
        ("slack_weight", float("inf")),
        ("objective_profile", "task_lsq_v1"),
    ],
)
def test_invalid_elastic_configuration_fails(keyword, value):
    with pytest.raises(ValueError, match="elastic"):
        solve_elastic_collision_step(
            [1.0],
            [0.0],
            [0.0],
            sparse.eye(1),
            sparse.eye(1),
            [-1.0],
            [1.0],
            [],
            sparse.eye(1),
            [-0.1],
            **{keyword: value},
        )
