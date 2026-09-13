import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, minimize

from gear_sonic.utils.g1_true23_pd_box_step import solve_box_quadratic
from gear_sonic.utils.g1_true23_pd_contact_step import contact_backward_pass, solve_contact_quadratic


@pytest.mark.parametrize("seed", range(8))
def test_general_inequalities_match_independent_scipy_optimum(seed):
    rng, n = np.random.default_rng(seed + 892), 8
    factor = rng.normal(size=(n, n))
    h, g = factor.T @ factor + np.eye(n), rng.normal(0, 3, n)
    c = np.vstack((np.eye(n), -np.eye(n), rng.normal(size=(7, n))))
    b = rng.uniform(0.05, 0.6, len(c))
    x, _, report = solve_contact_quadratic(h, g, c, b, np.zeros((n, 1)), np.zeros((len(c), 1)))
    reference = minimize(
        lambda value: 0.5 * value @ h @ value + g @ value,
        np.zeros(n),
        jac=lambda value: h @ value + g,
        constraints=[LinearConstraint(c, -np.inf, b)],
        method="SLSQP",
        options=dict(ftol=1e-12, maxiter=500),
    )
    assert reference.success
    np.testing.assert_allclose(x, reference.x, atol=3e-6, rtol=0)
    assert np.max(c @ x - b) < 1e-9
    assert report["maximum_scaled_stationarity"] < 1e-9


def test_active_contact_feedback_includes_motion_of_constraint_with_measured_state():
    h, g = np.eye(2) * 2, np.asarray([-2.0, -1.0])
    c = np.vstack((np.eye(2), -np.eye(2), [1.0, 1.0]))
    b = np.asarray([2.0, 2.0, 2.0, 2.0, 0.25])
    gx = np.asarray([[0.3, -0.2], [-0.4, 0.8]])
    cx = np.zeros((5, 2))
    cx[-1] = [1.0, -0.5]
    x, gain, _ = solve_contact_quadratic(h, g, c, b, gx, cx)
    np.testing.assert_allclose(x, [0.375, -0.125], atol=1e-12)
    np.testing.assert_allclose(c[-1] @ gain, -cx[-1], atol=1e-12)
    for axis in range(2):
        direction, epsilon = np.eye(2)[axis], 1e-6
        plus, _, _ = solve_contact_quadratic(
            h, g + epsilon * gx @ direction, c, b - epsilon * cx @ direction, gx, cx
        )
        minus, _, _ = solve_contact_quadratic(
            h, g - epsilon * gx @ direction, c, b + epsilon * cx @ direction, gx, cx
        )
        np.testing.assert_allclose((plus - minus) / (2 * epsilon), gain[:, axis], atol=1e-8)


@pytest.mark.parametrize("index", range(3))
def test_actual_ill_conditioned_contact_hessians_reproduce_verified_box_solution(index):
    fixture = Path(__file__).parent / "fixtures" / "g1_true23_pd_box_ill_conditioned.json"
    row = json.loads(fixture.read_text())["cases"][index]
    h, g, lo, hi = [np.asarray(row[key]) for key in ("h", "g", "lower", "upper")]
    c, b = np.vstack((np.eye(23), -np.eye(23))), np.r_[hi, -lo]
    x, _, _ = solve_contact_quadratic(h, g, c, b, np.zeros((23, 1)), np.zeros((46, 1)))
    expected, _ = solve_box_quadratic(h, g, lo, hi)
    np.testing.assert_allclose(x, expected, atol=3e-8, rtol=0)


def test_infeasible_contact_problem_is_rejected_not_silently_relaxed():
    with pytest.raises(ValueError, match="no verified feasible"):
        solve_contact_quadratic(
            np.eye(1), np.zeros(1), [[1.0], [-1.0]], [-1.0, -1.0], np.zeros((1, 1)), np.zeros((2, 1))
        )


def test_contact_backward_without_contacts_matches_independent_dense_full_horizon_solution():
    from gear_sonic.tests.test_g1_true23_pd_shooting import (
        test_backward_pass_matches_independent_dense_full_horizon_quadratic_solution as assert_dense_solution,
    )

    def backward(plant, local, controls, seed_controls, regularization):
        empty = dict(jacobian=np.zeros((0, 29)), distance=np.zeros(0), floor=np.zeros(0))
        increments, gains, _ = contact_backward_pass(
            plant, local, controls, seed_controls, [empty] * len(controls), regularization
        )
        return increments, gains

    assert_dense_solution(backward)


def test_contact_backward_propagates_next_state_contact_constraint_into_feedback():
    from types import SimpleNamespace

    from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE

    a, b = np.eye(81), np.zeros((81, 23))
    a[58:] = 0
    b[58:] = np.eye(23)
    b[6, 0] = 1.0
    terminal_g = np.zeros(81)
    terminal_g[6] = 0.5  # Unconstrained optimum would cross signed-distance zero.
    local = dict(
        a=a[None],
        b=b[None],
        gradient=np.zeros((1, 81)),
        hessian=np.zeros((1, 81, 81)),
        terminal_g=terminal_g,
        terminal_h=np.eye(81),
    )
    controls = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)[None]
    jacobian = np.zeros((1, 29))
    jacobian[0, 6] = 1.0
    geometry = [dict(jacobian=jacobian, distance=np.zeros(1), floor=np.zeros(1))]
    plant = SimpleNamespace(lower=controls[0] - 2, upper=controls[0] + 2)
    increments, feedback, reports = contact_backward_pass(plant, local, controls, controls, geometry, 0.0)
    assert increments[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert feedback[0, 0, 6] == pytest.approx(-1.0, abs=1e-12)
    assert 46 in reports[0]["active_rows"]
