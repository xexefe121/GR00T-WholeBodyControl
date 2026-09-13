import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

from gear_sonic.utils.g1_true23_pd_box_step import solve_box_quadratic


@pytest.mark.parametrize("case_index", range(3))
def test_actual_contact_quadratics_converge_at_roundoff_without_newton_step_stalling(case_index):
    fixture = Path(__file__).parent / "fixtures" / "g1_true23_pd_box_ill_conditioned.json"
    row = json.loads(fixture.read_text())["cases"][case_index]
    h, g, lower, upper = [np.asarray(row[key]) for key in ("h", "g", "lower", "upper")]
    assert np.linalg.cond(h) > 1e6
    x, free = solve_box_quadratic(h, g, lower, upper)
    assert np.all(x >= lower) and np.all(x <= upper)
    # Independently solve the free absolute optimum, not the Newton residual
    # recurrence used by the implementation. SPD plus these KKT signs proves
    # the unique constrained minimum to floating-point backward accuracy.
    fixed = ~free
    expected = np.linalg.solve(h[np.ix_(free, free)], -g[free] - h[np.ix_(free, fixed)] @ x[fixed])
    np.testing.assert_allclose(x[free], expected, atol=2e-8, rtol=0)
    derivative = h @ x + g
    scale = max(1.0, float(np.max(np.abs(h))), float(np.max(np.abs(g))))
    assert np.max(np.abs(derivative[free])) / scale < 1e-13
    assert np.all(derivative[x == lower] >= -scale * 1e-13)
    assert np.all(derivative[x == upper] <= scale * 1e-13)


def test_free_joint_is_resolved_when_a_coupled_joint_hits_its_bound():
    h, g = np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([-6.0, -2.0])
    x, free = solve_box_quadratic(h, g, np.full(2, -1.0), np.ones(2))
    np.testing.assert_allclose(x, [1.0, 0.5], atol=1e-12)
    np.testing.assert_array_equal(free, [False, True])
    clipped = np.clip(-np.linalg.solve(h, g), -1, 1)
    assert np.linalg.norm(x - clipped) > 1


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("scale", [1.0, 1e6])
def test_box_solver_matches_independent_scipy_constrained_optimization(seed, scale):
    rng, n = np.random.default_rng(seed), 23
    factor = rng.normal(size=(n, n))
    h, g = factor.T @ factor + np.eye(n) * 0.3, rng.normal(0, 3, n)
    lower, upper = -rng.uniform(0.05, 0.5, n), rng.uniform(0.05, 0.5, n)
    x, free = solve_box_quadratic(h * scale, g * scale, lower, upper)
    result = minimize(
        lambda q: 0.5 * q @ h @ q + g @ q,
        np.zeros(n),
        jac=lambda q: h @ q + g,
        bounds=list(zip(lower, upper)),
        method="SLSQP",
        options=dict(ftol=1e-12, maxiter=400),
    )
    assert result.success
    np.testing.assert_allclose(x, result.x, atol=2e-6, rtol=0)
    gradient = h @ x + g
    assert np.max(np.abs(gradient[free])) < 1e-9
    assert np.all(gradient[x == lower] >= -1e-9)
    assert np.all(gradient[x == upper] <= 1e-9)
    assert np.all(x >= lower) and np.all(x <= upper)


def test_active_set_feedback_derivative_matches_perturbed_qp_solution():
    rng, n = np.random.default_rng(42), 8
    factor = rng.normal(size=(n, n))
    h, g = factor.T @ factor + np.eye(n), rng.normal(size=n)
    lower, upper = np.full(n, -0.1), np.full(n, 0.1)
    _, free = solve_box_quadratic(h, g, lower, upper)
    direction, eps = rng.normal(size=n), 1e-6
    plus, _ = solve_box_quadratic(h, g + eps * direction, lower, upper)
    minus, _ = solve_box_quadratic(h, g - eps * direction, lower, upper)
    expected = np.zeros(n)
    expected[free] = -np.linalg.solve(h[np.ix_(free, free)], direction[free])
    np.testing.assert_allclose((plus - minus) / (2 * eps), expected, atol=1e-9, rtol=1e-8)
