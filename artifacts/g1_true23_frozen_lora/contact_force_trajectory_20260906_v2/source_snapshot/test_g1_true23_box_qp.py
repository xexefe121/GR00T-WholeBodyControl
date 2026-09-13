"""Numerical backend contracts; no physical motion qualification."""

import json
from types import SimpleNamespace

import clarabel
import numpy as np
import pytest
from scipy import sparse

from gear_sonic.utils.g1_true23_box_qp import solve_box_qp


def test_two_sided_rows_equality_and_unbounded_row_preserve_original_problem():
    matrix = sparse.csc_matrix([[1, 1], [1, 0], [0, 1], [1, -1]])
    lower, upper = np.array([1, 0.2, -np.inf, -np.inf]), np.array([1, 0.8, 0.4, np.inf])
    before = (matrix.copy(), lower.copy(), upper.copy())
    updates = []
    solution, report = solve_box_qp([1, 1], [0, 0], matrix, lower, upper, progress=updates.append)
    np.testing.assert_allclose(solution, [0.6, 0.4], atol=1e-8)
    assert report["accepted"] and report["cone_equality_rows"] == 1
    assert report["cone_inequality_rows"] == 3 and report["original_rows"] == 4
    assert report["independent_maximum_scaled_linear_violation"] <= 1e-8
    assert updates
    np.testing.assert_array_equal(matrix.toarray(), before[0].toarray())
    np.testing.assert_array_equal(lower, before[1])
    np.testing.assert_array_equal(upper, before[2])


@pytest.mark.parametrize(
    "patch",
    [
        {"diagonal": [-1]},
        {"linear_cost": [np.nan]},
        {"lower": [2], "upper": [1]},
        {"lower": [np.inf]},
        {"upper": [-np.inf]},
        {"lower": [np.nan]},
        {"matrix": [[np.inf]]},
        {"maximum_iterations": True},
        {"audit_tolerance": 1e-4},
        {"lower": 0.0},
    ],
)
def test_invalid_or_relaxed_backend_contract_rejected(patch):
    args = dict(diagonal=[1], linear_cost=[0], matrix=[[1]], lower=[0], upper=[1])
    args.update(patch)
    with pytest.raises(ValueError, match="box QP"):
        solve_box_qp(**args)


@pytest.mark.parametrize(
    "status,x,expected",
    [
        (clarabel.SolverStatus.AlmostSolved, [0.5], "AlmostSolved"),
        (clarabel.SolverStatus.Solved, [2.0], "independent_linear_constraints_failed"),
        (clarabel.SolverStatus.Solved, [np.nan], "nonfinite_or_wrong_shape_solution"),
        (clarabel.SolverStatus.Solved, [0.1, 0.2], "nonfinite_or_wrong_shape_solution"),
    ],
)
def test_solver_label_cannot_bypass_independent_audit(monkeypatch, status, x, expected):
    result = SimpleNamespace(status=status, x=x, iterations=10, r_prim=float("nan"), r_dual=0, solve_time=0.1)
    monkeypatch.setattr(clarabel, "DefaultSolver", lambda *args: SimpleNamespace(solve=lambda: result))
    solution, report = solve_box_qp([1], [0], [[1]], [0], [1])
    assert solution is None and not report["accepted"] and report["status"] == expected
    json.dumps(report, allow_nan=False)


def test_conflicting_original_rows_are_not_dropped():
    solution, report = solve_box_qp([1], [0], [[1], [1]], [0, 1], [0, 1])
    assert solution is None and not report["accepted"]
