"""Strict, independently audited offline box-QP backend.

Only diagonal positive-semidefinite objectives are accepted. Original rows of
``lower <= matrix @ x <= upper`` are audited after the cone-form solve; reduced
accuracy status, nonfinite results and excessive original residuals fail closed.
This numerical result is not robot or reference qualification.
"""

from __future__ import annotations

import clarabel
import numpy as np
from scipy import sparse


def solve_box_qp(
    diagonal, linear_cost, matrix, lower, upper, *, maximum_iterations=200, audit_tolerance=1e-8, progress=None
):
    diagonal, linear_cost, lower, upper = (
        np.asarray(value, dtype=float) for value in (diagonal, linear_cost, lower, upper)
    )
    matrix = sparse.csc_matrix(matrix, dtype=float)
    if (
        diagonal.ndim != 1
        or not len(diagonal)
        or linear_cost.shape != diagonal.shape
        or lower.ndim != 1
        or matrix.shape != (len(lower), len(diagonal))
        or upper.shape != lower.shape
        or not np.isfinite(diagonal).all()
        or np.any(diagonal < 0)
        or not np.isfinite(linear_cost).all()
        or not np.isfinite(matrix.data).all()
        or np.isnan(lower).any()
        or np.isnan(upper).any()
        or np.any(lower > upper)
        or np.isposinf(lower).any()
        or np.isneginf(upper).any()
        or type(maximum_iterations) is not int
        or maximum_iterations <= 0
        or isinstance(audit_tolerance, bool)
        or not np.isfinite(audit_tolerance)
        or not 0 < audit_tolerance <= 1e-8
    ):
        raise ValueError("box QP requires finite convex data, consistent bounds and strict audit settings")
    equality = np.isfinite(lower) & (lower == upper)
    finite_upper = np.isfinite(upper) & ~equality
    finite_lower = np.isfinite(lower) & ~equality
    cone_matrix = sparse.vstack([matrix[equality], matrix[finite_upper], -matrix[finite_lower]], format="csc")
    rhs = np.r_[upper[equality], upper[finite_upper], -lower[finite_lower]]
    neq = int(equality.sum())
    nineq = int(finite_upper.sum() + finite_lower.sum())
    cones = []
    if neq:
        cones.append(clarabel.ZeroConeT(neq))
    if nineq:
        cones.append(clarabel.NonnegativeConeT(nineq))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = maximum_iterations
    settings.max_threads = 1
    settings.direct_solve_method = "qdldl"
    settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = 1e-9
    solver = clarabel.DefaultSolver(
        sparse.diags(diagonal, format="csc"), linear_cost, cone_matrix, rhs, cones, settings
    )
    if progress:

        def callback(info):
            if info.iterations % 10 == 0:
                progress({"qp_iteration": int(info.iterations)})
            return False

        solver.set_termination_callback(callback)
    result = solver.solve()
    report = {
        "backend": "clarabel",
        "backend_version": clarabel.__version__,
        "status": str(result.status),
        "iterations": int(result.iterations),
        "primal_residual": float(result.r_prim),
        "dual_residual": float(result.r_dual),
        "solve_time_s": float(result.solve_time),
        "original_rows": matrix.shape[0],
        "cone_equality_rows": neq,
        "cone_inequality_rows": nineq,
        "maximum_threads": 1,
        "direct_solve_method": "qdldl",
        "requested_feasibility_tolerance": 1e-9,
        "requested_gap_tolerance": 1e-9,
        "accepted": False,
    }
    # Preserve serializable failure evidence even when a backend reports NaN.
    for key in ("primal_residual", "dual_residual", "solve_time_s"):
        if not np.isfinite(report[key]):
            report[key] = None
    if result.status != clarabel.SolverStatus.Solved or result.x is None:
        return None, report
    solution = np.asarray(result.x, dtype=float)
    if solution.shape != diagonal.shape or not np.isfinite(solution).all():
        report["status"] = "nonfinite_or_wrong_shape_solution"
        return None, report
    actual = matrix @ solution
    violation = max(np.maximum(lower - actual, 0).max(initial=0), np.maximum(actual - upper, 0).max(initial=0))
    report["independent_maximum_scaled_linear_violation"] = float(violation) if np.isfinite(violation) else None
    if not np.isfinite(violation) or violation > audit_tolerance:
        report["status"] = "independent_linear_constraints_failed"
        return None, report
    report["accepted"] = True
    return solution, report
