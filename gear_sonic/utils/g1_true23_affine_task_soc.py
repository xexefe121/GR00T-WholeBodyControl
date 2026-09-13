"""Equivalent task SQP with residual variables eliminated analytically.

min 0.5||r+Jd||² + 0.5||(x+d-x0)||²_P, with the same box/linear
constraints and affine residual norm constraints. No acceptance tolerance is
changed. Removing auxiliary residual equalities avoids auditing cancellation
error in variables that need not be optimized independently in the first place.
"""

import clarabel
import numpy as np
from scipy import sparse


def solve_affine_task_soc(
    posture,
    displacement,
    residual,
    jacobian,
    matrix,
    lower,
    upper,
    groups,
    *,
    objective_scale=1.0,
    objective_profile="task_lsq_v1",
):
    if objective_profile not in ("task_lsq_v1", "minimum_change_v2", "pose_escape_guided_v3"):
        raise ValueError("unknown affine task objective profile")
    posture, displacement, residual, lower, upper = [
        np.asarray(value, dtype=float) for value in (posture, displacement, residual, lower, upper)
    ]
    jacobian, matrix = sparse.csc_matrix(jacobian), sparse.csc_matrix(matrix)
    n = len(posture)
    if (
        posture.shape != (n,)
        or displacement.shape != (n,)
        or residual.ndim != 1
        or jacobian.shape != (len(residual), n)
        or matrix.shape != (len(lower), n)
        or upper.shape != lower.shape
        or np.any(posture < 0)
        or (objective_profile != "task_lsq_v1" and np.any(posture <= 0))
        or not all(np.isfinite(v).all() for v in (posture, displacement, residual, jacobian.data, matrix.data))
        or np.isnan(lower).any()
        or np.isnan(upper).any()
        or np.isposinf(lower).any()
        or np.isneginf(upper).any()
        or np.any(lower > upper)
        or not np.isfinite(objective_scale)
        or objective_scale <= 0
    ):
        raise ValueError("invalid finite affine task SOC problem")
    equal = np.isfinite(lower) & (lower == upper)
    high, low = np.isfinite(upper) & ~equal, np.isfinite(lower) & ~equal
    blocks, rhs, cones = [matrix[equal], matrix[high], -matrix[low]], [upper[equal], upper[high], -lower[low]], []
    if equal.sum():
        cones.append(clarabel.ZeroConeT(int(equal.sum())))
    if high.sum() + low.sum():
        cones.append(clarabel.NonnegativeConeT(int(high.sum() + low.sum())))
    for group in groups:
        indices, scales, radius = (
            np.asarray(group["indices"]),
            np.asarray(group["scales"], dtype=float),
            float(group["radius"]),
        )
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or not len(indices)
            or scales.shape != indices.shape
            or np.any(indices < 0)
            or np.any(indices >= len(residual))
            or not np.isfinite(scales).all()
            or not np.isfinite(radius)
            or radius < 0
        ):
            raise ValueError("invalid affine residual norm constraint")
        blocks.extend((sparse.csc_matrix((1, n)), -sparse.diags(scales) @ jacobian[indices]))
        rhs.append(np.r_[radius, scales * residual[indices]])
        cones.append(clarabel.SecondOrderConeT(1 + len(indices)))
    if objective_profile != "task_lsq_v1":
        # Explicitly different engineering objective: nearest CURRENT iterate.
        # Residuals still enter every original hard task norm constraint, but
        # unrelated task improvements cannot drive unnecessary pose movement.
        hessian = objective_scale * sparse.diags(posture)
        cost = (
            objective_scale * posture * displacement
            if objective_profile == "pose_escape_guided_v3"
            else np.zeros(n)
        )
    else:
        hessian = objective_scale * (jacobian.T @ jacobian + sparse.diags(posture))
        cost = objective_scale * (jacobian.T @ residual + posture * displacement)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter, settings.max_threads, settings.direct_solve_method = 200, 1, "qdldl"
    settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = 1e-10
    answer = clarabel.DefaultSolver(
        sparse.triu(hessian, format="csc"),
        cost,
        sparse.vstack(blocks, format="csc"),
        np.concatenate(rhs),
        cones,
        settings,
    ).solve()
    report = {
        "backend": "clarabel",
        "backend_version": clarabel.__version__,
        "status": str(answer.status),
        "formulation": "analytically_eliminated_task_residuals_v2",
        "iterations": int(answer.iterations),
        "solve_time_s": float(answer.solve_time),
        "accepted": False,
        "decision_variables": n,
        "eliminated_residual_variables": len(residual),
        "norm_constraint_count": len(groups),
        "objective_scale": objective_scale,
        "objective_profile": objective_profile,
        "requested_feasibility_tolerance": 1e-10,
        "original_row_audit_tolerance": 1e-8,
        "objective_and_constraints_changed": objective_profile != "task_lsq_v1",
        "objective_changed_from_task_lsq": objective_profile != "task_lsq_v1",
        "hard_constraints_changed": False,
        "qp_optimality_proven": False,
    }
    if answer.status not in (clarabel.SolverStatus.Solved, clarabel.SolverStatus.AlmostSolved) or answer.x is None:
        return None, report
    step = np.asarray(answer.x)
    if step.shape != (n,) or not np.isfinite(step).all():
        report["status"] = "nonfinite_or_wrong_shape_step"
        return None, report
    actual = matrix @ step
    linear_violation = float(
        max(np.maximum(lower - actual, 0).max(initial=0), np.maximum(actual - upper, 0).max(initial=0))
    )
    actual_residual = residual + jacobian @ step
    norm_violation = max(
        (
            max(float(np.linalg.norm(actual_residual[g["indices"]] * g["scales"])) - g["radius"], 0.0)
            for g in groups
        ),
        default=0.0,
    )
    report.update(independent_original_row_violation=linear_violation, independent_norm_violation=norm_violation)
    if not np.isfinite(linear_violation + norm_violation) or max(linear_violation, norm_violation) > 1e-8:
        report["status"] = "independent_original_constraints_failed"
        return None, report
    report["accepted"] = True
    report["inaccurate_solution_independently_feasible"] = answer.status == clarabel.SolverStatus.AlmostSolved
    return step, report
