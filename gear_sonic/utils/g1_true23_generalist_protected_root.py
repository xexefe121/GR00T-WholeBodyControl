"""Offline SE(3)+23 fitting with explicit protected-task norm constraints.

The existing engineering objective is unchanged. Foot positions/orientations,
COM regression, and the original per-frame weighted-cost budget are hard SOC
constraints on each linearized subproblem, not extra objective penalties.
Intermediate nonlinear iterates are not accepted reference motions. Final FK,
serialization, and the original independent expert gate remain authoritative.
"""

from __future__ import annotations

from dataclasses import asdict
from itertools import product

import clarabel
import numpy as np
from scipy import sparse

from gear_sonic.utils.g1_23dof_trajectory_projection import _constraint_system
from gear_sonic.utils.g1_true23_original_task_trajectory import TASK_SCALES


def residual_groups(problem, baseline):
    """Map exact existing task residuals to the unchanged expert-mask budgets."""
    offsets, cursor = {}, 0
    for task in problem.tasks:
        ps, rs = TASK_SCALES[task.name]
        for kind, scale in (("position", ps), ("orientation", rs)):
            if scale:
                offsets[task.name, kind] = (np.arange(cursor, cursor + 3), scale)
                cursor += 3
    groups = []
    for frame in range(len(problem.initial)):
        for name, kind, radius in (
            ("left_foot", "position", baseline.config.valid_max_foot_position_error_m),
            ("right_foot", "position", baseline.config.valid_max_foot_position_error_m),
            (
                "left_foot",
                "orientation",
                baseline.diagnostics["task_left_foot_orientation_error_before_rad"][frame]
                + baseline.config.valid_max_foot_orientation_regression_rad,
            ),
            (
                "right_foot",
                "orientation",
                baseline.diagnostics["task_right_foot_orientation_error_before_rad"][frame]
                + baseline.config.valid_max_foot_orientation_regression_rad,
            ),
            (
                "whole_robot_com",
                "position",
                baseline.diagnostics["task_whole_robot_com_position_error_before_m"][frame]
                + baseline.config.valid_max_com_regression_m,
            ),
        ):
            indices, scale = offsets[name, kind]
            groups.append(
                {
                    "name": f"{name}_{kind}",
                    "frame": frame,
                    "indices": frame * cursor + indices,
                    "scales": np.full(3, 1 / scale),
                    "radius": float(radius),
                }
            )
        cost_scale = np.zeros(cursor)
        for task in problem.tasks:
            contact = (task.contact_side == "left" and baseline.contact_flags[frame, 0]) or (
                task.contact_side == "right" and baseline.contact_flags[frame, 1]
            )
            multiplier = baseline.config.contact_weight_multiplier if contact else 1.0
            for kind, weight in (("position", task.position_weight), ("orientation", task.orientation_weight)):
                if weight:
                    indices, scale = offsets[task.name, kind]
                    cost_scale[indices] = np.sqrt(weight * multiplier) / scale
        before = baseline.diagnostics["weighted_task_error_before"][frame]
        cost_bound = before + baseline.config.priority_relative_tolerance * max(1.0, before)
        groups.append(
            {
                "name": "original_weighted_cost",
                "frame": frame,
                "indices": frame * cursor + np.arange(cursor),
                "scales": cost_scale,
                "radius": float(np.sqrt(cost_bound)),
            }
        )
    return groups, cursor


def audit_norms(residual, groups):
    categories = {}
    maximum = 0.0
    for group in groups:
        norm = float(np.linalg.norm(residual[group["indices"]] * group["scales"]))
        excess = max(norm - group["radius"], 0.0)
        maximum = max(maximum, excess / max(group["radius"], 1e-12))
        row = categories.setdefault(group["name"], {"failed_frames": [], "maximum_excess": 0.0})
        if excess > 0:
            row["failed_frames"].append(group["frame"])
        row["maximum_excess"] = max(row["maximum_excess"], excess)
    return {"passed": maximum == 0.0, "maximum_normalized_excess": maximum, "categories": categories}


def solve_box_soc(diagonal, cost, matrix, lower, upper, norm_groups):
    """Strict box/SOC solve with independent original-row and norm audits."""
    diagonal, cost, lower, upper = [np.asarray(x, dtype=float) for x in (diagonal, cost, lower, upper)]
    matrix = sparse.csc_matrix(matrix)
    if (
        diagonal.ndim != 1
        or cost.shape != diagonal.shape
        or np.any(diagonal < 0)
        or matrix.shape != (len(lower), len(diagonal))
        or upper.shape != lower.shape
        or not all(np.isfinite(x).all() for x in (diagonal, cost, matrix.data))
        or np.isnan(lower).any()
        or np.isnan(upper).any()
        or np.any(lower > upper)
        or np.isposinf(lower).any()
        or np.isneginf(upper).any()
    ):
        raise ValueError("invalid finite convex protected-task problem")
    equal = np.isfinite(lower) & (lower == upper)
    high, low = np.isfinite(upper) & ~equal, np.isfinite(lower) & ~equal
    parts = [matrix[equal], matrix[high], -matrix[low]]
    rhs = [upper[equal], upper[high], -lower[low]]
    cones = []
    if equal.sum():
        cones.append(clarabel.ZeroConeT(int(equal.sum())))
    if high.sum() + low.sum():
        cones.append(clarabel.NonnegativeConeT(int(high.sum() + low.sum())))
    row_indices, columns, values, cone_rhs, row = [], [], [], [], 0
    for group in norm_groups:
        indices = np.asarray(group["indices"])
        scales = np.asarray(group["scales"], dtype=float)
        radius = float(group["radius"])
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or not len(indices)
            or scales.shape != indices.shape
            or not np.isfinite(scales).all()
            or np.any(indices < 0)
            or np.any(indices >= len(diagonal))
            or not np.isfinite(radius)
            or radius < 0
        ):
            raise ValueError("invalid protected norm constraint")
        row_indices.extend(range(row + 1, row + 1 + len(indices)))
        columns.extend(indices.tolist())
        values.extend((-scales).tolist())
        cone_rhs.extend([radius, *np.zeros(len(indices))])
        row += 1 + len(indices)
        cones.append(clarabel.SecondOrderConeT(1 + len(indices)))
    parts.append(sparse.csc_matrix((values, (row_indices, columns)), shape=(row, len(diagonal))))
    rhs.append(np.asarray(cone_rhs))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter, settings.max_threads = 200, 1
    settings.direct_solve_method = "qdldl"
    settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = 1e-9
    answer = clarabel.DefaultSolver(
        sparse.diags(diagonal, format="csc"),
        cost,
        sparse.vstack(parts, format="csc"),
        np.concatenate(rhs),
        cones,
        settings,
    ).solve()
    report = {
        "backend": "clarabel",
        "backend_version": clarabel.__version__,
        "status": str(answer.status),
        "iterations": int(answer.iterations),
        "solve_time_s": float(answer.solve_time),
        "norm_constraint_count": len(norm_groups),
        "accepted": False,
        "original_row_audit_tolerance": 1e-8,
        "requested_feasibility_tolerance": 1e-9,
    }
    if not np.isfinite(report["solve_time_s"]):
        report["solve_time_s"] = None
    if answer.status != clarabel.SolverStatus.Solved or answer.x is None:
        return None, report
    solution = np.asarray(answer.x, dtype=float)
    if solution.shape != diagonal.shape or not np.isfinite(solution).all():
        report["status"] = "nonfinite_or_wrong_shape_solution"
        return None, report
    actual = matrix @ solution
    violation = float(
        max(np.maximum(lower - actual, 0).max(initial=0), np.maximum(actual - upper, 0).max(initial=0))
    )
    norm_violation = max(
        (max(float(np.linalg.norm(solution[g["indices"]] * g["scales"])) - g["radius"], 0.0) for g in norm_groups),
        default=0.0,
    )
    report.update(independent_original_row_violation=violation, independent_norm_violation=norm_violation)
    if not np.isfinite(violation + norm_violation) or max(violation, norm_violation) > 1e-8:
        report["status"] = "independent_original_constraints_failed"
        return None, report
    report["accepted"] = True
    return solution, report


def protected_step(problem, current, residual, jacobian, groups):
    cfg = problem.config
    trust = np.repeat([cfg.root_trust_m, cfg.rotation_trust_rad, cfg.joint_trust_rad], [3, 3, 23])
    lo, hi = np.maximum(problem.lower, current - trust), np.minimum(problem.upper, current + trust)
    operator, lower, upper = _constraint_system(
        len(current),
        lo,
        hi,
        problem.velocity * cfg.serialization_margin_fraction * 0.02,
        problem.acceleration * cfg.serialization_margin_fraction * 0.02**2,
        problem.initial_velocity * 0.02,
    )
    temporal = sparse.kron(operator, sparse.eye(29), format="csc")
    facets = np.zeros((8, 29))
    facets[:, 3:6] = list(product((-1.0, 1.0), repeat=3))
    rotation = sparse.kron(sparse.eye(len(current)), sparse.csc_matrix(facets), format="csc")
    n, m = current.size, len(residual)
    matrix = sparse.vstack(
        (
            sparse.hstack((temporal, sparse.csc_matrix((temporal.shape[0], m)))),
            sparse.hstack((rotation, sparse.csc_matrix((rotation.shape[0], m)))),
            sparse.hstack((jacobian, -sparse.eye(m))),
        ),
        format="csc",
    )
    temporal_origin, rotation_origin = temporal @ current.ravel(), rotation @ current.ravel()
    groups_with_variables = [
        {**g, "indices": g["indices"] + n, "radius": g["radius"] * cfg.serialization_margin_fraction}
        for g in groups
    ]
    scale = 1 / len(current)
    solution, report = solve_box_soc(
        scale * np.r_[problem.posture_weight, np.ones(m)],
        scale * np.r_[problem.posture_weight * (current - problem.initial).ravel(), np.zeros(m)],
        matrix,
        np.r_[lower.ravel() - temporal_origin, np.full(rotation.shape[0], -np.inf), -residual],
        np.r_[upper.ravel() - temporal_origin, cfg.maximum_root_rotation_l1_rad - rotation_origin, -residual],
        groups_with_variables,
    )
    if solution is None:
        return None, report
    candidate = current + solution[:n].reshape(current.shape)
    # Separate absolute-row audit catches errors in the increment formulation.
    absolute = np.r_[candidate.ravel(), solution[n:]]
    actual = matrix @ absolute
    equality = jacobian @ current.ravel() - residual
    absolute_lower = np.r_[lower.ravel(), np.full(rotation.shape[0], -np.inf), equality]
    absolute_upper = np.r_[upper.ravel(), np.full(rotation.shape[0], cfg.maximum_root_rotation_l1_rad), equality]
    violation = float(
        max(
            np.maximum(absolute_lower - actual, 0).max(initial=0),
            np.maximum(actual - absolute_upper, 0).max(initial=0),
        )
    )
    report["independent_absolute_row_violation"] = violation
    if not np.isfinite(violation) or violation > 1e-8:
        report.update(accepted=False, status="independent_absolute_row_audit_failed")
        return None, report
    return candidate, report


def fit_protected_task_path(problem, initial_variables, baseline, *, progress=None):
    """Restore hard protected feasibility, then stop once all position gates pass.

    A strict linearized solve may still have nonlinear residual. Line search
    may accept only decreased exact constraint excess until feasibility; once
    feasible, it must preserve every hard task bound. No slack variables or
    gate changes are used. Infeasible subproblems are not physical certificates.
    """
    current = np.array(initial_variables, dtype=float, copy=True)
    if current.shape != problem.initial.shape or not problem.audit(current)["passed"]:
        raise ValueError("protected refinement requires a full temporally feasible starting path")
    groups, rows_per_frame = residual_groups(problem, baseline)
    residual, _, _ = problem.evaluate(current, derivatives=False)
    if len(residual) != len(current) * rows_per_frame:
        raise ValueError("task residual layout differs from protected norm constraints")
    before, history, failure = audit_norms(residual, groups), [], None
    target_tasks = [
        i for i, task in enumerate(problem.tasks) if task.name in ("head_proxy", "left_hand", "right_hand")
    ]
    for iteration in range(problem.config.maximum_iterations):
        residual, jacobian, errors = problem.evaluate(current)
        current_audit = audit_norms(residual, groups)
        if current_audit["passed"] and np.all(np.percentile(errors[:, target_tasks, 0], 95, axis=0) <= 0.10):
            break
        candidate, solver = protected_step(problem, current, residual, jacobian, groups)
        row = {"iteration": iteration + 1, "qp": solver, "accepted": False}
        if candidate is None:
            failure = "strict protected linearized subproblem failed; not physical infeasibility"
        else:
            old_delta = (current - problem.initial).ravel()
            old_merit = residual @ residual + np.dot(problem.posture_weight * old_delta, old_delta)
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = current + fraction * (candidate - current)
                if not problem.audit(trial)["passed"]:
                    continue
                trial_residual, _, _ = problem.evaluate(trial, derivatives=False)
                audit = audit_norms(trial_residual, groups)
                if not current_audit["passed"]:
                    acceptable = audit["passed"] or audit["maximum_normalized_excess"] < (
                        current_audit["maximum_normalized_excess"] * (1 - 1e-8)
                    )
                else:
                    delta = (trial - problem.initial).ravel()
                    merit = trial_residual @ trial_residual + np.dot(problem.posture_weight * delta, delta)
                    acceptable = audit["passed"] and merit < old_merit - 1e-10 * max(1.0, old_merit)
                if acceptable:
                    current = trial
                    row.update(accepted=True, fraction=fraction, protected_audit=audit)
                    break
            if not row["accepted"]:
                failure = "nonlinear protected feasibility stalled; not physical infeasibility"
        history.append(row)
        if progress:
            progress(row)
        if failure:
            break
    residual, _, _ = problem.evaluate(current, derivatives=False)
    return current, {
        "kind": "g1_true23_hard_protected_root_refinement_v1",
        "config": asdict(problem.config),
        "before_protected_audit": before,
        "after_protected_audit": audit_norms(residual, groups),
        "after": problem.metrics(current),
        "iterations": history,
        "failure": failure,
        "path_constraints": problem.audit(current),
        "frames_dropped": 0,
        "protected_constraints_are_objective_penalties": False,
        "constraint_slack_variables_used": False,
        "nonlinear_intermediate_iterates_are_accepted_references": False,
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
