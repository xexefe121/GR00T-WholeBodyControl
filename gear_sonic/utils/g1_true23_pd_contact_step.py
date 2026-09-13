"""SIM-only contact-constrained local PD steps; nonlinear physics decides.

Linear constraints do not certify contact safety. Original physical geometry,
every physics substep, and the complete original motion must still be replayed.
"""

import clarabel
import numpy as np
from scipy import sparse

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_pd_trajectory_optimizer import CONTROL_WEIGHT, SLEW_WEIGHT
from gear_sonic.utils.g1_true23_self_collision import SelfCollisionLinearizer


def _equality_solution(h, g, matrix, bound, cost_state, constraint_state):
    n, m = len(g), len(bound)
    if m:
        if np.linalg.matrix_rank(matrix, tol=1e-11) != m:
            raise ValueError("dependent active contact constraints")
        basis = np.linalg.qr(matrix.T, mode="complete")[0][:, m:]
        particular = np.linalg.lstsq(matrix, bound, rcond=1e-12)[0]
        gain = np.linalg.lstsq(matrix, -constraint_state, rcond=1e-12)[0]
    else:
        basis, particular, gain = np.eye(n), np.zeros(n), np.zeros_like(cost_state)
    if basis.shape[1]:
        reduced = basis.T @ h @ basis
        x = particular - basis @ np.linalg.solve(reduced, basis.T @ (h @ particular + g))
        gain -= basis @ np.linalg.solve(reduced, basis.T @ (h @ gain + cost_state))
    else:
        x = particular
    multiplier = np.linalg.lstsq(matrix.T, -(h @ x + g), rcond=1e-12)[0] if m else np.zeros(0)
    return x, gain, multiplier


def solve_contact_quadratic(hessian, gradient, matrix, bound, cost_state, constraint_state):
    """min .5 u'Hu + g'u with Cu <= b; also differentiate b - Cx dx.

    Clarabel finds a feasible working point. A primal working-set polish then
    checks KKT conditions and derives feedback from the same active constraints.
    This is a local approximation, never a physical-feasibility certificate.
    """
    h, g, c, b, gx, cx = [
        np.asarray(value, dtype=float)
        for value in (hessian, gradient, matrix, bound, cost_state, constraint_state)
    ]
    n, m = len(g), len(b)
    if h.shape != (n, n) or c.shape != (m, n) or gx.ndim != 2 or gx.shape[0] != n:
        raise ValueError("contact quadratic shape mismatch")
    if cx.shape != (m, gx.shape[1]) or not all(np.isfinite(v).all() for v in (h, g, c, b, gx, cx)):
        raise ValueError("contact quadratic needs finite compatible state constraints")
    if not np.allclose(h, h.T, atol=1e-10, rtol=1e-12):
        raise ValueError("contact quadratic Hessian must be symmetric")
    np.linalg.cholesky(h)
    objective_scale = max(1.0, float(np.max(np.abs(h))), float(np.max(np.abs(g))))
    row_scale = np.maximum(np.maximum(np.linalg.norm(c, axis=1), np.abs(b)), 1e-12)
    hn, gn, gxn = h / objective_scale, g / objective_scale, gx / objective_scale
    cn, bn, cxn = c / row_scale[:, None], b / row_scale, cx / row_scale[:, None]
    settings = clarabel.DefaultSettings()
    settings.verbose, settings.max_threads, settings.max_iter = False, 1, 200
    settings.direct_solve_method = "qdldl"
    settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = 1e-11
    answer = clarabel.DefaultSolver(
        sparse.triu(sparse.csc_matrix(hn), format="csc"),
        gn,
        sparse.csc_matrix(cn),
        bn,
        [clarabel.NonnegativeConeT(m)],
        settings,
    ).solve()
    if answer.status not in (clarabel.SolverStatus.Solved, clarabel.SolverStatus.AlmostSolved):
        raise ValueError("contact quadratic has no verified feasible start: " + str(answer.status))
    x = np.asarray(answer.x)
    if not np.isfinite(x).all() or np.max(cn @ x - bn, initial=0.0) > 1e-8:
        raise ValueError("contact quadratic initial feasibility audit failed")
    working = []
    for iteration in range(40 * (n + m) + 1):
        selected = np.asarray(working, dtype=int)
        optimum, gain, multipliers = _equality_solution(hn, gn, cn[selected], bn[selected], gxn, cxn[selected])
        violation = cn @ optimum - bn
        if np.max(violation, initial=0.0) > 1e-10:
            direction = optimum - x
            rate = cn @ direction
            ratios = np.where(rate > 1e-14, np.maximum(0.0, bn - cn @ x) / np.maximum(rate, 1e-300), np.inf)
            ratios[selected] = np.inf
            blocker = int(np.argmin(ratios))
            step = min(1.0, float(ratios[blocker]))
            if not np.isfinite(step) or blocker in working:
                raise RuntimeError("contact working set cannot resolve an infeasible step")
            x = x + step * direction
            working.append(blocker)
            continue
        x = optimum
        if len(multipliers) and np.min(multipliers) < -1e-11:
            del working[int(np.argmin(multipliers))]
            continue
        stationarity = hn @ x + gn + cn[selected].T @ multipliers
        if np.max(np.abs(stationarity), initial=0.0) > 1e-8 or np.max(c @ x - b, initial=0.0) > 1e-8:
            raise RuntimeError("contact quadratic independent KKT or original-row audit failed")
        if np.max(np.abs(cn[selected] @ gain + cxn[selected]), initial=0.0) > 1e-8:
            raise RuntimeError("contact feedback violates its active state-dependent constraints")
        return (
            x,
            gain,
            dict(
                backend_status=str(answer.status),
                polish_iterations=iteration + 1,
                active_rows=working,
                maximum_original_row_violation=float(np.max(c @ x - b, initial=0.0)),
                maximum_scaled_stationarity=float(np.max(np.abs(stationarity), initial=0.0)),
            ),
        )
    raise RuntimeError("contact quadratic working set did not converge")


def linearize_self_envelopes(model, trajectory, limits, progress=None):
    """Next-control signed geometry; private query model never supplies forces."""
    query, columns, rows = SelfCollisionLinearizer(model, near_distance_m=0.03), np.arange(29), []
    for index, pose in enumerate(trajectory["qpos"][1:]):
        # Reserve 5 mm inside the query horizon so finite-difference probes
        # cannot disappear merely by crossing the query cutoff.
        contacts = [row for row in query.pose_rows(pose, columns) if row["distance_m"] < 0.025]
        jacobian = np.zeros((len(contacts), 29))
        if contacts:
            # Only articulated joints affect robot-robot signed distance. Use
            # central geometry differences, matching the protected cost.
            for joint in range(23):
                values = []
                for sign in (-1, 1):
                    perturbed = pose.copy()
                    perturbed[7 + joint] += sign * 1e-6
                    sampled = {row["geoms"]: row["distance_m"] for row in query.pose_rows(perturbed, columns)}
                    if any(row["geoms"] not in sampled for row in contacts):
                        raise ValueError("near-contact pair vanished during constraint differentiation")
                    values.append(np.asarray([sampled[row["geoms"]] for row in contacts]))
                jacobian[:, 6 + joint] = (values[1] - values[0]) / 2e-6
        rows.append(
            dict(
                jacobian=jacobian,
                distance=np.asarray([row["distance_m"] for row in contacts]),
                floor=np.asarray([limits.get(row["geoms"], 0.0) for row in contacts]),
                geoms=[row["geoms"] for row in contacts],
            )
        )
        if progress is not None and ((index + 1) % 100 == 0 or index + 2 == len(trajectory["qpos"])):
            progress(
                dict(
                    stage="linearize_self_contact_constraints",
                    completed=index + 1,
                    total=len(trajectory["qpos"]) - 1,
                )
            )
    return rows


def contact_backward_pass(
    plant, local, controls, seed_controls, geometry, regularization, progress=None, *, stage_models=None
):
    count = len(controls)
    increments, feedback = np.zeros((count, 23)), np.zeros((count, 23, 81))
    vg, vh, identity, reports = local["terminal_g"].copy(), local["terminal_h"].copy(), np.eye(23), []
    if stage_models is not None:
        if stage_models:
            raise ValueError("stage-model destination must start empty")
        stage_models.extend([None] * count)
    for i in reversed(range(count)):
        a, b = local["a"][i], local["b"][i]
        previous = controls[i - 1] if i else np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
        delta, slew = controls[i] - seed_controls[i], controls[i] - previous
        lx, lxx = local["gradient"][i].copy(), local["hessian"][i].copy()
        lx[58:] -= SLEW_WEIGHT * slew
        lxx[58:, 58:] += SLEW_WEIGHT * identity
        lu = CONTROL_WEIGHT * delta + SLEW_WEIGHT * slew
        lux = np.zeros((23, 81))
        lux[:, 58:] = -SLEW_WEIGHT * identity
        qx, qu = lx + a.T @ vg, lu + b.T @ vg
        qxx = lxx + a.T @ vh @ a
        quu = (CONTROL_WEIGHT + SLEW_WEIGHT) * identity + b.T @ vh @ b
        qux = lux + b.T @ vh @ a
        h = 0.5 * (quu + quu.T) + regularization * identity
        if stage_models is not None:
            stage_models[i] = dict(hessian=h.copy(), gradient=qu.copy(), gradient_state=qux.copy())
        geom = geometry[i]
        state_rows = np.zeros((len(geom["distance"]), 81))
        state_rows[:, :29] = -geom["jacobian"]
        matrix = np.vstack((identity, -identity, state_rows @ b))
        bound = np.r_[plant.upper - controls[i], controls[i] - plant.lower, geom["distance"] - geom["floor"]]
        constraint_state = np.vstack((np.zeros((46, 81)), state_rows @ a))
        try:
            k, gain, report = solve_contact_quadratic(h, qu, matrix, bound, qux, constraint_state)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(f"contact backward control {i}: {exc}") from exc
        report["control_index"] = i
        reports.append(report)
        increments[i], feedback[i] = k, gain
        vg = qx + gain.T @ qu + qux.T @ k + gain.T @ quu @ k
        vh = qxx + gain.T @ quu @ gain + gain.T @ qux + qux.T @ gain
        vh = 0.5 * (vh + vh.T)
        if not np.isfinite(vg).all() or not np.isfinite(vh).all():
            raise RuntimeError("contact trajectory quadratic approximation diverged")
        if progress is not None and (i % 200 == 0):
            progress(dict(stage="contact_backward", control=i, total=count))
    return increments, feedback, reports
