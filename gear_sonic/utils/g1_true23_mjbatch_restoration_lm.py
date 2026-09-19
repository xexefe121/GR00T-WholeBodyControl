"""Opt-in third restoration attempt; existing two attempts delegate unchanged."""
import time

import numpy as np

from gear_sonic.utils.g1_true23_feasibility_referee import inspect_native_segment
from gear_sonic.utils.g1_true23_mjbatch_ilqr_core import boxqp, ilqr
from gear_sonic.utils.g1_true23_mjbatch_restoration import restore_feasible_seed as original_restore


def control_lm_backward(A, B, lx, lxx, lu, luu, lo, hi, mu, diagnostics=None):
    """Fixed control LM, symmetric Q, factored unregularized value update."""
    T, nu, nx = len(lu), lu.shape[1], lx.shape[1]
    vx, vxx = lx[-1], lxx[-1]
    k, K = np.zeros((T + 1, nu)), np.empty((T, nu, nx))
    for t in reversed(range(T)):
        qu = lu[t] + B[t].T @ vx
        raw_quu = luu[t] + B[t].T @ vxx @ B[t]
        symmetric_quu = .5 * (raw_quu + raw_quu.T)
        quu_reg = symmetric_quu + mu * np.eye(nu)
        qux_reg = B[t].T @ vxx @ A[t]
        finite = bool(np.isfinite(quu_reg).all())
        eigen_min = None
        entry = dict(knot=t, mu=float(mu), finite=finite)
        if finite:
            eigen_min = float(np.linalg.eigvalsh(quu_reg).min())
            entry.update(raw_lower_triangle_eigen_min=float(np.linalg.eigvalsh(raw_quu).min()),
                         symmetric_eigen_min=float(np.linalg.eigvalsh(symmetric_quu).min()),
                         regularized_eigen_min=eigen_min,
                         regularized_eigen_max=float(np.linalg.eigvalsh(quu_reg).max()),
                         raw_to_symmetric_max=float(np.abs(raw_quu-symmetric_quu).max()))
        if diagnostics is not None:
            diagnostics.append(entry)
        if not finite or eigen_min <= 0:
            return None
        k[t], free = boxqp(quu_reg, qu, lo[t], hi[t], k[t + 1])
        K[t] = 0
        K[t, free] = -np.linalg.solve(quu_reg[np.ix_(free, free)], qux_reg[free])
        M = A[t] + B[t] @ K[t]
        vx = lx[t] + K[t].T @ (lu[t] + luu[t] @ k[t]) + M.T @ (vx + vxx @ B[t] @ k[t])
        vxx = lxx[t] + K[t].T @ luu[t] @ K[t] + M.T @ vxx @ M
        vxx = .5 * (vxx + vxx.T)
    return k[:-1], K


def restore_feasible_seed(tracker, native, data, contract, motion, original_targets, window_start, *,
                          restorer=None, threads=8, retry_zero_feedback=True, save_proposal=None,
                          retry_control_lm=False, save_diagnostics=None):
    """Third attempt only after unchanged guided and original-warm K0 both fail."""
    if retry_control_lm and not retry_zero_feedback:
        raise ValueError("third control-LM retry requires both original restoration attempts")
    warm = np.asarray(original_targets).copy()
    guided = [None]

    def capture(mode, targets):
        if mode == "guided":
            guided[0] = np.asarray(targets).copy()
        return save_proposal(mode, targets) if save_proposal is not None else {}

    best, result, restorer = original_restore(
        tracker, native, data, contract, motion, warm, window_start, restorer=restorer, threads=threads,
        retry_zero_feedback=retry_zero_feedback,
        save_proposal=capture if retry_control_lm else save_proposal,
    )
    if not retry_control_lm or best is not None:
        return best, result, restorer
    started = time.perf_counter()
    attempt = dict(mode="control_lm_zero_feedback", accepted=False,
                   initial_iterate="first guided final proposal",
                   regularization_anchor="original shifted warm targets, unchanged",
                   mu_schedule=dict(initial=1., reject_factor=10., accept_factor=.1, floor=1e-6, cap=1e6),
                   Hessian="Qsym=.5*(Quu+Quu.T); Qreg=Qsym+muI; Qux unregularized",
                   value_update="factored closed-loop algebra with original unregularized stage cost",
                   maximum_iterations=10, eigenvalue_clipping=False, main_solver_changed=False)
    result["attempts"].append(attempt)
    proposal = guided[0]
    if proposal is None or proposal.shape != warm.shape or not np.isfinite(proposal).all():
        attempt["skipped_reason"] = "guided final proposal unavailable, malformed, or nonfinite"
        attempt["elapsed_ms"] = (time.perf_counter() - started) * 1000
        result["elapsed_ms"] += attempt["elapsed_ms"]
        return best, result, restorer
    actual = np.r_[data.qpos, data.qvel]
    restorer.original_targets = warm.copy()
    restorer.zero_rollout_feedback = True
    restorer.window(window_start)
    ordinary_rollout = restorer.rollout
    searches, sequences, matrices, sweeps = [], [], [], []

    def observed_rollout(*args, **kwargs):
        states, targets, costs = ordinary_rollout(*args, **kwargs)
        searches.append(costs.copy())
        sequences.append(targets.copy())
        return states, targets, costs

    def observed_backward(*args):
        entries = []
        value = control_lm_backward(*args, diagnostics=entries)
        sweeps.append(dict(mu=float(args[-1]), succeeded=value is not None))
        matrices.append(entries)
        return value

    restorer.rollout = observed_rollout
    try:
        initial_states, _, initial_costs = restorer.rollout(actual, proposal)
        attempt["initial_merit"] = float(initial_costs[0]) if np.isfinite(initial_costs[0]) else None
        if np.isfinite(initial_costs[0]):
            _, targets, _, merit = ilqr(
                restorer, actual, proposal.copy(), iters=10,
                initial_rollout=(initial_states[:, 0], initial_costs[0]), backward_override=observed_backward,
            )
            attempt["final_merit"] = float(merit)
            if save_proposal is not None:
                attempt.update(save_proposal(attempt["mode"], targets))
            states, bounded, costs = tracker.rollout(actual, targets)
            attempt["nominal_feasibility"] = tracker.last_rollout_feasibility
            if np.isfinite(costs[0]):
                oracle, _ = inspect_native_segment(native, data, targets, contract, retain_trace=False)
                attempt["independent_actual_state_oracle"] = oracle
                if oracle["feasible"]:
                    best = (float(costs[0]), "restoration", states[:, 0].copy(), bounded[:, 0].copy())
                    attempt["accepted"] = True
        else:
            attempt["skipped_reason"] = "guided final proposal has nonfinite initial restoration merit"
    finally:
        restorer.rollout = ordinary_rollout
        restorer.zero_rollout_feedback = False
    np.testing.assert_array_equal(restorer.original_targets, warm)
    attempt.update(backward_sweeps=sweeps, backward_matrices=matrices,
                   search_merits=[x.tolist() for x in searches],
                   elapsed_ms=(time.perf_counter() - started) * 1000)
    if save_diagnostics is not None:
        attempt.update(save_diagnostics(attempt["mode"], dict(
            targets_by_search=np.asarray(sequences), merit_by_search=np.asarray(searches),
            original_regularization_anchor=warm, optimization_initial_targets=proposal,
        )))
    result["accepted"] = best is not None
    result["elapsed_ms"] += attempt["elapsed_ms"]
    return best, result, restorer
