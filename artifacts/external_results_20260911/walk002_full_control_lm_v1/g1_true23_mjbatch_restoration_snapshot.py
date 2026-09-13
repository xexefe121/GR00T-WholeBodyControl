"""Experimental seed-only physical-margin merit; never an execution policy."""

import time

import numpy as np

from gear_sonic.utils.g1_true23_feasibility_referee import inspect_native_segment
from gear_sonic.utils.g1_true23_mjbatch_ilqr_core import ilqr
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker


class Native23RestorationTracker(Native23Tracker):
    """Search finite, possibly infeasible counterfactuals; certify separately."""

    def __init__(self, servo, contract, motion, original_targets, *, threads=8, zero_rollout_feedback=False):
        targets = np.asarray(original_targets, float)
        if targets.shape != (30, 23) or not np.isfinite(targets).all():
            raise ValueError("restoration requires the declared finite H30 target seed")
        super().__init__(servo, contract, motion, horizon=30, threads=threads,
                         fd_epsilon=1e-6, hard_feasibility=False)
        np.testing.assert_array_equal(targets, np.clip(targets, self.lo, self.hi))
        self.original_targets = targets.copy()
        self.control_weight = 1e-4
        self.zero_rollout_feedback = bool(zero_rollout_feedback)

    def target_reference(self, t):
        return self.original_targets[np.minimum(t, self.T - 1)]

    def residual(self, t, features):
        del t
        state = features[..., 42:]
        joints, speed = state[..., 7:30], np.abs(state[..., 36:])
        caps = np.asarray(self.contract["native_velocity"])
        tilt = np.arccos(np.clip(1 - 2 * np.sum(state[..., 4:6] ** 2, axis=-1), -1, 1))
        return np.concatenate((
            np.maximum(self.lo + .02 - joints, 0) / .02,
            np.maximum(joints - self.hi + .02, 0) / .02,
            np.maximum(speed - .9 * caps, 0) / (.1 * caps),
            np.maximum(.30 - state[..., 2:3], 0) / .05,
            np.maximum(tilt[..., None] - 1.1, 0) / .1,
        ), axis=-1)

    def rollout(self, x0, us, gains=None):
        if gains is not None and self.zero_rollout_feedback:
            xs, k, K = gains
            gains = (xs, k, np.zeros_like(K))
        states, targets, totals = super().rollout(x0, us, gains)
        # Intermediate native range violations are allowed only in this private
        # merit search. Engine-invalid/nonfinite costs cannot win a line search.
        totals[~np.isfinite(totals)] = np.inf
        return states, targets, totals

    def merit_contract(self):
        return dict(joint_inner_margin_rad=.02, joint_margin_normalizer_rad=.02,
                    speed_fraction=.9, speed_normalizer_fraction=.1,
                    minimum_root_height_m=.30, height_normalizer_m=.05,
                    maximum_root_tilt_rad=1.1, tilt_normalizer_rad=.1,
                    target_regularization_weight=1e-4,
                    target_regularization_reference="original shifted warm targets",
                    zero_rollout_feedback=self.zero_rollout_feedback,
                    feedback_suppression_scope="private restoration line search only",
                    execution_policy=False, main_tracking_objective_changed=False)


def restore_feasible_seed(tracker, native, data, contract, motion, original_targets, window_start, *,
                          restorer=None, threads=8, retry_zero_feedback=True, save_proposal=None):
    """Return only a certified ordinary final proposal; never execute or change actual state.

    The optional callback archives each proposal and returns a JSON-safe receipt.
    The caller must invoke this only after all standard seed candidates fail.
    """
    if tracker.feasibility is None or tracker.T != 30:
        raise ValueError("restoration requires the native H30 hard-feasibility tracker")
    warm = np.asarray(original_targets).copy()
    actual = np.r_[data.qpos, data.qvel]
    started = time.perf_counter()
    if restorer is None:
        restorer = Native23RestorationTracker(tracker.model, contract, motion, warm, threads=threads)
    else:
        restorer.original_targets = warm.copy()
    result = dict(trigger="all standard seeds infeasible", attempts=[])
    best = None
    for zero_feedback in ((False, True) if retry_zero_feedback else (False,)):
        tick = time.perf_counter()
        restorer.zero_rollout_feedback = zero_feedback
        restorer.window(window_start)
        proposal_states, _, proposal_costs = restorer.rollout(actual, warm)
        attempt = dict(mode="zero_feedback" if zero_feedback else "guided", contract=restorer.merit_contract(),
                       initial_merit=float(proposal_costs[0]) if np.isfinite(proposal_costs[0]) else None)
        if np.isfinite(proposal_costs[0]):
            _, targets, _, merit = ilqr(restorer, actual, warm.copy(), iters=10,
                                       initial_rollout=(proposal_states[:, 0], proposal_costs[0]))
            states, bounded_targets, costs = tracker.rollout(actual, targets)
            attempt.update(final_merit=float(merit), nominal_feasibility=tracker.last_rollout_feasibility)
            if save_proposal is not None:
                attempt.update(save_proposal(attempt["mode"], targets))
            if np.isfinite(costs[0]):
                oracle, _ = inspect_native_segment(native, data, targets, contract, retain_trace=False)
                attempt["independent_actual_state_oracle"] = oracle
                if oracle["feasible"]:
                    best = (float(costs[0]), "restoration", states[:, 0].copy(), bounded_targets[:, 0].copy())
        attempt.update(accepted=best is not None, elapsed_ms=(time.perf_counter() - tick) * 1000)
        result["attempts"].append(attempt)
        if best is not None:
            break
    restorer.zero_rollout_feedback = False
    result.update(accepted=best is not None, elapsed_ms=(time.perf_counter() - started) * 1000)
    return best, result, restorer
