"""Experimental seed-only physical-margin merit; never an execution policy."""

import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker


class Native23RestorationTracker(Native23Tracker):
    """Search finite, possibly infeasible counterfactuals; certify separately."""

    def __init__(self, servo, contract, motion, original_targets, *, threads=8):
        targets = np.asarray(original_targets, float)
        if targets.shape != (30, 23) or not np.isfinite(targets).all():
            raise ValueError("restoration requires the declared finite H30 target seed")
        super().__init__(servo, contract, motion, horizon=30, threads=threads,
                         fd_epsilon=1e-6, hard_feasibility=False)
        np.testing.assert_array_equal(targets, np.clip(targets, self.lo, self.hi))
        self.original_targets = targets.copy()
        self.control_weight = 1e-4

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

    def rollout(self, *args, **kwargs):
        states, targets, totals = super().rollout(*args, **kwargs)
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
                    execution_policy=False, main_tracking_objective_changed=False)
