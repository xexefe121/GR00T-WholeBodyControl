"""Offline, exact-simulator one-step target-feasibility experiment.

This is not a hardware controller, real-time implementation, recovery policy,
or a recursive safety guarantee. It adds a predicted-next-state constraint to
the unchanged target/effort/slew intersection and reports failure explicitly.
"""

from __future__ import annotations

import copy
import time

import numpy as np
from scipy.optimize import minimize

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
)
from gear_sonic.utils.g1_true23_sim_acquisition import effort_feasible_target, effort_target_interval


class PredictiveTargetError(ValueError):
    def __init__(self, details):
        super().__init__("no verified next-step-feasible target found by bounded offline optimizer")
        self.details = details


def filter_with_preview(requested, previous, q, dq, kp, kd, effort, *, dt, slew_rate, predict, reserve_rad=1e-5):
    """Minimize target change subject to exact current and predicted constraints.

    ``predict(target)`` must return independent next q/dq without mutating live
    state. Sequential linearization handles coupled dynamics; every accepted
    candidate is independently checked against the nonlinear preview. Solver
    failure proves only that this bounded search failed, not infeasibility.
    """
    started = time.perf_counter()
    if slew_rate is None or not np.isfinite(reserve_rad) or not 0 < reserve_rad <= 0.001:
        raise ValueError("predictive experiment requires finite slew and a positive 0..0.001 rad reserve")
    greedy = effort_feasible_target(requested, previous, q, dq, kp, kd, effort, dt=dt, slew_rate=slew_rate)
    low, high = effort_target_interval(previous, q, dq, kp, kd, effort, dt=dt, slew_rate=slew_rate)
    kp, kd, effort = map(np.asarray, (kp, kd, effort))
    width = 0.95 * 0.25 * effort / kp
    hard_low = np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + 0.05
    hard_high = np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) - 0.05
    delta = slew_rate * dt
    calls = 0

    def preview(target):
        nonlocal calls
        calls += 1
        nq, ndq = predict(np.array(target, copy=True))
        next_low, next_high = effort_target_interval(target, nq, ndq, kp, kd, effort, dt=dt, slew_rate=slew_rate)
        return np.asarray(nq) + kd * np.asarray(ndq) / kp, next_high - next_low

    center, gap = preview(greedy)
    initial_gap = gap.copy()
    candidate = greedy.copy()
    iterations = 0
    optimizer_success = None
    if np.min(gap) < reserve_rad:
        # All 23 targets remain free within current numeric bounds. No legs
        # are held, substituted, or removed from SONIC control.
        for iterations in range(1, 5):
            jac = np.empty((23, 23))
            for i in range(23):
                # Probe only admissible current targets; fixed coordinates
                # need no derivative because their optimization bound is fixed.
                room_plus, room_minus = high[i] - candidate[i], candidate[i] - low[i]
                epsilon = min(2e-5, max(room_plus, room_minus))
                if epsilon <= 1e-12:
                    jac[:, i] = 0.0
                    continue
                signed = epsilon if room_plus >= room_minus else -epsilon
                perturbed = candidate.copy()
                perturbed[i] += signed
                changed_center, _ = preview(perturbed)
                jac[:, i] = (changed_center - center) / signed
            # Future existence: effort interval intersects both joint bounds
            # and next target's slew interval, with the stated tiny reserve.
            matrix = np.vstack((np.eye(23) - jac, jac - np.eye(23), -jac, jac))
            offset = center - jac @ candidate
            bound = (
                np.r_[
                    width + delta + offset,
                    width + delta - offset,
                    offset + width - hard_low,
                    hard_high + width - offset,
                ]
                - reserve_rad
            )
            scale = max(delta, 1e-6)
            constraint = {
                "type": "ineq",
                "fun": lambda x: bound - matrix @ (greedy + x * scale),
                "jac": lambda x: -matrix * scale,
            }
            result = minimize(
                lambda x: 0.5 * float(x @ x),
                (candidate - greedy) / scale,
                jac=lambda x: x,
                method="SLSQP",
                bounds=list(zip((low - greedy) / scale, (high - greedy) / scale)),
                constraints=[constraint],
                options={"maxiter": 50, "ftol": 1e-10, "disp": False},
            )
            optimizer_success = bool(result.success)
            if not np.isfinite(result.x).all():
                break
            candidate = np.clip(greedy + result.x * scale, low, high)
            center, gap = preview(candidate)
            if np.min(gap) >= reserve_rad * 0.99:
                break
    details = {
        "preview_calls": calls,
        "linearization_iterations": iterations,
        "optimizer_reported_success": optimizer_success,
        "greedy_minimum_next_interval_width_rad": float(np.min(initial_gap)),
        "accepted_minimum_next_interval_width_rad": float(np.min(gap)),
        "maximum_target_change_rad": float(np.max(np.abs(candidate - greedy))),
        "elapsed_s": time.perf_counter() - started,
        "next_reserve_rad": reserve_rad,
        "recursive_feasibility_proven": False,
        "hardware_authorized": False,
    }
    if np.min(gap) < reserve_rad * 0.99:
        details["limiting_joints"] = [HARDWARE_23_JOINT_NAMES[i] for i in np.flatnonzero(gap < reserve_rad * 0.99)]
        raise PredictiveTargetError(details)
    return candidate, details


class MujocoTargetPreview:
    """Exact copied-data preview; does not advance or alter the actual simulator."""

    def __init__(self, controller, kp, kd):
        self.controller = controller
        self.kp, self.kd = np.asarray(kp).copy(), np.asarray(kd).copy()
        self._copy_data = getattr(controller.module, "mj_copyData", None)
        self.probe = controller.module.MjData(controller.model) if callable(self._copy_data) else None

    def __call__(self, target):
        controller = self.controller
        if callable(self._copy_data):
            self._copy_data(self.probe, controller.model, controller.data)
        else:
            # MuJoCo 3.2.3 exposes the native data copy through __copy__, not
            # mj_copyData. Preserve its full state, including solver history.
            self.probe = copy.copy(controller.data)
        self.probe.ctrl[:] = self.kp * (target - self.probe.qpos[7:]) - self.kd * self.probe.qvel[6:]
        controller.module.mj_step(controller.model, self.probe)
        return self.probe.qpos[7:].copy(), self.probe.qvel[6:].copy()
