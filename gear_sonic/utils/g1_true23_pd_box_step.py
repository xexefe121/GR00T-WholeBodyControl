"""Bound-constrained local PD trajectory step; actual rollouts remain the referee."""

import numpy as np

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_pd_trajectory_optimizer import CONTROL_WEIGHT, SLEW_WEIGHT


def solve_box_quadratic(hessian, gradient, lower, upper):
    """Feasible primal active-set solve for a small strictly convex box QP.

    Unlike clipping an unconstrained answer, every free variable is re-solved
    when another variable hits a bound. The returned free set defines the
    local feedback derivative. This is an optimization step, not a physical
    feasibility certificate; independent nonlinear rollouts still decide.
    """
    h, g, lo, hi = map(lambda value: np.asarray(value, dtype=np.float64), (hessian, gradient, lower, upper))
    n = len(g)
    if h.shape != (n, n) or lo.shape != (n,) or hi.shape != (n,):
        raise ValueError("box quadratic shape mismatch")
    if not np.isfinite(np.r_[h.ravel(), g, lo, hi]).all() or np.any(lo >= hi):
        raise ValueError("box quadratic requires finite strict bounds")
    if not np.allclose(h, h.T, atol=1e-10, rtol=1e-12):
        raise ValueError("box quadratic Hessian must be symmetric")
    np.linalg.cholesky(h)
    scale = max(1.0, float(np.max(np.abs(h))), float(np.max(np.abs(g))))
    h, g = h / scale, g / scale
    x = np.clip(-np.linalg.solve(h, g), lo, hi)
    active = np.where(x <= lo, -1, np.where(x >= hi, 1, 0))
    for _ in range(20 * n + 1):
        derivative = h @ x + g
        free = active == 0
        # Ill-conditioned contact Hessians can amplify a roundoff-sized
        # residual into a Newton correction above any fixed step tolerance.
        # Test stationarity at matrix-product precision, not correction size.
        roundoff = 64 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(h) @ np.abs(x) + np.abs(g))))
        direction = np.zeros(n)
        if np.any(free):
            direction[free] = -np.linalg.solve(h[np.ix_(free, free)], derivative[free])
        if np.max(np.abs(derivative[free]), initial=0.0) <= roundoff:
            violation = np.where(active == -1, -derivative, np.where(active == 1, derivative, -np.inf))
            release = int(np.argmax(violation))
            if violation[release] <= roundoff:
                projected = np.where((active == -1) & (derivative >= 0), 0, derivative)
                projected = np.where((active == 1) & (derivative <= 0), 0, projected)
                if np.max(np.abs(projected)) > 1e-8:
                    raise RuntimeError("box quadratic failed independent projected stationarity check")
                return x, free
            active[release] = 0
            continue
        ratios = np.full(n, np.inf)
        negative, positive = direction < 0, direction > 0
        ratios[negative] = (lo[negative] - x[negative]) / direction[negative]
        ratios[positive] = (hi[positive] - x[positive]) / direction[positive]
        blocker = int(np.argmin(ratios))
        step = min(1.0, max(0.0, float(ratios[blocker])))
        x = np.clip(x + step * direction, lo, hi)
        if step < 1.0:
            active[blocker] = -1 if direction[blocker] < 0 else 1
    raise RuntimeError("box quadratic active set did not converge")


def box_backward_pass(plant, local, controls, seed_controls, regularization):
    count = len(controls)
    increments, feedback = np.zeros((count, 23)), np.zeros((count, 23, 81))
    vg, vh = local["terminal_g"].copy(), local["terminal_h"].copy()
    identity = np.eye(23)
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
        k, free = solve_box_quadratic(h, qu, plant.lower - controls[i], plant.upper - controls[i])
        gain = np.zeros((23, 81))
        if np.any(free):
            gain[free] = -np.linalg.solve(h[np.ix_(free, free)], qux[free])
        increments[i], feedback[i] = k, gain
        vg = qx + gain.T @ qu + qux.T @ k + gain.T @ quu @ k
        vh = qxx + gain.T @ quu @ gain + gain.T @ qux + qux.T @ gain
        vh = 0.5 * (vh + vh.T)
        if not np.isfinite(vg).all() or not np.isfinite(vh).all():
            raise RuntimeError("box trajectory quadratic approximation diverged")
    return increments, feedback
