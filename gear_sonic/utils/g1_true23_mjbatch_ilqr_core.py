# SPDX-License-Identifier: Apache-2.0
# Solver math adapted from Kevin Zakka's mjbatch/examples/g1_flip.py,
# commit 77966f85bcd8f7ef4351cb4a1a6f42e133d19725. No example robot or reference data.
"""Finite-difference iLQR math for the declared offline native23 experiment."""

from mjbatch import Batch
import mujoco
import numpy as np

EPS = 1e-6
ALPHAS = 0.5 ** np.arange(9)
TOL = 1e-4
ITERS = 5


class NoFeasiblePlan(RuntimeError):
    """No physically feasible incumbent exists; this is not a recovery policy."""

    def __init__(self, diagnostics):
        super().__init__("no feasible initial rollout")
        self.diagnostics = diagnostics


def quat_mul(a, b):
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdims=True)
    v = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate([w, v], axis=-1)


def quat_exp(v):
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = np.divide(v, angle, out=np.zeros_like(v), where=angle > 0)
    return np.concatenate([np.cos(angle / 2), np.sin(angle / 2) * axis], axis=-1)


def quat_log(q):
    norm = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    axis = np.divide(q[..., 1:], norm, out=np.zeros_like(q[..., 1:]), where=norm > 0)
    angle = 2 * np.arctan2(norm, q[..., :1])
    return np.where(angle > np.pi, angle - 2 * np.pi, angle) * axis


class Planner:
    """iLQR over a MuJoCo model. Subclasses supply cost(t, x, u) and expand(xs, us)."""

    def __init__(self, model, T, sub, num_threads=4, fd_epsilon=EPS, feasibility=None):
        if not np.isfinite(fd_epsilon) or fd_epsilon <= 0:
            raise ValueError("finite-difference epsilon must be finite and positive")
        self.fd_epsilon = float(fd_epsilon)
        (self.model, self.T, self.sub) = (model, T, sub)
        (self.nq, self.nv, self.nu) = (model.nq, model.nv, model.nu)
        self.nx = 2 * self.nv
        (self.lo, self.hi) = model.actuator_ctrlrange.T
        jt = model.jnt_type
        assert not (jt == 1).any(), "ball joints are not supported"
        self.free = list(zip(model.jnt_qposadr[jt == 0], model.jnt_dofadr[jt == 0], strict=True))
        quat = [a + i for (a, _) in self.free for i in range(3, 7)]
        spin = [d + i for (_, d) in self.free for i in range(3, 6)]
        (self.lin_q, self.lin_v) = (np.setdiff1d(range(self.nq), quat), np.setdiff1d(range(self.nv), spin))
        self.batch = Batch(model, T * (1 + self.nx + self.nu), num_threads=num_threads)
        (self.qpos, self.qvel) = (self.batch.bind("qpos"), self.batch.bind("qvel"))
        (self.ctrl, self.warm) = (self.batch.bind("ctrl"), self.batch.bind("qacc_warmstart"))
        self.line = Batch(model, len(ALPHAS), num_threads=num_threads, forward=feasibility is None)
        self.line_fields = [self.line.bind(f) for f in ("qpos", "qvel", "ctrl", "qacc_warmstart")]
        self.warning = self.line.bind("warning")
        self.feasibility = feasibility
        self.last_rollout_feasibility = None
        self.last_solve_feasibility = None
        if feasibility is not None:
            self.line_time = self.line.bind("time")
            self.line_force = self.line.bind("qfrc_actuator")

    def integrate(self, qpos, dq):
        out = qpos.copy()
        out[:, self.lin_q] += dq[:, self.lin_v]
        for a, d in self.free:
            quat = quat_mul(qpos[:, a + 3 : a + 7], quat_exp(dq[:, d + 3 : d + 6]))
            out[:, a + 3 : a + 7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
        return out

    def difference(self, x, y):
        dq = np.empty((len(x), self.nv))
        dq[:, self.lin_v] = y[:, self.lin_q] - x[:, self.lin_q]
        for a, d in self.free:
            conj = x[:, a + 3 : a + 7] * [1, -1, -1, -1]
            dq[:, d + 3 : d + 6] = quat_log(quat_mul(conj, y[:, a + 3 : a + 7]))
        return np.concatenate([dq, y[:, self.nq :] - x[:, self.nq :]], axis=1)

    def step(self, x, u):
        n = len(x)
        (self.qpos[:n], self.qvel[:n]) = (x[:, : self.nq], x[:, self.nq :])
        (self.ctrl[:n], self.warm[:n]) = (u, 0.0)
        ids = np.arange(n) if n < self.batch.num_sims else None
        self.batch.step(ids, nstep=1)
        self.probe(n)
        if self.sub > 1:
            self.batch.step(ids, nstep=self.sub - 1)
        return np.concatenate([self.qpos[:n], self.qvel[:n]], axis=1)

    def probe(self, n):
        pass

    def advance(self, x, u):
        (qpos, qvel, ctrl, warm) = self.line_fields
        (qpos[:], qvel[:], ctrl[:], warm[:]) = (x[:, : self.nq], x[:, self.nq :], u, 0.0)
        if self.feasibility is not None:
            # Keep the original 20ms dynamics contract, but inspect every actual
            # 2ms integration result. A rejected lane never becomes valid again.
            for substep in range(1, self.sub + 1):
                ids = np.flatnonzero(self._rollout_valid)
                if not len(ids):
                    break
                self.line.step(ids, nstep=1)
                self._record_feasibility(
                    qpos, qvel, force=self.line_force[:, 6:], warning=self.warning[:, :, 1],
                    time=self.line_time, expected_time=(self._rollout_knot * self.sub + substep)
                    * self.model.opt.timestep, substep=substep,
                )
            # Legacy advance forwards derived fields once per control, not once
            # per physics substep. Preserve that exact numerical contract.
            ids = np.flatnonzero(self._rollout_valid)
            if len(ids):
                self.line.forward(ids)
                self._record_feasibility(
                    qpos, qvel, force=self.line_force[:, 6:], warning=self.warning[:, :, 1],
                    time=self.line_time, expected_time=(self._rollout_knot + 1) * self.sub
                    * self.model.opt.timestep, substep=self.sub,
                )
            return np.concatenate([qpos, qvel], axis=1)
        self.line.step(nstep=self.sub)
        return np.concatenate([qpos, qvel], axis=1)

    def _record_feasibility(self, qpos, qvel, *, substep, **kwargs):
        invalid, reasons, metrics = self.feasibility.assess(qpos, qvel, **kwargs)
        active = self._rollout_valid.copy()
        for name in self._rollout_peaks:
            values = metrics[name]
            self._rollout_peaks[name][active] = np.maximum(
                self._rollout_peaks[name][active], np.nan_to_num(values[active], nan=np.inf)
            )
        for index in np.flatnonzero(active & invalid):
            self._rollout_first[index] = self.feasibility.witness(
                index, reasons, metrics, control=self._rollout_knot, substep=substep
            )
        self._rollout_valid &= ~invalid

    def linearize(self, xs, us):
        """Compute the Jacobians A_t = df/dx and B_t = df/du at every knot by forward differences."""
        (T, nx, nu, nv, cols) = (self.T, self.nx, self.nu, self.nv, 1 + self.nx + self.nu)
        (x, u) = (np.repeat(xs[:-1], cols, axis=0), np.repeat(us, cols, axis=0))
        delta = self.fd_epsilon * np.tile(np.eye(cols)[:, 1:], (T, 1))
        delta[:, nx:] *= np.where(u + self.fd_epsilon > self.hi, -1.0, 1.0)
        x[:, : self.nq] = self.integrate(x[:, : self.nq], delta[:, :nv])
        x[:, self.nq :] += delta[:, nv:nx]
        out = self.step(x, u + delta[:, nx:]).reshape(T, cols, -1)
        base = np.repeat(out[:, 0], cols - 1, axis=0)
        h = delta.reshape(T, cols, -1)[:, 1:].sum(axis=2).ravel()
        jac = self.difference(base, out[:, 1:].reshape(-1, out.shape[2])) / h[:, None]
        jac = np.swapaxes(jac.reshape(T, nx + nu, nx), 1, 2)
        return (jac[:, :, :nx], jac[:, :, nx:])

    def rollout(self, x0, us, gains=None):
        """Roll out us on every line search sim; with gains (xs, k, K), sim a plays u + a k + K dx."""
        (n, T) = (len(ALPHAS), self.T)
        (x, warned) = (np.tile(x0, (n, 1)), self.warning.copy())
        new_xs = np.empty((T + 1, n, self.nq + self.nv))
        (new_us, total) = (np.empty((T, n, self.nu)), np.zeros(n))
        if self.feasibility is not None:
            if np.asarray(us).shape != (T, self.nu):
                raise ValueError("invalid hard-feasibility control shape")
            self.warning[:] = 0
            self.line_time[:] = 0
            self._rollout_valid = np.ones(n, dtype=bool)
            self._rollout_first = [None] * n
            self._rollout_peaks = {name: np.zeros(n) for name in
                                   ("range_excess_rad", "speed_ratio", "effort_ratio")}
            self._rollout_knot = 0
            self._record_feasibility(x[:, :self.nq], x[:, self.nq:], substep=0)
            warned = self.warning.copy()
        new_xs[0] = x
        for t in range(T):
            if self.feasibility is not None and not self._rollout_valid.any():
                new_xs[t + 1:] = x
                new_us[t:] = np.clip(us[t:, None, :], self.lo, self.hi)
                total[:] = np.inf
                break
            u = np.tile(us[t], (n, 1))
            if gains is not None:
                (xs, k, K) = gains
                u += ALPHAS[:, None] * k[t] + self.difference(np.tile(xs[t], (n, 1)), x) @ K[t].T
            if self.feasibility is not None:
                self._rollout_knot = t
                for index in np.flatnonzero(self._rollout_valid & ~np.isfinite(u).all(axis=1)):
                    self._rollout_first[index] = dict(control=t, substep=0, reasons=["nonfinite_target"])
                    self._rollout_valid[index] = False
            u = np.clip(u, self.lo, self.hi)
            total += self.cost(t, x, u)
            x = self.advance(x, u)
            (new_xs[t + 1], new_us[t]) = (x, u)
        if self.feasibility is None or self._rollout_valid.any():
            total += self.cost(T, x, np.zeros((n, self.nu)))
        total[(self.warning != warned).any((1, 2))] = np.inf
        if self.feasibility is not None:
            for index in np.flatnonzero(self._rollout_valid & ~np.isfinite(total)):
                self._rollout_first[index] = dict(control=T, substep=0, reasons=["nonfinite_cost"])
                self._rollout_valid[index] = False
            total[~self._rollout_valid] = np.inf
            self.last_rollout_feasibility = dict(
                feasible=self._rollout_valid.tolist(), first_violation=self._rollout_first,
                peaks={name: [float(x) if np.isfinite(x) else None for x in value]
                       for name, value in self._rollout_peaks.items()},
                inspection="every2ms including the initial state; invalid lanes latched",
            )
        return (new_xs, new_us, total)


def boxqp(Q, q, lo, hi, k):
    """Minimize 1/2 k'Qk + q'k subject to lo <= k <= hi. Return k and the free set."""
    (k, index, n) = (np.clip(k, lo, hi), np.empty(len(q), np.int32), len(q))
    free = np.zeros(n, bool)
    free[index[: mujoco.mju_boxQP(k, np.empty((n, n + 7)), index, Q, q, lo, hi)]] = True
    return (k, free)


def backward(A, B, lx, lxx, lu, luu, lo, hi, mu):
    """Run the backward pass with box-constrained controls (Tassa 2012, 2014)."""
    (T, nu, nx) = (len(lu), lu.shape[1], lx.shape[1])
    (vx, vxx) = (lx[-1], lxx[-1])
    (k, K) = (np.zeros((T + 1, nu)), np.empty((T, nu, nx)))
    for t in reversed(range(T)):
        (qx, qu) = (lx[t] + A[t].T @ vx, lu[t] + B[t].T @ vx)
        qxx = lxx[t] + A[t].T @ vxx @ A[t]
        (quu, qux) = (luu[t] + B[t].T @ vxx @ B[t], B[t].T @ vxx @ A[t])
        reg = vxx + mu * np.eye(nx)
        (quu_reg, qux_reg) = (luu[t] + B[t].T @ reg @ B[t], B[t].T @ reg @ A[t])
        if not (np.isfinite(quu_reg).all() and np.linalg.eigvalsh(quu_reg).min() > 0.0):
            return None
        (k[t], free) = boxqp(quu_reg, qu, lo[t], hi[t], k[t + 1])
        K[t] = 0.0
        K[t, free] = -np.linalg.solve(quu_reg[np.ix_(free, free)], qux_reg[free])
        vx = qx + K[t].T @ quu @ k[t] + K[t].T @ qu + qux.T @ k[t]
        vxx = qxx + K[t].T @ quu @ K[t] + K[t].T @ qux + qux.T @ K[t]
        vxx = 0.5 * (vxx + vxx.T)
    return (k[:-1], K)


def ilqr(planner, x0, us, xs=None, watch=lambda xs: None, iters=ITERS, initial_rollout=None,
         backward_override=None):
    if planner.feasibility is not None:
        if backward_override is not None:
            raise ValueError("private restoration backward override cannot alter the hard main solver")
        return _ilqr_feasible(planner, x0, us, xs=xs, watch=watch, iters=iters, initial_rollout=initial_rollout)
    backward_function = backward if backward_override is None else backward_override
    (T, nu, nx) = (planner.T, planner.nu, planner.nx)
    if initial_rollout is not None:
        if xs is not None:
            raise ValueError("cached rollout and independent initial trajectory are mutually exclusive")
        xs, total = initial_rollout
        xs = xs.copy()
        if xs.shape != (T + 1, planner.nq + planner.nv) or not np.array_equal(xs[0], x0):
            raise ValueError("cached rollout does not begin at the actual planning state")
    elif xs is None:
        (xs, _, total) = planner.rollout(x0, us)
        (xs, total) = (xs[:, 0], total[0])
    else:
        total = np.inf
    watch(xs)

    def derivatives():
        return (*planner.linearize(xs, us), *planner.expand(xs, us), planner.lo - us, planner.hi - us)

    (mu, K, d) = (1.0, np.zeros((T, nu, nx)), derivatives())
    stop = False
    for _ in range(iters):
        accepted = False
        if (sweep := backward_function(*d, mu)) is not None:
            (k, K) = sweep
            (new_xs, new_us, totals) = planner.rollout(x0, us, (xs, k, K))
            best = int(np.argmin(totals))
            if accepted := bool(totals[best] < total):
                (xs, us, drop) = (new_xs[:, best], new_us[:, best], total - totals[best])
                (total, mu) = (totals[best], max(mu / 10.0, 1e-06))
                stop = drop < TOL * total
        if not accepted:
            mu *= 10.0
            stop = mu > 1000000.0
        watch(xs)
        if stop:
            break
        # The final derivative refresh was unused by the returned controller.
        if accepted and _ + 1 < iters:
            d = derivatives()
    return (xs, us, K, total)


def _ilqr_feasible(planner, x0, us, xs=None, watch=lambda xs: None, iters=ITERS, initial_rollout=None):
    """Feasible-incumbent iLQR: unsafe updates cannot replace safe commands.

    No restoration is implied. Even a supplied cached trajectory is independently
    rerolled because endpoint samples alone cannot certify substep feasibility.
    """
    if xs is not None and initial_rollout is not None:
        raise ValueError("cached rollout and independent initial trajectory are mutually exclusive")
    if initial_rollout is not None:
        cached = initial_rollout[0]
        if cached.shape != (planner.T + 1, planner.nq + planner.nv) or not np.array_equal(cached[0], x0):
            raise ValueError("cached rollout does not begin at the actual planning state")
    candidates, controls, totals = planner.rollout(x0, us)
    diagnostics = dict(initial_rollout=planner.last_rollout_feasibility, iterations=[],
                       cached_seed_revalidated=initial_rollout is not None,
                       incumbent_gain_semantics="last accepted rollout's generating backward sweep")
    planner.last_solve_feasibility = diagnostics
    if not np.isfinite(totals[0]):
        diagnostics["status"] = "no_feasible_seed"
        raise NoFeasiblePlan(diagnostics)
    xs, us, total = candidates[:, 0].copy(), controls[:, 0].copy(), float(totals[0])
    K = np.zeros((planner.T, planner.nu, planner.nx))
    watch(xs)
    mu, d = 1.0, None
    for iteration in range(iters):
        if d is None:
            d = (*planner.linearize(xs, us), *planner.expand(xs, us), planner.lo - us, planner.hi - us)
        if not all(np.isfinite(value).all() for value in d):
            diagnostics["iterations"].append(
                dict(iteration=iteration, accepted=False, reason="nonfinite_derivatives")
            )
            break
        sweep = backward(*d, mu)
        accepted = False
        entry = dict(iteration=iteration, accepted=False, regularization=mu)
        if sweep is not None and all(np.isfinite(value).all() for value in sweep):
            k, proposed_K = sweep
            new_xs, new_us, totals = planner.rollout(x0, us, (xs, k, proposed_K))
            entry["rollout_feasibility"] = planner.last_rollout_feasibility
            finite = np.flatnonzero(np.isfinite(totals))
            best = int(finite[np.argmin(totals[finite])]) if len(finite) else None
            if best is not None and totals[best] < total:
                drop = total - float(totals[best])
                xs, us = new_xs[:, best].copy(), new_us[:, best].copy()
                K, total = proposed_K.copy(), float(totals[best])
                mu, d = max(mu / 10, 1e-6), None
                entry.update(accepted=True, selected_lane=best, cost=total)
                accepted = True
                stop = drop < TOL * total
        if not accepted:
            mu *= 10
            stop = mu > 1e6
        diagnostics["iterations"].append(entry)
        watch(xs)
        if stop:
            break
    diagnostics.update(status="feasible_incumbent", accepted_cost=total)
    return xs, us, K, total
