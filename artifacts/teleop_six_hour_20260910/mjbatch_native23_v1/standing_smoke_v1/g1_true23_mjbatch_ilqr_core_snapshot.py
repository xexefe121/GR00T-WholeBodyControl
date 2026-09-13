# SPDX-License-Identifier: Apache-2.0
# Solver math adapted from Kevin Zakka's mjbatch/examples/g1_flip.py,
# commit 77966f85bcd8f7ef4351cb4a1a6f42e133d19725. No example robot or reference data.
"""Finite-difference iLQR math for the declared offline native23 experiment."""

import mujoco
import numpy as np
from mjbatch import Batch

EPS = 1e-6
ALPHAS = 0.5 ** np.arange(9)
TOL = 1e-4
ITERS = 5


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

    def __init__(self, model, T, sub, num_threads=4):
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
        self.line = Batch(model, len(ALPHAS), num_threads=num_threads, forward=True)
        self.line_fields = [self.line.bind(f) for f in ("qpos", "qvel", "ctrl", "qacc_warmstart")]
        self.warning = self.line.bind("warning")

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
        self.line.step(nstep=self.sub)
        return np.concatenate([qpos, qvel], axis=1)

    def linearize(self, xs, us):
        """Compute the Jacobians A_t = df/dx and B_t = df/du at every knot by forward differences."""
        (T, nx, nu, nv, cols) = (self.T, self.nx, self.nu, self.nv, 1 + self.nx + self.nu)
        (x, u) = (np.repeat(xs[:-1], cols, axis=0), np.repeat(us, cols, axis=0))
        delta = EPS * np.tile(np.eye(cols)[:, 1:], (T, 1))
        delta[:, nx:] *= np.where(u + EPS > self.hi, -1.0, 1.0)
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
        new_xs[0] = x
        for t in range(T):
            u = np.tile(us[t], (n, 1))
            if gains is not None:
                (xs, k, K) = gains
                u += ALPHAS[:, None] * k[t] + self.difference(np.tile(xs[t], (n, 1)), x) @ K[t].T
            u = np.clip(u, self.lo, self.hi)
            total += self.cost(t, x, u)
            x = self.advance(x, u)
            (new_xs[t + 1], new_us[t]) = (x, u)
        total += self.cost(T, x, np.zeros((n, self.nu)))
        total[(self.warning != warned).any((1, 2))] = np.inf
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


def ilqr(planner, x0, us, xs=None, watch=lambda xs: None, iters=ITERS):
    (T, nu, nx) = (planner.T, planner.nu, planner.nx)
    if xs is None:
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
        if (sweep := backward(*d, mu)) is not None:
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
        if accepted:
            d = derivatives()
    return (xs, us, K, total)
