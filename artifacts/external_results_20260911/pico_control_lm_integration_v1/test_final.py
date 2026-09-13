"""Scope and delegation witnesses for the opt-in third restoration attempt."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils import g1_true23_mjbatch_ilqr_core as core, g1_true23_mjbatch_restoration_lm as helper


class ScalarPlanner:
    feasibility = None
    T, nu, nx, nq, nv = 2, 1, 1, 1, 0
    lo, hi = np.array([-2.0]), np.array([2.0])

    def rollout(self, x0, us, gains=None):
        states = np.empty((3, 9, 1))
        controls = np.empty((2, 9, 1))
        states[0] = x0
        for t in range(2):
            controls[t] = us[t]
            if gains is not None:
                xs, k, K = gains
                controls[t] += core.ALPHAS[:, None] * k[t] + (states[t] - xs[t]) @ K[t].T
            controls[t] = np.clip(controls[t], self.lo, self.hi)
            states[t + 1] = states[t] + controls[t]
        costs = ((states - 1) ** 2).sum((0, 2)) + 0.1 * (controls**2).sum((0, 2))
        return states, controls, costs

    def linearize(self, states, targets):
        return np.ones((2, 1, 1)), np.ones((2, 1, 1))

    def expand(self, states, targets):
        return 2 * (states - 1), np.full((3, 1, 1), 2.0), 0.2 * targets, np.full((2, 1, 1), 0.2)


def test_default_solver_bitexact_original():
    old = Path("/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_full_hard_restoration_v1")
    if not old.exists():
        pytest.skip("frozen native323 fixture unavailable")
    spec = importlib.util.spec_from_file_location(
        "old_exact_solver", old / "g1_true23_mjbatch_ilqr_core_snapshot.py"
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    actual = core.ilqr(ScalarPlanner(), np.array([0.0]), np.zeros((2, 1)), iters=5)
    expected = original.ilqr(ScalarPlanner(), np.array([0.0]), np.zeros((2, 1)), iters=5)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_main_hard_solver_and_backward_are_unchanged():
    path = Path(
        "/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_full_hard_restoration_v1/"
        "g1_true23_mjbatch_ilqr_core_snapshot.py"
    )
    if not path.exists():
        pytest.skip("frozen native323 fixture unavailable")

    def functions(p):
        return {n.name: ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n, ast.FunctionDef)}

    before, after = functions(path), functions(Path(core.__file__))
    for name in ("_ilqr_feasible", "backward", "boxqp"):
        assert before[name] == after[name]


def test_evaluator_requires_explicit_existing_restoration(tmp_path):
    from gear_sonic.scripts.evaluate_g1_true23_mjbatch_mpc import run

    output = tmp_path / "must_not_exist"
    args = SimpleNamespace(
        horizon=30,
        commit=5,
        iterations=5,
        threads=1,
        feedback_clip=0.1,
        checkpoint_controls=100,
        restoration=False,
        restoration_control_lm_retry=True,
        output=output,
    )
    with pytest.raises(ValueError, match="requires explicit guided/K0 restoration"):
        run(args)
    assert not output.exists()


def test_override_cannot_affect_main_hard_solver():
    planner = ScalarPlanner()
    planner.feasibility = object()
    with pytest.raises(ValueError, match="cannot alter the hard main solver"):
        core.ilqr(planner, np.array([0.0]), np.zeros((2, 1)), backward_override=lambda *x: None)


def test_control_LM_uses_control_identity_and_unregularized_cross():
    # One-step scalar LQR: A=2,B=3,V=4,R=.5,mu=1. Exact Qreg=37.5;
    # Qux=24 (state-space damping would incorrectly use30 for this variant).
    d = (
        np.array([[[2.0]]]),
        np.array([[[3.0]]]),
        np.zeros((2, 1)),
        np.array([[[0.0]], [[4.0]]]),
        np.zeros((1, 1)),
        np.array([[[0.5]]]),
        np.full((1, 1), -10.0),
        np.full((1, 1), 10.0),
        1.0,
    )
    k, K = helper.control_lm_backward(*d)
    np.testing.assert_allclose(k, 0, atol=0, rtol=0)
    np.testing.assert_allclose(K, -24 / 37.5, atol=1e-14, rtol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_prior_success_returns_exact_tuple_without_third(monkeypatch, enabled):
    best = (1.0, "restoration", object(), object())
    result = {"accepted": True, "attempts": [], "elapsed_ms": 0.0}
    returned_restorer = object()
    monkeypatch.setattr(helper, "original_restore", lambda *a, **k: (best, result, returned_restorer))
    monkeypatch.setattr(helper, "ilqr", lambda *a, **k: pytest.fail("third solve called"))
    out = helper.restore_feasible_seed(
        None, None, None, None, None, np.zeros((30, 23)), 3810, retry_control_lm=enabled
    )
    assert out[0] is best and out[1] is result and out[2] is returned_restorer


@pytest.mark.parametrize("guided", [None, np.full((30, 23), np.nan)])
def test_unavailable_or_nonfinite_guided_skips_third(monkeypatch, guided):
    def fake(*args, **kwargs):
        if guided is not None:
            kwargs["save_proposal"]("guided", guided)
        return None, {"accepted": False, "attempts": [], "elapsed_ms": 0.0}, object()

    monkeypatch.setattr(helper, "original_restore", fake)
    monkeypatch.setattr(helper, "ilqr", lambda *a, **k: pytest.fail("invalid guided seed used"))
    best, result, _ = helper.restore_feasible_seed(
        None, None, None, None, None, np.zeros((30, 23)), 3810, retry_control_lm=True
    )
    assert best is None and "skipped_reason" in result["attempts"][-1]


def test_third_initial_iterate_is_guided_but_anchor_is_original(monkeypatch):
    warm, guided, solution = np.zeros((30, 23)), np.ones((30, 23)), np.full((30, 23), 2.0)
    restorer = SimpleNamespace(original_targets=None, zero_rollout_feedback=False, window=lambda start: None)
    restorer.rollout = lambda x, u: (np.zeros((31, 9, 59)), np.repeat(u[:, None], 9, axis=1), np.ones(9))

    def old(*args, **kwargs):
        kwargs["save_proposal"]("guided", guided)
        kwargs["save_proposal"]("zero_feedback", np.full((30, 23), 3.0))
        return (
            None,
            {"accepted": False, "attempts": [{"mode": "guided"}, {"mode": "zero_feedback"}], "elapsed_ms": 0.0},
            restorer,
        )

    def solve(p, x, u, **kwargs):
        np.testing.assert_array_equal(u, guided)
        np.testing.assert_array_equal(p.original_targets, warm)
        assert p.zero_rollout_feedback and kwargs["backward_override"] is not None
        return None, solution, None, 0.1

    checked_states = np.full((31, 9, 59), 4.0)
    tracker = SimpleNamespace(
        rollout=lambda x, u: (checked_states, np.repeat(u[:, None], 9, axis=1), np.full(9, 8.0)),
        last_rollout_feasibility={"feasible": [True] * 9},
    )
    monkeypatch.setattr(helper, "original_restore", old)
    monkeypatch.setattr(helper, "ilqr", solve)
    monkeypatch.setattr(helper, "inspect_native_segment", lambda *a, **k: ({"feasible": True}, None))
    data = SimpleNamespace(qpos=np.zeros(30), qvel=np.zeros(29))
    best, result, _ = helper.restore_feasible_seed(
        tracker, None, data, {}, None, warm, 3810, retry_control_lm=True
    )
    assert result["accepted"] and best[0] == 8.0
    np.testing.assert_array_equal(best[2], checked_states[:, 0])
    np.testing.assert_array_equal(best[3], solution)
    np.testing.assert_array_equal(warm, 0.0)
    assert not restorer.zero_rollout_feedback
