from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils import g1_true23_generalist_feasibility_restore as restore


def test_intermediate_envelopes_do_not_mutate_original_task_budgets(monkeypatch):
    original = [{"name": "foot", "frame": 0, "indices": np.array([0]), "scales": np.ones(1), "radius": 1.0}]
    observed = []

    def step(problem, current, residual, jacobian, groups):
        radius = groups[0]["radius"]
        observed.append(radius)
        return (np.asarray([[radius]]) if radius >= 1.5 else None), {"accepted": radius >= 1.5}

    monkeypatch.setattr(restore, "protected_step", step)
    problem = SimpleNamespace(config=SimpleNamespace(serialization_margin_fraction=0.995))
    trial, report = restore.intermediate_restoration_step(
        problem, np.array([[2.0]]), np.array([2.0]), None, original
    )
    assert 1.5 <= trial[0, 0] < 1.53
    assert original[0]["radius"] == 1
    assert len(observed) == 7
    assert not report["original_task_gate_qualified"]
    assert not report["joint_root_and_temporal_bounds_relaxed"]


def test_impossible_temporal_or_root_bounds_return_no_intermediate_pose(monkeypatch):
    monkeypatch.setattr(restore, "protected_step", lambda *args: (None, {"accepted": False}))
    problem = SimpleNamespace(config=SimpleNamespace(serialization_margin_fraction=0.995))
    groups = [{"name": "foot", "frame": 0, "indices": np.array([0]), "scales": np.ones(1), "radius": 1.0}]
    candidate, report = restore.intermediate_restoration_step(
        problem, np.array([[2.0]]), np.array([2.0]), None, groups
    )
    assert candidate is None
    assert len(report["attempts"]) == 1
    assert not report["intermediate_iterate_is_reference"]


def test_feasible_original_cannot_use_looser_intermediate_envelope():
    problem = SimpleNamespace(config=SimpleNamespace(serialization_margin_fraction=0.995))
    groups = [{"name": "foot", "frame": 0, "indices": np.array([0]), "scales": np.ones(1), "radius": 1.0}]
    with pytest.raises(ValueError, match="infeasible original"):
        restore.intermediate_restoration_step(problem, np.array([[0.5]]), np.array([0.5]), None, groups)


def test_restoration_requires_strict_original_nonlinear_improvement(monkeypatch):
    problem = SimpleNamespace(
        evaluate=lambda state, **kwargs: (state.ravel(), None, None),
        audit=lambda state: {"passed": True},
    )
    groups = [{"name": "foot", "frame": 0, "indices": np.array([0]), "scales": np.ones(1), "radius": 1.0}]

    def strict(problem, state, baseline):
        audit = restore.audit_norms(state.ravel(), groups)
        return state.copy(), {"before_protected_audit": audit, "after_protected_audit": audit, "iterations": []}

    monkeypatch.setattr(restore, "fit_protected_task_path", strict)
    monkeypatch.setattr(restore, "residual_groups", lambda *args: (groups, 1))
    monkeypatch.setattr(restore, "intermediate_restoration_step", lambda *args: (np.array([[3.0]]), {}))
    output, report = restore.fit_with_feasibility_restoration(problem, np.array([[2.0]]), None)
    np.testing.assert_array_equal(output, [[2.0]])
    assert not report["after_protected_audit"]["passed"]
    assert not report["teacher_accepted"]
    assert not report["original_acceptance_budgets_changed"]
    assert not report["intermediate_restoration"]["iterations"][0]["accepted"]
