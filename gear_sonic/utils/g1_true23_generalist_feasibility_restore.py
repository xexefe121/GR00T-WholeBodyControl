"""Bounded intermediate feasibility recovery for offline nonlinear fitting.

Only the intermediate linearized task envelopes may expand. Every joint/root
position, velocity and acceleration bound stays hard. Exact original nonlinear
task budgets must improve, and the existing final FK/serialization/expert gates
remain authoritative. Intermediate iterates are never accepted references.
"""

from __future__ import annotations

import numpy as np

from gear_sonic.utils.g1_true23_generalist_protected_root import (
    audit_norms,
    fit_protected_task_path,
    protected_step,
    residual_groups,
)


def intermediate_restoration_step(problem, current, residual, jacobian, groups, *, bisections=6):
    """Find a tight feasible intermediate envelope; return no acceptance claim."""
    if type(bisections) is not int or not 1 <= bisections <= 8:
        raise ValueError("restoration bisections must be within 1..8")
    original = audit_norms(residual, groups)
    if original["passed"]:
        raise ValueError("feasibility restoration requires an infeasible original task path")
    # Compensate for the solver's unchanged interior serialization margin.
    upper = (1 + original["maximum_normalized_excess"]) / problem.config.serialization_margin_fraction - 1 + 0.01
    if not np.isfinite(upper) or not 0 < upper <= 1000:
        raise ValueError("restoration envelope exceeds bounded normalized range")
    lower, candidate, attempts = 0.0, None, []
    for index in range(bisections + 1):
        envelope = upper if index == 0 else (lower + upper) / 2
        intermediate = [{**group, "radius": group["radius"] * (1 + envelope)} for group in groups]
        trial, solver = protected_step(problem, current, residual, jacobian, intermediate)
        attempts.append({"envelope_normalized_excess": envelope, "solver": solver})
        if trial is None:
            if index == 0:
                break
            lower = envelope
        else:
            candidate, upper = trial, envelope
    return candidate, {
        "kind": "g1_true23_intermediate_feasibility_envelope_v1",
        "attempts": attempts,
        "selected_envelope_normalized_excess": upper if candidate is not None else None,
        "original_task_gate_qualified": False,
        "intermediate_iterate_is_reference": False,
        "joint_root_and_temporal_bounds_relaxed": False,
    }


def fit_with_feasibility_restoration(problem, initial, baseline, *, maximum_restoration_iterations=8):
    if type(maximum_restoration_iterations) is not int or not 1 <= maximum_restoration_iterations <= 12:
        raise ValueError("restoration iteration budget must be within 1..12")
    current, strict = fit_protected_task_path(problem, initial, baseline)
    original_strict = strict
    if strict["after_protected_audit"]["passed"]:
        strict["intermediate_feasibility_restoration_used"] = False
        return current, strict
    groups, _ = residual_groups(problem, baseline)
    history, stop_reason = [], None
    for iteration in range(maximum_restoration_iterations):
        residual, jacobian, _ = problem.evaluate(current)
        before = audit_norms(residual, groups)
        if before["passed"]:
            break
        candidate, step_report = intermediate_restoration_step(problem, current, residual, jacobian, groups)
        row = {"iteration": iteration + 1, "accepted": False, "intermediate_restoration": step_report}
        if candidate is not None:
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = current + fraction * (candidate - current)
                if not problem.audit(trial)["passed"]:
                    continue
                trial_residual, _, _ = problem.evaluate(trial, derivatives=False)
                exact = audit_norms(trial_residual, groups)
                if exact["passed"] or exact["maximum_normalized_excess"] < before["maximum_normalized_excess"] * (
                    1 - 1e-6
                ):
                    current = trial
                    row.update(accepted=True, fraction=fraction, original_protected_audit=exact)
                    break
        history.append(row)
        if not row["accepted"]:
            stop_reason = "intermediate restoration did not improve original nonlinear task budgets"
            break
    # This last solve and all later serialization checks use original budgets.
    current, final = fit_protected_task_path(problem, current, baseline)
    final.update(
        kind="g1_true23_hard_protected_root_with_intermediate_restoration_v1",
        intermediate_feasibility_restoration_used=True,
        original_strict_attempt=original_strict,
        intermediate_restoration={
            "maximum_iterations": maximum_restoration_iterations,
            "iterations": history,
            "stop_reason": stop_reason,
        },
        before_protected_audit=original_strict["before_protected_audit"],
        iterations=[*original_strict["iterations"], *history, *final["iterations"]],
        original_acceptance_budgets_changed=False,
        intermediate_task_envelopes_are_relaxed=True,
        nonlinear_intermediate_iterates_are_accepted_references=False,
        teacher_accepted=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    return current, final
