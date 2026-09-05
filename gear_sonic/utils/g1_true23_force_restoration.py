"""Offline sequential contact/force/path restoration, never a controller.

Whole-path variables and both model-specific contact-force sets are solved
together. Nonlinear geometry and force fits are recomputed after every proposed
step. Full force LP, serialized references and real closed-loop replay remain
independent qualification work; no optimizer status grants readiness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import osqp
from scipy import sparse

from gear_sonic.utils.g1_23dof_trajectory_projection import _constraint_system, audit_trajectory_constraints


@dataclass(frozen=True)
class ForceRestorationConfig:
    maximum_iterations: int = 8
    qp_maximum_iterations: int = 30000
    root_trust_m: float = 0.015
    joint_trust_rad: float = 0.12
    force_residual_cost: float = 1000.0
    contact_slack_cost: float = 100.0
    path_audit_tolerance: float = 2e-7
    linear_audit_tolerance: float = 1e-8
    generalized_force_tolerance: float = 1e-5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError("force restoration settings must be finite and positive")
            if "iterations" in name and type(value) is not int:
                raise ValueError("force restoration iteration counts must be integers")


def force_restoration_step(
    current,
    desired,
    lower,
    upper,
    velocity,
    acceleration,
    initial_velocity,
    contact_values,
    contact_jacobian,
    forces,
    *,
    config,
):
    """Scaled whole-path QP with explicit six-row floating-base force balance.

    Unknown normalized force residuals and geometric slacks aid restoration.
    They are not allowed in final acceptance. No physical limit is changed.
    """
    steps = np.tile(np.r_[np.full(3, config.root_trust_m), np.full(23, config.joint_trust_rad)], len(current))
    trust = steps.reshape(current.shape)
    lo, hi = np.maximum(lower, current - trust), np.minimum(upper, current + trust)
    if np.any(lo > hi):
        raise ValueError("force restoration path lies outside original correction bounds")
    operator, temporal_lower, temporal_upper = _constraint_system(
        len(current), lo, hi, velocity * 0.02, acceleration * 0.02**2, initial_velocity * 0.02
    )
    temporal = sparse.kron(operator, sparse.eye(26), format="csc")
    row_scale = np.tile(steps[:26], operator.shape[0])
    temporal_delta = sparse.diags(1 / row_scale) @ temporal @ sparse.diags(steps)
    tl, tu = (
        (temporal_lower.ravel() - temporal @ current.ravel()) / row_scale,
        (temporal_upper.ravel() - temporal @ current.ravel()) / row_scale,
    )
    retained = contact_values <= np.asarray(abs(contact_jacobian) @ steps).ravel() + 1e-8
    geometry = contact_jacobian[retained] @ sparse.diags(steps) / 0.001
    gl = -contact_values[retained] / 0.001
    scale = forces["scale"]
    force_delta = sparse.diags(1 / scale) @ forces["jacobian"] @ sparse.diags(steps)
    force_map = sparse.diags(1 / scale) @ forces["force_map"]
    n, nf, nr, nc = current.size, force_map.shape[1], len(scale), len(gl)
    zero = sparse.csc_matrix
    matrix = sparse.vstack(
        [
            sparse.hstack([temporal_delta, zero((len(tl), nf + 2 * nr + nc))]),
            sparse.hstack([force_delta, -force_map, -sparse.eye(nr), sparse.eye(nr), zero((nr, nc))]),
            sparse.hstack([geometry, zero((nc, nf + 2 * nr)), sparse.eye(nc)]),
            sparse.hstack([zero((nf, n)), sparse.eye(nf), zero((nf, 2 * nr + nc))]),
            sparse.hstack([zero((2 * nr + nc, n + nf)), sparse.eye(2 * nr + nc)]),
        ],
        format="csc",
    )
    fl = -forces["required"] / scale
    constraint_lower = np.r_[tl, fl, gl, forces["lower"], np.zeros(2 * nr + nc)]
    constraint_upper = np.r_[tu, fl, np.full(nc, np.inf), forces["upper"], np.full(2 * nr + nc, np.inf)]
    path_weights = np.tile(np.r_[np.full(3, 1 / 0.08**2), np.full(23, 1 / 0.6**2)], len(current))
    diagonal = np.r_[path_weights * steps**2, np.full(nf + 2 * nr + nc, 1e-8)]
    q = np.r_[
        path_weights * steps * (current - desired).ravel(),
        np.zeros(nf),
        np.full(2 * nr, config.force_residual_cost),
        np.full(nc, config.contact_slack_cost),
    ]
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.diags(diagonal, format="csc"),
        q=q,
        A=matrix,
        l=constraint_lower,
        u=constraint_upper,
        verbose=False,
        eps_abs=1e-9,
        eps_rel=1e-9,
        max_iter=config.qp_maximum_iterations,
        polishing=True,
        adaptive_rho_interval=50,
    )
    solver.warm_start(
        x=np.r_[
            np.zeros(n),
            forces["seed"],
            np.maximum(forces["normalized_residual"], 0),
            np.maximum(-forces["normalized_residual"], 0),
            np.maximum(gl, 0),
        ]
    )
    result = solver.solve(raise_error=False)
    report = {
        "status": result.info.status,
        "iterations": result.info.iter,
        "primal_residual": float(result.info.prim_res),
        "dual_residual": float(result.info.dual_res),
        "solve_time_s": float(result.info.run_time),
        "path_variables": n,
        "force_variables": nf,
        "generalized_force_residual_variables": 2 * nr,
        "force_residual_objective": "absolute_normalized_generalized_force_exact_penalty",
        "contact_rows_total": len(contact_values),
        "contact_rows_retained": nc,
    }
    if result.info.status_val != 1 or result.x is None or not np.isfinite(result.x).all():
        return None, report
    linear = matrix @ result.x
    violation = max(
        np.maximum(constraint_lower - linear, 0).max(initial=0),
        np.maximum(linear - constraint_upper, 0).max(initial=0),
    )
    report["independent_maximum_scaled_linear_violation"] = float(violation)
    report["maximum_temporary_contact_slack_m"] = float(result.x[n + nf + 2 * nr :].max(initial=0) * 0.001)
    report["maximum_temporary_generalized_force_residual"] = float(
        np.max(np.abs((result.x[n + nf : n + nf + nr] - result.x[n + nf + nr : n + nf + 2 * nr]) * scale))
    )
    if violation > config.linear_audit_tolerance:
        report["status"] = "independent_linear_constraints_failed"
        return None, report
    return current + (steps * result.x[:n]).reshape(current.shape), report


def restore_force_trajectory(
    desired,
    lower,
    upper,
    velocity,
    acceleration,
    initial_velocity,
    contacts,
    force_system,
    *,
    config=ForceRestorationConfig(),
    progress=None,
):
    """Attempt every whole-path frame; return failed candidates without promotion."""
    current = np.array(desired, dtype=float, copy=True)

    def temporal(path):
        return audit_trajectory_constraints(
            path,
            lower_bounds=lower,
            upper_bounds=upper,
            dt=0.02,
            max_velocity=velocity,
            max_acceleration=acceleration,
            initial_velocity=initial_velocity,
            tolerance=config.path_audit_tolerance,
        )

    def merits(contact, force):
        return config.contact_slack_cost * contact[
            "summed_violation_m"
        ] / 0.001 + config.force_residual_cost * float(np.abs(force["normalized_residual"]).sum())

    def compact(contact, force):
        return {
            "contact": contact,
            "force": {
                key: force[key]
                for key in (
                    "summed_normalized_force_residual_squared",
                    "maximum_absolute_generalized_force_residual",
                    "models",
                )
            },
        }

    if not temporal(current).passed:
        raise ValueError("force restoration input violates immutable position/derivative bounds")
    contact = contacts.audit(current)
    force = force_system.evaluate(current, jacobian=False)
    before = compact(contact, force)
    history, failure = [], None
    for iteration in range(config.maximum_iterations):
        if (
            contact["passed"]
            and force["maximum_absolute_generalized_force_residual"] <= config.generalized_force_tolerance
        ):
            break
        values, jacobian = contacts.evaluate(current)
        linear_forces = force_system.evaluate(current)
        candidate, qp = force_restoration_step(
            current,
            desired,
            lower,
            upper,
            velocity,
            acceleration,
            initial_velocity,
            values,
            jacobian,
            linear_forces,
            config=config,
        )
        record = {"iteration": iteration + 1, "qp": qp}
        if candidate is None:
            failure = "force restoration QP failed solver/independent constraints"
            history.append(record)
            break
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
            proposal = current + fraction * (candidate - current)
            if not temporal(proposal).passed:
                continue
            next_contact = contacts.audit(proposal)
            next_force = force_system.evaluate(proposal, jacobian=False)
            if merits(next_contact, next_force) < merits(contact, force) - 1e-8:
                current, contact, force, accepted = proposal, next_contact, next_force, True
                record.update(fraction=fraction, after=compact(contact, force))
                break
        history.append(record)
        if progress:
            progress(
                {
                    "iteration": iteration + 1,
                    "accepted": accepted,
                    "contact_violated_frames": contact["violated_frames"],
                    "force_residual_squared": force["summed_normalized_force_residual_squared"],
                }
            )
        if not accepted:
            failure = "actual-contact force/geometry restoration stalled"
            break
    if failure is None and not (
        contact["passed"]
        and force["maximum_absolute_generalized_force_residual"] <= config.generalized_force_tolerance
    ):
        failure = "whole-path force restoration iteration limit"
    return current, {
        "kind": "g1_true23_whole_path_force_restoration_v1",
        "config": asdict(config),
        "before": before,
        "after": compact(contact, force),
        "iterations": history,
        "failure": failure,
        "serialized_reference_and_independent_support_lp_required": True,
        "force_linearization_uses_frozen_body_attached_points_not_closest_feature_derivatives": True,
        "static_friction_assistance_is_optimistic": True,
        "dynamic_feasibility_proven": False,
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
