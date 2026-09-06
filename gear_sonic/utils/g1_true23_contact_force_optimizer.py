"""Offline full-path force restoration with conservative local contact patches.

All frames retain independent native26 pose variables. Source position boxes,
50 Hz timing, root rotations, velocity, acceleration and initial velocity remain
immutable. The local objective minimizes the correction from the CURRENT pose,
not displacement toward a rejected force-infeasible source. No hardware APIs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from scipy import sparse

from gear_sonic.utils.g1_23dof_trajectory_projection import _constraint_system, audit_trajectory_constraints
from gear_sonic.utils.g1_true23_box_qp import solve_box_qp
from gear_sonic.utils.g1_true23_contact_patch import FrozenFloorSupportPatch
from gear_sonic.utils.g1_true23_force_trajectory import ForceLinearization
from gear_sonic.utils.g1_true23_reference_support import floor_contact_map


@dataclass(frozen=True)
class ContactForceConfig:
    maximum_iterations: int = 64
    qp_maximum_iterations: int = 200
    root_trust_m: float = 0.015
    joint_trust_rad: float = 0.12
    initial_trust_fraction: float = 1 / 16
    minimum_trust_fraction: float = 1 / 256
    patch_guard_m: float = 0.00005
    force_residual_cost: float = 1000.0
    contact_slack_cost: float = 100.0
    path_audit_tolerance: float = 2e-7
    linear_audit_tolerance: float = 1e-8
    generalized_force_tolerance: float = 1e-5
    effort_limit_fraction: float = 0.98

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError("contact-force settings must be finite and positive")
            if "iterations" in name and type(value) is not int:
                raise ValueError("contact-force iteration counts must be integers")
        if not (
            self.minimum_trust_fraction <= self.initial_trust_fraction <= 1
            and self.root_trust_m <= 0.015
            and self.joint_trust_rad <= 0.12
            and self.path_audit_tolerance < self.patch_guard_m < 0.002
            and self.effort_limit_fraction < 1
            and self.path_audit_tolerance <= 2e-7
            and self.linear_audit_tolerance <= 1e-8
            and self.generalized_force_tolerance <= 1e-5
        ):
            raise ValueError("contact-force settings may not relax original trust, limits or audit tolerances")


class PatchedForceLinearization(ForceLinearization):
    """Reuse the pinned inverse-force implementation; add actual candidate patches.

    The extra private position pass avoids changing historical source hashes.
    Every model has separate force variables but shares the entire pose path.
    Patches are rebuilt after every accepted update, not a fixed foot schedule.
    """

    def __init__(self, *args, patch_guard_m=0.00005, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isfinite(patch_guard_m) or not 0 < patch_guard_m < self.gap:
            raise ValueError("contact patch guard must be positive and smaller than the candidate gap")
        self.patch_guard = patch_guard_m

    def evaluate(self, path, *, jacobian=True):
        result = super().evaluate(path, jacobian=jacobian)
        if not jacobian:
            return result
        values, derivatives, patches, counts = [], [], [], {}
        for name, (inverse, model, data, plane) in self.models.items():
            qpos, _velocity, _acceleration = inverse.state(path)
            # Sequential patch audits share one dedicated scratch per model,
            # rather than retaining thousands of complete MuJoCo workspaces.
            patch_data = mujoco.MjData(model)
            model_derivatives = []
            count = 0
            for frame, position in enumerate(qpos):
                data.qpos[:] = position
                mujoco.mj_fwdPosition(model, data)
                _force_map, records = floor_contact_map(model, data, plane, self.gap)
                patch = FrozenFloorSupportPatch(
                    model, data, records, plane, gap_m=self.gap, guard_m=self.patch_guard
                )
                patch.data = patch_data
                value, derivative = patch.evaluate(position)
                values.append(value)
                model_derivatives.append(sparse.csc_matrix(derivative))
                patches.append((inverse, frame, patch))
                count += len(value)
            derivatives.append(sparse.block_diag(model_derivatives, format="csc"))
            counts[name] = count
        result.update(
            patch_values=np.concatenate(values),
            patch_jacobian=sparse.vstack(derivatives, format="csc"),
            frozen_patches=patches,
            patch_counts=counts,
        )
        return result

    @staticmethod
    def audit_frozen_patches(path, linearization):
        """Audit material points nonlinearly, independently of QP row residuals."""
        poses = {}
        worst = 0.0
        for inverse, frame, patch in linearization["frozen_patches"]:
            if inverse not in poses:
                poses[inverse] = inverse.state(path)[0]
            values, _ = patch.evaluate(poses[inverse][frame], jacobian=False)
            worst = max(worst, float(np.maximum(-values, 0).max(initial=0)))
        return worst


def contact_force_step(
    current,
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
    trust_fraction,
    progress=None,
):
    """Strict full-path QP with hard local patches and temporary restoration slacks."""
    if not config.minimum_trust_fraction <= trust_fraction <= 1:
        raise ValueError("trust fraction lies outside configured bounded restoration range")
    steps = np.tile(np.r_[np.full(3, config.root_trust_m), np.full(23, config.joint_trust_rad)], len(current))
    steps *= trust_fraction
    trust = steps.reshape(current.shape)
    lo, hi = np.maximum(lower, current - trust), np.minimum(upper, current + trust)
    if np.any(lo > hi):
        raise ValueError("current path lies outside original correction bounds")
    operator, temporal_lower, temporal_upper = _constraint_system(
        len(current), lo, hi, velocity * 0.02, acceleration * 0.02**2, initial_velocity * 0.02
    )
    temporal = sparse.kron(operator, sparse.eye(26), format="csc")
    row_scale = np.tile(steps[:26], operator.shape[0])
    step_matrix = sparse.diags(steps)
    temporal_delta = sparse.diags(1 / row_scale) @ temporal @ step_matrix
    tl = (temporal_lower.ravel() - temporal @ current.ravel()) / row_scale
    tu = (temporal_upper.ravel() - temporal @ current.ravel()) / row_scale
    retained = contact_values <= np.asarray(abs(contact_jacobian) @ steps).ravel() + 1e-8
    geometry = contact_jacobian[retained] @ step_matrix / 0.001
    gl = -contact_values[retained] / 0.001
    patch = forces["patch_jacobian"] @ step_matrix / 0.001
    pl = -forces["patch_values"] / 0.001
    scale = forces["scale"]
    force_delta = sparse.diags(1 / scale) @ forces["jacobian"] @ step_matrix
    force_map = sparse.diags(1 / scale) @ forces["force_map"]
    n, nf, nr, nc, npatch = current.size, force_map.shape[1], len(scale), len(gl), len(pl)
    zero = sparse.csc_matrix
    matrix = sparse.vstack(
        [
            sparse.hstack([temporal_delta, zero((len(tl), nf + 2 * nr + nc))]),
            sparse.hstack([force_delta, -force_map, -sparse.eye(nr), sparse.eye(nr), zero((nr, nc))]),
            sparse.hstack([geometry, zero((nc, nf + 2 * nr)), sparse.eye(nc)]),
            sparse.hstack([patch, zero((npatch, nf + 2 * nr + nc))]),
            sparse.hstack([zero((nf, n)), sparse.eye(nf), zero((nf, 2 * nr + nc))]),
            sparse.hstack([zero((2 * nr + nc, n + nf)), sparse.eye(2 * nr + nc)]),
        ],
        format="csc",
    )
    fl = -forces["required"] / scale
    constraint_lower = np.r_[tl, fl, gl, pl, forces["lower"], np.zeros(2 * nr + nc)]
    constraint_upper = np.r_[tu, fl, np.full(nc + npatch, np.inf), forces["upper"], np.full(2 * nr + nc, np.inf)]
    path_weights = np.tile(np.r_[np.full(3, 1 / 0.08**2), np.full(23, 1 / 0.6**2)], len(current))
    diagonal = np.r_[path_weights * steps**2, np.full(nf + 2 * nr + nc, 1e-8)]
    # Zero path linear term: penalize delta from current, NOT distance back to source.
    linear_cost = np.r_[
        np.zeros(n + nf), np.full(2 * nr, config.force_residual_cost), np.full(nc, config.contact_slack_cost)
    ]
    solution, report = solve_box_qp(
        diagonal,
        linear_cost,
        matrix,
        constraint_lower,
        constraint_upper,
        maximum_iterations=config.qp_maximum_iterations,
        audit_tolerance=config.linear_audit_tolerance,
        progress=progress,
    )
    report.update(
        path_variables=n,
        force_variables=nf,
        generalized_force_residual_variables=2 * nr,
        contact_rows_total=len(contact_values),
        contact_rows_retained=nc,
        hard_patch_rows=npatch,
        patch_counts=forces.get("patch_counts", {}),
        trust_fraction=trust_fraction,
        pose_objective_center="current_iterate",
        force_residual_objective="absolute_normalized_generalized_force_exact_penalty",
        all_frames_have_independent_pose_variables=True,
    )
    if solution is None:
        return None, report
    report["maximum_temporary_contact_slack_m"] = float(solution[n + nf + 2 * nr :].max(initial=0) * 0.001)
    residual = (solution[n + nf : n + nf + nr] - solution[n + nf + nr : n + nf + 2 * nr]) * scale
    report["maximum_temporary_generalized_force_residual"] = float(np.abs(residual).max(initial=0))
    return current + (steps * solution[:n]).reshape(current.shape), report


def restore_contact_force_trajectory(
    desired,
    lower,
    upper,
    velocity,
    acceleration,
    initial_velocity,
    contacts,
    force_system,
    *,
    config=ContactForceConfig(),
    progress=None,
):
    """Try every frame and both models, retaining all failed candidates in reports."""
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

    def merit(contact, force):
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

    def restored(contact, force):
        return (
            contact["passed"]
            and force["maximum_absolute_generalized_force_residual"] <= config.generalized_force_tolerance
        )

    if not temporal(current).passed:
        raise ValueError("contact-force input violates immutable position/derivative bounds")
    contact = contacts.audit(current)
    force = force_system.evaluate(current, jacobian=False)
    before = compact(contact, force)
    history, failure = [], None
    trust_fraction = config.initial_trust_fraction
    for iteration in range(config.maximum_iterations):
        if restored(contact, force):
            break
        values, jacobian = contacts.evaluate(current)
        linear_forces = force_system.evaluate(current)
        record = {"iteration": iteration + 1, "attempts": []}
        accepted = False
        while trust_fraction >= config.minimum_trust_fraction:
            candidate, qp = contact_force_step(
                current,
                lower,
                upper,
                velocity,
                acceleration,
                initial_velocity,
                values,
                jacobian,
                linear_forces,
                config=config,
                trust_fraction=trust_fraction,
                progress=progress,
            )
            attempt = {"qp": qp, "trials": []}
            record["attempts"].append(attempt)
            if candidate is not None:
                for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    proposal = current + fraction * (candidate - current)
                    path_passed = temporal(proposal).passed
                    patch_error = force_system.audit_frozen_patches(proposal, linear_forces)
                    # The linear patch sits guard_m INSIDE the immutable 2 mm
                    # candidate band to absorb curvature. Nonlinear evaluation
                    # may consume that reserve, but must stay 0.2 um inside the
                    # outer band. Requiring the reserve itself to stay unused
                    # wrongly shrinks valid steps toward zero near curved feet.
                    patch_passed = patch_error <= config.patch_guard_m - config.path_audit_tolerance
                    trial = {
                        "fraction": fraction,
                        "temporal_passed": path_passed,
                        "frozen_patch_guard_consumed_m": patch_error,
                        "frozen_patch_within_candidate_band": patch_passed,
                        "accepted": False,
                    }
                    attempt["trials"].append(trial)
                    if not path_passed or not patch_passed:
                        continue
                    next_contact = contacts.audit(proposal)
                    next_force = force_system.evaluate(proposal, jacobian=False)
                    trial.update(
                        merit=merit(next_contact, next_force),
                        maximum_absolute_generalized_force_residual=next_force[
                            "maximum_absolute_generalized_force_residual"
                        ],
                    )
                    if trial["merit"] < merit(contact, force) - 1e-8:
                        current, contact, force, accepted = proposal, next_contact, next_force, True
                        trial["accepted"] = True
                        record.update(fraction=fraction, after=compact(contact, force))
                        # Grow only after an unshortened, independently checked step.
                        trust_fraction = min(1.0, trust_fraction * 2) if fraction == 1 else trust_fraction
                        break
            if progress:
                progress(
                    {
                        "iteration": iteration + 1,
                        "qp_status": qp["status"],
                        "qp_accepted": qp["accepted"],
                        "trial_accepted": accepted,
                        "trust_fraction": qp.get("trust_fraction"),
                        "independent_linear_violation": qp.get("independent_maximum_scaled_linear_violation"),
                        "trials": attempt["trials"],
                    }
                )
            if accepted:
                break
            if trust_fraction == config.minimum_trust_fraction:
                break
            trust_fraction = max(config.minimum_trust_fraction, trust_fraction / 2)
        history.append(record)
        if progress:
            progress(
                {
                    "iteration": iteration + 1,
                    "accepted": accepted,
                    "contact_violated_frames": contact["violated_frames"],
                    "force_residual_squared": force["summed_normalized_force_residual_squared"],
                    "maximum_force_residual": force["maximum_absolute_generalized_force_residual"],
                    "next_trust_fraction": trust_fraction,
                }
            )
        if not accepted:
            failure = "strict QP or nonlinear contact-force restoration exhausted bounded trust retries"
            break
    if failure is None and not restored(contact, force):
        failure = "whole-path contact-force restoration iteration limit"
    return current, {
        "kind": "g1_true23_whole_path_contact_force_restoration_v2",
        "config": asdict(config),
        "before": before,
        "after": compact(contact, force),
        "iterations": history,
        "failure": failure,
        "frames_in": len(desired),
        "frames_out": len(current),
        "pose_variables": int(current.size),
        "all_frames_have_independent_pose_variables": True,
        "pose_objective_center": "current_iterate",
        "temporary_slacks_are_not_acceptance": True,
        "frozen_patch_is_local_hypothesis_not_contact_or_no_slip_proof": True,
        "nonlinear_merit_uses_l1_norm_of_bounded_l2_seed_not_exact_l1_force_optimum": True,
        "serialized_reference_and_independent_support_lp_required": True,
        "static_friction_assistance_is_optimistic": True,
        "dynamic_feasibility_proven": False,
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
