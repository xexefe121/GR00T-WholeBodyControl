"""Jointly refine whole-path derivative limits and actual contact geometry.

Sequential convex restoration uses temporary nonnegative contact slacks, but
acceptance requires an independent nonlinear check with no allowed slack.
Inferred support bodies are hypotheses, not measured contacts. No dynamics,
force feasibility, policy tracking, hardware or teacher acceptance is implied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from scipy import sparse

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_trajectory_projection import (
    _constraint_system,
    audit_trajectory_constraints,
)
from gear_sonic.utils.g1_true23_contact_geometry import lift_reset_floor_overlap

VARIABLE_DOFS = np.r_[0:3, 6:29]


@dataclass(frozen=True)
class ContactTrajectoryConfig:
    maximum_iterations: int = 20
    qp_maximum_iterations: int = 100000
    root_trust_m: float = 0.015
    joint_trust_rad: float = 0.12
    floor_clearance_m: float = 0.0002
    support_gap_m: float = 0.002
    contact_slack_cost: float = 100.0
    audit_tolerance: float = 2e-7

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError("contact refinement settings must be finite and positive")
            if "iterations" in name and type(value) is not int:
                raise ValueError("iteration counts must be integers")
        if not self.audit_tolerance < self.floor_clearance_m < self.support_gap_m <= 0.01:
            raise ValueError("contact tolerance/clearance/gap ordering is invalid")


class ContactLinearization:
    """Signed distance and world-z closest-point Jacobian on independent data."""

    def __init__(self, models, source_qpos, support_bodies, config):
        self.source = np.asarray(source_qpos, dtype=float)
        if self.source.ndim != 2 or self.source.shape[1] != 30 or not np.isfinite(self.source).all():
            raise ValueError("contact path requires finite [frames,30] native23 poses")
        if not models or len(support_bodies) != len(self.source):
            raise ValueError("every frame requires an explicit support hypothesis and collision models")
        self.supports = [tuple(names) for names in support_bodies]
        if any(len(set(names)) != len(names) for names in self.supports):
            raise ValueError("duplicate support body in frame")
        self.config, self.models = config, models
        self.materials = []
        self.row_labels = []
        for name, model in models.items():
            if (
                model.nq != 30
                or model.nv != 29
                or model.njnt != 24
                or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
                or [model.joint(i).name for i in range(1, 24)] != list(HARDWARE_23_JOINT_NAMES)
                or not np.array_equal(model.jnt_qposadr, np.r_[0, 7:30])
                or not np.array_equal(model.jnt_dofadr, np.r_[0, 6:29])
            ):
                raise ValueError("contact model must use exact native23 joint/state order")
            # Reuse the existing single-articulation / static-horizontal-floor validator.
            lift_reset_floor_overlap(model, model.qpos0[None])
            plane = int(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)[0])
            geoms = [
                i
                for i in range(model.ngeom)
                if model.geom_bodyid[i] > 0
                and (
                    (model.geom_contype[i] & model.geom_conaffinity[plane])
                    or (model.geom_contype[plane] & model.geom_conaffinity[i])
                )
            ]
            groups = {}
            for body in {body for names in self.supports for body in names}:
                bid = model.body(body).id
                groups[body] = [j for j, geom in enumerate(geoms) if model.geom_bodyid[geom] == bid]
                if not groups[body]:
                    raise ValueError(f"support body lacks floor-colliding geometry: {body}")
            self.materials.append((name, model, mujoco.MjData(model), plane, geoms, groups))
        for frame, supports in enumerate(self.supports):
            for name, _model, _data, _plane, geoms, _groups in self.materials:
                self.row_labels.extend((frame, name, "floor", geom) for geom in geoms)
                self.row_labels.extend((frame, name, "support", body) for body in supports)

    def evaluate(self, path, *, jacobian=True):
        path = np.asarray(path, dtype=float)
        if path.shape != (len(self.source), 26) or not np.isfinite(path).all():
            raise ValueError("contact variables require finite [frames,26] root-offset/joint path")
        values, blocks = [], []
        for frame, x in enumerate(path):
            frame_rows = []
            for _name, model, data, plane, geoms, groups in self.materials:
                data.qpos[:] = self.source[frame]
                data.qpos[:3] += x[:3]
                data.qpos[7:] = x[3:]
                mujoco.mj_kinematics(model, data)
                if jacobian:
                    mujoco.mj_comPos(model, data)
                distances, derivatives = [], []
                for geom in geoms:
                    segment = np.zeros(6)
                    distance = mujoco.mj_geomDistance(model, data, plane, geom, 3.0, segment)
                    if distance >= 3.0 or not np.isfinite(distance):
                        raise ValueError("collider lies outside the explicit 3 m distance query")
                    distances.append(distance)
                    if jacobian:
                        jacp = np.zeros((3, model.nv))
                        mujoco.mj_jac(model, data, jacp, None, segment[3:], int(model.geom_bodyid[geom]))
                        derivatives.append(jacp[2, VARIABLE_DOFS])
                distances = np.asarray(distances)
                values.extend(distances - self.config.floor_clearance_m)
                if jacobian:
                    frame_rows.extend(derivatives)
                for body in self.supports[frame]:
                    nearest = min(groups[body], key=lambda index: distances[index])
                    values.append(self.config.support_gap_m - distances[nearest])
                    if jacobian:
                        frame_rows.append(-derivatives[nearest])
            if jacobian:
                blocks.append(sparse.csr_matrix(np.asarray(frame_rows)))
        values = np.asarray(values)
        matrix = sparse.block_diag(blocks, format="csc") if jacobian else None
        return values, matrix

    def audit(self, path):
        values, _ = self.evaluate(path, jacobian=False)
        violation = np.maximum(-values, 0)
        bad = np.flatnonzero(violation > self.config.audit_tolerance)
        worst = sorted(bad, key=lambda index: violation[index], reverse=True)[:16]
        return {
            "passed": not len(bad),
            "constraint_count": len(values),
            "violated_constraints": len(bad),
            "violated_frames": len({self.row_labels[i][0] for i in bad}),
            "maximum_violation_m": float(violation.max(initial=0)),
            "summed_violation_m": float(violation.sum()),
            "worst": [
                {
                    "frame": self.row_labels[i][0],
                    "model": self.row_labels[i][1],
                    "kind": self.row_labels[i][2],
                    "target": self.row_labels[i][3],
                    "violation_m": float(violation[i]),
                }
                for i in worst
            ],
        }


def solve_linearized_restoration(
    current,
    desired,
    lower,
    upper,
    velocity,
    acceleration,
    initial_velocity,
    values,
    jacobian,
    *,
    config,
    trust_scale=1.0,
):
    """Sparse whole-horizon QP; contact slacks are diagnostic restoration variables."""
    import osqp

    shape = current.shape
    trust = trust_scale * np.r_[np.full(3, config.root_trust_m), np.full(23, config.joint_trust_rad)]
    # These omitted rows are redundant for this *linearized* trust box, not
    # discarded collision checks. Every nonlinear distance is re-audited.
    total_contact_rows = len(values)
    retained = values <= np.asarray(abs(jacobian) @ np.tile(trust, len(current))).ravel() + 1e-8
    values, jacobian = values[retained], jacobian[retained]
    lo, hi = np.maximum(lower, current - trust), np.minimum(upper, current + trust)
    if np.any(lo > hi):
        raise ValueError("current path is outside the immutable correction box")
    operator, lows, highs = _constraint_system(
        len(current), lo, hi, velocity * 0.02, acceleration * 0.02**2, initial_velocity * 0.02
    )
    temporal = sparse.kron(operator, sparse.eye(26), format="csc")
    n, m = current.size, len(values)
    constraints = sparse.vstack(
        [
            sparse.hstack([temporal, sparse.csc_matrix((temporal.shape[0], m))]),
            sparse.hstack([jacobian, sparse.eye(m)]),
            sparse.hstack([sparse.csc_matrix((m, n)), sparse.eye(m)]),
        ],
        format="csc",
    )
    constraint_lower = np.r_[lows.ravel(), jacobian @ current.ravel() - values, np.zeros(m)]
    constraint_upper = np.r_[highs.ravel(), np.full(2 * m, np.inf)]
    weight = np.tile(np.r_[np.full(3, 100.0), np.ones(23)], len(current))
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.diags(np.r_[weight, np.full(m, 1e-6)], format="csc"),
        q=np.r_[-weight * desired.ravel(), np.full(m, config.contact_slack_cost)],
        A=constraints,
        l=constraint_lower,
        u=constraint_upper,
        verbose=False,
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=config.qp_maximum_iterations,
        polishing=True,
        adaptive_rho_interval=50,
    )
    solver.warm_start(x=np.r_[current.ravel(), np.maximum(-values, 0)])
    result = solver.solve(raise_error=False)
    report = {
        "status": result.info.status,
        "iterations": result.info.iter,
        "primal_residual": float(result.info.prim_res),
        "dual_residual": float(result.info.dual_res),
        "solve_time_s": float(result.info.run_time),
        "trust_scale": trust_scale,
        "linearized_contact_rows_total": total_contact_rows,
        "linearized_contact_rows_retained": len(values),
    }
    if result.info.status_val != 1 or result.x is None or not np.isfinite(result.x).all():
        return None, report
    linear_value = constraints @ result.x
    violation = max(
        np.maximum(constraint_lower - linear_value, 0).max(initial=0),
        np.maximum(linear_value - constraint_upper, 0).max(initial=0),
    )
    report.update(
        maximum_constraint_violation=float(violation), maximum_contact_slack_m=float(result.x[n:].max(initial=0))
    )
    if violation > config.audit_tolerance:
        report["status"] = "independent_linear_constraint_audit_failed"
        return None, report
    return result.x[:n].reshape(shape), report


def refine_contact_trajectory(
    desired,
    lower,
    upper,
    velocity,
    acceleration,
    initial_velocity,
    contacts,
    *,
    config=ContactTrajectoryConfig(),
    progress=None,
):
    """Keep every frame, audit the nonlinear result; failure never means acceptance."""
    current = np.array(desired, dtype=float, copy=True)

    def temporal_audit(path):
        return audit_trajectory_constraints(
            path,
            lower_bounds=lower,
            upper_bounds=upper,
            dt=0.02,
            max_velocity=velocity,
            max_acceleration=acceleration,
            initial_velocity=initial_velocity,
            tolerance=config.audit_tolerance,
        )

    if not temporal_audit(current).passed:
        raise ValueError("input contact candidate must already satisfy whole-path derivative bounds")
    before = contacts.audit(current)
    after, history, failure = before, [], None
    for iteration in range(config.maximum_iterations):
        if after["passed"]:
            break
        values, jacobian = contacts.evaluate(current)
        candidate, qp = solve_linearized_restoration(
            current,
            desired,
            lower,
            upper,
            velocity,
            acceleration,
            initial_velocity,
            values,
            jacobian,
            config=config,
        )
        record = {"iteration": iteration, "qp": qp}
        if candidate is None:
            failure = "linearized restoration did not pass solver and independent constraints"
            history.append(record)
            break
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
            proposal = current + fraction * (candidate - current)
            check = contacts.audit(proposal)
            if temporal_audit(proposal).passed and (
                check["passed"] or check["summed_violation_m"] < after["summed_violation_m"] - 1e-10
            ):
                current, after, accepted = proposal, check, True
                record.update(fraction=fraction, contact_audit=after)
                break
        history.append(record)
        if progress:
            progress(
                {
                    "iteration": iteration + 1,
                    "accepted": accepted,
                    "maximum_violation_m": after["maximum_violation_m"],
                    "violated_frames": after["violated_frames"],
                }
            )
        if not accepted:
            failure = "nonlinear contact restoration stalled"
            break
    if not after["passed"] and failure is None:
        failure = "whole-path contact restoration iteration limit"
    return current, {
        "kind": "g1_true23_contact_trajectory_restoration_v1",
        "config": asdict(config),
        "before": before,
        "after": after,
        "temporal_audit": asdict(temporal_audit(current)),
        "iterations": history,
        "failure": failure,
        "contact_and_derivative_constraints_passed": after["passed"],
        "temporary_contact_slack_permitted_in_final_acceptance": False,
        "dynamic_feasibility_proven": False,
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
