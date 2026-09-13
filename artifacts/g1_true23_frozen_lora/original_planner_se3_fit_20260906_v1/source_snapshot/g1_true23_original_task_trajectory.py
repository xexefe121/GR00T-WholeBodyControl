"""Diagnostic whole-clip SE(3)/native23 fitting to original SONIC task poses.

Unlike a controller-rollout teacher or a joint-output slice, targets come from
the original 29-joint planner FK. All frames and all 23 actual joints remain.
Small pelvis-attitude corrections may compensate missing waist pitch/roll.
They are reference variables, not added motors or root-force actuators.

This weighted geometric fit is NOT contact, inverse-dynamics, controller,
teleoperation or hardware qualification. In particular, residual foot errors
are not relaxed acceptance thresholds, and source contact timing is not a
verified contact schedule. No robot/network operations exist in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

from gear_sonic.utils import g1_23dof_task_space_retarget as retarget
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_trajectory_projection import _constraint_system, audit_trajectory_constraints
from gear_sonic.utils.g1_true23_box_qp import solve_box_qp


@dataclass(frozen=True)
class OriginalTaskConfig:
    maximum_iterations: int = 24
    maximum_root_offset_m: float = 0.08
    maximum_root_rotation_l1_rad: float = 0.45
    maximum_joint_change_rad: float = 0.6
    root_offset_velocity_m_s: float = 0.75
    root_offset_acceleration_m_s2: float = 6.0
    root_rotation_coordinate_velocity_rad_s: float = 1.5
    root_rotation_coordinate_acceleration_rad_s2: float = 12.0
    joint_velocity_rad_s: float = 5.0
    joint_acceleration_rad_s2: float = 80.0
    serialization_margin_fraction: float = 0.995
    root_trust_m: float = 0.015
    rotation_trust_rad: float = 0.08
    joint_trust_rad: float = 0.12

    def __post_init__(self):
        if type(self.maximum_iterations) is not int or self.maximum_iterations <= 0:
            raise ValueError("iteration count must be a positive integer")
        for value in asdict(self).values():
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError("task-fit settings must be finite and positive")
        if self.maximum_root_rotation_l1_rad >= 0.5 or not 0 < self.serialization_margin_fraction < 1:
            raise ValueError("attitude correction must stay inside the existing 0.5-rad screen with a margin")
        if self.joint_velocity_rad_s > 5 or self.joint_acceleration_rad_s2 > 80:
            raise ValueError("task fitting cannot increase existing joint velocity/acceleration limits")


def skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def so3_left_jacobian(vector):
    """Map rotation-vector increments to world/left angular perturbations."""
    value = np.asarray(vector, dtype=float)
    theta = np.linalg.norm(value)
    cross = skew(value)
    if theta < 1e-5:
        return np.eye(3) + (0.5 - theta**2 / 24) * cross + (1 / 6 - theta**2 / 120) * cross @ cross
    return np.eye(3) + (1 - np.cos(theta)) / theta**2 * cross + (theta - np.sin(theta)) / theta**3 * cross @ cross


def so3_left_jacobian_inverse(vector):
    value = np.asarray(vector, dtype=float)
    theta = np.linalg.norm(value)
    cross = skew(value)
    coefficient = 1 / 12 + theta**2 / 720 if theta < 1e-5 else (1 - theta / (2 * np.tan(theta / 2))) / theta**2
    return np.eye(3) - 0.5 * cross + coefficient * cross @ cross


# Frozen engineering objective scales, not acceptance or manufacturer limits.
TASK_SCALES = {
    "left_foot": (200.0, 10.0),
    "right_foot": (200.0, 10.0),
    "whole_robot_com": (1 / 0.03, 0.0),
    "torso_orientation": (0.0, 10.0),
    "head_proxy": (20.0, 0.0),
    "left_hand": (20.0, 4.0),
    "right_hand": (20.0, 4.0),
    "left_elbow": (10.0, 0.0),
    "right_elbow": (10.0, 0.0),
}


class OriginalTaskPath:
    """Immutable original targets and exact task Jacobians for all 29 variables."""

    def __init__(self, source_model, target_model, source_qpos, seed_joints, *, config=OriginalTaskConfig()):
        self.config = config
        self.source = np.array(source_qpos, dtype=float, copy=True)
        seed = np.array(seed_joints, dtype=float, copy=True)
        source_layout = retarget._model_layout(source_model)
        self.layout = retarget._model_layout(target_model)
        if (source_model.nq, target_model.nq, target_model.nv) != (36, 30, 29) or self.layout.joint_names != tuple(
            HARDWARE_23_JOINT_NAMES
        ):
            raise ValueError("task fitting requires exact original29/native23 model layouts")
        if len(source_layout.joint_names) != 29 or not set(self.layout.joint_names).issubset(
            source_layout.joint_names
        ):
            raise ValueError("source and target joint identities differ")
        if (
            self.source.ndim != 2
            or self.source.shape[1] != 36
            or len(self.source) < 3
            or seed.shape != (len(self.source), 23)
        ):
            raise ValueError("full source/seed arrays must have matching [frames>=3,36/23] shapes")
        if (
            not np.isfinite(self.source).all()
            or not np.isfinite(seed).all()
            or np.any(np.linalg.norm(self.source[:, 3:7], axis=1) < 1e-12)
        ):
            raise ValueError("original source and native seed must be finite with nonzero quaternions")
        self.source[:, 3:7] /= np.linalg.norm(self.source[:, 3:7], axis=1, keepdims=True)
        self.source_rotation = Rotation.from_quat(self.source[:, [4, 5, 6, 3]])
        self.model, self.data = target_model, mujoco.MjData(target_model)
        self.tasks = retarget.validate_tasks(source_model, target_model, retarget.DEFAULT_TASKS)
        source_data = mujoco.MjData(source_model)
        self.targets = []
        for pose in self.source:
            source_data.qpos[:] = pose
            mujoco.mj_forward(source_model, source_data)
            self.targets.append(retarget._task_targets(source_model, source_data, self.tasks))
        self.initial = np.column_stack((np.zeros((len(seed), 6)), seed))
        low, high = retarget.safe_target_joint_bounds(
            target_model, native_action_clip=9.5, safe_limit_guard_rad=0.05
        )
        self.lower = np.column_stack(
            (
                np.full((len(seed), 3), -config.maximum_root_offset_m),
                np.full((len(seed), 3), -config.maximum_root_rotation_l1_rad),
                np.maximum(low, seed - config.maximum_joint_change_rad),
            )
        )
        self.upper = np.column_stack(
            (
                np.full((len(seed), 3), config.maximum_root_offset_m),
                np.full((len(seed), 3), config.maximum_root_rotation_l1_rad),
                np.minimum(high, seed + config.maximum_joint_change_rad),
            )
        )
        self.velocity = np.repeat(
            [
                config.root_offset_velocity_m_s,
                config.root_rotation_coordinate_velocity_rad_s,
                config.joint_velocity_rad_s,
            ],
            [3, 3, 23],
        )
        self.acceleration = np.repeat(
            [
                config.root_offset_acceleration_m_s2,
                config.root_rotation_coordinate_acceleration_rad_s2,
                config.joint_acceleration_rad_s2,
            ],
            [3, 3, 23],
        )
        self.initial_velocity = (self.initial[1] - self.initial[0]) / 0.02
        self.posture_weight = np.tile(np.r_[np.full(3, 100.0), np.full(3, 2.0), np.full(23, 0.1)], len(seed))
        if not self.audit(self.initial)["passed"]:
            raise ValueError("seed must satisfy unchanged whole-path position/derivative bounds")

    def qpos(self, variables):
        x = np.asarray(variables, dtype=float)
        if x.shape != self.initial.shape or not np.isfinite(x).all():
            raise ValueError("all finite frame variables must be supplied")
        rotation = Rotation.from_rotvec(x[:, 3:6]) * self.source_rotation
        return np.column_stack((self.source[:, :3] + x[:, :3], rotation.as_quat()[:, [3, 0, 1, 2]], x[:, 6:]))

    def audit(self, variables):
        temporal = audit_trajectory_constraints(
            variables,
            lower_bounds=self.lower,
            upper_bounds=self.upper,
            dt=0.02,
            max_velocity=self.velocity,
            max_acceleration=self.acceleration,
            initial_velocity=self.initial_velocity,
            tolerance=2e-7,
        )
        l1 = float(np.abs(variables[:, 3:6]).sum(axis=1).max())
        return {
            "passed": temporal.passed and l1 <= self.config.maximum_root_rotation_l1_rad + 2e-7,
            "temporal": asdict(temporal),
            "maximum_root_rotation_l1_rad": l1,
            "rotation_derivatives_are_rotvec_coordinates_not_physical_angular_acceleration": True,
        }

    def evaluate(self, variables, *, derivatives=True):
        poses = self.qpos(variables)
        all_residual, blocks, errors = [], [], []
        for frame, pose in enumerate(poses):
            self.data.qpos[:] = pose
            mujoco.mj_forward(self.model, self.data)
            root_jac = so3_left_jacobian(variables[frame, 3:6])
            residual, jacobians, frame_errors = [], [], []
            for task, target in zip(self.tasks, self.targets[frame], strict=True):
                body = self.model.body(task.target_body).id
                jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                if task.kind == "subtree_com":
                    position, rotation = self.data.subtree_com[body].copy(), np.eye(3)
                    if derivatives:
                        mujoco.mj_jacSubtreeCom(self.model, self.data, jp, body)
                else:
                    position, rotation = retarget._point_pose(self.data, body, task.target_point)
                    if derivatives:
                        mujoco.mj_jac(self.model, self.data, jp, jr, position, body)
                position_error = position - target.position
                orientation_error = Rotation.from_matrix(rotation @ target.rotation.T).as_rotvec()
                frame_errors.append((np.linalg.norm(position_error), np.linalg.norm(orientation_error)))
                ps, rs = TASK_SCALES[task.name]
                if ps:
                    residual.append(ps * position_error)
                    if derivatives:
                        jacobians.append(
                            ps
                            * np.column_stack(
                                (
                                    np.eye(3),
                                    -skew(position - pose[:3]) @ root_jac,
                                    jp[:, self.layout.dof_addresses],
                                )
                            )
                        )
                if rs:
                    residual.append(rs * orientation_error)
                    if derivatives:
                        jacobians.append(
                            rs
                            * so3_left_jacobian_inverse(orientation_error)
                            @ np.column_stack((np.zeros((3, 3)), root_jac, jr[:, self.layout.dof_addresses]))
                        )
            all_residual.append(np.concatenate(residual))
            errors.append(frame_errors)
            if derivatives:
                blocks.append(sparse.csc_matrix(np.vstack(jacobians)))
        return (
            np.concatenate(all_residual),
            sparse.block_diag(blocks, format="csc") if derivatives else None,
            np.array(errors),
        )

    def metrics(self, variables):
        residual, _, errors = self.evaluate(variables, derivatives=False)
        return {
            "frames_checked": len(variables),
            "weighted_task_squared_error": float(residual @ residual),
            "per_task": {
                task.name: {
                    "position_max_m": float(errors[:, index, 0].max()),
                    "position_mean_m": float(errors[:, index, 0].mean()),
                    "orientation_max_rad": float(errors[:, index, 1].max()),
                    "orientation_mean_rad": float(errors[:, index, 1].mean()),
                }
                for index, task in enumerate(self.tasks)
            },
            "both_feet_within_existing_5mm_screen": bool(np.max(errors[:, :2, 0]) <= 0.005),
        }

    def serialize(self, variables):
        pose = self.qpos(variables)
        return retarget.build_mjlab_motion_arrays(
            self.model,
            SimpleNamespace(
                joint_pos_hardware=pose[:, 7:],
                root_pos_w=pose[:, :3],
                root_quat_wxyz=pose[:, 3:7],
                fps=50,
            ),
        )

    def serialized_variables(self, motion):
        rotations = Rotation.from_quat(motion["body_quat_w"][:, 0][:, [1, 2, 3, 0]])
        return np.column_stack(
            (
                motion["body_pos_w"][:, 0] - self.source[:, :3],
                (rotations * self.source_rotation.inv()).as_rotvec(),
                motion["joint_pos"],
            )
        )


def solve_task_step(problem, current, residual, jacobian):
    cfg = problem.config
    trust = np.repeat([cfg.root_trust_m, cfg.rotation_trust_rad, cfg.joint_trust_rad], [3, 3, 23])
    lo, hi = np.maximum(problem.lower, current - trust), np.minimum(problem.upper, current + trust)
    margin = cfg.serialization_margin_fraction
    operator, lower, upper = _constraint_system(
        len(current),
        lo,
        hi,
        problem.velocity * margin * 0.02,
        problem.acceleration * margin * 0.02**2,
        problem.initial_velocity * 0.02,
    )
    temporal = sparse.kron(operator, sparse.eye(29), format="csc")
    facets = np.zeros((8, 29))
    facets[:, 3:6] = list(product((-1.0, 1.0), repeat=3))
    rotation_bound = sparse.kron(sparse.eye(len(current)), sparse.csc_matrix(facets), format="csc")
    n, m = current.size, len(residual)
    equality = jacobian @ current.ravel() - residual
    matrix = sparse.vstack(
        (
            sparse.hstack((temporal, sparse.csc_matrix((temporal.shape[0], m)))),
            sparse.hstack((rotation_bound, sparse.csc_matrix((rotation_bound.shape[0], m)))),
            sparse.hstack((jacobian, -sparse.eye(m))),
        ),
        format="csc",
    )
    solution, report = solve_box_qp(
        np.r_[problem.posture_weight, np.ones(m)],
        np.r_[-problem.posture_weight * problem.initial.ravel(), np.zeros(m)],
        matrix,
        np.r_[lower.ravel(), np.full(rotation_bound.shape[0], -np.inf), equality],
        np.r_[upper.ravel(), np.full(rotation_bound.shape[0], cfg.maximum_root_rotation_l1_rad), equality],
    )
    return None if solution is None else solution[:n].reshape(current.shape), report


def fit_original_task_path(problem, *, progress=None):
    current = problem.initial.copy()
    before = problem.metrics(current)
    history, failure = [], None

    def merit(x, residual):
        delta = x.ravel() - problem.initial.ravel()
        return float(residual @ residual + np.dot(problem.posture_weight * delta, delta))

    for iteration in range(problem.config.maximum_iterations):
        residual, jacobian, _ = problem.evaluate(current)
        previous_merit = merit(current, residual)
        candidate, qp = solve_task_step(problem, current, residual, jacobian)
        row = {"iteration": iteration + 1, "qp": qp, "accepted": False}
        if candidate is None:
            failure = "task QP failed strict solver/original-row audit"
        else:
            for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = current + fraction * (candidate - current)
                if not problem.audit(trial)["passed"]:
                    continue
                trial_residual, _, _ = problem.evaluate(trial, derivatives=False)
                score = merit(trial, trial_residual)
                if score < previous_merit - 1e-10 * max(1.0, previous_merit):
                    current = trial
                    row.update(accepted=True, fraction=fraction, merit=score)
                    break
            if not row["accepted"]:
                failure = "nonlinear task fit stalled; this is not physical infeasibility"
        history.append(row)
        if progress:
            progress(row)
        if failure is not None:
            break
    return current, {
        "kind": "g1_true23_original_planner_se3_task_fit_v1",
        "config": asdict(problem.config),
        "before": before,
        "after": problem.metrics(current),
        "iterations": history,
        "failure": failure,
        "iteration_budget_exhausted": failure is None and len(history) == problem.config.maximum_iterations,
        "path_constraints": problem.audit(current),
        "frames_dropped": 0,
        "time_scale": 1.0,
        "native_joint_count": 23,
        "root_attitude_reference_optimization_enabled": True,
        "contact_force_or_controller_qualification_performed": False,
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
