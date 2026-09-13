"""Bounded native five-axis arm IK for original29 hand intent.

The two hand points use ``neutral_wrist_hand_tasks``: wrist-roll-local
(.264, +/-.025, 0). Only the ten arm joints change; root, waist, and legs
retain their input values. This is reference geometry, not dynamic tracking.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES


HAND_POINTS_LOCAL = np.array(((.264, -.025, 0.), (.264, .025, 0.)))


@dataclass(frozen=True)
class ArmIKResult:
    qpos: np.ndarray
    position_errors_m: np.ndarray
    orientation_errors_rad: np.ndarray
    nfev: np.ndarray
    njev: np.ndarray
    success: np.ndarray
    costs: np.ndarray


class Native23ArmIK:
    """Reusable deterministic analytic-Jacobian solver; one instance per worker.

    Residual scales are metres for position, metres/radian for soft rotation
    and posture. Defaults give hand position priority with mild orientation
    and source-posture regularization. No history, frame index, or clip-specific
    tuning enters the objective.
    """

    def __init__(self, model, *, position_weight=1., orientation_weight=.03,
                 posture_weight=.002, max_nfev=60):
        names = tuple(model.joint(i).name for i in range(1, model.njnt))
        if (model.nq, model.nv) != (30, 29) or names != tuple(HARDWARE_23_JOINT_NAMES):
            raise ValueError("arm IK requires the native23 joint topology")
        weights = np.asarray((position_weight, orientation_weight, posture_weight))
        if not np.isfinite(weights).all() or position_weight <= 0 or np.any(weights[1:] < 0):
            raise ValueError("IK residual weights must be finite and nonnegative; position must be positive")
        if type(max_nfev) is not int or max_nfev < 1:
            raise ValueError("max_nfev must be a positive integer")
        self.model, self.data = model, mujoco.MjData(model)
        self.position_weight = float(position_weight)
        self.orientation_weight = float(orientation_weight)
        self.posture_weight = float(posture_weight)
        self.max_nfev = max_nfev
        joint_groups = tuple(tuple(f"{side}_{joint}_joint" for joint in
                             ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"))
                             for side in ("left", "right"))
        ids = np.array([[model.joint(name).id for name in group] for group in joint_groups])
        self.qpos_indices = model.jnt_qposadr[ids].copy()
        self.dof_indices = model.jnt_dofadr[ids].copy()
        self.lower, self.upper = model.jnt_range[ids, 0].copy(), model.jnt_range[ids, 1].copy()
        if not np.all(model.jnt_limited[ids]) or not np.all(self.lower < self.upper):
            raise ValueError("native arm joints must have finite positional bounds")
        self.body_ids = model.jnt_bodyid[ids[:, -1]].copy()

    def hand_poses(self, qpos):
        """Return proxy positions [left,right,xyz] and body rotation matrices."""
        self.data.qpos[:] = self._pose(qpos)
        mujoco.mj_fwdPosition(self.model, self.data)
        rotations = self.data.xmat[self.body_ids].reshape(2, 3, 3).copy()
        positions = self.data.xpos[self.body_ids] + np.einsum("nij,nj->ni", rotations, HAND_POINTS_LOCAL)
        return positions.copy(), rotations

    @staticmethod
    def _pose(value):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (30,) or not np.isfinite(value).all():
            raise ValueError("arm IK requires finite qpos[30]")
        if not np.isclose(np.linalg.norm(value[3:7]), 1., atol=1e-6, rtol=0):
            raise ValueError("arm IK requires a normalized root quaternion")
        return value.copy()

    def residual_jacobian(self, joints, *, side, qpos, target_position_w,
                          target_rotation_w, posture):
        """World point, soft rotation matrix and posture residual/Jacobian."""
        data, body = self.data, self.body_ids[side]
        data.qpos[:] = qpos
        data.qpos[self.qpos_indices[side]] = joints
        mujoco.mj_fwdPosition(self.model, data)
        rotation = data.xmat[body].reshape(3, 3)
        point = data.xpos[body] + rotation @ HAND_POINTS_LOCAL[side]
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jp, jr, point, body)
        columns = self.dof_indices[side]
        residuals = [self.position_weight * (point - target_position_w)]
        jacobians = [self.position_weight * jp[:, columns]]
        if target_rotation_w is not None and self.orientation_weight:
            scale = self.orientation_weight / np.sqrt(2.)
            residuals.append(scale * (rotation - target_rotation_w).T.ravel())
            angular = jr[:, columns]
            rotation_jac = np.concatenate([np.cross(angular.T, rotation[:, col]).T for col in range(3)], axis=0)
            jacobians.append(scale * rotation_jac)
        if self.posture_weight:
            residuals.append(self.posture_weight * (joints - posture))
            jacobians.append(self.posture_weight * np.eye(5))
        return np.concatenate(residuals), np.vstack(jacobians)

    def solve(self, qpos, target_positions_w, target_quaternions_w=None, *,
              target_rotations_w=None, posture_qpos=None, previous_qpos=None,
              max_step_rad=None):
        """Fit both hands; quaternion inputs are WXYZ. Optional matrices are 2x3x3.

        ``posture_qpos`` defines regularization independently of the solver seed.
        Supplying both ``previous_qpos`` and positive ``max_step_rad`` intersects
        physical arm bounds with the previous arm pose +/- the step. A scalar
        step or [2,5] array is accepted. This constrains causal reference changes
        without changing source timing; target errors remain explicit.
        Root/waist/legs are copied bit-for-bit from ``qpos``. Results report
        optimizer convergence separately from achieved metric errors.
        """
        pose = self._pose(qpos)
        posture = pose.copy() if posture_qpos is None else self._pose(posture_qpos)
        lower, upper = self.lower.copy(), self.upper.copy()
        if (previous_qpos is None) != (max_step_rad is None):
            raise ValueError("previous_qpos and max_step_rad must be supplied together")
        if previous_qpos is not None:
            previous = self._pose(previous_qpos)
            step = np.asarray(max_step_rad, dtype=np.float64)
            if step.shape not in ((), (2, 5)) or not np.isfinite(step).all() or np.any(step <= 0):
                raise ValueError("max_step_rad must be finite positive scalar or [2,5]")
            lower = np.maximum(lower, previous[self.qpos_indices] - step)
            upper = np.minimum(upper, previous[self.qpos_indices] + step)
            if np.any(lower >= upper):
                raise ValueError("previous arm step box has no interior within native joint bounds")
        positions = np.asarray(target_positions_w, dtype=np.float64)
        if positions.shape != (2, 3) or not np.isfinite(positions).all():
            raise ValueError("hand target positions must be finite [2,3]")
        if target_quaternions_w is not None and target_rotations_w is not None:
            raise ValueError("pass hand quaternions or rotation matrices, not both")
        rotations = None
        if target_quaternions_w is not None:
            quats = np.asarray(target_quaternions_w, dtype=np.float64)
            if quats.shape != (2, 4) or not np.isfinite(quats).all() or not np.allclose(np.linalg.norm(quats, axis=1), 1., atol=1e-6, rtol=0):
                raise ValueError("hand quaternions must be normalized finite WXYZ[2,4]")
            rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
        elif target_rotations_w is not None:
            rotations = np.asarray(target_rotations_w, dtype=np.float64)
            if (rotations.shape != (2, 3, 3) or not np.isfinite(rotations).all()
                or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-6, rtol=0)
                or not np.allclose(np.linalg.det(rotations), 1., atol=1e-6, rtol=0)):
                raise ValueError("hand rotation matrices must be finite proper SO(3)[2,3,3]")
        evaluations, jac_evaluations, converged, costs = [], [], [], []
        for side in range(2):
            indices = self.qpos_indices[side]
            kwargs = dict(side=side, qpos=pose, target_position_w=positions[side],
                          target_rotation_w=None if rotations is None else rotations[side], posture=posture[indices])
            cache = {}

            def evaluate(joints):
                if "q" not in cache or not np.array_equal(joints, cache["q"]):
                    cache["q"] = joints.copy()
                    cache["value"] = self.residual_jacobian(joints, **kwargs)
                return cache["value"]

            initial = np.clip(pose[indices], lower[side], upper[side])
            residual, _ = evaluate(initial)
            if np.max(np.abs(residual)) < 1e-12:
                pose[indices] = initial
                evaluations.append(0)
                jac_evaluations.append(0)
                converged.append(True)
                costs.append(float(.5 * residual @ residual))
                continue
            result = least_squares(lambda x: evaluate(x)[0], initial, jac=lambda x: evaluate(x)[1],
                                   bounds=(lower[side], upper[side]), max_nfev=self.max_nfev,
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10)
            if not np.isfinite(result.x).all():
                raise ValueError("arm IK returned nonfinite joints")
            pose[indices] = result.x
            evaluations.append(result.nfev)
            jac_evaluations.append(result.njev)
            converged.append(result.success)
            costs.append(result.cost)
        achieved_positions, achieved_rotations = self.hand_poses(pose)
        angle_errors = (np.full(2, np.nan) if rotations is None else
                        Rotation.from_matrix(rotations.transpose(0, 2, 1) @ achieved_rotations).magnitude())
        return ArmIKResult(pose, np.linalg.norm(achieved_positions - positions, axis=1), angle_errors,
                           np.asarray(evaluations), np.asarray(jac_evaluations), np.asarray(converged), np.asarray(costs))


def solve_native23_arms(model, qpos, target_positions_w, target_quaternions_w=None, *,
                       target_rotations_w=None, posture_qpos=None, position_weight=1.,
                       orientation_weight=.03, posture_weight=.002, max_nfev=60,
                       previous_qpos=None, max_step_rad=None):
    """Convenience wrapper; reuse ``Native23ArmIK`` for a complete trajectory."""
    return Native23ArmIK(model, position_weight=position_weight, orientation_weight=orientation_weight,
                         posture_weight=posture_weight, max_nfev=max_nfev).solve(
        qpos, target_positions_w, target_quaternions_w, target_rotations_w=target_rotations_w,
        posture_qpos=posture_qpos, previous_qpos=previous_qpos, max_step_rad=max_step_rad)
