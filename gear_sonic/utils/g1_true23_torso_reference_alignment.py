"""SIM-only, frame-local compensation of missing waist and wrist references.

This is reference conversion, never measured state, a physical command, or a
qualified teacher. All29 source angles remain intact. The original29 torso
frame with missing axes zeroed defines a virtual pelvis-to-torso transform;
inverting that transform transfers waist roll/pitch into the reference pelvis.
The native torso's separate fixed geometric offset is measured, not erased.
Bounded native leg IK then fits the original world feet, and each five-axis arm
fits the original hand proxy. No future frames, speed change, state reanchoring,
extra actuators, policy changes, or full-trajectory feasibility claims.
"""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES, SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_23dof_task_space_retarget import safe_target_joint_bounds
from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks
from gear_sonic.utils.g1_true23_original29_reference import SOURCE_JOINT_NAMES


@dataclass(frozen=True)
class AlignmentConfig:
    leg_orientation_scale_m: float = 0.10
    arm_orientation_scale_m: float = 0.02
    leg_posture_scale_m: float = 1e-6
    arm_posture_scale_m: float = 0.003
    max_function_evaluations: int = 50

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"alignment setting must be finite positive: {name}")
        if type(self.max_function_evaluations) is not int:
            raise ValueError("alignment evaluation limit must be an integer")


def _pose(value, width):
    pose = np.asarray(value)
    if pose.shape != (width,) or pose.dtype.kind not in "fi" or not np.isfinite(pose).all():
        raise ValueError("alignment requires one finite real full source pose")
    if not np.isclose(np.linalg.norm(pose[3:7]), 1.0, atol=1e-6, rtol=0):
        raise ValueError("alignment requires normalized WXYZ, no implicit repair")
    return pose.astype(np.float64, copy=True)


class TorsoReferenceAlignment:
    """Deterministic one-frame reference transform with bounded limb IK.

    Roots have no imposed optimization box: the exact derived correction is
    returned for independent, unchanged trajectory/fidelity screens. This class
    cannot accept a motion or relax those screens on the caller's behalf.
    """

    def __init__(self, source_model, native_model, *, config=AlignmentConfig()):
        if (source_model.nq, source_model.nv, source_model.nu) != (36, 35, 29):
            raise ValueError("alignment requires exact original29 source topology")
        if (native_model.nq, native_model.nv, native_model.nu) != (30, 29, 23):
            raise ValueError("alignment requires exact native23 target topology")
        source_names = tuple(source_model.joint(i).name for i in range(1, source_model.njnt))
        native_names = tuple(native_model.joint(i).name for i in range(1, native_model.njnt))
        if source_names != SOURCE_JOINT_NAMES or native_names != HARDWARE_23_JOINT_NAMES:
            raise ValueError("alignment joint names/order differ from native23 contract")
        self.source, self.native, self.config = source_model, native_model, config
        self.source_data = mujoco.MjData(source_model)
        self.zero_data = mujoco.MjData(source_model)
        self.native_data = mujoco.MjData(native_model)
        self.keep = np.asarray(SOURCE_MJ29_KEEP_INDICES, dtype=int)
        self.missing = np.setdiff1d(np.arange(29), self.keep)
        self.lower, self.upper = safe_target_joint_bounds(
            native_model, native_action_clip=9.5, safe_limit_guard_rad=0.05
        )
        self.tasks, self.hand_convention = neutral_wrist_hand_tasks(source_model, native_model)
        self.limbs = [
            (next(task for task in self.tasks if task.name == name), np.asarray(indices))
            for name, indices in (
                ("left_foot", range(0, 6)),
                ("right_foot", range(6, 12)),
                ("left_hand", range(13, 18)),
                ("right_hand", range(18, 23)),
            )
        ]
        self.source_torso = source_model.body("torso_link").id
        self.native_torso = native_model.body("torso_link").id

    def pelvis_pose(self, source_pose):
        """Invert actual FK chains, not assumed Euler orders or body offsets."""
        source_pose = _pose(source_pose, 36)
        self.source_data.qpos[:] = source_pose
        self.zero_data.qpos[:] = 0
        self.zero_data.qpos[3] = 1
        self.zero_data.qpos[7 + self.keep] = source_pose[7 + self.keep]
        mujoco.mj_fwdPosition(self.source, self.source_data)
        mujoco.mj_fwdPosition(self.source, self.zero_data)
        desired_rotation = self.source_data.xmat[self.source_torso].reshape(3, 3)
        relative_rotation = self.zero_data.xmat[self.source_torso].reshape(3, 3)
        root_rotation = desired_rotation @ relative_rotation.T
        root_position = (
            self.source_data.xpos[self.source_torso] - root_rotation @ self.zero_data.xpos[self.source_torso]
        )
        quaternion = Rotation.from_matrix(root_rotation).as_quat()[[3, 0, 1, 2]]
        # Choose the source hemisphere for reproducible interpolation. This is
        # reference quaternion sign, not a measured pose or time-series filter.
        if np.dot(quaternion, source_pose[3:7]) < 0:
            quaternion = -quaternion
        if np.all(source_pose[7 + self.missing] == 0):
            # Exact identity is important for unchanged stationary acquisition.
            # Check the geometric assertion before retaining original bits.
            np.testing.assert_allclose(root_position, source_pose[:3], atol=1e-12, rtol=0)
            np.testing.assert_allclose(quaternion, source_pose[3:7], atol=1e-12, rtol=0)
            return source_pose[:7].copy()
        return np.r_[root_position, quaternion]

    def limb_residual_jacobian(self, joints, *, pose, task, indices, desired, posture):
        """World point + rotation-matrix residual and exact hinge Jacobian."""
        data = self.native_data
        data.qpos[:] = pose
        data.qpos[7 + indices] = joints
        mujoco.mj_fwdPosition(self.native, data)
        body = self.native.body(task.target_body).id
        rotation = data.xmat[body].reshape(3, 3)
        point = data.xpos[body] + rotation @ task.target_point
        leg = task.name.endswith("foot")
        rotation_scale = (
            self.config.leg_orientation_scale_m if leg else self.config.arm_orientation_scale_m
        ) / np.sqrt(2)
        posture_scale = self.config.leg_posture_scale_m if leg else self.config.arm_posture_scale_m
        residual = np.r_[
            point - desired[0],
            rotation_scale * (rotation - desired[1]).T.ravel(),
            posture_scale * (joints - posture),
        ]
        jacp, jacr = np.zeros((3, self.native.nv)), np.zeros((3, self.native.nv))
        mujoco.mj_jac(self.native, data, jacp, jacr, point, body)
        columns = 6 + indices
        angular = jacr[:, columns]
        rotation_jac = np.concatenate([np.cross(angular.T, rotation[:, column]).T for column in range(3)], axis=0)
        jacobian = np.vstack(
            (jacp[:, columns], rotation_scale * rotation_jac, posture_scale * np.eye(len(indices)))
        )
        return residual, jacobian

    def convert(self, source_pose):
        source_pose = _pose(source_pose, 36)
        original_joints = source_pose[7 + self.keep]
        pose = np.r_[self.pelvis_pose(source_pose), original_joints]
        # The unoptimized waist yaw is not silently clipped to make a receipt.
        if not self.lower[12] <= pose[19] <= self.upper[12]:
            raise ValueError("retained waist yaw outside unchanged reference envelope")
        iterations = []
        for task, indices in self.limbs:
            body = self.source.body(task.source_body).id
            desired_rotation = self.source_data.xmat[body].reshape(3, 3).copy()
            desired_point = self.source_data.xpos[body] + desired_rotation @ task.source_point
            kwargs = dict(
                pose=pose,
                task=task,
                indices=indices,
                desired=(desired_point, desired_rotation),
                posture=original_joints[indices],
            )
            initial = np.clip(original_joints[indices], self.lower[indices], self.upper[indices])
            residual, _ = self.limb_residual_jacobian(initial, **kwargs)
            if np.max(np.abs(residual)) < 1e-12:
                pose[7 + indices] = initial
                iterations.append(0)
                continue
            result = least_squares(
                lambda q: self.limb_residual_jacobian(q, **kwargs)[0],
                initial,
                jac=lambda q: self.limb_residual_jacobian(q, **kwargs)[1],
                bounds=(self.lower[indices], self.upper[indices]),
                max_nfev=self.config.max_function_evaluations,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            if not np.isfinite(result.x).all():
                raise ValueError("bounded reference IK returned nonfinite joints")
            pose[7 + indices] = result.x
            iterations.append(int(result.nfev))
        if np.any(pose[7:] < self.lower) or np.any(pose[7:] > self.upper):
            raise ValueError("reference IK escaped unchanged joint envelope")
        return pose, {"limb_function_evaluations": iterations, "kinematic_or_dynamic_acceptance": False}

    def contract(self):
        return dict(
            kind="native23_frame_local_torso_reference_alignment_v1",
            config=asdict(self.config),
            lower_joint_bounds_rad=self.lower.tolist(),
            upper_joint_bounds_rad=self.upper.tolist(),
            source_joint_names=list(SOURCE_JOINT_NAMES),
            native_joint_names=list(HARDWARE_23_JOINT_NAMES),
            source_frame_or_timing_removed=False,
            previous_or_future_frames_consumed=False,
            original_reference_preserved_for_independent_scoring=True,
            native_static_torso_offset_removed=False,
            exact_virtual_torso_not_full_task_or_feasibility_claim=True,
            independent_trajectory_and_contact_screens_required=True,
            hand_convention=self.hand_convention,
            hardware_authorized=False,
            deployment_ready=False,
        )
