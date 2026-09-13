"""Explicit neutral-wrist hand proxies for diagnostic native23 task fitting.

The source SONIC proxy is 18 cm from wrist-yaw, not wrist-roll. Native23 has
only wrist-roll; applying the same numeric offset to that body shortens the
task arm by the two removed link translations. Derive the neutral transform
from the models instead. Source trajectories retain every missing-joint angle.

This is a coordinate convention, NOT a contact landmark or identical hand
geometry claim. It deliberately changes neither DEFAULT_TASKS nor the VR,
causal-feature, training, ONNX or deployed native23 interfaces. Those require
a separately versioned, end-to-end migration before using a new convention.
"""

from __future__ import annotations

from dataclasses import asdict, replace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils import g1_23dof_task_space_retarget as retarget
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_original_task_trajectory import OriginalTaskConfig, OriginalTaskPath

HAND_FRAME_CONVENTION = "native23_source_neutral_wrist_hand_proxy_v1"


def neutral_wrist_hand_tasks(source, target):
    """Return a new task tuple and explicit derivation; never mutate the models."""
    if (source.nq, source.nv, target.nq, target.nv) != (36, 35, 30, 29):
        raise ValueError("hand-frame derivation requires original29/native23 models")
    source_layout, target_layout = retarget._model_layout(source), retarget._model_layout(target)
    if target_layout.joint_names != tuple(HARDWARE_23_JOINT_NAMES) or not set(target_layout.joint_names).issubset(
        source_layout.joint_names
    ):
        raise ValueError("hand-frame derivation requires the exact shared native23 joints")
    tasks = retarget.validate_tasks(source, target, retarget.DEFAULT_TASKS)
    source_data, target_data = mujoco.MjData(source), mujoco.MjData(target)
    for model, data in ((source, source_data), (target, target_data)):
        data.qpos[:] = 0
        data.qpos[2:4] = [0.8, 1]
        mujoco.mj_fwdPosition(model, data)
    corrected, evidence = [], []
    for task in tasks:
        if task.name not in ("left_hand", "right_hand"):
            corrected.append(task)
            continue
        side = task.name.split("_")[0]
        source_roll = int(source.joint(f"{side}_wrist_roll_joint").bodyid[0])
        pitch = int(source.joint(f"{side}_wrist_pitch_joint").bodyid[0])
        yaw = int(source.joint(f"{side}_wrist_yaw_joint").bodyid[0])
        target_roll = int(target.joint(f"{side}_wrist_roll_joint").bodyid[0])
        if (
            source.body(task.source_body).id != yaw
            or target.body(task.target_body).id != target_roll
            or source.body_parentid[yaw] != pitch
            or source.body_parentid[pitch] != source_roll
        ):
            raise ValueError("hand task no longer matches the removed pitch/yaw wrist chain")
        source_rotation = source_data.xmat[source_roll].reshape(3, 3)
        target_rotation = target_data.xmat[target_roll].reshape(3, 3)
        if not np.allclose(
            source_data.xpos[source_roll], target_data.xpos[target_roll], atol=1e-9, rtol=0
        ) or not (np.allclose(source_rotation, target_rotation, atol=1e-9, rtol=0)):
            raise ValueError("neutral shared wrist-roll frames do not coincide; cannot assume a mapping")
        source_point, hand_rotation = retarget._point_pose(source_data, yaw, task.source_point)
        relative_rotation = source_rotation.T @ hand_rotation
        if not np.allclose(relative_rotation, np.eye(3), atol=1e-9, rtol=0):
            raise ValueError("nonidentity neutral hand rotation needs an explicit orientation convention")
        local = source_rotation.T @ (source_point - source_data.xpos[source_roll])
        new_task = replace(task, target_point=tuple(float(x) for x in local))
        corrected.append(new_task)
        legacy_point, _ = retarget._point_pose(target_data, target_roll, task.target_point)
        new_point, _ = retarget._point_pose(target_data, target_roll, new_task.target_point)
        evidence.append(
            {
                "name": task.name,
                "source_roll_body": source.body(source_roll).name,
                "source_task_body": task.source_body,
                "target_roll_body": task.target_body,
                "source_point_unchanged": list(task.source_point),
                "legacy_target_point": list(task.target_point),
                "corrected_target_point": list(new_task.target_point),
                "neutral_source_hand_rotation_in_roll": relative_rotation.tolist(),
                "neutral_legacy_position_error_m": float(np.linalg.norm(legacy_point - source_point)),
                "neutral_corrected_position_error_m": float(np.linalg.norm(new_point - source_point)),
            }
        )
    if len(evidence) != 2:
        raise ValueError("both hand tasks must remain present")
    corrected = retarget.validate_tasks(source, target, corrected)
    return corrected, {
        "task_point_convention": HAND_FRAME_CONVENTION,
        "hands": evidence,
        "tasks": [asdict(task) for task in corrected],
        "source_task_points_and_actual_source_joint_motion_unchanged": True,
        "default_teleop_and_checkpoint_convention_modified": False,
        "proxy_is_a_physical_contact_landmark": False,
        "source_and_target_hand_meshes_identical": False,
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


class NeutralWristHandTaskPath(OriginalTaskPath):
    """Same all-frame SE(3)/23-joint problem with explicit target-only proxies."""

    def __init__(self, source_model, target_model, source_qpos, seed_joints, *, config=OriginalTaskConfig()):
        tasks, evidence = neutral_wrist_hand_tasks(source_model, target_model)
        super().__init__(source_model, target_model, source_qpos, seed_joints, config=config)
        # Parent caches targets only from source_body/source_point. Verify those
        # fields and task order before retaining the exact original source cache.
        for old, new in zip(self.tasks, tasks, strict=True):
            if replace(new, target_point=old.target_point) != old:
                raise ValueError("hand-frame correction changed more than target point coordinates")
        self.tasks = tasks
        self.hand_frame_evidence = evidence

    def metrics(self, variables):
        return {**super().metrics(variables), "task_point_convention": HAND_FRAME_CONVENTION}


def task_pose_errors(source, target, source_qpos, native_qpos, *, tasks):
    """Measure both conventions explicitly; no substitution of cached source FK."""
    tasks = retarget.validate_tasks(source, target, tasks)
    source_qpos, native_qpos = np.asarray(source_qpos), np.asarray(native_qpos)
    if (
        source_qpos.ndim != 2
        or source_qpos.shape[1] != source.nq
        or len(source_qpos) < 1
        or native_qpos.shape != (len(source_qpos), target.nq)
        or not np.isfinite(source_qpos).all()
        or not np.isfinite(native_qpos).all()
        or not np.allclose(np.linalg.norm(source_qpos[:, 3:7], axis=1), 1, atol=1e-6, rtol=0)
        or not np.allclose(np.linalg.norm(native_qpos[:, 3:7], axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("task errors need finite complete matching source/native pose arrays")
    source_data, target_data = mujoco.MjData(source), mujoco.MjData(target)
    positions, rotations = [], []
    for source_pose, target_pose in zip(source_qpos, native_qpos, strict=True):
        source_data.qpos[:] = source_pose
        target_data.qpos[:] = target_pose
        mujoco.mj_fwdPosition(source, source_data)
        mujoco.mj_fwdPosition(target, target_data)
        source_pos, source_quat = retarget._task_pose_arrays(source, source_data, tasks, source=True)
        target_pos, target_quat = retarget._task_pose_arrays(target, target_data, tasks, source=False)
        positions.append(np.linalg.norm(source_pos - target_pos, axis=1))
        rotations.append(
            (
                Rotation.from_quat(source_quat[:, [1, 2, 3, 0]]).inv()
                * Rotation.from_quat(target_quat[:, [1, 2, 3, 0]])
            ).magnitude()
        )
    positions, rotations = np.asarray(positions), np.asarray(rotations)
    return {
        task.name: {
            "position_max_m": float(positions[:, i].max()),
            "position_mean_m": float(positions[:, i].mean()),
            "orientation_max_rad": float(rotations[:, i].max()),
        }
        for i, task in enumerate(tasks)
    }
