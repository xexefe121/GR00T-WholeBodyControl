"""Separate row-deletion geometry from measured tracking, without actuation.

The ideal native poses exactly retain source root and 23 joint angles. They
are forward kinematics, not simulated motion, a teacher or a reachability bound.
Another retargeter may move retained joints and obtain different task errors.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES, SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks
from gear_sonic.utils.g1_true23_original29_reference import SOURCE_JOINT_NAMES, VR_BODIES, VR_OFFSETS

TASK_NAMES = ("left_hand", "right_hand", "head_proxy")
WAIST = ("waist_roll_joint", "waist_pitch_joint")
WRISTS = (
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _poses(value, width):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] != width or not len(value) or not np.isfinite(value).all():
        raise ValueError("requires finite nonempty full qpos matrix")
    if not np.allclose(np.linalg.norm(value[:, 3:7], axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("requires normalized source and measured root quaternions")
    return value.astype(np.float64, copy=True)


def task_fk(model, poses, bodies, offsets):
    """Evaluate copied poses in private MjData; never integrate or change model."""
    poses = _poses(poses, model.nq)
    ids = [model.body(name).id for name in bodies]
    data = mujoco.MjData(model)
    positions, quaternions = [], []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_fwdPosition(model, data)
        positions.append(
            [
                data.xpos[b] + data.xmat[b].reshape(3, 3) @ offset
                for b, offset in zip(ids, offsets, strict=True)
            ]
        )
        quaternions.append(data.xquat[ids].copy())
    return np.asarray(positions), np.asarray(quaternions)


def geometry_arrays(source_model, native_model, source_qpos29):
    poses = _poses(source_qpos29, 36)
    if (source_model.nq, source_model.nv, source_model.nu) != (36, 35, 29):
        raise ValueError("requires exact original29 source model")
    if (native_model.nq, native_model.nv, native_model.nu) != (30, 29, 23):
        raise ValueError("requires exact physical native23 model")
    source_names = tuple(source_model.joint(i).name for i in range(1, source_model.njnt))
    native_names = tuple(native_model.joint(i).name for i in range(1, native_model.njnt))
    if source_names != SOURCE_JOINT_NAMES or native_names != HARDWARE_23_JOINT_NAMES:
        raise ValueError("model joint identities or order differ")
    if set(source_names) - set(native_names) != set(WAIST + WRISTS):
        raise ValueError("unexpected missing native axes")
    native_q = np.concatenate((poses[:, :7], poses[:, 7:][:, SOURCE_MJ29_KEEP_INDICES]), axis=1)
    tasks, convention = neutral_wrist_hand_tasks(source_model, native_model)
    tasks = [next(task for task in tasks if task.name == name) for name in TASK_NAMES]
    native_bodies = tuple(task.target_body for task in tasks)
    native_offsets = np.asarray([task.target_point for task in tasks])
    out = dict(source_qpos29=poses, ideal_native_qpos23=native_q)
    for key, axes in (
        ("original", ()),
        ("waist_zero", WAIST),
        ("wrists_zero", WRISTS),
        ("all_missing_zero", WAIST + WRISTS),
    ):
        value = poses.copy()
        for name in axes:
            value[:, 7 + source_names.index(name)] = 0
        out[key + "_position_w"], out[key + "_quaternion_wxyz"] = task_fk(
            source_model, value, VR_BODIES, VR_OFFSETS
        )
    out["ideal_native_position_w"], out["ideal_native_quaternion_wxyz"] = task_fk(
        native_model, native_q, native_bodies, native_offsets
    )
    # The actual native torso frame is 1 cm lower at neutral than the zeroed
    # source torso frame. Joint deletion alone therefore does not reproduce
    # every native task. Measure this structural term instead of assuming it
    # is zero or changing either model/landmark to manufacture equality.
    out["native_minus_zeroed_source_position_w"] = (
        out["ideal_native_position_w"] - out["all_missing_zero_position_w"]
    )
    return out, dict(
        task_order=list(TASK_NAMES),
        native_bodies=list(native_bodies),
        native_offsets=native_offsets.tolist(),
        hand_convention=convention,
        waist_axes=list(WAIST),
        wrist_axes=list(WRISTS),
        pure_missing_axes_and_structural_native_geometry_separated=True,
        source_sample_times_or_retained_pose_changed=False,
        kinematic_not_integrated=True,
        optimal_retarget_or_reachability_bound=False,
        deployment_ready=False,
        hardware_authorized=False,
    )


def position_summary(error):
    error = np.asarray(error)
    if error.ndim != 3 or error.shape[1:] != (3, 3) or not len(error) or not np.isfinite(error).all():
        raise ValueError("task errors require finite nonempty [frames,3 tasks,3 coordinates]")
    norms = np.linalg.norm(error, axis=-1)
    return dict(
        p95_m=np.percentile(norms, 95, axis=0).tolist(),
        rms_m=np.sqrt(np.mean(norms**2, axis=0)).tolist(),
        max_m=norms.max(axis=0).tolist(),
    )


def angle_summary(actual_wxyz, desired_wxyz):
    a, b = np.asarray(actual_wxyz), np.asarray(desired_wxyz)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (3, 4) or not len(a):
        raise ValueError("task rotations require matching [frames,3 tasks,4]")
    rotations = Rotation.from_quat(a.reshape(-1, 4)[:, [1, 2, 3, 0]]).inv() * Rotation.from_quat(
        b.reshape(-1, 4)[:, [1, 2, 3, 0]]
    )
    values = rotations.magnitude().reshape(-1, 3)
    return dict(
        p95_rad=np.percentile(values, 95, axis=0).tolist(),
        rms_rad=np.sqrt(np.mean(values**2, axis=0)).tolist(),
    )


def decompose_position_error(
    actual_centered, ideal_centered, source_centered, *, zeroed_source_centered=None
):
    actual, ideal, source = (
        np.asarray(value, dtype=np.float64) for value in (actual_centered, ideal_centered, source_centered)
    )
    if not (actual.shape == ideal.shape == source.shape):
        raise ValueError("position decomposition needs identically phased task arrays")
    total, controller, geometry = actual - source, actual - ideal, ideal - source
    for value in (total, controller, geometry):
        position_summary(value)
    identity_error = float(np.max(np.abs(total - controller - geometry)))
    total_mse = np.mean(np.sum(total**2, axis=-1), axis=0)
    controller_mse = np.mean(np.sum(controller**2, axis=-1), axis=0)
    geometry_mse = np.mean(np.sum(geometry**2, axis=-1), axis=0)
    cross = 2 * np.mean(np.sum(controller * geometry, axis=-1), axis=0)
    np.testing.assert_allclose(total_mse, controller_mse + geometry_mse + cross, atol=1e-12, rtol=0)
    if identity_error > 1e-12:
        raise ValueError("position vector decomposition does not close")
    result = dict(
        total=position_summary(total),
        controller_to_retained_pose=position_summary(controller),
        reference_embodiment_gap=position_summary(geometry),
        vector_identity_max_error_m=identity_error,
        total_mse_m2=total_mse.tolist(),
        controller_mse_m2=controller_mse.tolist(),
        geometry_mse_m2=geometry_mse.tolist(),
        signed_cross_term_m2=cross.tolist(),
        percentile_components_are_not_additive=True,
        causal_attribution_fraction_claimed=False,
        controller_term_includes_root_orientation_error=True,
        global_translation_removed_but_world_axes_preserved=True,
    )
    if zeroed_source_centered is not None:
        zeroed = np.asarray(zeroed_source_centered, dtype=np.float64)
        if zeroed.shape != source.shape:
            raise ValueError("zeroed-source geometry must use the same phased task arrays")
        missing, structural = zeroed - source, ideal - zeroed
        np.testing.assert_allclose(geometry, missing + structural, atol=1e-12, rtol=0)
        result["pure_missing_axes"] = position_summary(missing)
        result["structural_native_minus_zeroed_source"] = position_summary(structural)
    return result
