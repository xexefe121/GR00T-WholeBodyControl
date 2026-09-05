"""Whole-path geometric floor conditioning of native23 references.

This preserves every joint sample, phase and orientation. Only a smooth root-z
offset may change. Floor clearance is not support-contact/COM optimization,
force feasibility, tracking quality, or deployment authorization.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from types import SimpleNamespace

import mujoco
import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_trajectory_projection import (
    audit_trajectory_constraints,
    project_nearest_trajectory,
)
from gear_sonic.utils.g1_true23_contact_geometry import audit_reset_contacts, lift_reset_floor_overlap
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion


def compiled_model_sha256(model):
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def motion_qpos(model, motion):
    count = validate_library_motion(motion)
    joints = [i for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    names = tuple(model.joint(i).name for i in joints)
    if names != tuple(HARDWARE_23_JOINT_NAMES) or model.nq != 30:
        raise ValueError("reference geometry requires exact unprefixed native23 joint layout")
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1 or model.jnt_qposadr[free[0]] != 0:
        raise ValueError("reference geometry requires one leading floating root")
    if model.body(int(model.jnt_bodyid[free[0]])).name != "pelvis":
        raise ValueError("reference root must be pelvis")
    result = np.tile(model.qpos0, (count, 1))
    result[:, :3] = motion["body_pos_w"][:, 0]
    result[:, 3:7] = motion["body_quat_w"][:, 0]
    result[:, model.jnt_qposadr[joints]] = motion["joint_pos"]
    return result


def reference_geometry(model, motion):
    qpos = motion_qpos(model, motion)
    contacts = audit_reset_contacts(model, qpos)
    distances = np.array(
        [
            row["minimum_floor_contact_distance_m"]
            if row["minimum_floor_contact_distance_m"] is not None
            else np.nan
            for row in contacts["rows"]
        ]
    )
    overlapping = np.isfinite(distances) & (distances < 0)
    return {
        "compiled_mjb_sha256": compiled_model_sha256(model),
        "frames": len(qpos),
        "frames_with_floor_overlap": int(overlapping.sum()),
        "worst_floor_overlap_m": float(max(0, -np.nanmin(distances))) if np.isfinite(distances).any() else 0.0,
        "floor_contact_audit": contacts,
    }


def condition_reference_floor(
    motion,
    models,
    *,
    output_model,
    clearance_m=1e-5,
    maximum_lift_m=0.2,
    maximum_offset_velocity_m_s=0.75,
    maximum_offset_acceleration_m_s2=6.0,
):
    """Bounded minimum-norm root-z path clearing every supplied flat model.

    Smoothness bounds apply to the correction, not the original root path.
    The source clip remains whole and at original tempo. The independent
    post-serialization contact check is authoritative for floor clearance.
    """
    count = validate_library_motion(motion)
    if not models:
        raise ValueError("floor conditioning requires explicit collision models")
    if not np.isfinite([clearance_m, maximum_lift_m]).all() or not 0 < clearance_m < maximum_lift_m:
        raise ValueError("floor conditioning requires finite ordered positive bounds")
    input_fk = audit_reference_kinematics(SimpleNamespace(module=mujoco, model=output_model), motion)
    if not input_fk["position_fk_consistent"] or not input_fk["orientation_fk_consistent"]:
        raise ValueError("source body channels are inconsistent with native23 kinematics")
    before = {name: reference_geometry(model, motion) for name, model in models.items()}
    lower = np.zeros(count)
    for name, model in models.items():
        # Reuse the tested topology/plane-direction check. This does not apply
        # its independent per-frame lifts to the trajectory being conditioned.
        lift_reset_floor_overlap(
            model, motion_qpos(model, motion)[:1], clearance_m=clearance_m, maximum_lift_m=maximum_lift_m
        )
        for i, row in enumerate(before[name]["floor_contact_audit"]["rows"]):
            distance = row["minimum_floor_contact_distance_m"]
            if distance is not None and distance < 0:
                lower[i] = max(lower[i], clearance_m - distance)
    if np.any(lower > maximum_lift_m):
        raise ValueError("reference penetration exceeds maximum lift; no frames may be dropped")
    upper = np.full((count, 1), maximum_lift_m)
    # Reserve float32 serialization margin. Initial correction velocity is zero;
    # unlike retiming this does not change the clip duration or phase mapping.
    projection = project_nearest_trajectory(
        np.zeros((count, 1)),
        lower_bounds=lower[:, None],
        upper_bounds=upper,
        dt=0.02,
        max_velocity=maximum_offset_velocity_m_s * 0.995,
        max_acceleration=maximum_offset_acceleration_m_s2 * 0.995,
    )
    root = motion["body_pos_w"][:, 0].astype(np.float64).copy()
    root[:, 2] += projection.projected_path[:, 0]
    root = root.astype(np.float32)
    actual_offset = root[:, 2].astype(np.float64) - motion["body_pos_w"][:, 0, 2].astype(np.float64)
    audit = audit_trajectory_constraints(
        actual_offset[:, None],
        lower_bounds=lower[:, None],
        upper_bounds=upper,
        dt=0.02,
        max_velocity=maximum_offset_velocity_m_s,
        max_acceleration=maximum_offset_acceleration_m_s2,
        tolerance=1e-7,
    )
    if not audit.passed:
        raise ValueError("serialized root correction failed trajectory bounds")
    # Rigid vertical translation: do not silently re-estimate existing joint
    # or angular velocity channels. Only add the correction's vertical
    # derivative (the existing body-velocity central-difference convention).
    arrays = {key: value.copy() for key, value in motion.items()}
    arrays["body_pos_w"][:, :, 2] = (
        motion["body_pos_w"][:, :, 2].astype(np.float64) + actual_offset[:, None]
    ).astype(np.float32)
    arrays["body_lin_vel_w"][:, :, 2] = (
        motion["body_lin_vel_w"][:, :, 2].astype(np.float64) + np.gradient(actual_offset, 0.02)[:, None]
    ).astype(np.float32)
    validate_library_motion(arrays)
    if not np.array_equal(arrays["joint_pos"], motion["joint_pos"]):
        raise RuntimeError("floor conditioning changed source joint samples")
    if not np.array_equal(arrays["body_pos_w"][:, 0, :2], motion["body_pos_w"][:, 0, :2]):
        raise RuntimeError("floor conditioning changed horizontal root path")
    after = {name: reference_geometry(model, arrays) for name, model in models.items()}
    if any(report["frames_with_floor_overlap"] for report in after.values()):
        raise ValueError("serialized corrected path retains floor overlap")
    root_q_error = np.minimum(
        np.linalg.norm(arrays["body_quat_w"][:, 0] - motion["body_quat_w"][:, 0], axis=1),
        np.linalg.norm(arrays["body_quat_w"][:, 0] + motion["body_quat_w"][:, 0], axis=1),
    )
    if np.max(root_q_error) > 2e-7:
        raise RuntimeError("floor conditioning changed root orientation")
    report = {
        "kind": "g1_true23_whole_path_floor_conditioning_v1",
        "source_frame_count": count,
        "output_frame_count": len(arrays["joint_pos"]),
        "frames_removed": 0,
        "controlled_joint_count": 23,
        "maximum_joint_sample_change_rad": 0.0,
        "joint_velocity_and_all_orientation_channels_preserved": True,
        "vertical_velocity_correction_convention": "numpy_gradient_at_50hz",
        "input_kinematic_audit": input_fk,
        "time_scale": 1.0,
        "root_z_offsets_m": actual_offset.tolist(),
        "maximum_root_lift_m": float(actual_offset.max()),
        "minimum_required_lifts_m": lower.tolist(),
        "correction_bounds": {
            "maximum_lift_m": maximum_lift_m,
            "clearance_m": clearance_m,
            "maximum_offset_velocity_m_s": maximum_offset_velocity_m_s,
            "maximum_offset_acceleration_m_s2": maximum_offset_acceleration_m_s2,
            "initial_offset_velocity_m_s": 0.0,
            "original_root_derivatives_bounded": False,
        },
        "projection_iterations": projection.iterations,
        "serialized_correction_audit": asdict(audit),
        "before": before,
        "after": after,
        "geometric_floor_clearance_passed": True,
        "support_contacts_or_com_optimized": False,
        "self_collision_free_proven": False,
        "dynamic_feasibility_proven": False,
        "full_clip_tracking_qualified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    return arrays, report
