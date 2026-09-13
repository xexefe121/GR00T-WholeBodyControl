"""One fixed planar source registration before an offline simulation replay.

Joint motion, height, timing, relative travel and all internal geometry remain
unchanged. All world-space channels receive the SAME fixed SE(2) transform.
Nothing follows measured robot state during replay. This module has no robot
transport, controller reset, live calibration, or qualification authority.
"""

from copy import deepcopy

import numpy as np


def register_motion_start(motion, *, target_root_xy=(0.0, 0.0), target_yaw_rad=0.0):
    target = np.asarray(target_root_xy, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all() or not np.isfinite(target_yaw_rad):
        raise ValueError("start registration requires finite target XY and heading")
    positions = np.asarray(motion["body_pos_w"])
    quaternions = np.asarray(motion["body_quat_w"])
    joints = np.asarray(motion["joint_pos"])
    n = len(joints)
    if (
        n < 2
        or joints.shape != (n, 23)
        or positions.shape != (n, 24, 3)
        or quaternions.shape != (n, 24, 4)
        or not all(
            value.dtype.kind == "f" and np.isfinite(value).all() for value in (joints, positions, quaternions)
        )
        or not np.allclose(np.linalg.norm(quaternions, axis=-1), 1.0, atol=1e-5, rtol=0)
    ):
        raise ValueError("start registration requires complete finite native23 physical body channels")
    for key in ("body_lin_vel_w", "body_ang_vel_w"):
        value = np.asarray(motion[key])
        if value.shape != positions.shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError("start registration requires matching finite world velocity channels")
    joint_velocity = np.asarray(motion["joint_vel"])
    if joint_velocity.shape != joints.shape or not np.isfinite(joint_velocity).all():
        raise ValueError("start registration requires unchanged finite native23 joint velocities")
    w, x, y, z = np.asarray(quaternions[0, 0], dtype=float)
    source_yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    angle = float(np.arctan2(np.sin(target_yaw_rad - source_yaw), np.cos(target_yaw_rad - source_yaw)))
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0.0, 0.0, 1.0]])
    translation = np.zeros(3)
    translation[:2] = target - (rotation @ positions[0, 0])[:2]
    result = deepcopy(motion)
    result["body_pos_w"] = (np.asarray(positions, dtype=float) @ rotation.T + translation).astype(positions.dtype)
    for key in ("body_lin_vel_w", "body_ang_vel_w"):
        result[key] = (np.asarray(motion[key], dtype=float) @ rotation.T).astype(motion[key].dtype)
    # Left multiplication by yaw quaternion rotates every body in the world.
    qw, qx, qy, qz = np.moveaxis(np.asarray(quaternions, dtype=float), -1, 0)
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    result["body_quat_w"] = np.stack(
        (c * qw - s * qz, c * qx - s * qy, c * qy + s * qx, c * qz + s * qw), axis=-1
    ).astype(quaternions.dtype)
    proof = {
        "kind": "g1_true23_once_only_source_start_se2_registration_v1",
        "source_first_root_xy": positions[0, 0, :2].tolist(),
        "source_first_root_heading_rad": source_yaw,
        "target_root_xy": target.tolist(),
        "target_heading_rad": float(target_yaw_rad),
        "fixed_world_yaw_rotation_rad": angle,
        "fixed_world_translation_m": translation.tolist(),
        "frame_count": n,
        "joint_positions_and_velocities_bit_exact": all(
            np.array_equal(result[key], motion[key]) for key in ("joint_pos", "joint_vel")
        ),
        "world_height_channels_bit_exact": np.array_equal(result["body_pos_w"][..., 2], positions[..., 2]),
        "all_world_positions_orientations_and_velocities_transformed_consistently": True,
        "reference_world_arrays_bit_exact": False,
        "full_source_frames_and_timing_preserved": True,
        "relative_motion_excursion_scaled": False,
        "measured_robot_state_used": False,
        "midrun_reregistration_or_pose_reset": False,
        "same_unregistered_benchmark_claimed": False,
        "registered_reference_geometric_or_dynamic_qualification": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    return result, proof
