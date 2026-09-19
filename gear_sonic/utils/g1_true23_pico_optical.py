"""Strict, explicitly optical SMPL reference from a paired PICO recording.

No sensor-only reconstruction, robot transport, or physical actuation occurs.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)


def global_rotations(local):
    local = np.asarray(local, dtype=np.float64)
    if local.ndim != 4 or local.shape[1:] != (24, 3, 3) or not np.isfinite(local).all():
        raise ValueError("requires finite SMPL24 local rotation matrices")
    result = local.copy()
    for joint, parent in enumerate(PARENTS[1:], 1):
        result[:, joint] = result[:, parent] @ local[:, joint]
    return result


def validate_pair(gt, sensors):
    """Validate released numeric arrays; do not substitute optical data for sensors."""
    expected_gt = dict(betas=(10,), global_orient=(3,), body_pose=(69,), transl=(3,), joints=(24, 3))
    expected_sensor = dict(sensor_coordinates=(3, 3), sensor_orientation=(5, 3, 3), sensor_acceleration=(5, 3))
    count = len(gt["joints"])
    if count < 12:
        raise ValueError("recording too short")
    for group, expected in ((gt, expected_gt), (sensors, expected_sensor)):
        for name, tail in expected.items():
            value = np.asarray(group[name])
            if value.shape != (count, *tail) or value.dtype.kind != "f" or not np.isfinite(value).all():
                raise ValueError(f"invalid paired tensor: {name}")
    if gt.get("gender") not in ("female", "male", "neutral"):
        raise ValueError("unknown published SMPL gender")
    if np.max(np.abs(gt["betas"] - gt["betas"][:1])) > 1e-7:
        raise ValueError("subject shape varies within recording")
    matrices = np.asarray(sensors["sensor_orientation"], dtype=np.float64)
    orthogonality = float(np.max(np.abs(matrices.swapaxes(-1, -2) @ matrices - np.eye(3))))
    determinant = float(np.max(np.abs(np.linalg.det(matrices) - 1)))
    if max(orthogonality, determinant) > 1e-5:
        raise ValueError("sensor orientation is not a proper rotation")
    aa = np.concatenate((gt["global_orient"], gt["body_pose"]), axis=1).reshape(count, 24, 3)
    local = Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(count, 24, 3, 3)
    world = global_rotations(local)
    positions = np.asarray(gt["joints"], dtype=np.float64)
    offsets = np.zeros_like(positions)
    for joint, parent in enumerate(PARENTS[1:], 1):
        offsets[:, joint] = np.einsum("nij,nj->ni", world[:, parent].swapaxes(-1, -2), positions[:, joint] - positions[:, parent])
    error = np.linalg.norm(offsets - offsets[:1], axis=-1)
    maximum = float(error.max())
    if maximum > 1e-5:
        raise ValueError(f"optical rotation/position FK mismatch: {maximum} m")
    head_gap = np.linalg.norm(sensors["sensor_coordinates"][:, 0] - positions[:, 15], axis=1)
    return local, dict(
        frames=count, optical_local_rotation_world_position_coherence_passed=True,
        rest_offset_max_variation_m=maximum, coherence_tolerance_m=1e-5,
        sensor_rotation_orthogonality_max=orthogonality, sensor_rotation_determinant_max=determinant,
        sensor_head_to_optical_head_distance_p95_m=float(np.percentile(head_gap, 95)),
        sensor_and_optical_head_same_physical_point_assumed=False,
        sensor_timestamps_available=False, sensor_only_full_body_reconstruction=False,
    )


def resample_optical(positions, local, source_hz=60, target_hz=50):
    """Time-preserving whole-duration interpolation; never extrapolate an endpoint."""
    positions = np.asarray(positions, dtype=np.float64)
    if source_hz <= 0 or target_hz <= 0 or len(positions) != len(local):
        raise ValueError("invalid paired resampling rates or lengths")
    times = np.arange(len(positions), dtype=np.float64) / source_hz
    queries = np.arange(int(np.floor(times[-1] * target_hz)) + 1, dtype=np.float64) / target_hz
    sampled = np.stack([np.interp(queries, times, row) for row in positions.reshape(len(positions), -1).T], axis=1).reshape(-1, 24, 3)
    rotations = np.stack([Slerp(times, Rotation.from_matrix(local[:, j]))(queries).as_matrix() for j in range(24)], axis=1)
    return sampled, global_rotations(rotations), times, queries


def register_initial_se2(poses):
    """One constant world XY/yaw transform; preserve vertical motion and all joints."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 36 or not np.isfinite(poses).all():
        raise ValueError("requires finite 29-joint poses")
    rotations = Rotation.from_quat(poses[:, [4, 5, 6, 3]])
    forward = rotations[0].as_matrix()[:2, 0]
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError("initial root cannot define heading")
    yaw = float(np.arctan2(forward[1], forward[0]))
    registration = Rotation.from_euler("z", -yaw)
    translation = np.r_[poses[0, :2], 0.0]
    result = poses.copy()
    result[:, :3] = registration.apply(poses[:, :3] - translation)
    result[:, 3:7] = (registration * rotations).as_quat()[:, [3, 0, 1, 2]]
    np.testing.assert_array_equal(result[:, 7:], poses[:, 7:])
    np.testing.assert_array_equal(result[:, 2], poses[:, 2])
    return result, dict(subtracted_translation_w_m=translation.tolist(), applied_yaw_rad=-yaw,
                        applied_once_to_all_frames=True, per_frame_reanchoring=False, vertical_shift_m=0.0)
