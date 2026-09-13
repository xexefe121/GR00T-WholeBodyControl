"""Received 29-axis reference for the released low-latency G1 encoder.

This is not a robot command and never invents absent physical measurements.
All 29 source axes are virtual motion intent. The exact upstream command
reshape is intentionally preserved, including its flattened q-then-dq layout.
"""

import numpy as np

from gear_sonic.scripts.simulate_g1_sonic_library_motions import ISAAC_TO_MUJOCO_INDEX
from gear_sonic.teleop.buffered_source_horizon import _relative_orientation_6d


def full_pose_reference_contract():
    return dict(
        kind="sonic_low_latency_received_g1_encoder640_v1",
        received_source_samples=11,
        source_period_s=0.02,
        encoder_anchor_age_s=0.2,
        reference_axes="all29 source axes retained, canonical IL29 input order",
        motion_layout="concat(flat(q0..q9),flat(forward_dq0..dq9)).reshape(10,58)",
        orientation_layout="q0..q9 relative to current measured pelvis; first two matrix columns row-major",
        encoder_layout="concat(motion10x58,orientation10x6,axis=-1).flatten()",
        ordinary_per_frame_q_dq_interleaving_equivalent=False,
        unreceived_future_samples_consumed=False,
        hardware_authorized=False,
        deployment_ready=False,
    )


def full_pose_encoder640(received_qpos29, measured_quaternion_wxyz):
    poses = np.asarray(received_qpos29)
    measured = np.asarray(measured_quaternion_wxyz)
    if poses.shape != (11, 36) or poses.dtype != np.float32 or not np.isfinite(poses).all():
        raise ValueError("G1 reference requires exactly eleven finite float32 full29 poses")
    if measured.shape != (4,) or measured.dtype != np.float32 or not np.isfinite(measured).all():
        raise ValueError("G1 reference requires finite float32 measured WXYZ")
    if (
        np.max(np.abs(np.linalg.norm(poses[:, 3:7], axis=-1) - 1)) > 1e-4
        or abs(np.linalg.norm(measured) - 1) > 1e-4
    ):
        raise ValueError("G1 reference quaternions must be normalized")
    q = poses[:, 7:][:, ISAAC_TO_MUJOCO_INDEX]
    dq = (q[1:] - q[:-1]) / np.float32(0.02)
    motion = np.concatenate((q[:-1].reshape(-1), dq.reshape(-1))).reshape(10, 58)
    orientation = np.asarray([_relative_orientation_6d(measured, quat) for quat in poses[:-1, 3:7]])
    result = np.concatenate((motion, orientation), axis=-1).reshape(640)
    if result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError("G1 reference overflows finite float32 encoder input")
    return result
