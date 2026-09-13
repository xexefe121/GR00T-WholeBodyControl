"""SIM-only optical PICO pose adapter for the released SONIC SMPL encoder.

This is not a reconstruction of sparse headset sensors. Public optical SMPL
poses are converted to SONIC's neutral SMPL-X body/thumbnail geometry. Preserve
the upstream nonzero neutral pelvis offset; do not silently pelvis-center it.
"""

import gc
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

from gear_sonic.scripts.init_g1_23dof_checkpoint import _load_pinned_legacy_release
from gear_sonic.trl.utils import torch_transform as upstream
from gear_sonic.utils.g1_23dof_contract import LOW_LATENCY_RELEASE_SHA256
from gear_sonic.utils.g1_29dof_low_latency_teacher import (
    DECODER_DIMS,
    ENCODER_DIMS,
    ExactLowLatencyTeacher,
    _extract_component,
    _numpy_float32_matrix,
    exact_fsq32,
)
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

HUMAN_SHA256 = "4de0bae69caf31e8829a2d3e8adecd887f29115af60a0b8d59237dfbfea1c975"
SMPL_DIMS = (336, 2048, 1024, 512, 512, 64)
OUTPUT_JOINTS = (*range(22), 39, 54)


def human_reference(root_axis_angle, body_axis_angle, queries, human_path, yaw):
    """Whole-recording 60-to-50 Hz SLERP; input is already Z-up, not Y-up."""
    root = np.asarray(root_axis_angle, dtype=np.float64)
    body = np.asarray(body_axis_angle, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    if root.ndim != 2 or root.shape[1:] != (3,) or body.shape != (len(root), 69):
        raise ValueError("requires paired optical root3/body69 axis angles")
    if not all(np.isfinite(x).all() for x in (root, body, queries)):
        raise ValueError("nonfinite optical source")
    times = np.arange(len(root), dtype=np.float64) / 60
    expected = np.arange(int(np.floor(times[-1] * 50)) + 1) / 50
    if not np.array_equal(queries, expected):
        raise ValueError("requires the complete unscaled 50 Hz optical timeline")
    human_path = Path(human_path)
    if file_sha256(human_path) != HUMAN_SHA256:
        raise ValueError("not the pinned upstream neutral human geometry")
    info = torch.load(human_path, map_location="cpu", weights_only=True)
    if set(info) != {"J", "parents_list", "rot_mats"} or info["J"].shape != (55, 3):
        raise ValueError("unexpected neutral human geometry schema")
    aa = np.concatenate((root, body[:, :63]), axis=1).reshape(-1, 22, 3)
    sampled = np.stack(
        [Slerp(times, Rotation.from_rotvec(aa[:, j]))(queries).as_rotvec() for j in range(22)],
        axis=1,
    ).astype(np.float32)
    # Safe weights-only preload prevents upstream's legacy default loader call.
    old_info = upstream.human_joints_info
    upstream.human_joints_info = info
    try:
        with torch.inference_mode():
            root_t = torch.from_numpy(sampled[:, 0])
            body_t = torch.from_numpy(sampled[:, 1:].reshape(-1, 63))
            joints = upstream.compute_human_joints(body_t, root_t, use_thumb_joints=True)
            raw_q = upstream.angle_axis_to_quaternion(root_t)
            offset = torch.tensor([0.5, -0.5, -0.5, -0.5]).expand_as(raw_q)
            corrected_q = upstream.quat_mul(raw_q, offset)
            local = upstream.quat_apply(upstream.quat_inv(corrected_q)[:, None].repeat(1, 24, 1), joints)
    finally:
        upstream.human_joints_info = old_info
    corrected = Rotation.from_quat(corrected_q.numpy()[:, [1, 2, 3, 0]])
    registered = Rotation.from_euler("z", float(yaw)) * corrected
    result = dict(
        local_joints24=local.numpy().copy(),
        root_quaternion_wxyz=registered.as_quat()[:, [3, 0, 1, 2]].astype(np.float32),
        local_axis_angle22=sampled,
        neutral_fk_joints24_unregistered=joints.numpy().copy(),
        neutral_rest_joints55=info["J"].numpy().copy(),
        neutral_parents55=np.asarray(info["parents_list"], dtype=np.int64),
        timestamps_s=queries.copy(),
    )
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError("nonfinite derived neutral human reference")
    return result


def smpl_input(local_joints4, relative_root4, wrist6_4):
    """Training order: concatenate 72/6/6 inside each frame, THEN flatten."""
    joints = np.asarray(local_joints4, dtype=np.float32)
    root = np.asarray(relative_root4, dtype=np.float32)
    wrists = np.asarray(wrist6_4, dtype=np.float32)
    if joints.shape != (4, 24, 3) or root.shape != (4, 3, 3) or wrists.shape != (4, 6):
        raise ValueError("requires four frames of local24 joints, root matrix, and IL29 wrist6")
    value = np.concatenate((joints.reshape(4, 72), root[..., :2].reshape(4, 6), wrists), axis=1)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite SMPL encoder input")
    return np.ascontiguousarray(value.reshape(336), dtype=np.float32)


def smpl_relative_from_teleop(relative6, gmr_anchor_wxyz, human_root4_wxyz):
    """Re-express existing measured-relative orientation without adding feedback.

    R_m^-1 R_h = (R_m^-1 R_g) (R_g^-1 R_h). The first factor is
    already in the baseline's float32 267-vector. Recover its third column by
    cross product, not an estimator or source-state substitution. Recorded runs
    must independently compare with the actual measured pelvis quaternion.
    """
    first_two = np.asarray(relative6, dtype=np.float64).reshape(3, 2)
    relative = np.column_stack((first_two, np.cross(first_two[:, 0], first_two[:, 1])))
    anchor = Rotation.from_quat(np.asarray(gmr_anchor_wxyz)[[1, 2, 3, 0]]).as_matrix()
    human = Rotation.from_quat(np.asarray(human_root4_wxyz)[:, [1, 2, 3, 0]]).as_matrix()
    return relative @ anchor.T @ human


class OpticalSMPLTeacher:
    """Two exact released encoders, one unchanged released decoder; CPU only."""

    def __init__(self, checkpoint_path):
        checkpoint, digest, _ = _load_pinned_legacy_release(Path(checkpoint_path))
        if digest != LOW_LATENCY_RELEASE_SHA256:
            raise ValueError("requires the pinned low-latency release")
        state = checkpoint["policy_state_dict"]
        self.encoders = {}
        for route, dims in (("teleop", ENCODER_DIMS), ("smpl", SMPL_DIMS)):
            self.encoders[route] = _extract_component(
                state,
                prefix=f"actor_module.encoders.{route}.module.",
                dims=dims,
                device=torch.device("cpu"),
                context=route,
            )
        self.decoder = _extract_component(
            state,
            prefix="actor_module.decoders.g1_dyn.module.",
            dims=DECODER_DIMS,
            device=torch.device("cpu"),
            context="released decoder",
        )
        self.checkpoint_sha256 = digest
        del state, checkpoint
        gc.collect()

    def forward(self, semantic, history, route):
        if route not in self.encoders:
            raise ValueError("unknown encoder route")
        semantic, _ = _numpy_float32_matrix(semantic, 336 if route == "smpl" else 267, route)
        history, _ = _numpy_float32_matrix(history, 930, "history")
        if len(semantic) != len(history):
            raise ValueError("encoder/history batch mismatch")
        with torch.inference_mode():
            latent = ExactLowLatencyTeacher._mlp(self.encoders[route], torch.from_numpy(semantic))
            token = exact_fsq32(latent)
            action = ExactLowLatencyTeacher._mlp(
                self.decoder, torch.cat((token, torch.from_numpy(history)), dim=-1)
            )
        return action.numpy().copy(), token.numpy().copy()
