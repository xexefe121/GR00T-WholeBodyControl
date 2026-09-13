"""Inference-only residual CPU helpers; no RSL-RL or MJLab dependency."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_compact_features import make_goal
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_numpy
from gear_sonic.utils.g1_true23_step1b_mujoco import _quaternion_matrix


KIND = "native23_frozen_bfm_bounded_target_residual_v1"


def residual_features_numpy(qpos, qvel, previous_combined_action, base_target, motion, original_vr21,
                            frame, measured_feet_w, foot_body_indices):
    """Match the residual learner's188 values from measured simulator state.

    frame is the held target frame (CPU evaluator control+11). previous action
    is the combined BFM/residual preclip action-equivalent. Foot indices address
    the native motion's24-body arrays, not MuJoCo's world-inclusive body list.
    """
    qpos, qvel = np.asarray(qpos), np.asarray(qvel)
    desired = np.asarray(motion["body_pos_w"][frame, 0], dtype=np.float32)
    previous = np.asarray(motion["body_pos_w"][frame - 1, 0], dtype=np.float32)
    feedback = root_feedback_numpy(desired, qpos[:3], (desired - previous) / np.float32(.02), qvel[:3], qpos[3:7])
    q_ref = np.asarray(motion["joint_pos"][frame], dtype=np.float32)
    dq_ref = (q_ref - np.asarray(motion["joint_pos"][frame - 1], dtype=np.float32)) / np.float32(.02)
    feet_error = (motion["body_pos_w"][frame, list(foot_body_indices)] - measured_feet_w).astype(np.float32)
    fields = (q_ref, dq_ref, feedback, np.asarray(motion["body_quat_w"][frame, 0], dtype=np.float32),
              qpos[3:7].astype(np.float32), np.asarray(original_vr21[frame], dtype=np.float32), feet_error,
              np.array([desired[2], qpos[2]], np.float32))
    with torch.inference_mode():
        goal = make_goal(*(torch.from_numpy(value[None]) for value in fields))[0].numpy()
    gravity = _quaternion_matrix(qpos[3:7]).T @ np.array([0., 0., -1.])
    result = np.r_[qvel[3:6], qpos[7:] - np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE), qvel[6:],
                   previous_combined_action, gravity, goal, base_target].astype(np.float32)
    if result.shape != (188,) or not np.isfinite(result).all():
        raise ValueError("residual replay requires188 finite feature values")
    return result


def functional_residual(actor_state, features):
    x = (features - actor_state["obs_normalizer._mean"]) / (actor_state["obs_normalizer._std"] + .01)
    for index in (0, 2, 4):
        x = F.linear(x, actor_state[f"mlp.{index}.weight"], actor_state[f"mlp.{index}.bias"])
        if index != 4:
            x = F.elu(x)
    return x


class BFMResidualCPU:
    def __init__(self, checkpoint):
        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if saved["kind"] != KIND or saved["manifest"]["contract"]["residual_radius_rad"] != .15:
            raise ValueError("not a compatible BFM residual checkpoint")
        self.state = saved["actor_state"]
        if self.state["mlp.0.weight"].shape != (256, 188) or self.state["mlp.4.weight"].shape != (23, 256):
            raise ValueError("BFM residual architecture differs")
        self.completed_updates = saved["completed_updates"]
        self.manifest = saved["manifest"]

    @torch.inference_mode()
    def predict(self, features):
        value = np.asarray(features, dtype=np.float32)
        if value.shape != (188,) or not np.isfinite(value).all():
            raise ValueError("BFM residual requires a finite188-vector")
        raw = functional_residual(self.state, torch.from_numpy(value[None]))[0]
        return (.15 * torch.tanh(raw)).numpy().copy()
