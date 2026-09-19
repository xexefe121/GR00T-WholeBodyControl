"""Batched frozen BFM features and bounded native23 joint-target residuals.

This is a separate simulation learner. Its zero residual follows the public
BFM policy, including measured-heading goal feedback and preclip action memory.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gear_sonic.utils.g1_true23_bfmzero_inference import BFMZeroInference, reference_features
from gear_sonic.utils.g1_true23_compact_features import quaternion_matrix


KIND = "native23_frozen_bfm_bounded_target_residual_v1"
RESIDUAL_RADIUS_RAD = 0.15


def heading_matrix(quaternion):
    rotation = quaternion_matrix(quaternion)
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    c, s = torch.cos(yaw), torch.sin(yaw)
    zeros, ones = torch.zeros_like(c), torch.ones_like(c)
    heading = torch.stack((c, -s, zeros, s, c, zeros, zeros, zeros, ones), -1).reshape(*yaw.shape, 3, 3)
    return heading, yaw


def bfm_state_and_terms(qpos, qvel, last_action, default_q):
    relative = qpos[:, 7:] - default_q
    gyro = qvel[:, 3:6] * 0.25
    gravity = -quaternion_matrix(qpos[:, 3:7])[:, 2, :]
    state = torch.cat((relative, qvel[:, 6:], gravity, gyro), -1)
    terms = torch.cat((last_action, gyro, relative, qvel[:, 6:], gravity), -1)
    return state, terms


class BFMHistoryTorch:
    """Four prior frames, grouped by alphabetic term then newest-first time."""

    WIDTHS = (23, 3, 23, 23, 3)

    def __init__(self, count, device):
        self.data = torch.zeros(count, 4, 75, device=device)

    def reset(self, ids):
        self.data[ids] = 0

    def before_update(self, terms):
        history = torch.cat([part.flatten(1) for part in self.data.split(self.WIDTHS, -1)], -1)
        self.data[:, 1:] = self.data[:, :-1].clone()
        self.data[:, 0] = terms
        return history


class FrozenBFMController:
    def __init__(self, weights, contract, motion, *, count, device, horizon=8, position_gain=1.0, yaw_gain=2.0):
        self.policy = BFMZeroInference(Path(weights))
        self.policy.weights = {key: value.to(device) for key, value in self.policy.weights.items()}
        self.device = device
        self.horizon = horizon
        self.position_gain, self.yaw_gain = position_gain, yaw_gain
        self.default_q = torch.as_tensor(contract["default_q"], dtype=torch.float32, device=device)
        self.action_scale = torch.as_tensor(.25 * contract["training_effort"] / contract["kp"], dtype=torch.float32, device=device)
        state, privileged = reference_features(motion, contract)
        self.reference_state = torch.as_tensor(state, device=device)
        self.reference_privileged = torch.as_tensor(privileged, device=device)
        self.reference_position = torch.as_tensor(motion["body_pos_w"][:, 0], dtype=torch.float32, device=device)
        self.reference_quaternion = torch.as_tensor(motion["body_quat_w"][:, 0], dtype=torch.float32, device=device)
        self.reference_velocity = torch.as_tensor(motion["body_lin_vel_w"][:, 0], dtype=torch.float32, device=device)
        self.reference_heading, self.reference_yaw = heading_matrix(self.reference_quaternion)
        self.history = BFMHistoryTorch(count, device)
        self.last_action = torch.zeros(count, 23, device=device)
        self.base_action = torch.zeros_like(self.last_action)
        self.last_goal = torch.zeros(count, 256, device=device)

    def reset(self, ids):
        self.history.reset(ids)
        self.last_action[ids] = 0
        self.base_action[ids] = 0

    def corrected_inputs(self, frame, final_frame, qpos):
        indices = torch.minimum(frame[:, None] + torch.arange(self.horizon, device=self.device), final_frame[:, None])
        states = self.reference_state[indices].clone()
        privileged = self.reference_privileged[indices].clone()
        actual_heading, actual_yaw = heading_matrix(qpos[:, 3:7])
        yaw_difference = self.reference_yaw[frame] - actual_yaw
        yaw_error = torch.atan2(torch.sin(yaw_difference), torch.cos(yaw_difference))
        omega = torch.zeros(len(frame), 3, device=self.device)
        omega[:, 2] = (self.yaw_gain * yaw_error).clamp(-.8, .8)
        delta = self.position_gain * (self.reference_position[frame] - qpos[:, :3])
        delta[:, 2] = 0
        delta *= (.6 / delta.norm(dim=-1).clamp_min(1e-8)).clamp(max=1)[:, None]
        root_velocity = self.reference_velocity[indices]
        actual_local = torch.einsum("nji,nhj->nhi", actual_heading, root_velocity + delta[:, None])
        reference_local = torch.einsum("nhji,nhj->nhi", self.reference_heading[indices], root_velocity)
        local_delta = actual_local - reference_local
        local_positions = torch.cat((torch.zeros(len(frame), self.horizon, 1, 3, device=self.device),
                                     privileged[..., 1:73].reshape(len(frame), self.horizon, 24, 3)), -2)
        omega_rows = omega[:, None, None, :].expand_as(local_positions)
        correction = local_delta[:, :, None, :] + torch.cross(omega_rows, local_positions, dim=-1)
        privileged[..., 223:298] += correction.flatten(-2)
        privileged[..., 298:373] += omega_rows.flatten(-2)
        states[..., -3:] += omega[:, None]
        return states, privileged

    @torch.inference_mode()
    def observe(self, qpos, qvel, frame, final_frame):
        state, terms = bfm_state_and_terms(qpos, qvel, self.last_action, self.default_q)
        history = self.history.before_update(terms)
        states, privileged = self.corrected_inputs(frame, final_frame, qpos)
        latent = self.policy.backward(states.flatten(0, 1), privileged.flatten(0, 1))
        goal = 16 * F.normalize(latent.reshape(len(frame), self.horizon, 256).mean(1), dim=-1)
        self.last_goal.copy_(goal)
        self.base_action.copy_(5 * self.policy.actor(state, self.last_action, history, goal))
        return self.base_action.clone(), state, history, goal

    def targets(self, residual_raw, lower, upper):
        delta = RESIDUAL_RADIUS_RAD * torch.tanh(residual_raw)
        combined = self.base_action + delta / self.action_scale
        target = self.default_q + combined * self.action_scale
        self.last_action.copy_(combined)
        return target.clamp(lower, upper), delta, combined


def residual_contract():
    return dict(kind=KIND, residual_radius_rad=RESIDUAL_RADIUS_RAD, position_gain=1.0, yaw_gain=2.0,
                goal_horizon=8, goal_feedback="measured_heading_v2", residual_input_dim=188,
                residual_action="u sampled in R23; delta_q=.15*tanh(u); clipped physical joint target",
                last_action="combined preclip action-equivalent", bfm_weights_frozen=True,
                hardware_authorized=False, deployment_ready=False)
