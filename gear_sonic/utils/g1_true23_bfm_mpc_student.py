"""Bounded goal-conditioned correction to the frozen native23 BFM controller.

The BFM caller owns its exact preclip combined action history. This head never
loads a recording identity, source frame index, clock, MPC state, or MPC gain.
"""
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_mpc_student import GoalFeatures, OFFSETS

KIND = "native23_bfm_mpc_actual_residual_v1"
FEATURES = 1069
BASE_CONTRACT = dict(position_gain=1., yaw_gain=2., horizon=8,
                     arm_reference=False, goal_gyro_convention="published-world-unscaled")


class ResidualFeatures:
    def __init__(self, motion, original29, contract, limits):
        self.goals = GoalFeatures(motion, original29, contract)
        self.default = np.asarray(contract["default_q"], dtype=np.float64)
        self.scale = .25 * np.asarray(contract["training_effort"]) / np.asarray(contract["kp"])
        self.limits = np.asarray(limits)

    def __call__(self, qpos, qvel, frame, base_target, previous_combined_action):
        previous_target = np.clip(self.default + self.scale * previous_combined_action,
                                  self.limits[:, 0], self.limits[:, 1])
        result = np.r_[self.goals(qpos, qvel, previous_target, frame),
                       np.asarray(base_target) - self.default,
                       np.asarray(previous_combined_action)].astype(np.float32)
        if result.shape != (FEATURES,) or not np.isfinite(result).all():
            raise ValueError("invalid BFM residual student features")
        return result


def make_actor():
    import torch
    actor = torch.nn.Sequential(torch.nn.Linear(FEATURES, 256), torch.nn.ELU(),
                                torch.nn.Linear(256, 256), torch.nn.ELU(),
                                torch.nn.Linear(256, 23))
    torch.nn.init.zeros_(actor[-1].weight)
    torch.nn.init.zeros_(actor[-1].bias)
    return actor


class BFMMPCResidualCPU:
    def __init__(self, checkpoint, motion, original29, contract, limits):
        import torch
        self.torch = torch
        self.saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if (self.saved["kind"] != KIND or self.saved["goal_offsets"] != OFFSETS.tolist()
                or self.saved["base_contract"] != BASE_CONTRACT):
            raise ValueError("BFM residual student contract mismatch")
        self.radius = float(self.saved["radius_rad"])
        if not np.isfinite(self.radius) or not 0 < self.radius <= 1.:
            raise ValueError("invalid declared residual radius")
        self.actor = make_actor()
        self.actor.load_state_dict(self.saved["actor_state"])
        self.actor.eval()
        self.mean, self.std = self.saved["feature_mean"], self.saved["feature_std"]
        if self.mean.shape != (FEATURES,) or self.std.shape != (FEATURES,) or not (self.std > 0).all():
            raise ValueError("invalid residual normalization")
        self.features = ResidualFeatures(motion, original29, contract, limits)

    def predict(self, qpos, qvel, frame, base_target, previous_combined_action):
        features = self.features(qpos, qvel, frame, base_target, previous_combined_action)
        with self.torch.inference_mode():
            x = self.torch.from_numpy(features)[None]
            delta = self.radius * self.torch.tanh(self.actor((x - self.mean) / self.std))[0]
        result = delta.numpy()
        if not np.isfinite(result).all():
            raise ValueError("nonfinite residual student output")
        return result


def zero_checkpoint(path, radius=.25):
    import torch
    actor = make_actor()
    checkpoint = dict(kind=KIND, goal_offsets=OFFSETS.tolist(), base_contract=BASE_CONTRACT,
                      radius_rad=float(radius), actor_state=actor.state_dict(),
                      feature_mean=torch.zeros(FEATURES), feature_std=torch.ones(FEATURES),
                      step=0, full_body_tracking_qualified=False, hardware_authorized=False)
    torch.save(checkpoint, path)

