"""Compact reference-residual tracker, distinct from every SONIC checkpoint."""

import numpy as np
import torch
from torch import nn
from rsl_rl.models.mlp_model import MLPModel

from gear_sonic.trl.mjlab.native23_generalist_actor import BoundedGaussianDistribution
from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_source_action_codec import (
    SOURCE_SCALE_NATIVE_IL23,
    source_scaled_precompensation,
)


def reachable_source_bounds():
    targets = [
        safe_target_transform_numpy(source_scaled_precompensation(np.full(23, value, np.float32))[0])[1].astype(
            np.float64
        )
        for value in (-9.99, 9.99)
    ]
    default = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    index = list(MUJOCO_TO_ISAACLAB_DOF)
    return tuple(
        ((value - default)[index] / np.asarray(SOURCE_SCALE_NATIVE_IL23)).astype(np.float32) for value in targets
    )


class CompactNative23Actor(MLPModel):
    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        *,
        hidden_dims=(256, 256, 256),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
    ):
        if output_dim != 23 or obs_set != "actor" or tuple(obs_groups[obs_set]) != ("native_controller",):
            raise ValueError("compact actor requires its distinct165-input/23-output interface")
        if obs["native_controller"].shape[-1] != 165 or tuple(hidden_dims) != (256, 256, 256):
            raise ValueError("compact actor topology mismatch")
        if activation != "elu" or obs_normalization is not True or distribution_cfg is not None:
            raise ValueError("compact actor recipe changed")
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=True,
            distribution_cfg=None,
        )
        self.distribution = BoundedGaussianDistribution(init_std=0.20, std_min=0.03, std_max=0.50)
        final = [layer for layer in self.mlp.modules() if isinstance(layer, nn.Linear)][-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        lower, upper = reachable_source_bounds()
        self.register_buffer("center", torch.from_numpy((lower + upper) / 2))
        self.register_buffer("half_range", torch.from_numpy((upper - lower) / 2))
        self.register_buffer("default_hw", torch.tensor(SAFE_TARGET_DEFAULT_Q_HARDWARE, dtype=torch.float32))
        self.register_buffer("source_scale", torch.tensor(SOURCE_SCALE_NATIVE_IL23, dtype=torch.float32))
        self.register_buffer("hardware_to_native", torch.tensor(MUJOCO_TO_ISAACLAB_DOF))

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state
        if masks is not None:
            from rsl_rl.utils import unpad_trajectories

            obs = unpad_trajectories(obs, masks)
        features = obs["native_controller"]
        q_ref = features[..., 75:98]
        source_ref = (q_ref - self.default_hw).index_select(-1, self.hardware_to_native) / self.source_scale
        normalized_ref = ((source_ref - self.center) / self.half_range).clamp(-0.9999, 0.9999)
        residual = self.mlp(self.get_latent(obs))
        mean = self.center + self.half_range * torch.tanh(torch.atanh(normalized_ref) + residual)
        if not torch.isfinite(mean).all():
            raise ValueError("compact actor emitted nonfinite actions")
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def as_onnx(self, *args, **kwargs):
        raise RuntimeError("compact tracker requires separately verified multi-input deployment export")
