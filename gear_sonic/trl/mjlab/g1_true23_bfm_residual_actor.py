"""Small trainable target-residual actor; BFM base is frozen outside PPO."""

from torch import nn

from rsl_rl.models.mlp_model import MLPModel
from gear_sonic.trl.mjlab.native23_generalist_actor import BoundedGaussianDistribution


class BFMResidualActor(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, *, hidden_dims=(256, 256), activation="elu",
                 obs_normalization=True, distribution_cfg=None, cnn_cfg=None):
        if output_dim != 23 or obs["bfm_residual"].shape[-1] != 188 or cnn_cfg is not None:
            raise ValueError("BFM residual actor requires188 observations and23 native outputs")
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims=hidden_dims, activation=activation,
                         obs_normalization=obs_normalization, distribution_cfg=None)
        self.distribution = BoundedGaussianDistribution(init_std=.08, std_min=.01, std_max=.30)
        final = [layer for layer in self.mlp.modules() if isinstance(layer, nn.Linear)][-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            from rsl_rl.utils import unpad_trajectories
            obs = unpad_trajectories(obs, masks)
        mean = self.mlp(self.get_latent(obs))
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean
