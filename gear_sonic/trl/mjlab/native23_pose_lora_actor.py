"""Pose-conditioned first-affine LoRA with a frozen SONIC base and root branch.

Unlike root-only adaptation, this branch can correct the base at zero root
feedback. It uses the existing source token and masked physical history, not
new privileged observations or fictitious absent-motor feedback. Research only.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import _tensor_state_sha256
from gear_sonic.trl.mjlab.native23_frozen_decoder_actor import True23FrozenDecoderActorModel

ACTOR_KIND = "g1_native23_frozen_base_pose_lora_root_adapter_v1"
ARCHITECTURE = "frozen_first_affine_pose_lora16_plus_root9_v1"
TRAINABLE = "pose_lora_root_conditioner_bounded_exploration_and_critic_only"
RANK = 16
POSE_INIT_SEED = 20260909


def pose_adapter_contract():
    return {
        "kind": ARCHITECTURE,
        "input": "frozen_source_token64_and_masked_physical_history930",
        "input_dimension": 994,
        "output_dimension": 4096,
        "rank": RANK,
        "scaling": 1.0,
        "bias": False,
        "a_initialization": "kaiming_uniform_a_sqrt5_local_generator",
        "a_initialization_seed": POSE_INIT_SEED,
        "b_initialization": "exact_zero",
        "initialization_consumes_global_rng": False,
        "trainable_parameters": RANK * (994 + 4096),
        "zero_root_pose_correction_possible": True,
        "new_actor_observations": False,
        "released_base_tensors_frozen": True,
        "physical_limits_or_action_projection_changed": False,
        "tracking_improvement_proven": False,
    }


def pose_conditioned_decoder_forward(actor, decoder_input, root_feedback):
    update = F.linear(F.linear(decoder_input, actor.pose_lora_a), actor.pose_lora_b)
    decoder = actor.core.decoder
    hidden = decoder.module[0](decoder_input) + actor.root_conditioner(root_feedback)
    hidden = hidden + update
    for layer in decoder.module[1:]:
        hidden = layer(hidden)
    return hidden


def initialize_pose_parameters(weight):
    a = nn.Parameter(weight.new_empty(RANK, 994))
    b = nn.Parameter(weight.new_zeros(4096, RANK))
    generator = torch.Generator(device=weight.device).manual_seed(POSE_INIT_SEED)
    nn.init.kaiming_uniform_(a, a=math.sqrt(5.0), generator=generator)
    return a, b


class True23PoseLoraActorModel(True23FrozenDecoderActorModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tuple(self.core.decoder_dims[:2]) != (994, 4096):
            raise ValueError("pose LoRA requires the pinned low-latency first affine")
        weight = self.core.decoder.module[0].weight
        self.pose_lora_a, self.pose_lora_b = initialize_pose_parameters(weight)
        self._initial_pose_a_sha256 = _tensor_state_sha256({"a": self.pose_lora_a})

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state
        if masks is not None:
            from rsl_rl.utils import unpad_trajectories

            obs = unpad_trajectories(obs, masks)
        semantic = obs[self.tokenizer_obs_group]
        if self.tokenizer_has_encoder_index:
            if not torch.isfinite(semantic[..., :1]).all():
                raise ValueError("pose LoRA encoder route contains non-finite values")
            semantic = semantic[..., 1:]
        feedback = obs[self.root_feedback_obs_group]
        proprioception = obs[self.proprioception_obs_group]
        if feedback.shape != (*proprioception.shape[:-1], 9) or not torch.isfinite(feedback).all():
            raise ValueError("pose LoRA root input must be finite, matching-batch [...,9]")
        if feedback.dtype != proprioception.dtype or feedback.device != proprioception.device:
            raise ValueError("pose LoRA root dtype/device must match policy observations")
        decoded = torch.cat(
            (self.core.encode(semantic), self.core.codec.encode_proprioception(proprioception)), -1
        )
        mean = pose_conditioned_decoder_forward(self, decoded, feedback)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def artifact_contract(self):
        return {
            **super().artifact_contract(),
            "kind": ACTOR_KIND,
            "architecture": ARCHITECTURE,
            "pose_adapter": pose_adapter_contract(),
            "actor_trainable": "pose_lora_root_conditioner_and_bounded_exploration_only",
            "root_only_checkpoint_relabelling_allowed": False,
        }

    def parameter_groups(self):
        groups = super().parameter_groups()
        if not self.pose_lora_a.requires_grad or not self.pose_lora_b.requires_grad:
            raise RuntimeError("pose LoRA became frozen")
        return {
            "root_conditioner": groups["root_conditioner"],
            "pose_adapter": [self.pose_lora_a, self.pose_lora_b],
            "exploration": groups["exploration"],
        }

    def initial_conditioner_is_zero(self):
        return (
            super().initial_conditioner_is_zero()
            and not bool(torch.count_nonzero(self.pose_lora_b).item())
            and _tensor_state_sha256({"a": self.pose_lora_a}) == self._initial_pose_a_sha256
        )
