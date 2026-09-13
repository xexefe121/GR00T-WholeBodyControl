"""Decoder-wide LoRA under the corrected original-source native23 contract.

The released base remains frozen. The old first-affine adapter is retained,
with independently zero-effect rank16 adapters on the remaining eight affine
layers. This distinct actor cannot be relabelled as a first-affine checkpoint.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import _tensor_state_sha256
from gear_sonic.trl.mjlab.native23_frozen_decoder_actor import True23FrozenDecoderActorModel
from gear_sonic.trl.mjlab.native23_pose_lora_actor import (
    POSE_INIT_SEED,
    RANK,
    initialize_pose_parameters,
    pose_adapter_contract,
)

ACTOR_KIND = "g1_native23_frozen_base_decoder_lora_root_adapter_v1"
ARCHITECTURE = "frozen_all_affine_decoder_lora16_plus_root9_v1"
TRAINABLE = "decoder_lora_root_conditioner_bounded_exploration_and_critic_only"
DIMS = (994, 4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 23)


def decoder_adapter_contract():
    return dict(
        kind=ARCHITECTURE,
        input="frozen_source_token64_and_masked_physical_history930",
        linear_layer_dimensions=list(DIMS),
        linear_module_indices=list(range(0, 18, 2)),
        rank=RANK,
        scaling=1.0,
        bias=False,
        a_initialization="kaiming_uniform_a_sqrt5_local_generator",
        a_initialization_seeds=[POSE_INIT_SEED + i for i in range(9)],
        b_initialization="exact_zero",
        initialization_consumes_global_rng=False,
        trainable_parameters=RANK * sum(a + b for a, b in zip(DIMS[:-1], DIMS[1:], strict=True)),
        first_affine_initialization_identical_to_previous_pose_adapter=True,
        new_actor_observations=False,
        released_base_tensors_frozen=True,
        physical_limits_or_action_projection_changed=False,
        tracking_improvement_proven=False,
    )


def initialize_tail_parameters(layers):
    a, b = nn.ParameterList(), nn.ParameterList()
    if len(layers) != 8:
        raise ValueError("decoder-wide LoRA requires eight tail affine layers")
    for i, layer in enumerate(layers, 1):
        if not isinstance(layer, nn.Linear) or tuple(layer.weight.shape) != (DIMS[i + 1], DIMS[i]):
            raise ValueError("decoder-wide LoRA tail dimensions differ from the pinned native23 core")
        first = nn.Parameter(layer.weight.new_empty(RANK, DIMS[i]))
        second = nn.Parameter(layer.weight.new_zeros(DIMS[i + 1], RANK))
        generator = torch.Generator(device=layer.weight.device).manual_seed(POSE_INIT_SEED + i)
        nn.init.kaiming_uniform_(first, a=math.sqrt(5.0), generator=generator)
        a.append(first)
        b.append(second)
    return a, b


def decoder_conditioned_forward(actor, decoder_input, root_feedback):
    decoder = actor.core.decoder
    first_update = F.linear(F.linear(decoder_input, actor.pose_lora_a), actor.pose_lora_b)
    hidden = decoder.module[0](decoder_input) + actor.root_conditioner(root_feedback)
    hidden = hidden + first_update
    adapter = 0
    for layer in decoder.module[1:]:
        if isinstance(layer, nn.Linear):
            update = F.linear(F.linear(hidden, actor.decoder_lora_a[adapter]), actor.decoder_lora_b[adapter])
            hidden = layer(hidden) + update
            adapter += 1
        else:
            hidden = layer(hidden)
    if adapter != 8:
        raise RuntimeError("decoder-wide LoRA did not execute every expected tail adapter")
    return hidden


class True23DecoderLoraActorModel(True23FrozenDecoderActorModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tuple(self.core.decoder_dims) != DIMS:
            raise ValueError("decoder-wide LoRA requires the pinned low-latency native23 decoder")
        self.pose_lora_a, self.pose_lora_b = initialize_pose_parameters(self.core.decoder.module[0].weight)
        self._initial_pose_a_sha256 = _tensor_state_sha256({"a": self.pose_lora_a})
        self.decoder_lora_a, self.decoder_lora_b = initialize_tail_parameters(
            [self.core.decoder.module[i] for i in range(2, 18, 2)]
        )
        self._initial_tail_a_sha256 = _tensor_state_sha256(self.decoder_lora_a.state_dict())

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state
        if masks is not None:
            from rsl_rl.utils import unpad_trajectories

            obs = unpad_trajectories(obs, masks)
        semantic = obs[self.tokenizer_obs_group]
        if self.tokenizer_has_encoder_index:
            if not torch.isfinite(semantic[..., :1]).all():
                raise ValueError("decoder-wide LoRA encoder route contains non-finite values")
            semantic = semantic[..., 1:]
        feedback = obs[self.root_feedback_obs_group]
        proprioception = obs[self.proprioception_obs_group]
        if feedback.shape != (*proprioception.shape[:-1], 9) or not torch.isfinite(feedback).all():
            raise ValueError("decoder-wide LoRA root input must be finite matching-batch [...,9]")
        if feedback.dtype != proprioception.dtype or feedback.device != proprioception.device:
            raise ValueError("decoder-wide LoRA root dtype/device must match policy observations")
        decoded = torch.cat(
            (self.core.encode(semantic), self.core.codec.encode_proprioception(proprioception)), -1
        )
        mean = decoder_conditioned_forward(self, decoded, feedback)
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
            "decoder_adapter": decoder_adapter_contract(),
            "actor_trainable": "all_decoder_lora_root_conditioner_and_bounded_exploration_only",
            "first_affine_only_checkpoint_relabelling_allowed": False,
        }

    def parameter_groups(self):
        groups = super().parameter_groups()
        if not self.pose_lora_a.requires_grad or not self.pose_lora_b.requires_grad:
            raise RuntimeError("decoder-wide LoRA first adapter became frozen")
        adapters = [self.pose_lora_a, self.pose_lora_b]
        for a, b in zip(self.decoder_lora_a, self.decoder_lora_b, strict=True):
            if not a.requires_grad or not b.requires_grad:
                raise RuntimeError("decoder-wide LoRA tail adapter became frozen")
            adapters.extend((a, b))
        return {
            "root_conditioner": groups["root_conditioner"],
            "decoder_adapters": adapters,
            "exploration": groups["exploration"],
        }

    def initial_conditioner_is_zero(self):
        return (
            super().initial_conditioner_is_zero()
            and not bool(torch.count_nonzero(self.pose_lora_b).item())
            and _tensor_state_sha256({"a": self.pose_lora_a}) == self._initial_pose_a_sha256
            and all(not bool(torch.count_nonzero(b).item()) for b in self.decoder_lora_b)
            and _tensor_state_sha256(self.decoder_lora_a.state_dict()) == self._initial_tail_a_sha256
        )
