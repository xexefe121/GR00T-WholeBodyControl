"""Native23 adaptation of the original SONIC core, with explicit920ms timing.

Seven affine adapters, not the low-latency decoder's nine. The encoder and
physical23-row base decoder stay frozen. A separate zero-initialized Root9
projection preserves the starting policy and requires a future live estimator.
No legacy export or low-latency checkpoint relabelling is permitted.
"""

import copy
import gc
import math

import torch
from torch import nn
from torch.nn import functional as F

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import FrozenPlatformTrue23Core, _tensor_state_sha256
from gear_sonic.trl.mjlab.native23_generalist_actor import (
    BoundedGaussianDistribution,
    Native23GeneralistCore,
    True23Native23GeneralistActorModel,
)
from gear_sonic.trl.mjlab.true23_actor import _SonicActorModule
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state
from gear_sonic.utils.g1_23dof_contract import REFERENCE_PROFILE_NORMAL
from gear_sonic.utils.g1_true23_normal_reference import (
    NORMAL_TIMING,
    normal_reference_contract,
    normal_root_feedback_contract,
)

ACTOR_KIND = "g1_native23_normal_core_lora16_root9_training_actor_v1"
CORE_KIND = "g1_native23_frozen_original_sonic_core_v1"
DIMS = (994, 2048, 2048, 1024, 1024, 512, 512, 23)
RANK = 16
INIT_SEED = 20260910


def normal_adapter_contract():
    return dict(
        kind="original_sonic_native23_seven_affine_lora16_v1",
        dimensions=list(DIMS),
        rank=RANK,
        scaling=1.0,
        bias=False,
        a_seed_by_layer=[INIT_SEED + i for i in range(7)],
        a_initialization="kaiming_uniform_sqrt5_private_generator",
        b_initialization="exact_zero",
        global_rng_consumed=False,
        parameters=RANK * sum(a + b for a, b in zip(DIMS[:-1], DIMS[1:], strict=True)),
        retained_base_encoder_and_decoder_frozen=True,
        decoder_outputs=23,
        phantom_action_outputs=False,
    )


class FrozenNormalNative23Core(Native23GeneralistCore):
    """Reuse the checked RSL core interface without invoking its low-latency init."""

    def __init__(self, *, warm_start_path, source_checkpoint_path):
        nn.Module.__init__(self)
        # Constructors' discarded random weights must not change rollout/critic RNG.
        with torch.random.fork_rng(devices=[]):
            released = FrozenPlatformTrue23Core(
                warm_start_path=warm_start_path,
                source_checkpoint_path=source_checkpoint_path,
                lora_rank=1,
                lora_alpha=1,
            )
            if released.reference_profile != REFERENCE_PROFILE_NORMAL:
                raise ValueError("normal native23 actor refuses a low-latency release")
            if (*released.source_decoder_dims[:-1], 23) != DIMS:
                raise ValueError("normal native23 source decoder dimensions changed")
            self.actor_module = _SonicActorModule(DIMS)
        self.codec = released.codec
        self.decoder_dims = DIMS
        for name in (
            "warm_start_path",
            "source_checkpoint_path",
            "warm_start_metadata",
            "warm_start_initialization_report",
            "reference_profile",
            "source_checkpoint_sha256",
            "source_revision",
            "initial_std",
            "_initial_policy_sha256",
        ):
            setattr(self, name, copy.deepcopy(getattr(released, name)))
        self.warm_start_stage = "checkpoint_initialization"
        state = released.export_true23_policy_state(released.initial_std)
        self.load_state_dict({key: value for key, value in state.items() if key != "std"}, strict=True)
        actual = inspect_true23_policy_state(
            {"policy_state_dict": self.export_policy_state(self.initial_std)},
            reference_profile=REFERENCE_PROFILE_NORMAL,
        )
        if actual != self._initial_policy_sha256:
            raise ValueError("normal core differs from its approved native23 initialization")
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self._frozen_encoder_sha256 = self.frozen_encoder_sha256()
        self._frozen_decoder_sha256 = _tensor_state_sha256(self.decoder.state_dict())
        del released, state
        gc.collect()

    def assert_frozen_encoder_unchanged(self):
        if any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("normal released base unexpectedly became trainable")
        if self.frozen_encoder_sha256() != self._frozen_encoder_sha256:
            raise RuntimeError("normal frozen encoder changed")
        if _tensor_state_sha256(self.decoder.state_dict()) != self._frozen_decoder_sha256:
            raise RuntimeError("normal frozen decoder changed")

    def artifact_contract(self):
        return {
            **super().artifact_contract(),
            "kind": CORE_KIND,
            "trainable_decoder_parameters": 0,
            "decoder_training": "frozen_original_sonic_native23_rows",
            "decoder_frozen": True,
            "frozen_decoder_sha256": self._frozen_decoder_sha256,
        }


def normal_conditioned_forward(actor, decoder_input, root_feedback):
    hidden = decoder_input
    affine = 0
    for layer in actor.core.decoder.module:
        if isinstance(layer, nn.Linear):
            update = F.linear(F.linear(hidden, actor.lora_a[affine]), actor.lora_b[affine])
            hidden = layer(hidden) + update
            if affine == 0:
                hidden = hidden + actor.root_conditioner(root_feedback)
            affine += 1
        else:
            hidden = layer(hidden)
    if affine != 7:
        raise RuntimeError("normal actor did not execute its seven decoder affines")
    return hidden


class True23NormalLoraActorModel(True23Native23GeneralistActorModel):
    """Distinct versioned RSL5 actor; inherits mechanics, not low-latency setup."""

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        *,
        warm_start_path,
        source_checkpoint_path,
        reference_timing=NORMAL_TIMING,
        tokenizer_obs_group="tokenizer",
        proprioception_obs_group="policy",
        root_feedback_obs_group="root_feedback",
        distribution_cfg=None,
        std_min=0.02,
        std_max=0.5,
        hidden_dims=(),
        activation="silu",
        obs_normalization=False,
        cnn_cfg=None,
    ):
        nn.Module.__init__(self)
        if output_dim != 23 or obs_set != "actor" or reference_timing != NORMAL_TIMING:
            raise ValueError("normal actor requires physical23 outputs and explicit920ms timing")
        if hidden_dims or activation != "silu" or obs_normalization or cnn_cfg is not None:
            raise ValueError("normal actor forbids topology, activation or normalization overrides")
        required = (tokenizer_obs_group, proprioception_obs_group, root_feedback_obs_group)
        if tuple(obs_groups.get(obs_set, ())) != required:
            raise ValueError("normal actor requires tokenizer, policy and separate Root9 groups")
        semantic, history, root = (obs[key] for key in required)
        width = semantic.shape[-1]
        if (
            width not in (267, 268)
            or history.shape[-1] != 930
            or root.shape != (*history.shape[:-1], 9)
            or semantic.shape[:-1] != history.shape[:-1]
        ):
            raise ValueError("normal actor requires matching267/930/9 observation batches")
        cfg = dict(distribution_cfg or {})
        if set(cfg) - {"class_name", "init_std", "std_type"} or (
            cfg.get("class_name", "GaussianDistribution")
            not in (
                "GaussianDistribution",
                "rsl_rl.modules:GaussianDistribution",
                "rsl_rl.modules.distribution:GaussianDistribution",
            )
            or cfg.get("std_type", "scalar") != "scalar"
        ):
            raise ValueError("normal actor requires bounded scalar Gaussian exploration")
        self.distribution = BoundedGaussianDistribution(
            init_std=cfg.get("init_std", 0.1), std_min=std_min, std_max=std_max
        )
        self.tokenizer_obs_group = tokenizer_obs_group
        self.proprioception_obs_group = proprioception_obs_group
        self.root_feedback_obs_group = root_feedback_obs_group
        self.tokenizer_has_encoder_index = width == 268
        self.reference_timing = reference_timing
        self.core = FrozenNormalNative23Core(
            warm_start_path=warm_start_path, source_checkpoint_path=source_checkpoint_path
        )
        with torch.random.fork_rng(devices=[]):
            self.root_conditioner = nn.Linear(9, DIMS[1], bias=False)
        nn.init.zeros_(self.root_conditioner.weight)
        self.lora_a, self.lora_b = nn.ParameterList(), nn.ParameterList()
        for i, (left, right) in enumerate(zip(DIMS[:-1], DIMS[1:], strict=True)):
            a, b = nn.Parameter(torch.empty(RANK, left)), nn.Parameter(torch.zeros(right, RANK))
            nn.init.kaiming_uniform_(
                a, a=math.sqrt(5.0), generator=torch.Generator(device="cpu").manual_seed(INIT_SEED + i)
            )
            self.lora_a.append(a)
            self.lora_b.append(b)
        self._initial_a_sha256 = _tensor_state_sha256(self.lora_a.state_dict())

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state
        if masks is not None:
            from rsl_rl.utils import unpad_trajectories

            obs = unpad_trajectories(obs, masks)
        semantic, history, root = (
            obs[key]
            for key in (self.tokenizer_obs_group, self.proprioception_obs_group, self.root_feedback_obs_group)
        )
        if self.tokenizer_has_encoder_index:
            if not torch.isfinite(semantic[..., :1]).all():
                raise ValueError("normal encoder route contains nonfinite values")
            semantic = semantic[..., 1:]
        if (
            semantic.shape != (*history.shape[:-1], 267)
            or history.shape[-1] != 930
            or root.shape != (*history.shape[:-1], 9)
            or any(
                value.dtype != torch.float32 or value.device != history.device or not torch.isfinite(value).all()
                for value in (semantic, history, root)
            )
        ):
            raise ValueError("normal actor requires finite matching float32 inputs on one device")
        decoded = torch.cat((self.core.encode(semantic), self.core.codec.encode_proprioception(history)), -1)
        mean = normal_conditioned_forward(self, decoded, root)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def artifact_contract(self):
        return {
            **super().artifact_contract(),
            "kind": ACTOR_KIND,
            "reference": normal_reference_contract(),
            "root_feedback": normal_root_feedback_contract(),
            "normal_decoder_adapter": normal_adapter_contract(),
            "root_conditioner_shape": [DIMS[1], 9],
            "actor_trainable": "seven_decoder_lora_root9_bounded_exploration_only",
            "low_latency_checkpoint_relabelling_allowed": False,
            "legacy_single_input_export_permitted": False,
        }

    def parameter_groups(self):
        self.core.assert_frozen_encoder_unchanged()
        adapters = [p for pair in zip(self.lora_a, self.lora_b, strict=True) for p in pair]
        groups = {
            "root_conditioner": list(self.root_conditioner.parameters()),
            "decoder_adapters": adapters,
            "exploration": [self.distribution.raw_std],
        }
        if not all(p.requires_grad for values in groups.values() for p in values):
            raise RuntimeError("normal trainable adapter or exploration became frozen")
        return groups

    def initial_conditioner_is_zero(self):
        return (
            not bool(torch.count_nonzero(self.root_conditioner.weight))
            and all(not bool(torch.count_nonzero(b)) for b in self.lora_b)
            and _tensor_state_sha256(self.lora_a.state_dict()) == self._initial_a_sha256
        )

    def validate_training_artifact(self, artifact):
        super().validate_training_artifact(artifact)
        prefix = "core.actor_module.decoders.g1_dyn."
        frozen = {
            key.removeprefix(prefix): value
            for key, value in artifact["state_dict"].items()
            if key.startswith(prefix)
        }
        if _tensor_state_sha256(frozen) != self.core._frozen_decoder_sha256:
            raise ValueError("normal training artifact changes frozen decoder")

    def validate_base_training_artifact(self, artifact):
        del artifact
        raise ValueError("normal actor cannot relabel another actor family's checkpoint")

    def export_true23_policy_state(self):
        raise RuntimeError("normal Root9 actor requires a separately qualified multi-input export")
