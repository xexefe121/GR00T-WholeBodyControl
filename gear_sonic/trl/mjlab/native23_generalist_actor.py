"""Native-23 full-decoder SONIC actor; the released tokenizer stays frozen.

This is a simulator-training architecture, not a deployment qualification. The
267 -> FSQ64 tokenizer is copied from the hash-pinned low-latency release. All
nine decoder affine layers are trainable, and the final layer has only the 23
physical output rows. No absent actuator exists in this network's output.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import gc
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import (
    FrozenPlatformTrue23Core,
    G1True23AnalyticCodec,
    _DECODER_PREFIX,
    _ENCODER_DIMS,
    _ENCODER_PREFIX,
    _SOURCE_DECODER_DIMS,
    _extract_component,
    _load_pinned_legacy_release,
    _tensor_state_sha256,
)
from gear_sonic.trl.mjlab.true23_actor import _SonicActorModule
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state
from gear_sonic.utils.g1_23dof_checkpoint_io import (
    checkpoint_stage,
    load_safe_true23_checkpoint,
)
from gear_sonic.utils.g1_23dof_contract import (
    APPROVED_WARM_START_RELEASES,
    DEPLOYMENT_DECODER_INPUT_DIM,
    DEPLOYMENT_HISTORY_LENGTH,
    LOW_LATENCY_INITIAL_POLICY_STATE_SHA256,
    LOW_LATENCY_RELEASE_SHA256,
    NATIVE_IL23_TO_CANONICAL_IL29,
    OBS_LAYOUT_PADDED_IL29,
    REFERENCE_PROFILE_LOW_LATENCY,
    TARGET_DOF,
    TELEOP_ENCODER_INPUT_DIM,
    TOKEN_DIM,
)

GENERALIST_KIND = "g1_native23_full_decoder_frozen_sonic_tokenizer_v1"
STD_MIN = 0.02
STD_MAX = 0.5
INITIAL_STD = 0.1


class Native23GeneralistCore(nn.Module):
    """Exact released paired encoder and row-mapped, fully trainable decoder."""

    def __init__(self, *, warm_start_path: str | Path, source_checkpoint_path: str | Path) -> None:
        super().__init__()
        self.warm_start_path = Path(warm_start_path).expanduser().resolve()
        self.source_checkpoint_path = Path(source_checkpoint_path).expanduser().resolve()
        warm = load_safe_true23_checkpoint(self.warm_start_path, map_location="cpu")
        if checkpoint_stage(warm) != "checkpoint_initialization":
            raise ValueError("generalist requires untouched true23 initialization")
        metadata = warm["g1_23dof_metadata"]
        expected_metadata = {
            "reference_profile": REFERENCE_PROFILE_LOW_LATENCY,
            "observation_layout": OBS_LAYOUT_PADDED_IL29,
            "history_length": DEPLOYMENT_HISTORY_LENGTH,
            "decoder_input_dim": DEPLOYMENT_DECODER_INPUT_DIM,
            "decoder_output_dim": TARGET_DOF,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError("generalist requires exact low-latency H10 true23 metadata")
        initialization = warm.get("g1_23dof_initialization_report")
        if not isinstance(initialization, Mapping) or (
            initialization.get("source_checkpoint_sha256") != LOW_LATENCY_RELEASE_SHA256
            or initialization.get("reference_profile") != REFERENCE_PROFILE_LOW_LATENCY
        ):
            raise ValueError("generalist initialization source lineage mismatch")
        initial_hash = inspect_true23_policy_state(warm, reference_profile=REFERENCE_PROFILE_LOW_LATENCY)
        if initial_hash != LOW_LATENCY_INITIAL_POLICY_STATE_SHA256:
            raise ValueError("generalist initialization is not the approved row mapping")
        self.warm_start_metadata = copy.deepcopy(dict(metadata))
        self.warm_start_initialization_report = copy.deepcopy(dict(initialization))
        self.warm_start_stage = "checkpoint_initialization"
        del warm
        gc.collect()

        source, source_hash, release = _load_pinned_legacy_release(self.source_checkpoint_path)
        if source_hash != LOW_LATENCY_RELEASE_SHA256 or dict(release) != dict(
            APPROVED_WARM_START_RELEASES[LOW_LATENCY_RELEASE_SHA256]
        ):
            raise ValueError("generalist source must be the pinned low-latency release")
        source_state = source.get("policy_state_dict")
        if not isinstance(source_state, Mapping):
            raise ValueError("generalist source lacks policy_state_dict")
        self.reference_profile = REFERENCE_PROFILE_LOW_LATENCY
        self.source_checkpoint_sha256 = source_hash
        self.source_revision = release["source_revision"]
        source_dims = _SOURCE_DECODER_DIMS[self.reference_profile]
        self.decoder_dims = (*source_dims[:-1], TARGET_DOF)
        encoder_layers = _extract_component(source_state, prefix=_ENCODER_PREFIX, dims=_ENCODER_DIMS)
        decoder_layers = _extract_component(source_state, prefix=_DECODER_PREFIX, dims=source_dims)
        self.actor_module = _SonicActorModule(self.decoder_dims)
        self.codec = G1True23AnalyticCodec()
        keep = torch.tensor(NATIVE_IL23_TO_CANONICAL_IL29, dtype=torch.long)
        mapped: dict[str, torch.Tensor] = {}
        for prefix, layers in ((_ENCODER_PREFIX, encoder_layers), (_DECODER_PREFIX, decoder_layers)):
            for index, (weight, bias) in enumerate(layers):
                if prefix == _DECODER_PREFIX and index == len(layers) - 1:
                    weight, bias = weight.index_select(0, keep), bias.index_select(0, keep)
                mapped[f"{prefix}{2 * index}.weight"] = weight
                mapped[f"{prefix}{2 * index}.bias"] = bias
        self.load_state_dict(mapped, strict=True)
        noise_keys = [key for key in source_state if key.rsplit(".", 1)[-1] in {"std", "log_std"}]
        if len(noise_keys) != 1:
            raise ValueError("generalist source must have exactly one exploration tensor")
        noise = source_state[noise_keys[0]]
        if not isinstance(noise, torch.Tensor) or tuple(noise.shape) != (29,):
            raise ValueError("generalist source exploration tensor must have shape [29]")
        noise = noise.detach().cpu().float()
        if noise_keys[0].rsplit(".", 1)[-1] == "log_std":
            noise = noise.exp()
        self.initial_std = noise.index_select(0, keep).contiguous()
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(True)
        self._initial_policy_sha256 = initial_hash
        del source, source_state, mapped, encoder_layers, decoder_layers
        gc.collect()
        actual_hash = inspect_true23_policy_state(
            {"policy_state_dict": self.export_policy_state(self.initial_std)},
            reference_profile=self.reference_profile,
        )
        if actual_hash != initial_hash:
            raise RuntimeError("generalist zero-step policy differs from exact mapped release")
        self._frozen_encoder_sha256 = self.frozen_encoder_sha256()

    @property
    def encoder(self) -> nn.Module:
        return self.actor_module.encoders["teleop"]

    @property
    def decoder(self) -> nn.Module:
        return self.actor_module.decoders["g1_dyn"]

    def encode(self, semantic: torch.Tensor) -> torch.Tensor:
        if semantic.shape[-1] != TELEOP_ENCODER_INPUT_DIM:
            raise ValueError("generalist tokenizer requires 267 values")
        # Frozen means no encoder/input gradient graph, even when PPO enables
        # gradients for every observation tensor. FSQ retains exact rounding.
        with torch.no_grad():
            token = FrozenPlatformTrue23Core._fsq(self.encoder(semantic))
        return token

    def forward(self, semantic: torch.Tensor, proprioception: torch.Tensor) -> torch.Tensor:
        proprioception = self.codec.encode_proprioception(proprioception)
        return self.decoder(torch.cat((self.encode(semantic), proprioception), dim=-1))

    def export_policy_state(self, std: torch.Tensor) -> dict[str, torch.Tensor]:
        if tuple(std.shape) != (TARGET_DOF,) or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("generalist exported std must be finite positive [23]")
        result = {
            name: value.detach().cpu().float().contiguous().clone() for name, value in self.state_dict().items()
        }
        if any(not torch.isfinite(value).all() for value in result.values()):
            raise ValueError("generalist policy contains non-finite tensors")
        result["std"] = std.detach().cpu().float().contiguous().clone()
        return result

    def frozen_encoder_sha256(self) -> str:
        return _tensor_state_sha256(self.encoder.state_dict())

    def assert_frozen_encoder_unchanged(self) -> None:
        if any(parameter.requires_grad for parameter in self.encoder.parameters()):
            raise RuntimeError("generalist encoder became trainable")
        if not all(parameter.requires_grad for parameter in self.decoder.parameters()):
            raise RuntimeError("generalist decoder layer became frozen")
        if self.frozen_encoder_sha256() != self._frozen_encoder_sha256:
            raise RuntimeError("generalist frozen encoder tensor changed")

    def artifact_contract(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": GENERALIST_KIND,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_revision": self.source_revision,
            "reference_profile": self.reference_profile,
            "initial_true23_policy_sha256": self._initial_policy_sha256,
            "frozen_encoder_sha256": self._frozen_encoder_sha256,
            "encoder_input_dim": TELEOP_ENCODER_INPUT_DIM,
            "token_dim": TOKEN_DIM,
            "decoder_dims": list(self.decoder_dims),
            "trainable_decoder_parameters": sum(p.numel() for p in self.decoder.parameters()),
            "encoder_frozen": True,
            "fsq_frozen": True,
            "decoder_training": "all_affine_weights_and_biases",
            "physical_output_to_canonical29": list(NATIVE_IL23_TO_CANONICAL_IL29),
            "proprioception_codec": self.codec.contract(),
            "zero_step_mean_exact_row_mapping": True,
            "deployment_ready": False,
            "hardware_authorized": False,
        }


class BoundedGaussianDistribution(nn.Module):
    """Trainable per-action sigmoid std with finite, strictly positive bounds.

    The untransformed parameter is checkpointed for exact optimizer resume.
    Exported SONIC policies contain the transformed 23-value direct ``std``.
    No global torch distribution-validation setting is modified.
    """

    def __init__(
        self, *, init_std: float = INITIAL_STD, std_min: float = STD_MIN, std_max: float = STD_MAX
    ) -> None:
        super().__init__()
        if any(isinstance(v, bool) or not math.isfinite(v) for v in (init_std, std_min, std_max)):
            raise ValueError("generalist std settings must be finite numbers")
        if not 0 < std_min < init_std < std_max:
            raise ValueError("generalist std requires 0 < min < init < max")
        self.output_dim = TARGET_DOF
        self.std_min, self.std_max = float(std_min), float(std_max)
        self.init_std = float(init_std)
        fraction = (init_std - std_min) / (std_max - std_min)
        self.raw_std = nn.Parameter(torch.full((TARGET_DOF,), math.log(fraction / (1 - fraction))))
        self._distribution: Normal | None = None

    @property
    def bounded_std(self) -> torch.Tensor:
        if not torch.isfinite(self.raw_std).all():
            raise ValueError("generalist raw exploration parameter is non-finite")
        return self.std_min + (self.std_max - self.std_min) * self.raw_std.sigmoid()

    def update(self, mean: torch.Tensor) -> None:
        if mean.shape[-1] != TARGET_DOF or not torch.isfinite(mean).all():
            raise ValueError("generalist distribution mean must be finite with 23 actions")
        self._distribution = Normal(mean, self.bounded_std.expand_as(mean), validate_args=False)

    def _current(self) -> Normal:
        if self._distribution is None:
            raise RuntimeError("generalist distribution needs update before sampling/statistics")
        return self._distribution

    def sample(self) -> torch.Tensor:
        return self._current().sample()

    @property
    def mean(self) -> torch.Tensor:
        return self._current().mean

    @property
    def std(self) -> torch.Tensor:
        return self._current().stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self._current().entropy().sum(-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return self.mean, self.std

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._current().log_prob(outputs).sum(-1)

    @staticmethod
    def kl_divergence(old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        return torch.distributions.kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(-1)

    def contract(self) -> dict[str, Any]:
        return {
            "parameterization": "per_action_bounded_sigmoid",
            "std_min": self.std_min,
            "std_max": self.std_max,
            "init_std": self.init_std,
            "trainable": True,
            "source_exploration_reused": False,
        }


class True23Native23GeneralistActorModel(nn.Module):
    """RSL-RL 5 model interface for full native23 dynamics-decoder training."""

    is_recurrent = False

    def __init__(
        self,
        obs: Any,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        warm_start_path: str,
        source_checkpoint_path: str,
        std_min: float = STD_MIN,
        std_max: float = STD_MAX,
        tokenizer_obs_group: str = "tokenizer",
        proprioception_obs_group: str = "policy",
        distribution_cfg: dict[str, Any] | None = None,
        hidden_dims: tuple[int, ...] | list[int] = (),
        activation: str = "silu",
        obs_normalization: bool = False,
        cnn_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if output_dim != TARGET_DOF or obs_set != "actor":
            raise ValueError("generalist requires 23-output actor set")
        if obs_normalization or hidden_dims or activation != "silu" or cnn_cfg is not None:
            raise ValueError("generalist forbids normalization or topology/activation overrides")
        if tuple(obs_groups.get(obs_set, ())) != (tokenizer_obs_group, proprioception_obs_group):
            raise ValueError("generalist actor observation groups mismatch")
        width = obs[tokenizer_obs_group].shape[-1]
        if width not in (267, 268) or obs[proprioception_obs_group].shape[-1] != 930:
            raise ValueError("generalist requires teleop267 (or routed268) and H10-930")
        cfg = dict(distribution_cfg or {})
        if set(cfg) - {"class_name", "init_std", "std_type"}:
            raise ValueError("generalist received unknown distribution settings")
        if (
            cfg.get("class_name", "GaussianDistribution")
            not in (
                "GaussianDistribution",
                "rsl_rl.modules:GaussianDistribution",
                "rsl_rl.modules.distribution:GaussianDistribution",
            )
            or cfg.get("std_type", "scalar") != "scalar"
        ):
            raise ValueError("generalist requires bounded scalar Gaussian exploration")
        self.distribution = BoundedGaussianDistribution(
            init_std=cfg.get("init_std", INITIAL_STD), std_min=std_min, std_max=std_max
        )
        self.tokenizer_obs_group, self.proprioception_obs_group = tokenizer_obs_group, proprioception_obs_group
        self.tokenizer_has_encoder_index = width == 268
        self.core = Native23GeneralistCore(
            warm_start_path=warm_start_path, source_checkpoint_path=source_checkpoint_path
        )

    @property
    def reference_profile(self) -> str:
        return self.core.reference_profile

    def forward(
        self,
        obs: Any,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del hidden_state
        if masks is not None:
            from rsl_rl.utils import unpad_trajectories

            obs = unpad_trajectories(obs, masks)
        semantic = obs[self.tokenizer_obs_group]
        if self.tokenizer_has_encoder_index:
            if not torch.isfinite(semantic[..., :1]).all():
                raise ValueError("generalist encoder route contains non-finite values")
            semantic = semantic[..., 1:]
        mean = self.core(semantic, obs[self.proprioception_obs_group])
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def reset(self, dones: torch.Tensor | None = None, hidden_state: Any = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> None:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def update_normalization(self, obs: Any) -> None:
        del obs

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def export_true23_policy_state(self) -> dict[str, torch.Tensor]:
        self.core.assert_frozen_encoder_unchanged()
        return self.core.export_policy_state(self.distribution.bounded_std)

    def artifact_contract(self) -> dict[str, Any]:
        return {**self.core.artifact_contract(), "exploration": self.distribution.contract()}

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        self.core.assert_frozen_encoder_unchanged()
        return {"decoder": list(self.core.decoder.parameters()), "exploration": [self.distribution.raw_std]}

    def export_training_artifact(self) -> dict[str, Any]:
        self.core.assert_frozen_encoder_unchanged()
        state = {name: value.detach().cpu().contiguous().clone() for name, value in self.state_dict().items()}
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("generalist training state contains non-finite values")
        return {
            "contract": self.artifact_contract(),
            "state_dict": state,
            "state_sha256": _tensor_state_sha256(state),
        }

    def validate_training_artifact(self, artifact: Mapping[str, Any]) -> None:
        if not isinstance(artifact, Mapping) or set(artifact) != {"contract", "state_dict", "state_sha256"}:
            raise ValueError("generalist actor artifact fields mismatch")
        if artifact["contract"] != self.artifact_contract():
            raise ValueError("generalist actor artifact contract mismatch")
        state, expected = artifact["state_dict"], self.state_dict()
        if not isinstance(state, Mapping) or set(state) != set(expected):
            raise ValueError("generalist actor state keys mismatch")
        for name, value in state.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != expected[name].dtype
                or value.shape != expected[name].shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"generalist actor state tensor invalid: {name}")
        if artifact["state_sha256"] != _tensor_state_sha256(state):
            raise ValueError("generalist actor state hash mismatch")
        prefix = "core.actor_module.encoders.teleop."
        frozen = {name.removeprefix(prefix): value for name, value in state.items() if name.startswith(prefix)}
        if _tensor_state_sha256(frozen) != self.core._frozen_encoder_sha256:
            raise ValueError("generalist artifact changes pinned encoder")

    def load_training_artifact(self, artifact: Mapping[str, Any]) -> None:
        # Validate every field before the first in-place copy; no partially
        # loaded decoder on a malformed or encoder-mismatched artifact.
        self.validate_training_artifact(artifact)
        self.load_state_dict(artifact["state_dict"], strict=True)
        self.distribution._distribution = None
        self.core.assert_frozen_encoder_unchanged()

    def as_jit(self) -> nn.Module:
        raise RuntimeError("generalist requires independently qualified split SONIC export")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        raise RuntimeError("generalist requires independently qualified split SONIC export")


__all__ = ["BoundedGaussianDistribution", "Native23GeneralistCore", "True23Native23GeneralistActorModel"]
