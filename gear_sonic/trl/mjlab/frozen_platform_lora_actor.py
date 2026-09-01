"""Frozen SONIC platform with decoder-only LoRA for true 23-DoF G1.

The released controller remains the source of truth.  True23 observations are
already represented in its canonical padded-29 proprioception space.  This
module therefore needs only an analytic codec:

* encode: fix the six absent canonical joints to zero in every history block;
* decode: select the 23 physical joints from the released 29-action output.

Every released encoder/decoder tensor is frozen.  Zero-effect LoRA adapters on
the dynamics decoder are the only trainable actor parameters.  A rank-16
adapter on the default SONIC decoder contains exactly 245,744 parameters, the
configuration reported by the cross-embodiment transfer recipe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import gc
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from gear_sonic.scripts.init_g1_23dof_checkpoint import (
    _load_pinned_legacy_release,
)
from gear_sonic.utils.g1_23dof_artifact import (
    canonical_json_bytes,
    inspect_true23_policy_state,
)
from gear_sonic.utils.g1_23dof_checkpoint_io import (
    checkpoint_stage,
    load_safe_true23_checkpoint,
)
from gear_sonic.utils.g1_23dof_contract import (
    APPROVED_WARM_START_RELEASES,
    DEPLOYMENT_DECODER_INPUT_DIM,
    DEPLOYMENT_HISTORY_LENGTH,
    NATIVE_IL23_TO_CANONICAL_IL29,
    OBS_LAYOUT_PADDED_IL29,
    SOURCE_DOF,
    SOURCE_IL29_EXCLUDED_INDICES,
    TARGET_DOF,
    TELEOP_ENCODER_INPUT_DIM,
    TELEOP_FSQ_LEVEL,
    TELEOP_TOKEN_COUNT,
    TELEOP_TOKEN_WIDTH,
    TOKEN_DIM,
)

_ENCODER_PREFIX = "actor_module.encoders.teleop.module."
_DECODER_PREFIX = "actor_module.decoders.g1_dyn.module."
_ENCODER_DIMS = (267, 2048, 1024, 512, 512, 64)
_SOURCE_DECODER_DIMS = {
    "true23_step5_0p1s": (994, 2048, 2048, 1024, 1024, 512, 512, 29),
    "released_low_latency_step1_0p02s": (
        994,
        4096,
        4096,
        2048,
        2048,
        1024,
        1024,
        512,
        512,
        29,
    ),
}


def lora_parameter_count(dims: Sequence[int], rank: int) -> int:
    """Count bias-free LoRA A/B tensors on every linear layer."""

    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    if len(dims) < 2 or any(
        isinstance(width, bool) or not isinstance(width, int) or width <= 0
        for width in dims
    ):
        raise ValueError("LoRA dimensions must be positive integers")
    return rank * sum(
        input_dim + output_dim
        for input_dim, output_dim in zip(dims[:-1], dims[1:], strict=True)
    )


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _expected_component_keys(prefix: str, dims: Sequence[int]) -> set[str]:
    return {
        f"{prefix}{2 * index}.{suffix}"
        for index in range(len(dims) - 1)
        for suffix in ("weight", "bias")
    }


def _extract_component(
    state: Mapping[str, Any],
    *,
    prefix: str,
    dims: Sequence[int],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    expected = _expected_component_keys(prefix, dims)
    actual = {name for name in state if name.startswith(prefix)}
    if actual != expected:
        raise ValueError(
            f"frozen platform component mismatch for {prefix}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, (input_dim, output_dim) in enumerate(
        zip(dims[:-1], dims[1:], strict=True)
    ):
        module_index = 2 * index
        weight = state[f"{prefix}{module_index}.weight"]
        bias = state[f"{prefix}{module_index}.bias"]
        if (
            not isinstance(weight, torch.Tensor)
            or not isinstance(bias, torch.Tensor)
            or weight.dtype != torch.float32
            or bias.dtype != torch.float32
            or tuple(weight.shape) != (output_dim, input_dim)
            or tuple(bias.shape) != (output_dim,)
            or not torch.isfinite(weight).all()
            or not torch.isfinite(bias).all()
        ):
            raise ValueError(
                f"frozen platform tensor contract failed at {prefix}{module_index}"
            )
        result.append(
            (
                weight.detach().cpu().contiguous(),
                bias.detach().cpu().contiguous(),
            )
        )
    return tuple(result)


class G1True23AnalyticCodec(nn.Module):
    """Fixed padded-29 input and native-23 output map."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "action_keep_indices",
            torch.tensor(NATIVE_IL23_TO_CANONICAL_IL29, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "missing_indices",
            torch.tensor(SOURCE_IL29_EXCLUDED_INDICES, dtype=torch.long),
            persistent=False,
        )
        keep_mask = torch.ones(
            DEPLOYMENT_DECODER_INPUT_DIM - TOKEN_DIM,
            dtype=torch.float32,
        )
        offset = 3 * DEPLOYMENT_HISTORY_LENGTH
        for _ in range(3):
            block = keep_mask[offset : offset + DEPLOYMENT_HISTORY_LENGTH * SOURCE_DOF]
            block = block.reshape(DEPLOYMENT_HISTORY_LENGTH, SOURCE_DOF)
            block[:, list(SOURCE_IL29_EXCLUDED_INDICES)] = 0.0
            offset += DEPLOYMENT_HISTORY_LENGTH * SOURCE_DOF
        self.register_buffer(
            "proprioception_keep_mask",
            keep_mask,
            persistent=False,
        )

    def validate_padded_proprioception(self, value: torch.Tensor) -> None:
        if value.shape[-1] != DEPLOYMENT_DECODER_INPUT_DIM - TOKEN_DIM:
            raise ValueError("true23 codec requires padded-29 H10 proprioception")
        if not torch.isfinite(value).all():
            raise ValueError("true23 codec input contains NaN or Inf")
        offset = 3 * DEPLOYMENT_HISTORY_LENGTH
        for name in ("joint_pos_rel", "joint_vel", "previous_action"):
            block_size = DEPLOYMENT_HISTORY_LENGTH * SOURCE_DOF
            block = value[..., offset : offset + block_size].reshape(
                *value.shape[:-1], DEPLOYMENT_HISTORY_LENGTH, SOURCE_DOF
            )
            missing = block.index_select(-1, self.missing_indices)
            if torch.count_nonzero(missing):
                raise ValueError(
                    f"true23 codec requires zero-filled absent joints in {name}"
                )
            offset += block_size
        offset += 3 * DEPLOYMENT_HISTORY_LENGTH
        if offset != value.shape[-1]:
            raise AssertionError("padded-29 proprioception layout drift")

    def decode_action(self, source_action: torch.Tensor) -> torch.Tensor:
        if source_action.shape[-1] != SOURCE_DOF:
            raise ValueError("source action must contain 29 canonical joints")
        result = source_action.index_select(-1, self.action_keep_indices)
        if result.shape[-1] != TARGET_DOF:
            raise AssertionError("true23 action projection drift")
        return result

    def encode_proprioception(self, value: torch.Tensor) -> torch.Tensor:
        """Enforce fixed-zero absent slots without a device synchronization."""

        if value.shape[-1] != DEPLOYMENT_DECODER_INPUT_DIM - TOKEN_DIM:
            raise ValueError("true23 codec requires padded-29 H10 proprioception")
        return value * self.proprioception_keep_mask

    def contract(self) -> dict[str, Any]:
        return {
            "kind": "g1_29_to_true23_analytic_codec_v1",
            "encode": "canonical_il29_h10_with_absent_slots_fixed_zero",
            "decode": "native_il23_semantic_row_selection",
            "source_dof": SOURCE_DOF,
            "target_dof": TARGET_DOF,
            "missing_source_indices": list(SOURCE_IL29_EXCLUDED_INDICES),
            "target_to_source_indices": list(NATIVE_IL23_TO_CANONICAL_IL29),
            "learned_parameters": 0,
        }


class _FrozenLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.clone(), requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight, self.bias)


class _FrozenLoRALinear(_FrozenLinear):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        *,
        rank: int,
        alpha: float,
    ) -> None:
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("LoRA alpha must be finite and positive")
        super().__init__(weight, bias)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Parameter(weight.new_empty((rank, weight.shape[1])))
        self.lora_b = nn.Parameter(weight.new_zeros((weight.shape[0], rank)))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = super().forward(value)
        update = F.linear(F.linear(value, self.lora_a), self.lora_b)
        return base + update * self.scaling

    def merged_weight(self) -> torch.Tensor:
        return self.weight + (self.lora_b @ self.lora_a) * self.scaling

    def canonical_merged_weight(self) -> torch.Tensor:
        """Merge on CPU with fixed rank order for portable artifact hashes."""

        merged = self.weight.detach().cpu().float().contiguous().clone()
        lora_a = self.lora_a.detach().cpu().float().contiguous()
        lora_b = self.lora_b.detach().cpu().float().contiguous()
        for rank_index in range(self.rank):
            update = lora_b[:, rank_index : rank_index + 1] * lora_a[
                rank_index : rank_index + 1, :
            ]
            update.mul_(self.scaling)
            merged.add_(update)
        return merged


class _FrozenMlp(nn.Module):
    def __init__(
        self,
        layers: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_FrozenLinear(weight, bias) for weight, bias in layers]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index + 1 != len(self.layers):
                value = F.silu(value)
        return value


class _FrozenLoRAMlp(nn.Module):
    def __init__(
        self,
        layers: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _FrozenLoRALinear(
                    weight,
                    bias,
                    rank=rank,
                    alpha=alpha,
                )
                for weight, bias in layers
            ]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index + 1 != len(self.layers):
                value = F.silu(value)
        return value


class FrozenPlatformTrue23Core(nn.Module):
    """Released teleop encoder + FSQ + LoRA dynamics decoder + codec."""

    def __init__(
        self,
        *,
        warm_start_path: str | Path,
        source_checkpoint_path: str | Path,
        lora_rank: int,
        lora_alpha: float,
    ) -> None:
        super().__init__()
        self.warm_start_path = Path(warm_start_path).expanduser().resolve()
        self.source_checkpoint_path = (
            Path(source_checkpoint_path).expanduser().resolve()
        )
        warm_start = load_safe_true23_checkpoint(
            self.warm_start_path,
            map_location="cpu",
        )
        if checkpoint_stage(warm_start) != "checkpoint_initialization":
            raise ValueError("frozen-platform LoRA requires untouched true23 initialization")
        metadata = warm_start["g1_23dof_metadata"]
        if (
            metadata.get("observation_layout") != OBS_LAYOUT_PADDED_IL29
            or metadata.get("history_length") != DEPLOYMENT_HISTORY_LENGTH
            or metadata.get("decoder_input_dim") != DEPLOYMENT_DECODER_INPUT_DIM
            or metadata.get("decoder_output_dim") != TARGET_DOF
        ):
            raise ValueError("true23 initialization metadata is incompatible with codec")
        initialization = warm_start.get("g1_23dof_initialization_report")
        if not isinstance(initialization, Mapping):
            raise ValueError("true23 initialization lacks source provenance")

        source, source_sha256, release = _load_pinned_legacy_release(
            self.source_checkpoint_path
        )
        approved = APPROVED_WARM_START_RELEASES.get(source_sha256)
        if approved is None or dict(release) != dict(approved):
            raise ValueError("source checkpoint is not an approved frozen SONIC release")
        if (
            initialization.get("source_checkpoint_sha256") != source_sha256
            or initialization.get("reference_profile") != release["reference_profile"]
            or metadata.get("reference_profile") != release["reference_profile"]
        ):
            raise ValueError("source checkpoint and true23 initialization lineage differ")
        source_state = source.get("policy_state_dict")
        if not isinstance(source_state, Mapping):
            raise ValueError("frozen source lacks policy_state_dict")

        self.reference_profile = str(release["reference_profile"])
        try:
            self.source_decoder_dims = _SOURCE_DECODER_DIMS[self.reference_profile]
        except KeyError as exc:
            raise ValueError("unsupported frozen source reference profile") from exc
        encoder_layers = _extract_component(
            source_state,
            prefix=_ENCODER_PREFIX,
            dims=_ENCODER_DIMS,
        )
        decoder_layers = _extract_component(
            source_state,
            prefix=_DECODER_PREFIX,
            dims=self.source_decoder_dims,
        )
        self.encoder = _FrozenMlp(encoder_layers)
        self.decoder = _FrozenLoRAMlp(
            decoder_layers,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        self.codec = G1True23AnalyticCodec()
        self.lora_rank = lora_rank
        self.lora_alpha = float(lora_alpha)
        self.source_checkpoint_sha256 = source_sha256
        self.source_revision = release.get("source_revision")

        noise_names = [
            name
            for name, value in source_state.items()
            if name.rsplit(".", 1)[-1] in {"std", "log_std"}
            and isinstance(value, torch.Tensor)
            and tuple(value.shape) == (SOURCE_DOF,)
        ]
        if len(noise_names) != 1:
            raise ValueError("frozen source must contain exactly one 29-action std")
        source_noise = source_state[noise_names[0]].detach().cpu().float()
        if noise_names[0].rsplit(".", 1)[-1] == "log_std":
            source_noise = source_noise.exp()
        keep = torch.tensor(NATIVE_IL23_TO_CANONICAL_IL29, dtype=torch.long)
        self.initial_std = source_noise.index_select(0, keep).contiguous()
        if not torch.isfinite(self.initial_std).all() or (self.initial_std <= 0).any():
            raise ValueError("frozen source action standard deviation is invalid")

        self.warm_start_metadata = copy.deepcopy(dict(metadata))
        self.warm_start_initialization_report = copy.deepcopy(dict(initialization))
        self._initial_policy_sha256 = inspect_true23_policy_state(
            warm_start,
            reference_profile=self.reference_profile,
        )
        actual_initial = self.merged_true23_policy_sha256(self.initial_std)
        if actual_initial != self._initial_policy_sha256:
            raise RuntimeError(
                "zero-effect LoRA path does not reproduce true23 initialization"
            )
        self._frozen_state_sha256 = self.frozen_state_sha256()
        del source_state, source, warm_start
        gc.collect()

    @staticmethod
    def _fsq(latent: torch.Tensor) -> torch.Tensor:
        half_l = (TELEOP_FSQ_LEVEL - 1) * (1.0 + 1.0e-3) / 2.0
        shift = math.atanh(0.5 / half_l)
        bounded = torch.tanh(latent + shift) * half_l - 0.5
        return torch.round(bounded).div(TELEOP_FSQ_LEVEL // 2)

    def encode(self, semantic: torch.Tensor) -> torch.Tensor:
        if semantic.shape[-1] != TELEOP_ENCODER_INPUT_DIM:
            raise ValueError("frozen teleop encoder requires 267 values")
        latent = self.encoder(semantic).reshape(
            *semantic.shape[:-1],
            TELEOP_TOKEN_COUNT,
            TELEOP_TOKEN_WIDTH,
        )
        return self._fsq(latent).reshape(*semantic.shape[:-1], TOKEN_DIM)

    def forward(
        self,
        semantic: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        proprioception = self.codec.encode_proprioception(proprioception)
        token = self.encode(semantic)
        source_action = self.decoder(torch.cat((token, proprioception), dim=-1))
        return self.codec.decode_action(source_action)

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for index, layer in enumerate(self.decoder.layers):
            result[f"decoder.layers.{index}.lora_a"] = (
                layer.lora_a.detach().cpu().contiguous().clone()
            )
            result[f"decoder.layers.{index}.lora_b"] = (
                layer.lora_b.detach().cpu().contiguous().clone()
            )
        return result

    def load_lora_state_dict(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> None:
        if strict is not True:
            raise ValueError("LoRA state restore must be strict")
        expected = self.lora_state_dict()
        if set(state) != set(expected):
            raise ValueError(
                "LoRA state keys mismatch; "
                f"missing={sorted(set(expected) - set(state))}, "
                f"extra={sorted(set(state) - set(expected))}"
            )
        with torch.no_grad():
            for index, layer in enumerate(self.decoder.layers):
                for suffix in ("lora_a", "lora_b"):
                    name = f"decoder.layers.{index}.{suffix}"
                    value = state[name]
                    target = getattr(layer, suffix)
                    if (
                        not isinstance(value, torch.Tensor)
                        or tuple(value.shape) != tuple(target.shape)
                        or value.dtype != torch.float32
                        or not torch.isfinite(value).all()
                    ):
                        raise ValueError(f"invalid LoRA tensor: {name}")
                    target.copy_(value.to(device=target.device, dtype=target.dtype))

    def export_true23_policy_state(
        self,
        std: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if tuple(std.shape) != (TARGET_DOF,):
            raise ValueError("true23 exported std must have shape [23]")
        result: dict[str, torch.Tensor] = {}
        for index, layer in enumerate(self.encoder.layers):
            module_index = 2 * index
            result[f"{_ENCODER_PREFIX}{module_index}.weight"] = (
                layer.weight.detach().cpu().float().contiguous().clone()
            )
            result[f"{_ENCODER_PREFIX}{module_index}.bias"] = (
                layer.bias.detach().cpu().float().contiguous().clone()
            )
        keep = torch.tensor(NATIVE_IL23_TO_CANONICAL_IL29, dtype=torch.long)
        for index, layer in enumerate(self.decoder.layers):
            module_index = 2 * index
            weight = layer.canonical_merged_weight()
            bias = layer.bias.detach().cpu().float().contiguous()
            if index + 1 == len(self.decoder.layers):
                weight = weight.index_select(0, keep)
                bias = bias.index_select(0, keep)
            result[f"{_DECODER_PREFIX}{module_index}.weight"] = weight.clone()
            result[f"{_DECODER_PREFIX}{module_index}.bias"] = bias.clone()
        result["std"] = std.detach().cpu().float().contiguous().clone()
        return result

    def merged_true23_policy_sha256(self, std: torch.Tensor) -> str:
        """Hash merged policy with at most one full layer materialized."""

        if tuple(std.shape) != (TARGET_DOF,):
            raise ValueError("true23 merged std must have shape [23]")
        entries: list[tuple[str, Any]] = []
        for index, layer in enumerate(self.encoder.layers):
            module_index = 2 * index
            entries.extend(
                (
                    (f"{_ENCODER_PREFIX}{module_index}.weight", layer.weight),
                    (f"{_ENCODER_PREFIX}{module_index}.bias", layer.bias),
                )
            )
        for index, layer in enumerate(self.decoder.layers):
            module_index = 2 * index
            entries.extend(
                (
                    (
                        f"{_DECODER_PREFIX}{module_index}.weight",
                        ("merged_weight", layer, index),
                    ),
                    (f"{_DECODER_PREFIX}{module_index}.bias", ("bias", layer, index)),
                )
            )
        entries.append(("std", std))
        keep = self.codec.action_keep_indices.detach().cpu()
        digest = hashlib.sha256()
        for name, source in sorted(entries, key=lambda item: item[0]):
            if isinstance(source, tuple):
                kind, layer, index = source
                tensor = (
                    layer.canonical_merged_weight()
                    if kind == "merged_weight"
                    else layer.bias
                )
                if index + 1 == len(self.decoder.layers):
                    tensor = tensor.index_select(
                        0,
                        keep.to(device=tensor.device),
                    )
            else:
                tensor = source
            contiguous = tensor.detach().cpu().float().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(canonical_json_bytes(list(contiguous.shape)))
            digest.update(contiguous.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def frozen_state_sha256(self) -> str:
        state: dict[str, torch.Tensor] = {}
        for name, parameter in self.named_parameters():
            if not name.endswith(("lora_a", "lora_b")):
                state[name] = parameter
        return _tensor_state_sha256(state)

    def assert_frozen_platform_unchanged(self) -> None:
        if self.frozen_state_sha256() != self._frozen_state_sha256:
            raise RuntimeError("frozen SONIC platform tensor changed")

    def adapter_contract(self) -> dict[str, Any]:
        trainable = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.endswith(("lora_a", "lora_b"))
        )
        expected_trainable = lora_parameter_count(
            self.source_decoder_dims,
            self.lora_rank,
        )
        if trainable != expected_trainable:
            raise RuntimeError("decoder LoRA parameter count drift")
        return {
            "schema_version": 1,
            "kind": "g1_true23_frozen_sonic_decoder_lora",
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_revision": self.source_revision,
            "reference_profile": self.reference_profile,
            "initial_true23_policy_sha256": self._initial_policy_sha256,
            "frozen_state_sha256": self._frozen_state_sha256,
            "source_decoder_dims": list(self.source_decoder_dims),
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_scaling": self.lora_alpha / self.lora_rank,
            "lora_layers": len(self.decoder.layers),
            "trainable_actor_parameter_count": trainable,
            "codec": self.codec.contract(),
            "zero_effect_initialization": True,
            "merged_policy_hash_basis": (
                "fixed_rank_order_cpu_float32_outer_product_v1"
            ),
            "deployment_ready": False,
            "hardware_authorized": False,
        }


class True23FrozenPlatformLoraActorModel(nn.Module):
    """RSL-RL actor interface for :class:`FrozenPlatformTrue23Core`."""

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
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        tokenizer_obs_group: str = "tokenizer",
        proprioception_obs_group: str = "policy",
        distribution_cfg: dict[str, Any] | None = None,
        hidden_dims: tuple[int, ...] | list[int] = (),
        activation: str = "silu",
        obs_normalization: bool = False,
    ) -> None:
        del hidden_dims, activation
        super().__init__()
        if output_dim != TARGET_DOF or obs_set != "actor":
            raise ValueError("frozen-platform actor requires 23-output actor set")
        if obs_normalization:
            raise ValueError("frozen platform forbids extra observation normalization")
        required = (tokenizer_obs_group, proprioception_obs_group)
        if tuple(obs_groups[obs_set]) != required:
            raise ValueError("frozen-platform actor observation groups mismatch")
        tokenizer_dim = int(obs[tokenizer_obs_group].shape[-1])
        if tokenizer_dim not in {
            TELEOP_ENCODER_INPUT_DIM,
            TELEOP_ENCODER_INPUT_DIM + 1,
        }:
            raise ValueError("frozen-platform tokenizer must be 267 or routed 268")
        if int(obs[proprioception_obs_group].shape[-1]) != 930:
            raise ValueError("frozen-platform proprioception must be padded H10-930")
        self.tokenizer_obs_group = tokenizer_obs_group
        self.proprioception_obs_group = proprioception_obs_group
        self.tokenizer_has_encoder_index = tokenizer_dim == 268
        self.core = FrozenPlatformTrue23Core(
            warm_start_path=warm_start_path,
            source_checkpoint_path=source_checkpoint_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        if distribution_cfg is None:
            distribution_cfg = {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            }
        distribution_cfg = dict(distribution_cfg)
        class_name = distribution_cfg.pop("class_name", "GaussianDistribution")
        if class_name not in {
            "GaussianDistribution",
            "rsl_rl.modules:GaussianDistribution",
            "rsl_rl.modules.distribution:GaussianDistribution",
        } or distribution_cfg.get("std_type", "scalar") != "scalar":
            raise ValueError("frozen-platform actor requires scalar GaussianDistribution")
        from rsl_rl.modules.distribution import GaussianDistribution

        self.distribution = GaussianDistribution(
            output_dim,
            init_std=float(distribution_cfg.get("init_std", 1.0)),
            std_type="scalar",
        )
        with torch.no_grad():
            self.distribution.std_param.copy_(self.core.initial_std)
        self.distribution.std_param.requires_grad_(False)

    @property
    def reference_profile(self) -> str:
        return self.core.reference_profile

    def _split(self, obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = obs[self.tokenizer_obs_group]
        if self.tokenizer_has_encoder_index:
            route = tokenizer[..., :1]
            if not torch.isfinite(route).all():
                raise ValueError("encoder route contains NaN or Inf")
            tokenizer = tokenizer[..., 1:]
        return tokenizer, obs[self.proprioception_obs_group]

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
        semantic, proprioception = self._split(obs)
        mean = self.core(semantic, proprioception)
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
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def update_normalization(self, obs: Any) -> None:
        del obs

    def export_true23_policy_state(self) -> dict[str, torch.Tensor]:
        return self.core.export_true23_policy_state(self.distribution.std_param)

    def export_lora_sidecar(self) -> dict[str, Any]:
        self.core.assert_frozen_platform_unchanged()
        return {
            "contract": self.core.adapter_contract(),
            "adapter_state_dict": self.core.lora_state_dict(),
            "adapter_state_sha256": _tensor_state_sha256(
                self.core.lora_state_dict()
            ),
        }

    def as_jit(self) -> nn.Module:
        raise RuntimeError("merge and hash-validate frozen LoRA before deployment export")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        del verbose
        raise RuntimeError("merge and hash-validate frozen LoRA before deployment export")


__all__ = [
    "FrozenPlatformTrue23Core",
    "G1True23AnalyticCodec",
    "True23FrozenPlatformLoraActorModel",
    "lora_parameter_count",
]
