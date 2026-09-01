from functools import lru_cache
from pathlib import Path

import pytest
import torch

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import (
    FrozenPlatformTrue23Core,
    G1True23AnalyticCodec,
    _FrozenLoRALinear,
    lora_parameter_count,
)
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state
from gear_sonic.utils.g1_23dof_contract import (
    LOW_LATENCY_INITIAL_POLICY_STATE_SHA256,
    NATIVE_IL23_TO_CANONICAL_IL29,
    NORMAL_INITIAL_POLICY_STATE_SHA256,
    SOURCE_IL29_EXCLUDED_INDICES,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _normal_core() -> FrozenPlatformTrue23Core:
    root = _repo_root()
    return FrozenPlatformTrue23Core(
        warm_start_path=root / "sonic_release/g1_23dof_rev_1_0_init.pt",
        source_checkpoint_path=root / "sonic_release/last.pt",
        lora_rank=16,
        lora_alpha=16.0,
    )


@lru_cache(maxsize=1)
def _low_latency_core() -> FrozenPlatformTrue23Core:
    root = _repo_root()
    return FrozenPlatformTrue23Core(
        warm_start_path=(
            root / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"
        ),
        source_checkpoint_path=root / "low_latency/last.pt",
        lora_rank=8,
        lora_alpha=8.0,
    )


def test_analytic_codec_is_exact_semantic_projection() -> None:
    codec = G1True23AnalyticCodec()
    source = torch.arange(29, dtype=torch.float32).repeat(2, 1)
    actual = codec.decode_action(source)
    expected = source[:, list(NATIVE_IL23_TO_CANONICAL_IL29)]

    assert torch.equal(actual, expected)
    assert codec.contract()["learned_parameters"] == 0


def test_analytic_codec_rejects_nonzero_absent_joint_history() -> None:
    codec = G1True23AnalyticCodec()
    proprioception = torch.zeros(1, 930)
    codec.validate_padded_proprioception(proprioception)
    proprioception[0, 30 + SOURCE_IL29_EXCLUDED_INDICES[0]] = 1.0

    with pytest.raises(ValueError, match="zero-filled absent joints"):
        codec.validate_padded_proprioception(proprioception)
    encoded = codec.encode_proprioception(proprioception)
    assert encoded[0, 30 + SOURCE_IL29_EXCLUDED_INDICES[0]] == 0.0


def test_zero_initialized_lora_is_no_effect_and_merge_exact() -> None:
    torch.manual_seed(3)
    weight = torch.randn(7, 5)
    bias = torch.randn(7)
    layer = _FrozenLoRALinear(weight, bias, rank=2, alpha=2.0)
    value = torch.randn(4, 5)

    assert torch.equal(layer(value), torch.nn.functional.linear(value, weight, bias))
    assert torch.equal(layer.merged_weight(), weight)
    assert not layer.weight.requires_grad
    assert not layer.bias.requires_grad
    assert layer.lora_a.requires_grad
    assert layer.lora_b.requires_grad

    with torch.no_grad():
        layer.lora_b.fill_(0.25)
    expected = torch.nn.functional.linear(value, layer.merged_weight(), bias)
    assert torch.allclose(layer(value), expected, rtol=1e-6, atol=1e-6)


def test_reported_adapter_budgets_match_decoder_topologies() -> None:
    normal_dims = (994, 2048, 2048, 1024, 1024, 512, 512, 29)
    low_latency_dims = (
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
    )

    assert lora_parameter_count(normal_dims, 16) == 245_744
    assert lora_parameter_count(low_latency_dims, 8) == 253_944


def test_normal_release_zero_lora_reproduces_hash_bound_true23_warm_start() -> None:
    core = _normal_core()
    contract = core.adapter_contract()
    merged = core.export_true23_policy_state(core.initial_std)
    actual_hash = inspect_true23_policy_state(
        {"policy_state_dict": merged},
        reference_profile=core.reference_profile,
    )

    assert contract["trainable_actor_parameter_count"] == 245_744
    assert contract["initial_true23_policy_sha256"] == (
        NORMAL_INITIAL_POLICY_STATE_SHA256
    )
    assert actual_hash == NORMAL_INITIAL_POLICY_STATE_SHA256
    assert core.merged_true23_policy_sha256(core.initial_std) == actual_hash
    assert all(
        torch.count_nonzero(value) == 0
        for name, value in core.lora_state_dict().items()
        if name.endswith("lora_b")
    )
    core.assert_frozen_platform_unchanged()


def test_low_latency_zero_lora_reproduces_approved_true23_warm_start() -> None:
    core = _low_latency_core()
    contract = core.adapter_contract()
    actual_hash = inspect_true23_policy_state(
        {
            "policy_state_dict": core.export_true23_policy_state(
                core.initial_std
            )
        },
        reference_profile=core.reference_profile,
    )

    assert contract["trainable_actor_parameter_count"] == 253_944
    assert contract["initial_true23_policy_sha256"] == (
        LOW_LATENCY_INITIAL_POLICY_STATE_SHA256
    )
    assert actual_hash == LOW_LATENCY_INITIAL_POLICY_STATE_SHA256
    assert core.merged_true23_policy_sha256(core.initial_std) == actual_hash
    core.assert_frozen_platform_unchanged()
