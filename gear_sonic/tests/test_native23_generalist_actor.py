"""Real released-weight checks; no simulator, DDS participant or robot use."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from gear_sonic.trl.mjlab import native23_generalist_actor as module
from gear_sonic.trl.mjlab.frozen_platform_lora_actor import _tensor_state_sha256
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state
from gear_sonic.utils.g1_23dof_checkpoint_io import load_safe_true23_checkpoint
from gear_sonic.utils.g1_23dof_contract import (
    LOW_LATENCY_INITIAL_POLICY_STATE_SHA256,
    LOW_LATENCY_RELEASE_SHA256,
    NATIVE_IL23_TO_CANONICAL_IL29,
    SOURCE_IL29_EXCLUDED_INDICES,
)

ROOT = Path(__file__).resolve().parents[2]
WARM = ROOT / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"
SOURCE = ROOT / "low_latency/last.pt"


def observations(batch: int = 2, *, routed: bool = False) -> dict[str, torch.Tensor]:
    return {"tokenizer": torch.zeros(batch, 268 if routed else 267), "policy": torch.zeros(batch, 930)}


@pytest.fixture(scope="module")
def actor():
    return module.True23Native23GeneralistActorModel(
        observations(),
        {"actor": ["tokenizer", "policy"]},
        "actor",
        23,
        warm_start_path=str(WARM),
        source_checkpoint_path=str(SOURCE),
    )


def test_real_paired_release_exact_initial_network_and_23_rows(actor):
    contract = actor.artifact_contract()
    assert contract["source_checkpoint_sha256"] == LOW_LATENCY_RELEASE_SHA256
    assert contract["decoder_dims"] == [994, 4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 23]
    assert contract["physical_output_to_canonical29"] == list(NATIVE_IL23_TO_CANONICAL_IL29)
    assert not set(NATIVE_IL23_TO_CANONICAL_IL29) & set(SOURCE_IL29_EXCLUDED_INDICES)
    state = actor.core.export_policy_state(actor.core.initial_std)
    assert (
        inspect_true23_policy_state({"policy_state_dict": state}, reference_profile=actor.reference_profile)
        == LOW_LATENCY_INITIAL_POLICY_STATE_SHA256
    )
    assert state["actor_module.decoders.g1_dyn.module.16.weight"].shape == (23, 512)
    assert state["actor_module.decoders.g1_dyn.module.16.bias"].shape == (23,)
    assert not contract["deployment_ready"] and not contract["hardware_authorized"]
    assert contract["exploration"]["init_std"] == 0.1
    assert not contract["exploration"]["source_exploration_reused"]


def test_every_decoder_layer_receives_gradient_encoder_does_not(actor):
    actor.zero_grad(set_to_none=True)
    obs = observations()
    obs["tokenizer"].requires_grad_(True)
    obs["policy"].normal_(std=0.01).requires_grad_(True)
    mean = actor(obs)
    assert mean.shape == (2, 23)
    mean.square().sum().backward()
    decoder = list(actor.core.decoder.named_parameters())
    assert len(decoder) == 18
    assert all(
        p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for _, p in decoder
    )
    assert all(not p.requires_grad and p.grad is None for p in actor.core.encoder.parameters())
    assert obs["tokenizer"].grad is None
    assert obs["policy"].grad is not None
    actor.core.assert_frozen_encoder_unchanged()
    actor.zero_grad(set_to_none=True)


def test_exact_fsq_grid_and_absent_proprioception_slots_cannot_influence_action(actor):
    obs = observations()
    with torch.no_grad():
        baseline = actor(obs)
        tokens = actor.core.encode(obs["tokenizer"])
        assert tokens.shape == (2, 64)
        assert not tokens.requires_grad
        assert torch.equal(tokens * 16, (tokens * 16).round())
        offset = 30
        for _ in range(3):
            block = obs["policy"][:, offset : offset + 290].reshape(2, 10, 29)
            block[:, :, list(SOURCE_IL29_EXCLUDED_INDICES)] = 999
            offset += 290
        assert torch.equal(actor(obs), baseline)


def test_rsl_interface_and_trainable_parameter_partition(actor):
    actor.zero_grad(set_to_none=True)
    sampled = actor(observations(), stochastic_output=True)
    assert sampled.shape == (2, 23)
    assert actor.output_mean.shape == actor.output_std.shape == (2, 23)
    assert actor.output_entropy.shape == actor.get_output_log_prob(sampled).shape == (2,)
    assert len(actor.output_distribution_params) == 2
    assert torch.equal(
        actor.get_kl_divergence(actor.output_distribution_params, actor.output_distribution_params), torch.zeros(2)
    )
    (-actor.output_entropy.mean()).backward()
    assert actor.distribution.raw_std.grad is not None
    assert actor.distribution.raw_std.grad.abs().sum() > 0
    groups = actor.parameter_groups()
    assert list(groups) == ["decoder", "exploration"]
    grouped_ids = [id(p) for group in groups.values() for p in group]
    assert len(grouped_ids) == 19 == len(set(grouped_ids))
    assert set(grouped_ids) == {id(p) for p in actor.parameters() if p.requires_grad}
    actor.reset()
    actor.detach_hidden_state()
    actor.update_normalization(observations())
    assert actor.get_hidden_state() is None and not actor.is_recurrent
    actor.zero_grad(set_to_none=True)


@pytest.mark.parametrize("raw", [-1e30, -100, 0, 100, 1e30])
def test_exploration_remains_bounded_for_finite_raw_parameter(raw):
    distribution = module.BoundedGaussianDistribution()
    with torch.no_grad():
        distribution.raw_std.fill_(raw)
    distribution.update(torch.zeros(3, 23))
    assert torch.isfinite(distribution.std).all()
    assert (distribution.std >= 0.02).all() and (distribution.std <= 0.5).all()
    assert torch.isfinite(distribution.entropy).all()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_exploration_fails_closed(value):
    distribution = module.BoundedGaussianDistribution()
    with torch.no_grad():
        distribution.raw_std[0] = value
    with pytest.raises(ValueError, match="non-finite"):
        distribution.update(torch.zeros(1, 23))


@pytest.mark.parametrize(
    "settings",
    [
        {"std_min": 0},
        {"std_min": 0.1},
        {"std_max": 0.1},
        {"init_std": 0.5},
        {"init_std": float("nan")},
        {"init_std": True},
        {"std_min": -0.1},
    ],
)
def test_bad_std_settings_rejected(settings):
    with pytest.raises(ValueError, match="std"):
        module.BoundedGaussianDistribution(**settings)


def test_distribution_requires_update_and_does_not_set_global_torch_validation():
    before = torch.distributions.Distribution._validate_args
    distribution = module.BoundedGaussianDistribution()
    with pytest.raises(RuntimeError, match="needs update"):
        distribution.sample()
    assert torch.distributions.Distribution._validate_args == before
    with pytest.raises(ValueError, match="23 actions"):
        distribution.update(torch.zeros(1, 29))


@pytest.mark.parametrize(
    "change",
    [
        {"output_dim": 29},
        {"obs_set": "critic"},
        {"obs_normalization": True},
        {"hidden_dims": [512]},
        {"activation": "relu"},
        {"distribution_cfg": {"init_std": 0.1, "unknown": 2}},
        {"distribution_cfg": {"std_type": "log"}},
    ],
)
def test_invalid_actor_configuration_rejected_before_source_loading(change, monkeypatch):
    monkeypatch.setattr(
        module, "Native23GeneralistCore", lambda **kwargs: pytest.fail("invalid actor reached source loading")
    )
    settings = dict(
        obs=observations(),
        obs_groups={"actor": ["tokenizer", "policy"]},
        obs_set="actor",
        output_dim=23,
        warm_start_path=str(WARM),
        source_checkpoint_path=str(SOURCE),
    )
    settings.update(change)
    with pytest.raises(ValueError):
        module.True23Native23GeneralistActorModel(**settings)


def test_routed_observation_and_invalid_width(actor, monkeypatch):
    monkeypatch.setattr(module, "Native23GeneralistCore", lambda **kwargs: actor.core)
    routed = module.True23Native23GeneralistActorModel(
        observations(routed=True),
        {"actor": ["tokenizer", "policy"]},
        "actor",
        23,
        warm_start_path=str(WARM),
        source_checkpoint_path=str(SOURCE),
    )
    with torch.no_grad():
        assert torch.equal(routed(observations(routed=True)), actor(observations()))
    bad = observations(routed=True)
    bad["tokenizer"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="route"):
        routed(bad)
    with pytest.raises(ValueError, match="267"):
        actor.core.encode(torch.zeros(1, 266))
    with pytest.raises(ValueError, match="H10"):
        actor.core(torch.zeros(1, 267), torch.zeros(1, 929))


@pytest.mark.parametrize("method", ["as_jit", "as_onnx"])
def test_unqualified_deployment_export_rejected(actor, method):
    with pytest.raises(RuntimeError, match="qualified split"):
        getattr(actor, method)()


def test_artifact_exact_roundtrip_and_rejects_rehashed_frozen_encoder_changes(actor):
    artifact = actor.export_training_artifact()
    param = next(actor.core.decoder.parameters())
    with torch.no_grad():
        param[0, 0] += 0.1
        actor.distribution.raw_std.add_(0.2)
    actor.load_training_artifact(artifact)
    assert _tensor_state_sha256(actor.state_dict()) == artifact["state_sha256"]
    name = next(n for n in artifact["state_dict"] if n.startswith("core.actor_module.encoders."))
    original = artifact["state_dict"][name].flatten()[0].item()
    artifact["state_dict"][name].flatten()[0] += 0.1
    artifact["state_sha256"] = _tensor_state_sha256(artifact["state_dict"])
    before_decoder = param[0, 0].item()
    with pytest.raises(ValueError, match="pinned encoder"):
        actor.load_training_artifact(artifact)
    assert param[0, 0].item() == before_decoder
    assert dict(actor.state_dict())[name].flatten()[0].item() == original


@pytest.mark.parametrize("tamper", ["contract", "missing_key", "extra_key", "hash", "shape", "dtype", "nan"])
def test_invalid_artifacts_rejected_before_copy(actor, tamper):
    artifact = actor.export_training_artifact()
    if tamper == "contract":
        artifact["contract"]["hardware_authorized"] = True
    elif tamper == "extra_key":
        artifact["extra"] = True
    elif tamper == "missing_key":
        del artifact["state_dict"]["distribution.raw_std"]
    elif tamper == "hash":
        artifact["state_sha256"] = "0" * 64
    elif tamper == "shape":
        artifact["state_dict"]["distribution.raw_std"] = torch.zeros(29)
    elif tamper == "dtype":
        artifact["state_dict"]["distribution.raw_std"] = torch.zeros(23, dtype=torch.float64)
    else:
        artifact["state_dict"]["distribution.raw_std"][0] = float("nan")
    before = actor.distribution.raw_std.detach().clone()
    with pytest.raises(ValueError):
        actor.load_training_artifact(artifact)
    assert torch.equal(actor.distribution.raw_std, before)


def test_trained_or_relabelled_warm_start_rejected_before_source_read(monkeypatch):
    warm = load_safe_true23_checkpoint(WARM, map_location="cpu")
    monkeypatch.setattr(module, "load_safe_true23_checkpoint", lambda *args, **kwargs: warm)
    monkeypatch.setattr(module, "checkpoint_stage", lambda value: "trained")
    monkeypatch.setattr(
        module, "_load_pinned_legacy_release", lambda value: pytest.fail("invalid stage reached source read")
    )
    with pytest.raises(ValueError, match="untouched"):
        module.Native23GeneralistCore(warm_start_path=WARM, source_checkpoint_path=SOURCE)
    monkeypatch.setattr(module, "checkpoint_stage", lambda value: "checkpoint_initialization")
    warm["g1_23dof_metadata"] = copy.deepcopy(warm["g1_23dof_metadata"])
    warm["g1_23dof_metadata"]["reference_profile"] = "true23_step5_0p1s"
    with pytest.raises(ValueError, match="low-latency"):
        module.Native23GeneralistCore(warm_start_path=WARM, source_checkpoint_path=SOURCE)
