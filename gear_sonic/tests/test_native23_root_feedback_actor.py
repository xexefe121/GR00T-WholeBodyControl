"""Real SONIC weights: distinct root input without changing frozen semantics."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from gear_sonic.trl.mjlab.native23_generalist_actor import True23Native23GeneralistActorModel
from gear_sonic.trl.mjlab.native23_root_feedback_actor import True23RootFeedbackActorModel
from gear_sonic.trl.mjlab.frozen_platform_lora_actor import _tensor_state_sha256
from gear_sonic.utils.g1_23dof_contract import SOURCE_IL29_EXCLUDED_INDICES
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract

ROOT = Path(__file__).resolve().parents[2]
WARM = ROOT / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"
SOURCE = ROOT / "low_latency/last.pt"


def observations(batch=2, *, routed=False):
    return {
        "tokenizer": torch.zeros(batch, 268 if routed else 267),
        "policy": torch.zeros(batch, 930),
        "root_feedback": torch.zeros(batch, 9),
    }


@pytest.fixture(scope="module")
def actor():
    return True23RootFeedbackActorModel(
        observations(),
        {"actor": ["tokenizer", "policy", "root_feedback"]},
        "actor",
        23,
        warm_start_path=str(WARM),
        source_checkpoint_path=str(SOURCE),
    )


def base_artifact(actor):
    artifact = actor.export_training_artifact()
    artifact["contract"] = True23Native23GeneralistActorModel.artifact_contract(actor)
    del artifact["state_dict"]["root_conditioner.weight"]
    artifact["state_sha256"] = _tensor_state_sha256(artifact["state_dict"])
    return artifact


def test_zero_conditioner_preserves_exact_base_mean_for_arbitrary_feedback(actor):
    obs = observations()
    obs["policy"].normal_(std=0.1)
    obs["root_feedback"].normal_(std=10)
    with torch.no_grad():
        expected = actor.core(obs["tokenizer"], obs["policy"])
        actual = actor(obs)
    assert torch.equal(actual, expected)
    assert actual.shape == (2, 23)
    assert torch.count_nonzero(actor.root_conditioner.weight) == 0
    contract = actor.artifact_contract()
    assert contract["schema_version"] == 2
    assert contract["decoder_inputs"] == {"obs_dict": 994, "root_feedback": 9}
    assert contract["root_feedback_contract"] == root_feedback_contract()
    assert contract["trainable_root_conditioner_parameters"] == 36864
    assert not contract["legacy_single_input_export_permitted"]


def test_nonzero_root_conditioner_changes_action_not_frozen_token(actor):
    try:
        with torch.no_grad():
            actor.root_conditioner.weight.fill_(0.001)
            obs = observations()
            before = actor(obs)
            token = actor.core.encode(obs["tokenizer"])
            obs["root_feedback"][:, 0] = 2.0
            after = actor(obs)
            assert not torch.equal(before, after)
            assert torch.equal(token, actor.core.encode(obs["tokenizer"]))
    finally:
        with torch.no_grad():
            actor.root_conditioner.weight.zero_()


def test_all_decoder_and_conditioner_gradients_nonzero_encoder_frozen(actor):
    actor.zero_grad(set_to_none=True)
    obs = observations()
    obs["root_feedback"].normal_()
    obs["policy"].normal_(std=0.1)
    obs["tokenizer"].requires_grad_(True)
    actor(obs).square().sum().backward()
    for parameter in (*actor.core.decoder.parameters(), *actor.root_conditioner.parameters()):
        assert parameter.requires_grad and parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad)
    assert all(not p.requires_grad and p.grad is None for p in actor.core.encoder.parameters())
    assert obs["tokenizer"].grad is None
    actor.zero_grad(set_to_none=True)


def test_group_partition_and_rsl_distribution(actor):
    groups = actor.parameter_groups()
    assert list(groups) == ["decoder", "root_conditioner", "exploration"]
    assert [len(group) for group in groups.values()] == [18, 1, 1]
    grouped = [id(p) for group in groups.values() for p in group]
    assert len(grouped) == len(set(grouped)) == 20
    assert set(grouped) == {id(p) for p in actor.parameters() if p.requires_grad}
    actions = actor(observations(), stochastic_output=True)
    assert actions.shape == actor.output_mean.shape == actor.output_std.shape == (2, 23)
    assert torch.isfinite(actor.output_entropy).all()
    assert torch.all((actor.output_std >= 0.02) & (actor.output_std <= 0.5))


def test_absent_motor_slots_still_cannot_influence_action(actor):
    obs = observations()
    with torch.no_grad():
        before = actor(obs)
        for offset in (30, 320, 610):
            obs["policy"][:, offset : offset + 290].reshape(2, 10, 29)[
                :, :, list(SOURCE_IL29_EXCLUDED_INDICES)
            ] = 999.0
        assert torch.equal(before, actor(obs))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_feedback_rejected(actor, bad):
    obs = observations()
    obs["root_feedback"][0, 3] = bad
    with pytest.raises(ValueError, match="finite"):
        actor(obs)


@pytest.mark.parametrize("shape", [(2, 8), (2, 10), (1, 9)])
def test_feedback_shape_rejected(actor, shape):
    obs = observations()
    obs["root_feedback"] = torch.zeros(shape)
    with pytest.raises(ValueError, match="matching-batch"):
        actor(obs)


def test_legacy_policy_export_rejected(actor):
    with pytest.raises(RuntimeError, match="legacy single-input"):
        actor.export_true23_policy_state()


def test_base_transfer_preserves_trained_decoder_noise_and_zero_conditioner(actor):
    base = base_artifact(actor)
    before = actor.export_training_artifact()
    try:
        base["state_dict"]["core.actor_module.decoders.g1_dyn.module.16.bias"][0] += 0.125
        base["state_dict"]["distribution.raw_std"][0] += 0.1
        base["state_sha256"] = _tensor_state_sha256(base["state_dict"])
        actor.load_base_training_artifact(base)
        state = actor.state_dict()
        assert all(torch.equal(state[name], value) for name, value in base["state_dict"].items())
        assert not torch.count_nonzero(state["root_conditioner.weight"])
    finally:
        actor.load_training_artifact(before)


@pytest.mark.parametrize("tamper", ["encoder", "hash", "contract", "missing", "nan"])
def test_bad_base_transfer_rejected_without_partial_copy(actor, tamper):
    base = base_artifact(actor)
    before = _tensor_state_sha256(actor.state_dict())
    if tamper == "encoder":
        base["state_dict"]["core.actor_module.encoders.teleop.module.0.bias"][0] += 1
    elif tamper == "contract":
        base["contract"]["reference_profile"] = "wrong"
    elif tamper == "missing":
        del base["state_dict"]["distribution.raw_std"]
    elif tamper == "nan":
        base["state_dict"]["distribution.raw_std"][0] = float("nan")
    else:
        base["state_sha256"] = "0" * 64
    if tamper != "hash":
        base["state_sha256"] = _tensor_state_sha256(base["state_dict"])
    with pytest.raises(ValueError):
        actor.load_base_training_artifact(base)
    assert _tensor_state_sha256(actor.state_dict()) == before


def test_trained_conditioner_cannot_be_erased_by_base_transfer(actor):
    base = base_artifact(actor)
    try:
        with torch.no_grad():
            actor.root_conditioner.weight[0, 0] = 0.1
        with pytest.raises(ValueError, match="untouched zero"):
            actor.load_base_training_artifact(base)
    finally:
        with torch.no_grad():
            actor.root_conditioner.weight.zero_()


def test_versioned_artifact_roundtrip_includes_conditioner(actor):
    before = actor.export_training_artifact()
    try:
        with torch.no_grad():
            actor.root_conditioner.weight.fill_(0.01)
        changed = actor.export_training_artifact()
        actor.load_training_artifact(before)
        actor.load_training_artifact(changed)
        assert torch.equal(actor.root_conditioner.weight, changed["state_dict"]["root_conditioner.weight"])
        old = deepcopy(changed)
        old["contract"] = True23Native23GeneralistActorModel.artifact_contract(actor)
        with pytest.raises(ValueError, match="contract"):
            actor.load_training_artifact(old)
    finally:
        actor.load_training_artifact(before)
