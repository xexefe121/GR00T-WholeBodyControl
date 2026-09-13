"""Real released weights; frozen-base gradients, partition and snapshot guards."""

from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import _tensor_state_sha256
from gear_sonic.trl.mjlab.native23_generalist_runner import generalist_optimizer_groups
from gear_sonic.trl.mjlab.native23_original_intent_actor import original_intent_compatibility
from gear_sonic.trl.mjlab.native23_pose_lora_actor import ACTOR_KIND, True23PoseLoraActorModel
from gear_sonic.trl.mjlab.native23_pose_lora_runner import (
    CHECKPOINT_HEADER,
    MULTIPLIERS,
    Native23PoseLoraRunner,
    training_contract,
    validate_pose_lora_checkpoint,
)
from gear_sonic.trl.mjlab.native23_root_feedback_runner import validate_root_feedback_checkpoint
from gear_sonic.utils.g1_23dof_contract import SOURCE_IL29_EXCLUDED_INDICES
from gear_sonic.utils.g1_23dof_mjlab_training import build_file_manifest, build_mjlab_training_lineage

ROOT = Path(os.environ.get("G1_TRUE23_TEST_ASSET_ROOT", str(Path(__file__).resolve().parents[2])))
WARM = ROOT / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt"


def observations():
    generator = torch.Generator().manual_seed(97)
    return {
        "tokenizer": torch.randn(2, 267, generator=generator) * 0.1,
        "policy": torch.randn(2, 930, generator=generator) * 0.1,
        "root_feedback": torch.randn(2, 9, generator=generator),
    }


@pytest.fixture(scope="module")
def actor():
    return True23PoseLoraActorModel(
        observations(),
        {"actor": ["tokenizer", "policy", "root_feedback"]},
        "actor",
        23,
        warm_start_path=str(WARM),
        source_checkpoint_path=str(ROOT / "low_latency/last.pt"),
        release_compatibility=original_intent_compatibility("a" * 64, "b" * 64),
    )


def test_zero_step_exact_released_mean_and_honest_contract(actor):
    obs = observations()
    with torch.no_grad():
        assert torch.equal(actor(obs), actor.core(obs["tokenizer"], obs["policy"]))
    contract = actor.artifact_contract()
    assert contract["kind"] == ACTOR_KIND
    assert contract["decoder_frozen"] and contract["trainable_decoder_parameters"] == 0
    assert contract["frozen_decoder_parameters"] > 1_000_000
    assert contract["pose_adapter"]["trainable_parameters"] == 81440
    assert not contract["full_decoder_checkpoint_relabelling_allowed"]


def test_gradient_flows_through_decoder_without_changing_its_weights(actor):
    before = actor.export_training_artifact()
    try:
        groups = generalist_optimizer_groups(actor, torch.nn.Linear(4, 1))
        assert [g["name"] for g in groups] == list(MULTIPLIERS)
        assert [len(g["params"]) for g in groups] == [1, 2, 1, 2]
        optimizer = torch.optim.Adam(groups, lr=1e-4)
        obs = observations()
        obs["tokenizer"].requires_grad_(True)
        output = actor(obs)
        (output.square().sum() + actor.distribution.raw_std.sum()).backward()
        gradient = actor.root_conditioner.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
        assert all(not p.requires_grad and p.grad is None for p in actor.core.parameters())
        assert obs["tokenizer"].grad is None
        optimizer.step()
        actor.core.assert_frozen_encoder_unchanged()
        assert not torch.equal(before["state_dict"]["root_conditioner.weight"], actor.root_conditioner.weight)
        assert all(
            torch.equal(value, actor.state_dict()[name])
            for name, value in before["state_dict"].items()
            if name.startswith("core.")
        )
        with torch.no_grad():
            assert not torch.equal(output, actor(obs))
    finally:
        actor.zero_grad(set_to_none=True)
        actor.load_training_artifact(before)


@pytest.mark.parametrize("component", ["encoder", "decoder"])
def test_freeze_guard_rejects_trainable_base(actor, component):
    parameter = next(getattr(actor.core, component).parameters())
    try:
        parameter.requires_grad_(True)
        with pytest.raises(RuntimeError, match="became trainable"):
            actor.parameter_groups()
    finally:
        parameter.requires_grad_(False)


@pytest.mark.parametrize("component", ["encoders.teleop", "decoders.g1_dyn"])
def test_rehashed_base_weight_tamper_rejected_before_mutation(actor, component):
    before = actor.export_training_artifact()
    invalid = deepcopy(before)
    invalid["state_dict"][f"core.actor_module.{component}.module.0.bias"][0] += 0.125
    invalid["state_sha256"] = _tensor_state_sha256(invalid["state_dict"])
    with pytest.raises(ValueError, match="pinned"):
        actor.load_training_artifact(invalid)
    assert _tensor_state_sha256(actor.state_dict()) == before["state_sha256"]


def test_absent_motor_feedback_remains_ignored(actor):
    obs = observations()
    with torch.no_grad():
        expected = actor(obs)
        for offset in (30, 320, 610):
            obs["policy"][:, offset : offset + 290].reshape(2, 10, 29)[
                :, :, list(SOURCE_IL29_EXCLUDED_INDICES)
            ] = 999
        assert torch.equal(expected, actor(obs))


def test_pose_initialization_preserves_global_rng_and_is_repeatable():
    from gear_sonic.trl.mjlab.native23_pose_lora_actor import initialize_pose_parameters

    before = torch.random.get_rng_state()
    a, b = initialize_pose_parameters(torch.empty(1))
    a2, b2 = initialize_pose_parameters(torch.empty(1))
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(a, a2) and torch.equal(b, b2)
    assert torch.count_nonzero(a) and not torch.count_nonzero(b)


def test_zero_root_training_can_change_pose_and_preserve_base(actor):
    before = actor.export_training_artifact()
    obs = observations()
    obs["root_feedback"].zero_()
    try:
        optimizer = torch.optim.Adam(actor.parameter_groups()["pose_adapter"], lr=1e-4)
        for step in range(2):
            actor.zero_grad(set_to_none=True)
            loss = actor(obs).square().mean()
            loss.backward()
            assert not torch.count_nonzero(actor.root_conditioner.weight.grad)
            assert torch.count_nonzero(actor.pose_lora_b.grad)
            assert bool(torch.count_nonzero(actor.pose_lora_a.grad)) == bool(step)
            optimizer.step()
        with torch.no_grad():
            correction = actor(obs) - actor.core(obs["tokenizer"], obs["policy"])
        assert torch.count_nonzero(correction)
        assert not torch.equal(correction[0], correction[1])
        assert all(
            torch.equal(value, actor.state_dict()[name])
            for name, value in before["state_dict"].items()
            if name.startswith("core.")
        )
        assert not actor.initial_conditioner_is_zero()
    finally:
        actor.zero_grad(set_to_none=True)
        actor.load_training_artifact(before)


def test_cpu_reader_executes_nonzero_pose_branch_not_legacy_root_only(actor):
    from gear_sonic.utils.g1_true23_original_intent_checkpoint import OriginalIntentCPUActor
    from gear_sonic.utils.g1_true23_pose_lora_checkpoint import PoseLoraCPUActor

    before = actor.export_training_artifact()
    obs = {key: value[:1].clone() for key, value in observations().items()}
    obs["root_feedback"].zero_()
    try:
        with torch.no_grad():
            actor.pose_lora_b.fill_(0.025)
            expected = actor(obs).numpy()[0]
        inputs = [obs[key].numpy()[0].copy() for key in ("tokenizer", "policy", "root_feedback")]
        actual, trace = PoseLoraCPUActor(actor).infer(*inputs)
        omitted, old_trace = OriginalIntentCPUActor(actor).infer(*inputs)
        import numpy as np

        assert np.array_equal(expected, actual)
        assert not np.array_equal(omitted, actual)
        assert np.array_equal(trace, old_trace)
        assert trace.shape == (994,)
    finally:
        actor.load_training_artifact(before)


@pytest.mark.parametrize("method", ["as_jit", "as_onnx", "export_true23_policy_state"])
def test_legacy_export_still_rejected(actor, method):
    with pytest.raises(RuntimeError):
        getattr(actor, method)()


@pytest.fixture(scope="module")
def lineage(tmp_path_factory):
    directory = tmp_path_factory.mktemp("pose_lora_lineage")
    manifests = {}
    for kind in ("source_files", "robot_assets", "motion_dataset"):
        path = directory / kind
        path.write_text("non-executable unit-test fixture")
        manifests[kind] = build_file_manifest({kind: path}, kind=kind)
    return build_mjlab_training_lineage(
        WARM,
        resolved_config={"native23_pose_lora": training_contract()},
        source_manifest=manifests["source_files"],
        asset_manifest=manifests["robot_assets"],
        dataset_manifest=manifests["motion_dataset"],
    )


def runner_fixture(actor, lineage, directory):
    runner = object.__new__(Native23PoseLoraRunner)
    critic = torch.nn.Linear(4, 1)
    optimizer = torch.optim.Adam(generalist_optimizer_groups(actor, critic), lr=5e-7)
    for group in optimizer.param_groups:
        group["lr"] = 5e-7 * MULTIPLIERS[group["name"]]
    runner.alg = SimpleNamespace(get_policy=lambda: actor, critic=critic, optimizer=optimizer, learning_rate=5e-7)
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=16))
    runner.completed_update_count = runner.current_learning_iteration = 2
    runner._training_lineage, runner._lineage_sha256 = lineage, lineage["lineage_sha256"]
    runner._training_state_poisoned = False
    runner._generalist_contract = actor.artifact_contract()
    runner._generalist_optimizer_ids = tuple(id(p) for g in optimizer.param_groups for p in g["params"])
    runner.checkpoint_dir = directory
    return runner


def test_new_snapshot_roundtrip_old_reader_and_resume_rejected(actor, lineage, tmp_path):
    runner = runner_fixture(actor, lineage, tmp_path)
    path = runner._numbered_checkpoint_path(2)
    assert path.name == "pose_lora_model_2.pt"
    runner.save(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["header"] == CHECKPOINT_HEADER
    validate_pose_lora_checkpoint(saved, actor=actor, lineage=lineage)
    actor.load_training_artifact(saved["actor"])
    with pytest.raises(ValueError, match="header"):
        validate_root_feedback_checkpoint(saved, actor=actor, lineage=lineage)
    with pytest.raises(FileExistsError):
        runner.save(path)
    with pytest.raises(ValueError, match="fresh-only"):
        runner.load(path)


@pytest.mark.parametrize(
    "tamper", ["decoder_group", "empty_root", "ownership", "lr", "critic", "counter", "header", "lineage"]
)
def test_snapshot_tampering_rejected(actor, lineage, tmp_path, tamper):
    payload = runner_fixture(actor, lineage, tmp_path)._checkpoint_payload()
    groups = payload["optimizer_state_dict"]["param_groups"]
    if tamper == "decoder_group":
        groups[0]["name"] = "decoder"
    elif tamper == "empty_root":
        groups[0]["params"] = []
    elif tamper == "ownership":
        groups[0]["params"], groups[1]["params"] = groups[1]["params"], groups[0]["params"]
    elif tamper == "lr":
        groups[0]["lr"] *= 2
    elif tamper == "critic":
        payload["critic_state_sha256"] = "0" * 64
    elif tamper == "counter":
        payload["trainer_state"]["current_learning_iteration"] += 1
    elif tamper == "header":
        payload["header"]["hardware_authorized"] = True
    else:
        payload["lineage_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_pose_lora_checkpoint(payload, actor=actor, lineage=lineage)
