"""Exercise custom resume with actual actor, mocked runner transport only."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from gear_sonic.tests.test_native23_generalist_actor import WARM, actor as actor
from gear_sonic.trl.mjlab.native23_generalist_runner import (
    Native23GeneralistRunner,
    generalist_optimizer_groups,
    validate_generalist_checkpoint,
)
from gear_sonic.utils.g1_23dof_mjlab_training import (
    build_file_manifest,
    build_mjlab_training_lineage,
)


@pytest.fixture(scope="module")
def lineage(tmp_path_factory):
    directory = tmp_path_factory.mktemp("generalist_lineage_fixture")
    source = directory / "fixture.py"
    asset = directory / "fixture.xml"
    motion = directory / "fixture.npz"
    source.write_text("# non-executable lineage fixture\n")
    asset.write_text("<mujoco/>\n")
    motion.write_bytes(b"non-executable motion fixture")
    return build_mjlab_training_lineage(
        WARM,
        resolved_config={"simulation_only_test_fixture": True},
        source_manifest=build_file_manifest({"fixture.py": source}, kind="source_files"),
        asset_manifest=build_file_manifest({"fixture.xml": asset}, kind="robot_assets"),
        dataset_manifest=build_file_manifest({"fixture.npz": motion}, kind="motion_dataset"),
    )


def fixture_runner(actor, lineage, directory):
    runner = object.__new__(Native23GeneralistRunner)
    critic = torch.nn.Linear(4, 1)
    runner.alg = SimpleNamespace(
        get_policy=lambda: actor,
        critic=critic,
        optimizer=torch.optim.Adam(generalist_optimizer_groups(actor, critic), lr=1e-4),
        learning_rate=1e-4,
    )
    runner.cfg = {"algorithm": {"learning_rate": 1e-4}}
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=128))
    runner.completed_update_count = runner.current_learning_iteration = 2
    runner._training_lineage = lineage
    runner._lineage_sha256 = lineage["lineage_sha256"]
    runner._training_state_poisoned = False
    runner._generalist_contract = actor.artifact_contract()
    runner._generalist_optimizer_ids = tuple(
        id(p) for group in runner.alg.optimizer.param_groups for p in group["params"]
    )
    runner.checkpoint_dir = directory
    return runner


def test_optimizer_has_all_decoder_noise_critic_and_no_encoder(actor):
    critic = torch.nn.Linear(4, 1)
    groups = generalist_optimizer_groups(actor, critic)
    assert [group["name"] for group in groups] == ["decoder", "exploration", "critic"]
    assert [len(group["params"]) for group in groups] == [18, 1, 2]
    ids = {id(p) for group in groups for p in group["params"]}
    assert not ids & {id(p) for p in actor.core.encoder.parameters()}
    critic.requires_grad_(False)
    with pytest.raises(ValueError, match="requires decoder, exploration and critic"):
        generalist_optimizer_groups(actor, critic)


def test_actual_raw_std_optimizer_checkpoint_exact_roundtrip(actor, lineage, tmp_path):
    original = actor.export_training_artifact()
    runner = fixture_runner(actor, lineage, tmp_path)
    try:
        runner.alg.optimizer.zero_grad(set_to_none=True)
        actor.distribution.raw_std.square().mean().backward()
        runner.alg.optimizer.step()
        runner.alg.optimizer.zero_grad(set_to_none=True)
        saved_raw = actor.distribution.raw_std.detach().clone()
        saved_mean_parameter = next(actor.core.decoder.parameters()).detach().flatten()[0].item()
        path = runner._numbered_checkpoint_path(2)
        assert path.name == "native23_generalist_model_2.pt"
        runner.save(str(path))
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["header"]["role"] == "simulator_training_resume_only"
        assert not checkpoint["header"]["deployment_ready"]
        assert not checkpoint["header"]["promotion_eligible"]
        assert not checkpoint["header"]["hardware_authorized"]
        assert checkpoint["optimizer_state_dict"]["state"]
        with torch.no_grad():
            actor.distribution.raw_std.add_(1.0)
            next(actor.core.decoder.parameters()).flatten()[0] += 1.0
            runner.alg.critic.weight.add_(1.0)
        runner.env.unwrapped.common_step_counter = 256
        result = runner.load(str(path))
        assert torch.equal(actor.distribution.raw_std, saved_raw)
        assert next(actor.core.decoder.parameters()).flatten()[0].item() == saved_mean_parameter
        assert torch.equal(runner.alg.critic.weight, checkpoint["critic_state_dict"]["weight"])
        assert runner.env.unwrapped.common_step_counter == 128
        assert runner.completed_update_count == runner.current_learning_iteration == 2
        assert result["actor_state_sha256"] == checkpoint["actor"]["state_sha256"]
        assert (
            runner.alg.optimizer.state_dict()["state"].keys() == checkpoint["optimizer_state_dict"]["state"].keys()
        )
        for index, state in runner.alg.optimizer.state_dict()["state"].items():
            for name, value in state.items():
                assert torch.equal(value, checkpoint["optimizer_state_dict"]["state"][index][name])
        with pytest.raises(FileExistsError, match="overwrite"):
            runner.save(str(path))
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        actor.load_training_artifact(original)


@pytest.mark.parametrize(
    "tamper", ["header", "extra", "counter", "lineage", "critic_hash", "group", "optimizer_nan"]
)
def test_checkpoint_rejects_contract_counter_lineage_optimizer_corruption(actor, lineage, tmp_path, tamper):
    runner = fixture_runner(actor, lineage, tmp_path)
    checkpoint = runner._checkpoint_payload()
    if tamper == "header":
        checkpoint["header"]["deployment_ready"] = True
    elif tamper == "extra":
        checkpoint["unknown"] = True
    elif tamper == "counter":
        checkpoint["trainer_state"]["current_learning_iteration"] = 3
    elif tamper == "lineage":
        checkpoint["lineage_sha256"] = "0" * 64
    elif tamper == "critic_hash":
        checkpoint["critic_state_sha256"] = "0" * 64
    elif tamper == "group":
        checkpoint["optimizer_state_dict"]["param_groups"][0]["name"] = "encoder"
    else:
        checkpoint["optimizer_state_dict"]["state"] = {0: {"exp_avg": torch.tensor(float("nan"))}}
    with pytest.raises(ValueError):
        validate_generalist_checkpoint(checkpoint, actor=actor, lineage=lineage)


def test_old_or_partial_resume_and_poisoned_runner_rejected(actor, lineage, tmp_path):
    runner = fixture_runner(actor, lineage, tmp_path)
    checkpoint = runner._checkpoint_payload()
    with pytest.raises(ValueError, match="predate"):
        validate_generalist_checkpoint(checkpoint, actor=actor, lineage=lineage, minimum_update_count=3)
    with pytest.raises(ValueError, match="complete and strict"):
        runner.load("does_not_exist.pt", load_cfg={})
    with pytest.raises(ValueError, match="complete and strict"):
        runner.load("does_not_exist.pt", strict=False)
    runner._training_state_poisoned = True
    with pytest.raises(RuntimeError, match="partially mutated"):
        runner.save(str(tmp_path / "poisoned.pt"))
    with pytest.raises(RuntimeError, match="partially mutated"):
        runner.load("does_not_exist.pt")


def test_optimizer_drift_detected_before_checkpoint(actor, lineage, tmp_path):
    runner = fixture_runner(actor, lineage, tmp_path)
    runner.alg.optimizer.param_groups[0]["params"].append(next(actor.core.encoder.parameters()))
    with pytest.raises(RuntimeError, match="partition changed"):
        runner._checkpoint_payload()


def test_initial_network_validation_allows_only_explicit_noise_reinitialization(actor, lineage, tmp_path):
    runner = fixture_runner(actor, lineage, tmp_path)
    runner._validate_initial_policy()
    altered = copy.deepcopy(lineage)
    altered["warm_start"]["initial_policy_state_sha256"] = "0" * 64
    runner._training_lineage = altered
    with pytest.raises(ValueError, match="paired-source"):
        runner._validate_initial_policy()
