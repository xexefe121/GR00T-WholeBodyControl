"""Root conditioner participates in optimizer, resume and version rejection."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from gear_sonic.tests.test_native23_root_feedback_actor import actor as actor
from gear_sonic.tests.test_native23_generalist_runner import lineage as lineage
from gear_sonic.tests.test_native23_generalist_actor import WARM
from gear_sonic.utils.g1_23dof_mjlab_training import build_mjlab_training_lineage
from gear_sonic.trl.mjlab.native23_generalist_runner import (
    generalist_optimizer_groups,
    validate_generalist_checkpoint,
)
from gear_sonic.trl.mjlab.native23_root_feedback_runner import (
    CHECKPOINT_HEADER,
    Native23RootFeedbackRunner,
    validate_root_feedback_checkpoint,
    optimizer_multipliers,
    validate_optimizer_rates,
    OPTIMIZER_PROFILES,
)


def fixture_runner(actor, lineage, directory):
    runner = object.__new__(Native23RootFeedbackRunner)
    critic = torch.nn.Linear(4, 1)
    runner.alg = SimpleNamespace(
        get_policy=lambda: actor,
        critic=critic,
        optimizer=torch.optim.Adam(generalist_optimizer_groups(actor, critic), lr=1e-4),
        learning_rate=1e-4,
    )
    runner.cfg = {"algorithm": {"learning_rate": 1e-4}}
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=32))
    runner.completed_update_count = runner.current_learning_iteration = 2
    runner._training_lineage, runner._lineage_sha256 = lineage, lineage["lineage_sha256"]
    runner._training_state_poisoned = False
    runner._generalist_contract = actor.artifact_contract()
    runner._generalist_optimizer_ids = tuple(id(p) for g in runner.alg.optimizer.param_groups for p in g["params"])
    runner.checkpoint_dir = directory
    return runner


def test_conditioner_optimizer_group_is_complete_and_disjoint(actor):
    groups = generalist_optimizer_groups(actor, torch.nn.Linear(4, 1))
    assert [group["name"] for group in groups] == ["decoder", "root_conditioner", "exploration", "critic"]
    assert [len(group["params"]) for group in groups] == [18, 1, 1, 2]


@pytest.mark.parametrize("profile", ["legacy_uniform", "feedback_priority"])
def test_root_checkpoint_roundtrip_retains_conditioner_optimizer(actor, lineage, tmp_path, profile):
    if profile == "feedback_priority":
        materials = lineage["materials"]
        lineage = build_mjlab_training_lineage(
            WARM,
            resolved_config={"native23_root_feedback": {"optimizer_profile": profile}},
            source_manifest=materials["source_files"],
            asset_manifest=materials["robot_assets"],
            dataset_manifest=materials["motion_dataset"],
        )
    before = actor.export_training_artifact()
    runner = fixture_runner(actor, lineage, tmp_path)
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = runner.alg.learning_rate * OPTIMIZER_PROFILES[profile][group["name"]]
    try:
        runner.alg.optimizer.zero_grad(set_to_none=True)
        (actor.root_conditioner.weight.sum() + actor.distribution.raw_std.sum()).backward()
        runner.alg.optimizer.step()
        runner.alg.optimizer.zero_grad(set_to_none=True)
        assert torch.count_nonzero(actor.root_conditioner.weight)
        path = runner._numbered_checkpoint_path(2)
        assert path.name == "root_feedback_model_2.pt"
        runner.save(str(path))
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["header"] == CHECKPOINT_HEADER
        with torch.no_grad():
            actor.root_conditioner.weight.add_(1)
            actor.distribution.raw_std.add_(1)
        runner.env.unwrapped.common_step_counter = 64
        runner.load(str(path))
        assert torch.equal(actor.root_conditioner.weight, saved["actor"]["state_dict"]["root_conditioner.weight"])
        assert torch.equal(actor.distribution.raw_std, saved["actor"]["state_dict"]["distribution.raw_std"])
        assert runner.env.unwrapped.common_step_counter == 32
        current = runner.alg.optimizer.state_dict()
        assert current["param_groups"] == saved["optimizer_state_dict"]["param_groups"]
        assert runner._algorithm_learning_rate() == runner.alg.learning_rate
        assert [group["lr"] for group in current["param_groups"]] == pytest.approx(
            [runner.alg.learning_rate * value for value in OPTIMIZER_PROFILES[profile].values()]
        )
        for key, state in current["state"].items():
            for name, value in state.items():
                assert torch.equal(value, saved["optimizer_state_dict"]["state"][key][name])
        with pytest.raises(FileExistsError, match="overwrite"):
            runner.save(str(path))
        with pytest.raises(ValueError, match="header"):
            validate_generalist_checkpoint(saved, actor=actor, lineage=lineage)
    finally:
        actor.zero_grad(set_to_none=True)
        actor.load_training_artifact(before)


@pytest.mark.parametrize(
    "tamper",
    [
        "old_header",
        "missing_conditioner",
        "root_group",
        "root_group_empty",
        "swapped_parameter",
        "lineage",
        "counter",
        "critic",
    ],
)
def test_bad_root_checkpoint_rejected(actor, lineage, tmp_path, tamper):
    runner = fixture_runner(actor, lineage, tmp_path)
    checkpoint = runner._checkpoint_payload()
    if tamper == "old_header":
        from gear_sonic.trl.mjlab.native23_generalist_runner import CHECKPOINT_HEADER as OLD_HEADER

        checkpoint["header"] = deepcopy(OLD_HEADER)
    elif tamper == "missing_conditioner":
        del checkpoint["actor"]["state_dict"]["root_conditioner.weight"]
    elif tamper == "root_group":
        checkpoint["optimizer_state_dict"]["param_groups"][1]["name"] = "encoder"
    elif tamper == "root_group_empty":
        checkpoint["optimizer_state_dict"]["param_groups"][1]["params"] = []
    elif tamper == "swapped_parameter":
        groups = checkpoint["optimizer_state_dict"]["param_groups"]
        groups[1]["params"], groups[2]["params"] = groups[2]["params"], groups[1]["params"]
    elif tamper == "lineage":
        checkpoint["lineage_sha256"] = "0" * 64
    elif tamper == "counter":
        checkpoint["trainer_state"]["current_learning_iteration"] = 3
    else:
        checkpoint["critic_state_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_root_feedback_checkpoint(checkpoint, actor=actor, lineage=lineage)


def test_poisoned_runner_cannot_resume_or_save(actor, lineage, tmp_path):
    runner = fixture_runner(actor, lineage, tmp_path)
    runner._training_state_poisoned = True
    with pytest.raises(RuntimeError):
        runner.load("missing.pt")
    with pytest.raises(RuntimeError):
        runner.save(str(tmp_path / "no.pt"))


def test_feedback_priority_rates_are_explicit_and_differential():
    config = {"native23_root_feedback": {"optimizer_profile": "feedback_priority"}}
    multipliers = optimizer_multipliers(config)
    groups = [{"name": key, "lr": 5e-7 * value} for key, value in multipliers.items()]
    np_rates = [group["lr"] for group in groups]
    assert np_rates == pytest.approx([5e-7, 1e-4, 5e-7, 3e-4])
    assert validate_optimizer_rates(groups, 5e-7, multipliers) == 5e-7
    groups[1]["lr"] = 5e-7
    with pytest.raises(ValueError, match="bound profile"):
        validate_optimizer_rates(groups, 5e-7, multipliers)


def test_legacy_uniform_rate_contract_retained():
    assert optimizer_multipliers({}) == OPTIMIZER_PROFILES["legacy_uniform"]
    with pytest.raises(ValueError, match="multipliers"):
        optimizer_multipliers(
            {
                "native23_root_feedback": {
                    "optimizer_profile": "feedback_priority",
                    "optimizer_learning_rate_multipliers": {"decoder": 1.0},
                }
            }
        )
    with pytest.raises(ValueError, match="unknown"):
        optimizer_multipliers({"native23_root_feedback": {"optimizer_profile": "unbound"}})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_critic_is_rejected_before_digest_comparison(actor, lineage, tmp_path, bad):
    runner = fixture_runner(actor, lineage, tmp_path)
    checkpoint = runner._checkpoint_payload()
    next(iter(checkpoint["critic_state_dict"].values())).view(-1)[0] = bad
    checkpoint["critic_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_root_feedback_checkpoint(checkpoint, actor=actor, lineage=lineage)


def test_unbound_checkpoint_learning_rate_is_rejected(actor, lineage, tmp_path):
    runner = fixture_runner(actor, lineage, tmp_path)
    checkpoint = runner._checkpoint_payload()
    checkpoint["optimizer_state_dict"]["param_groups"][1]["lr"] *= 2
    with pytest.raises(ValueError, match="bound profile"):
        validate_root_feedback_checkpoint(checkpoint, actor=actor, lineage=lineage)
