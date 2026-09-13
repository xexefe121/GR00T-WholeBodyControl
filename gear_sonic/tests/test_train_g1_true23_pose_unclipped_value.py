"""The diagnostic-backed candidate changes a single optimizer choice."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from gear_sonic.scripts.train_g1_true23_pose_unclipped_value import (
    configure_agent,
    require_unclipped,
    value_contract,
)


def cfg():
    return SimpleNamespace(
        actor={"frozen": True},
        critic={"hidden_dims": [512, 256, 128]},
        algorithm=SimpleNamespace(
            use_clipped_value_loss=True,
            clip_param=0.2,
            max_grad_norm=0.5,
            value_loss_coef=1.0,
            learning_rate=5e-7,
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.002,
        ),
    )


def test_only_critic_value_clipping_changes():
    original = cfg()
    before = deepcopy(original)
    actual = configure_agent(original)
    assert original == before
    assert actual.algorithm.use_clipped_value_loss is False
    actual.algorithm.use_clipped_value_loss = True
    assert actual == original


@pytest.mark.parametrize(
    "key,value",
    [
        ("use_clipped_value_loss", False),
        ("clip_param", 0.3),
        ("value_loss_coef", 0.5),
    ],
)
def test_unrecognized_predecessor_rejected(key, value):
    value_cfg = cfg()
    setattr(value_cfg.algorithm, key, value)
    with pytest.raises(ValueError):
        configure_agent(value_cfg)


def test_actual_runtime_and_lineage_must_both_preserve_the_choice():
    agent = configure_agent(cfg())
    resolved = {"agent": {"algorithm": vars(agent.algorithm).copy()}, "native23_unclipped_value": value_contract()}
    runner = SimpleNamespace(
        alg=agent.algorithm, training_lineage={"materials": {"resolved_config": {"payload": resolved}}}
    )
    require_unclipped(runner)
    runner.alg.use_clipped_value_loss = True
    with pytest.raises(ValueError, match="runtime/config mismatch"):
        require_unclipped(runner)
    runner.alg.use_clipped_value_loss = False
    resolved["agent"]["algorithm"]["clip_param"] = 0.3
    with pytest.raises(ValueError, match="runtime/config mismatch"):
        require_unclipped(runner)
    resolved["agent"]["algorithm"]["clip_param"] = 0.2
    resolved["native23_unclipped_value"]["deployment_ready"] = True
    with pytest.raises(ValueError, match="executed lineage"):
        require_unclipped(runner)


def test_real_factory_before_shared_cli_overrides():
    from gear_sonic.trl.mjlab.config import true23_mjlab_ppo_runner_cfg

    original = true23_mjlab_ppo_runner_cfg()
    assert original.algorithm.max_grad_norm == 1.0
    before = asdict(original)
    result = configure_agent(original)
    expected = deepcopy(before)
    expected["algorithm"]["use_clipped_value_loss"] = False
    assert asdict(result) == expected
    assert asdict(original) == before
