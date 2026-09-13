"""Normal-core launch config cannot silently inherit the low-latency trainer."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")

from gear_sonic.scripts.train_g1_true23_normal_lora import agent_configuration, run
from gear_sonic.trl.mjlab.native23_normal_lora_runner import MULTIPLIERS, normal_training_contract
from gear_sonic.utils.g1_true23_normal_reference import NORMAL_TIMING


def test_normal_launch_has_distinct_actor_and_correct_timing():
    args = SimpleNamespace(
        warm_start=Path("normal_init.pt"), source_checkpoint=Path("normal.pt"), seed=7, updates=2, rollout_steps=8
    )
    cfg = agent_configuration(args)
    assert cfg.actor.class_name.endswith(":True23NormalLoraActorModel")
    assert cfg.actor.reference_timing == NORMAL_TIMING
    assert cfg.obs_groups["actor"] == ("tokenizer", "policy", "root_feedback")
    assert cfg.obs_groups["critic"] == ("critic", "root_feedback", "original_intent_value_reference")
    assert cfg.actor.distribution_cfg["init_std"] == 0.1
    assert cfg.algorithm.clip_param == 0.2
    assert not cfg.algorithm.use_clipped_value_loss
    assert cfg.algorithm.schedule == "fixed" and cfg.algorithm.desired_kl is None
    assert cfg.clip_actions == 10
    assert [cfg.algorithm.learning_rate * MULTIPLIERS[k] for k in MULTIPLIERS] == pytest.approx(
        [1e-4, 1e-4, 5e-7, 3e-4]
    )
    assert normal_training_contract()["fresh_only"] is True


@pytest.mark.parametrize(
    "updates,envs,steps", [(0, 4, 8), (101, 4, 8), (2, 3, 8), (2, 129, 8), (2, 4, 7), (2, 4, 17)]
)
def test_unbounded_or_invalid_training_refused_before_inputs(updates, envs, steps):
    with pytest.raises(ValueError, match="limited to"):
        run(SimpleNamespace(updates=updates, num_envs=envs, rollout_steps=steps))
