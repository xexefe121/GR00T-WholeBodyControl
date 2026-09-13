"""Reduction rounding is permitted; reward changes and bad bootstrap are not."""

import pytest
import torch

from gear_sonic.scripts.train_g1_true23_compact_tracker_v4 import audit_reward


def row():
    components = torch.tensor([[2.3, -100.0] + [-0.011] * 19], dtype=torch.float32)
    base = (components.double().sum(-1) + 3.0517578125e-5).float()
    return {
        "weighted_base_components": components,
        "base_reward": base,
        "shaping_reward": torch.tensor([0.3]),
        "world_quality_bonus": torch.zeros(1),
        "foot_precision_bonus": torch.zeros(1),
        "terminated": torch.tensor([True]),
        "timeouts": torch.tensor([False]),
    }


def test_terminal_float32_reduction_error_has_derived_bound():
    value = row()
    rewards = value["base_reward"] + value["shaping_reward"]
    errors = audit_reward(value, rewards, rewards.clone(), torch.ones(1), 0.99)
    assert errors["base"] > 2e-5
    assert errors["returned"] == errors["stored"] == 0


@pytest.mark.parametrize("change", ["base", "returned", "stored", "nan", "reset_bonus"])
def test_real_reward_errors_still_fail(change):
    value = row()
    rewards = value["base_reward"] + value["shaping_reward"]
    stored = rewards.clone()
    if change == "base":
        value["weighted_base_components"][0, 0] += 0.01
    elif change == "returned":
        rewards += 0.001
    elif change == "stored":
        stored += 0.01
    elif change == "nan":
        rewards[:] = float("nan")
    else:
        value["world_quality_bonus"][:] = 1
    with pytest.raises(ValueError):
        audit_reward(value, rewards, stored, torch.ones(1), 0.99)
