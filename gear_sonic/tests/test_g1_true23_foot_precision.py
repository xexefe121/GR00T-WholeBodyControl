"""Foot objective arithmetic and exact learner-state comparison contracts."""

import copy

import pytest
import torch

from gear_sonic.trl.mjlab.native23_pico_foot_precision_runner import equal_state, training_contract
from gear_sonic.utils.g1_true23_foot_precision_reward import (
    FootPrecisionStep,
    foot_precision_bonus,
    foot_precision_contract,
)


def test_bonus_has_declared_scale_and_bounds():
    costs = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0, 1000.0])
    flags = torch.zeros(6, dtype=torch.bool)
    result = foot_precision_bonus(costs, flags, flags)
    torch.testing.assert_close(result, 2 / (1 + costs / 9), rtol=0, atol=0)
    assert result[0] == 2 and result[3] == 1
    assert ((result >= 0) & (result <= 2)).all()
    assert (result[1:] < result[:-1]).all()
    assert foot_precision_contract()["reward_scale_is_not_acceptance_threshold"]


def test_done_and_timeout_never_reward_reset_geometry():
    cost = torch.tensor([0.0, 0.0, 0.0, 9.0])
    terminal = torch.tensor([True, False, True, False])
    timeout = torch.tensor([False, True, True, False])
    torch.testing.assert_close(foot_precision_bonus(cost, terminal, timeout), torch.tensor([0.0, 0.0, 0.0, 1.0]))


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
def test_invalid_cost_rejected(bad):
    flags = torch.zeros(1, dtype=torch.bool)
    with pytest.raises(ValueError):
        foot_precision_bonus(torch.tensor([bad]), flags, flags)


def test_invalid_done_contract_rejected():
    cost = torch.ones(2)
    with pytest.raises(ValueError):
        foot_precision_bonus(cost, torch.ones(2), torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError):
        foot_precision_bonus(cost, torch.zeros(1, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))


def test_overlay_runs_original_once_and_preserves_every_other_result():
    class Original:
        rows = []
        calls = 0

        def __call__(self, actions):
            self.calls += 1
            assert actions is action
            self.rows.append(
                dict(
                    world_cost_parts_after_including_reset_states=torch.tensor([[0.0, 9.0, 2.0], [0.0, 0.0, 0.0]]),
                    returned_reward=torch.tensor([3.0, -100.0]),
                )
            )
            return observation, torch.tensor([3.0, -100.0]), terminal, timeout, extras

    action, observation, extras = object(), object(), {}
    terminal, timeout = torch.tensor([False, True]), torch.zeros(2, dtype=torch.bool)
    original = Original()
    result = FootPrecisionStep(original)(action)
    assert original.calls == 1 and result[0] is observation and result[2] is terminal
    assert result[3] is timeout and result[4] is extras
    torch.testing.assert_close(result[1], torch.tensor([4.0, -100.0]))
    torch.testing.assert_close(
        original.rows[-1]["pre_foot_precision_returned_reward"], torch.tensor([3.0, -100.0])
    )
    torch.testing.assert_close(original.rows[-1]["foot_precision_bonus"], torch.tensor([1.0, 0.0]))


def test_exact_state_comparison_checks_moments_dtype_shape_and_counters():
    state = {
        "state": {3: {"step": torch.tensor(4000.0), "exp_avg": torch.tensor([1.0, 2.0])}},
        "param_groups": [{"params": [3], "lr": 1e-4}],
        "updates": 500,
    }
    assert equal_state(state, copy.deepcopy(state))
    for field in ("moment", "dtype", "counter", "group"):
        changed = copy.deepcopy(state)
        if field == "moment":
            changed["state"][3]["exp_avg"][0] += 1e-5
        elif field == "dtype":
            changed["state"][3]["step"] = changed["state"][3]["step"].double()
        elif field == "counter":
            changed["updates"] = 0
        else:
            changed["param_groups"][0]["params"] = [4]
        assert not equal_state(state, changed)


def test_continuation_contract_does_not_claim_simulator_resume():
    contract = training_contract()
    assert contract["actor_critic_and_adam_preserved"]
    assert contract["environment_and_rng_reinitialized"]
    assert not contract["uninterrupted_simulator_resume_claimed"]
    assert not contract["fresh_actor_or_critic"]
    assert contract["policy_architecture_unchanged"]
