from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from gear_sonic.utils.g1_true23_lifecycle_ppo import (
    LifecyclePPO,
    LifecyclePolicy,
    clipped_ppo_losses,
    generalized_advantages,
    make_critic,
    terminal_episode_rewards,
)
from gear_sonic.utils.g1_true23_lifecycle_rollout import lifecycle_training_plan


def episode_result(completed=3, requested=535, extra=True, return_count=250):
    full = completed == requested
    report = dict(
        completed_transitions=completed,
        requested_transitions=requested,
        failure=None if full else {"type": "TargetIntersectionError"},
        motion_fidelity={"passed": full},
        return_hold=dict(
            requested_transitions=250,
            completed_transitions=return_count,
            existing_guard_screen_passed=return_count == 250,
        ),
    )
    arrays = {
        key: np.zeros(completed)
        for key in (
            "joint_rmse_rad",
            "relative_body_error_m",
            "pelvis_orientation_error_rad",
            "pelvis_position_error_m",
        )
    }
    arrays["policy_inference_returned"] = np.ones(completed + int(extra), dtype=bool)
    return report, arrays


def test_early_return_cannot_earn_completion_bonus():
    rewards, assessment = terminal_episode_rewards(*episode_result())
    np.testing.assert_array_equal(rewards, [2, 2, 2, -40])
    assert assessment["partial_or_rejected_terminal_action"] is True
    assert assessment["full_motion"] is False and assessment["full_return"] is True
    assert assessment["fixed_controller_actions_enter_ppo"] is False


def test_full_motion_and_return_receive_joint_terminal_bonus():
    rewards, assessment = terminal_episode_rewards(*episode_result(535, 535, False))
    assert len(rewards) == 535 and rewards[-1] == 52
    assert assessment["terminal_assessment"] == 50


def test_failed_first_action_is_real_negative_experience_not_fake_completion():
    report, arrays = episode_result(completed=0, return_count=0)
    report["return_hold"] = {"not_run_reason": "acquisition_failed"}
    rewards, assessment = terminal_episode_rewards(report, arrays)
    np.testing.assert_array_equal(rewards, [-50])
    assert assessment["actual_completed_controls"] == 0


def test_completing_time_without_tracking_cannot_earn_motion_bonus():
    report, arrays = episode_result(535, 535, False)
    report["motion_fidelity"]["passed"] = False
    _, assessment = terminal_episode_rewards(report, arrays)
    assert assessment["terminal_assessment"] == -40


def test_incomplete_return_does_not_earn_joint_success_bonus():
    _, assessment = terminal_episode_rewards(*episode_result(535, 535, False, 249))
    assert assessment["full_motion"] is True and assessment["full_return"] is False
    assert assessment["terminal_assessment"] == pytest.approx(34.96)


@pytest.mark.parametrize("change", ["calls", "inference", "metric", "exception", "return", "no_action"])
def test_bad_rollout_accounting_rejected(change):
    report, arrays = episode_result()
    if change == "calls":
        arrays["policy_inference_returned"] = np.ones(6, dtype=bool)
    elif change == "inference":
        arrays["policy_inference_returned"][0] = False
    elif change == "metric":
        arrays["joint_rmse_rad"][0] = np.nan
    elif change == "exception":
        report["failure"] = {"type": "RuntimeError"}
    elif change == "return":
        report["return_hold"]["requested_transitions"] = 20
    else:
        report, arrays = episode_result(0, extra=False)
    with pytest.raises(ValueError):
        terminal_episode_rewards(report, arrays)


def test_gae_terminal_does_not_bootstrap_into_another_episode():
    advantage, returns = generalized_advantages([1, 2], [0.5, 0.25], gamma=1, lam=1)
    np.testing.assert_array_equal(advantage, [2.5, 1.75])
    np.testing.assert_array_equal(returns, [3, 2])
    separate, _ = generalized_advantages([2], [0.25], gamma=1, lam=1)
    assert separate[0] == advantage[-1]


def test_zero_discount_keeps_only_current_reward():
    advantage, returns = generalized_advantages([1, 2], [0.5, 0.25], gamma=0)
    np.testing.assert_array_equal(advantage, [0.5, 1.75])
    np.testing.assert_array_equal(returns, [1, 2])


@pytest.mark.parametrize("rewards,values", [([], []), ([1], [1, 2]), ([np.nan], [0])])
def test_gae_invalid_data_rejected(rewards, values):
    with pytest.raises(ValueError):
        generalized_advantages(rewards, values)


def test_ppo_clipping_handles_positive_and_negative_advantages():
    ratio = torch.tensor([2.0, 0.5])
    policy, value = clipped_ppo_losses(
        ratio.log(),
        torch.zeros(2),
        torch.tensor([1.0, -1.0]),
        torch.tensor([1.0, -1.0]),
        torch.zeros(2),
        torch.zeros(2),
    )
    assert float(policy) == pytest.approx(-0.2)
    assert float(value) == 1


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(994, 23) * 0.001, requires_grad=False)
        self.lora_a = nn.Parameter(torch.randn(994, 2) * 0.01)
        self.lora_b = nn.Parameter(torch.randn(2, 23) * 0.01)

    def forward(self, value):
        return value @ self.weight + value @ self.lora_a @ self.lora_b


class TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(267, 64)
        self.encoder.requires_grad_(False)
        self.decoder = TinyDecoder()
        self.codec = SimpleNamespace(encode_proprioception=lambda x: x, decode_action=lambda x: x)
        self.initial_std = torch.ones(23) * 0.38
        self.frozen = {name: p.clone() for name, p in self.named_parameters() if not p.requires_grad}

    def encode(self, value):
        return self.encoder(value)

    def forward(self, semantic, history):
        return self.decoder(torch.cat((self.encode(semantic), history), dim=-1))

    def assert_frozen_platform_unchanged(self):
        assert all(torch.equal(dict(self.named_parameters())[name], value) for name, value in self.frozen.items())


def test_deterministic_policy_is_exact_core_and_preserves_history():
    core, critic, calls = TinyCore(), make_critic("cpu"), []
    policy = LifecyclePolicy(core, critic, guard=lambda: calls.append(1), seed=1, stochastic=False)
    semantic, history = np.zeros(267, dtype=np.float32), np.zeros(930, dtype=np.float32)
    raw, decoder = policy.infer(semantic, history)
    with torch.no_grad():
        expected = core(torch.from_numpy(semantic)[None], torch.from_numpy(history)[None])[0].numpy()
    np.testing.assert_array_equal(raw, expected)
    np.testing.assert_array_equal(decoder[64:], history)
    assert calls == [1]


def test_stochastic_policy_records_actual_gaussian_log_probability():
    core, critic = TinyCore(), make_critic("cpu")
    policies = [LifecyclePolicy(core, critic, guard=lambda: None, seed=42, stochastic=True) for _ in range(2)]
    args = (np.zeros(267, dtype=np.float32), np.zeros(930, dtype=np.float32))
    first, _ = policies[0].infer(*args)
    second, _ = policies[1].infer(*args)
    np.testing.assert_array_equal(first, second)
    row = policies[0].records[0]
    expected = (
        torch.distributions.Normal(torch.from_numpy(row["mean"]), core.initial_std)
        .log_prob(torch.from_numpy(first))
        .sum()
    )
    assert row["log_prob"] == float(expected)


def test_real_gradient_update_is_lora_and_critic_only_with_exact_action_counts():
    torch.manual_seed(49)
    core, critic = TinyCore(), make_critic("cpu")
    ppo = LifecyclePPO(core, critic, None, guard=lambda: None, epochs=2, batch_size=4)
    policy = LifecyclePolicy(core, critic, guard=lambda: None, seed=50, stochastic=True)
    source = np.linspace(-0.1, 0.1, 267, dtype=np.float32)
    history = np.zeros(930, dtype=np.float32)
    episodes = []
    for length in (3, 6):
        policy.records = []
        for _ in range(length):
            raw, decoder = policy.infer(source, history)
            assert raw.shape == (23,) and decoder.shape == (994,)
            np.testing.assert_array_equal(decoder[64:], history)
        episodes.append(
            dict(
                records=list(policy.records),
                rewards=np.arange(length, dtype=np.float32) - 3,
                policy_update_at_collection=0,
            )
        )
    before = core.decoder.lora_b.clone()
    critic_before = critic[0].weight.clone()
    result = ppo.update(episodes)
    assert result["actual_active_actions"] == result["total_active_actions"] == 9
    assert result["minibatches"] == 6 and result["update"] == 1
    assert not torch.equal(before, core.decoder.lora_b)
    assert not torch.equal(critic_before, critic[0].weight)
    core.assert_frozen_platform_unchanged()
    with pytest.raises(ValueError, match="stale"):
        ppo.update(episodes)


def test_optimizer_rejects_unfrozen_platform():
    core = TinyCore()
    core.encoder.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="only decoder LoRA"):
        LifecyclePPO(core, make_critic("cpu"), None, guard=lambda: None)


def test_training_plan_retains_missing_request_and_does_not_duplicate_dance():
    rows = [
        dict(
            label=f"motion{i}.reference",
            unavailable=i == 1,
            historical_start=False,
            stationary_prerequisite_only=False,
        )
        for i in range(8)
    ]
    rows += [
        dict(
            label="happy_dance.historical",
            unavailable=False,
            historical_start=True,
            stationary_prerequisite_only=False,
        ),
        dict(
            label="standing.reference",
            unavailable=False,
            historical_start=False,
            stationary_prerequisite_only=True,
        ),
        dict(
            label="standing.acquired", unavailable=False, historical_start=False, stationary_prerequisite_only=True
        ),
    ]
    plan = lifecycle_training_plan(rows)
    assert len(plan) == 9 and sum(row["unavailable"] for row in plan) == 1
    assert all(row["training_return_requested_controls"] == 250 for row in plan)
    with pytest.raises(ValueError):
        lifecycle_training_plan(rows[:-1])
