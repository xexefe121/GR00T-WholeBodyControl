"""Test numerical diagnostics independently of robot/simulator access."""

import copy

import numpy as np
import pytest
from rsl_rl.models import MLPModel
import torch

from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256
from gear_sonic.utils.g1_true23_value_diagnostic import (
    calibration,
    cpu_state,
    critic_adam_state,
    fit_probe,
    loss_terms,
    stats,
    verify_gae,
)


def test_clip_blocks_improving_predictions_but_not_worsening_ones():
    values = torch.tensor([[-1.0], [1.0], [0.1], [-20.0]], requires_grad=True)
    old = torch.zeros_like(values)
    target = torch.full_like(values, -10.0)
    raw, clipped, blocked = loss_terms(values, old, target, 0.2)
    clipped.sum().backward()
    assert blocked[:, 0].tolist() == [True, False, False, False]
    assert values.grad[:, 0].tolist() == [0.0, 22.0, pytest.approx(20.2), -20.0]
    assert raw[0].item() == 81.0
    assert clipped[0].item() == pytest.approx(96.04)


def test_loss_is_installed_ppo_formula():
    gen = torch.Generator().manual_seed(51)
    values = torch.randn(511, 1, generator=gen)
    old = torch.randn(511, 1, generator=gen)
    returns = torch.randn(511, 1, generator=gen) * 200
    _, actual, _ = loss_terms(values, old, returns, 0.2)
    clipped = old + (values - old).clamp(-0.2, 0.2)
    expected = torch.max((values - returns).pow(2), (clipped - returns).pow(2))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("clip", [0, -1, float("inf"), float("nan")])
def test_invalid_clip(clip):
    with pytest.raises(ValueError, match="finite and positive"):
        loss_terms(torch.ones(1), torch.ones(1), torch.ones(1), clip)


def test_empty_or_nonfinite_stats_rejected():
    for value in ([], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError, match="finite samples"):
            stats(value)


def test_calibration_zero_variance_is_unavailable_not_perfect():
    report = calibration(torch.ones(3), torch.ones(3))
    assert report["mse"] == 0
    assert report["explained_variance"] is None


def test_timeout_reward_and_gae_verified_and_tampering_rejected():
    from rsl_rl.algorithms import PPO

    class Storage:
        num_transitions_per_env = 4
        values = torch.tensor([[[2.0], [-3.0]], [[4.0], [-8.0]], [[1.0], [-2.0]], [[0.0], [-1.0]]])
        dones = torch.tensor([[[0], [1]], [[1], [0]], [[0], [0]], [[0], [1]]], dtype=torch.uint8)
        returns = torch.zeros(4, 2, 1)

    class Algorithm:
        storage = Storage()
        gamma, lam = 0.99, 0.95
        normalize_advantage_per_mini_batch = False

        @staticmethod
        def critic(obs):
            return obs

    alg = Algorithm()
    last = torch.tensor([[9.0], [-1.0]])
    raw = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1) - 5
    timeouts = torch.zeros_like(raw)
    timeouts[1, 0] = 1
    alg.storage.rewards = raw + alg.gamma * alg.storage.values * timeouts
    PPO.compute_returns(alg, last)
    block = {key: getattr(alg.storage, key) for key in ("values", "rewards", "dones", "returns", "advantages")}
    block.update(raw_rewards=raw, timeouts=timeouts, last_values=last)
    assert verify_gae(block, alg.gamma, alg.lam)["verified"]
    corrupt = copy.deepcopy(block)
    corrupt["rewards"][1, 0] += 1
    with pytest.raises(AssertionError):
        verify_gae(corrupt, alg.gamma, alg.lam)


def test_adam_extraction_preserves_moments_and_remaps_only_critic_ids():
    actor = torch.nn.Linear(2, 1)
    critic = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(
        [
            {"name": "actor", "params": actor.parameters(), "lr": 1e-4},
            {"name": "critic", "params": critic.parameters(), "lr": 3e-4},
        ]
    )
    (actor(torch.ones(4, 2)).sum() + critic(torch.ones(4, 2)).sum()).backward()
    opt.step()
    before = copy.deepcopy(opt.state_dict())
    result = critic_adam_state(opt)
    assert result["param_groups"][0]["params"] == [0, 1]
    assert result["param_groups"][0]["lr"] == 3e-4
    for index, parameter in enumerate(critic.parameters()):
        for key, value in opt.state[parameter].items():
            torch.testing.assert_close(result["state"][index][key], value, atol=0, rtol=0)
    assert before["param_groups"] == opt.state_dict()["param_groups"]
    result["state"][0]["exp_avg"].zero_()
    assert opt.state[next(critic.parameters())]["exp_avg"].abs().sum() > 0


def test_critic_probe_clones_inputs_and_exposes_clipping_plateau():
    torch.manual_seed(7)
    observations = {"critic": torch.ones(32, 2)}
    critic = MLPModel(observations, {"critic": ["critic"]}, "critic", 1, hidden_dims=(4,), obs_normalization=True)
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
    optimizer = torch.optim.Adam([{"name": "critic", "params": critic.parameters(), "lr": 0.1}])
    original = cpu_state(critic)
    snapshot = critic_adam_state(optimizer)
    kwargs = dict(
        clip_param=0.2,
        max_grad_norm=0.5,
        value_loss_coef=1.0,
        indices=torch.arange(32),
        num_mini_batches=4,
        num_epochs=2,
    )
    clipped = fit_probe(
        critic, observations, torch.zeros(32, 1), -torch.ones(32, 1) * 10, snapshot, use_value_clip=True, **kwargs
    )
    unclipped = fit_probe(
        critic, observations, torch.zeros(32, 1), -torch.ones(32, 1) * 10, snapshot, use_value_clip=False, **kwargs
    )
    assert _state_sha256(cpu_state(critic)) == _state_sha256(original)
    assert critic.training
    assert snapshot["state"] == {}
    assert unclipped["after"]["mse"] < clipped["after"]["mse"]
    assert clipped["minibatches"][-1]["flat_clipped_branch_fraction"] == 1.0
    assert clipped["initial_critic_sha256"] == unclipped["initial_critic_sha256"]
    np.testing.assert_array_equal(observations["critic"].numpy(), np.ones((32, 2)))


def test_probe_requires_permutation_and_exact_batches():
    observations = {"critic": torch.ones(8, 2)}
    critic = MLPModel(observations, {"critic": ["critic"]}, "critic", 1, hidden_dims=(4,))
    opt = torch.optim.Adam([{"name": "critic", "params": critic.parameters()}])
    for indices, minibatches in ((torch.zeros(8, dtype=torch.int64), 4), (torch.arange(8), 3)):
        with pytest.raises(ValueError):
            fit_probe(
                critic,
                observations,
                torch.zeros(8, 1),
                torch.zeros(8, 1),
                critic_adam_state(opt),
                use_value_clip=True,
                clip_param=0.2,
                max_grad_norm=0.5,
                value_loss_coef=1,
                indices=indices,
                num_mini_batches=minibatches,
                num_epochs=2,
            )
