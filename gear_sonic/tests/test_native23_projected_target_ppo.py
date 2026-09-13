"""Upstream-update parity and explicit physical target-loss tests."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from rsl_rl.algorithms.ppo import PPO
import torch
from torch import nn

from gear_sonic.trl.mjlab.native23_projected_target_ppo import (
    PROFILE,
    ProjectedTargetPPO,
    projection_objective_contract,
    projection_objective_transition,
    reachable_source_bounds,
    source_projection_loss,
)
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_SCALE_NATIVE_IL23, source_scaled_precompensation


def test_loss_has_zero_gradient_inside_and_points_inward_outside_unchanged_bounds():
    low, high = reachable_source_bounds()
    value = np.tile((low + high) / 2, (2, 1))
    value[0, 0], value[1, 4] = low[0] - 0.5, high[4] + 0.7
    mean = torch.tensor(value, dtype=torch.float32, requires_grad=True)
    loss, projected = source_projection_loss(mean)
    loss.backward()
    assert projected.sum() == 2
    assert mean.grad[0, 0] < 0 and mean.grad[1, 4] > 0
    assert torch.count_nonzero(mean.grad) == 2
    expected = ((0.5 * SOURCE_SCALE_NATIVE_IL23[0]) ** 2 + (0.7 * SOURCE_SCALE_NATIVE_IL23[4]) ** 2) / 2
    assert loss.item() == pytest.approx(expected, abs=1e-7)
    np.testing.assert_array_equal(mean.detach().numpy(), value.astype(np.float32))


def test_closest_target_matches_existing_v2_projection_in_physical_units():
    low, high = reachable_source_bounds()
    rng = np.random.default_rng(2486)
    for raw in np.clip(rng.uniform(-1, 2, (64, 23)) * (high - low) + low, -9.5, 9.5).astype(np.float32):
        _, correction = source_scaled_precompensation(raw)
        selected = np.clip(raw.astype(float), low, high)
        np.testing.assert_allclose(
            raw * np.asarray(SOURCE_SCALE_NATIVE_IL23) + correction,
            selected * np.asarray(SOURCE_SCALE_NATIVE_IL23),
            rtol=0,
            atol=3e-7,
        )


@pytest.mark.parametrize(
    "value",
    [
        torch.zeros(23),
        torch.zeros(2, 29),
        torch.zeros(2, 23, dtype=torch.float64),
        torch.full((2, 23), float("nan")),
    ],
)
def test_loss_rejects_wrong_semantics(value):
    with pytest.raises(ValueError, match="finite float32"):
        source_projection_loss(value)


def test_objective_change_is_explicit_and_old_defaults_are_not_relabelled():
    args = SimpleNamespace(ppo_auxiliary_objective=PROFILE, allow_ppo_objective_transition=False)
    with pytest.raises(ValueError, match="explicit"):
        projection_objective_transition({}, args)
    args.allow_ppo_objective_transition = True
    receipt = projection_objective_transition({}, args)
    assert receipt["ppo_loss_changed"] and not receipt["same_ppo_objective_resume_claimed"]
    assert receipt["actor_critic_optimizer_and_learning_rates_preserved"]
    current = {"ppo_auxiliary_objective": projection_objective_contract(PROFILE)}
    args.allow_ppo_objective_transition = False
    assert not projection_objective_transition(current, args)["ppo_loss_changed"]
    current["ppo_auxiliary_objective"]["weight"] *= 2
    with pytest.raises(ValueError, match="contract differs"):
        projection_objective_transition(current, args)


class Observations(dict):
    @property
    def batch_size(self):
        return self["obs"].shape[:1]


class Actor(nn.Module):
    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 23)
        self.std = nn.Parameter(torch.full((23,), 0.1))

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        mean = self.linear(obs["obs"])
        self.dist = torch.distributions.Normal(mean, self.std.expand_as(mean), validate_args=False)
        return self.dist.sample() if stochastic_output else mean

    def get_output_log_prob(self, value):
        return self.dist.log_prob(value).sum(-1)

    @property
    def output_distribution_params(self):
        return self.dist.mean, self.dist.stddev

    @property
    def output_entropy(self):
        return self.dist.entropy().sum(-1)


class Critic(nn.Module):
    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 1)

    def forward(self, obs, masks=None, hidden_state=None):
        return self.linear(obs["obs"])


class Storage:
    def __init__(self, batches):
        self.batches, self.cleared = batches, False

    def mini_batch_generator(self, num_mini_batches, num_learning_epochs):
        for _ in range(num_learning_epochs):
            yield from self.batches[:num_mini_batches]

    def clear(self):
        self.cleared = True


def _fixture():
    torch.manual_seed(4123)
    actor, critic = Actor(), Critic()
    actor.linear.bias.data[17] = 4.5
    batches = []
    with torch.no_grad():
        for _ in range(2):
            obs = Observations(obs=torch.randn(16, 5))
            actions = actor(obs, stochastic_output=True)
            batches.append(
                SimpleNamespace(
                    observations=obs,
                    masks=None,
                    hidden_states=(None, None),
                    actions=actions,
                    old_actions_log_prob=actor.get_output_log_prob(actions)[:, None],
                    old_distribution_params=tuple(x.clone() for x in actor.output_distribution_params),
                    values=critic(obs),
                    returns=torch.randn(16, 1),
                    advantages=torch.randn(16, 1),
                )
            )
    del actor.dist
    return actor, critic, batches


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("clipped_value", [False, True])
def test_zero_weight_update_matches_upstream_actual_ppo_parameters_and_losses(normalize, clipped_value):
    actor, critic, batches = _fixture()
    kwargs = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-4,
        schedule="fixed",
        normalize_advantage_per_mini_batch=normalize,
        use_clipped_value_loss=clipped_value,
    )
    old = PPO(deepcopy(actor), deepcopy(critic), Storage(deepcopy(batches)), **kwargs)
    new = ProjectedTargetPPO(
        deepcopy(actor), deepcopy(critic), Storage(deepcopy(batches)), mean_projection_weight=0, **kwargs
    )
    torch.manual_seed(6734)
    expected = old.update()
    torch.manual_seed(6734)
    actual = new.update()
    for key, value in expected.items():
        assert actual[key] == value
    for before, after in ((old.actor, new.actor), (old.critic, new.critic)):
        for key, value in before.state_dict().items():
            torch.testing.assert_close(after.state_dict()[key], value, rtol=0, atol=0)
    assert old.storage.cleared and new.storage.cleared
    assert new.projection_runtime_receipt()["completed_minibatches"] == 4
    assert actual["source_projection"] > 0


def test_positive_weight_updates_impossible_mean_even_when_ppo_advantage_is_zero():
    actor, critic, batches = _fixture()
    for batch in batches:
        batch.advantages.zero_()
    alg = ProjectedTargetPPO(
        actor,
        critic,
        Storage(batches),
        num_learning_epochs=1,
        num_mini_batches=2,
        learning_rate=1e-3,
        schedule="fixed",
        value_loss_coef=0,
        entropy_coef=0,
    )
    before = source_projection_loss(actor(batches[0].observations))[0].item()
    alg.update()
    after = source_projection_loss(actor(batches[0].observations))[0].item()
    assert after < before


def test_unsupported_optimizer_scope_is_not_silently_approximated():
    actor, critic, batches = _fixture()
    with pytest.raises(ValueError, match="fixed-rate"):
        ProjectedTargetPPO(actor, critic, Storage(batches), schedule="adaptive")
    with pytest.raises(ValueError, match="versioned"):
        ProjectedTargetPPO(actor, critic, Storage(batches), mean_projection_weight=0.051)
