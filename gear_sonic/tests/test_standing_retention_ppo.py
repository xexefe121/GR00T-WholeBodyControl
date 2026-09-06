import copy

import pytest
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
import torch

from gear_sonic.trl.mjlab import standing_retention_ppo as retention


def make(*, clipped=True, normalize=False):
    torch.manual_seed(5)
    obs = TensorDict(dict(policy=torch.randn(4, 5), critic=torch.randn(4, 7)), batch_size=[4])
    groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = MLPModel(
        obs,
        groups,
        "actor",
        3,
        hidden_dims=[8],
        distribution_cfg=dict(class_name="GaussianDistribution", init_std=0.3, std_type="scalar"),
    )
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[8])
    storage = RolloutStorage("rl", 4, 4, obs, [3], "cpu")
    algorithm = PPO(
        actor,
        critic,
        storage,
        num_learning_epochs=2,
        num_mini_batches=2,
        schedule="fixed",
        desired_kl=None,
        learning_rate=1e-3,
        entropy_coef=0.002,
        use_clipped_value_loss=clipped,
        normalize_advantage_per_mini_batch=normalize,
    )
    with torch.no_grad():
        for index in range(4):
            algorithm.act(obs)
            algorithm.process_env_step(obs, torch.randn(4), torch.tensor([0, 0, index == 2, 0]), {})
        algorithm.compute_returns(obs)
    return algorithm


def exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            exact(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("clipped", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_zero_weight_matches_stock_ppo_weights_optimizer_losses_and_rng(clipped, normalize):
    original, candidate = make(clipped=clipped, normalize=normalize), make(clipped=clipped, normalize=normalize)
    retention.install(candidate, None, weight=0)
    torch.manual_seed(99)
    expected = original.update()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(99)
    actual = candidate.update()
    assert {key: actual[key] for key in expected} == expected
    assert actual["standing_retention"] == 0
    exact(original.actor.state_dict(), candidate.actor.state_dict())
    exact(original.critic.state_dict(), candidate.critic.state_dict())
    exact(original.optimizer.state_dict(), candidate.optimizer.state_dict())
    exact(rng, torch.get_rng_state())
    assert candidate.standing_minibatch_step == retention.optimizer_step(candidate.optimizer) == 4


class Anchor:
    def __init__(self, actor, *, nonfinite=False):
        self.actor, self.steps, self.nonfinite = actor, [], nonfinite

    def loss(self, step):
        self.steps.append(step)
        value = sum(parameter.square().mean() for parameter in self.actor.parameters())
        return value * float("nan") if self.nonfinite else value


def test_auxiliary_gradient_updates_actor_without_extra_optimizer_steps_or_critic_loss():
    original, candidate = make(), make()
    anchor = Anchor(candidate.actor)
    retention.install(candidate, anchor, weight=10)
    torch.manual_seed(99)
    original.update()
    torch.manual_seed(99)
    result = candidate.update()
    assert result["standing_retention"] > 0 and anchor.steps == [0, 1, 2, 3]
    assert retention.optimizer_step(candidate.optimizer) == 4
    assert any(
        not torch.equal(a, b)
        for a, b in zip(original.actor.parameters(), candidate.actor.parameters(), strict=True)
    )
    exact(original.critic.state_dict(), candidate.critic.state_dict())


def test_nonfinite_retention_stops_before_any_optimizer_step():
    candidate = make()
    before = copy.deepcopy(candidate.actor.state_dict())
    retention.install(candidate, Anchor(candidate.actor, nonfinite=True), weight=1)
    with pytest.raises(ValueError, match="nonfinite"):
        candidate.update()
    assert retention.optimizer_step(candidate.optimizer) == 0
    exact(before, candidate.actor.state_dict())


@pytest.mark.parametrize("weight", [-1, True, float("nan"), float("inf"), 1001])
def test_invalid_weight_rejected(weight):
    with pytest.raises(ValueError, match="weight"):
        retention.install(make(), None, weight=weight)


@pytest.mark.parametrize(
    "field,value",
    [("schedule", "adaptive"), ("desired_kl", 0.01), ("is_multi_gpu", True), ("symmetry", {}), ("rnd", object())],
)
def test_unsupported_ppo_modes_are_not_silently_changed(field, value):
    candidate = make()
    setattr(candidate, field, value)
    with pytest.raises(ValueError, match="supports only"):
        retention.install(candidate, None, weight=0)


def test_upstream_change_requires_revalidation(monkeypatch):
    monkeypatch.setattr(retention, "file_sha256", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="upstream PPO changed"):
        retention.install(make(), None, weight=0)


def test_mismatched_resume_step_rejects_before_update():
    candidate = make()
    retention.install(candidate, None, weight=0)
    candidate.standing_minibatch_step = 1
    with pytest.raises(ValueError, match="actual Adam state"):
        candidate.update()


def test_divergent_adam_parameter_counters_reject():
    candidate = make()
    candidate.update()
    first = next(iter(candidate.optimizer.state.values()))
    first["step"] += 1
    with pytest.raises(ValueError, match="diverged"):
        retention.optimizer_step(candidate.optimizer)
