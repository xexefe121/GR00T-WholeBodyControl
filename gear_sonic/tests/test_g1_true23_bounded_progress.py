"""Independent reward algebra, terminal semantics and SIM-only boundaries."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.utils import g1_true23_bounded_progress as math


def test_cost_is_bounded_monotone_and_preserves_near_zero_gradient():
    x = torch.tensor([0, 1e-9, 0.1, 1, 100, 1e30], dtype=torch.float64, requires_grad=True)
    actual = math.bounded_cost(x)
    np.testing.assert_allclose(actual.detach().numpy(), 1 - 1 / (1 + x.detach().numpy()), atol=1e-15)
    assert (actual >= 0).all() and (actual <= 1).all()
    assert (torch.diff(actual) > 0).all()
    actual.sum().backward()
    torch.testing.assert_close(x.grad, 1 / (1 + x.detach()).square())
    assert x.grad[0] == 1


@pytest.mark.parametrize("value", [[], [-0.001], [float("nan")], [float("inf")], [[1.0]]])
def test_bad_costs_rejected(value):
    with pytest.raises(ValueError):
        math.bounded_cost(torch.tensor(value))


@pytest.mark.parametrize("value", [torch.tensor([True]), torch.tensor([1]), torch.tensor([1 + 1j]), [1.0]])
def test_non_real_float_vector_rejected(value):
    with pytest.raises(ValueError):
        math.bounded_cost(value)


def flags():
    return torch.tensor([False, True, False, True]), torch.tensor([False, False, True, True])


def test_terminal_timeout_and_simultaneous_flags_use_correct_bootstrap_phase():
    before = torch.tensor([-2.0, -3, -4, -5], dtype=torch.float64)
    after = torch.tensor([-1.0, -9000, -7000, -8000], dtype=torch.float64)
    terminated, timeouts = flags()
    actual = math.potential_difference(before, after, terminated, timeouts)
    np.testing.assert_allclose(actual.numpy(), [1.01, 3, 0.04, 0.05], atol=1e-12)
    changed = after.clone()
    changed[1:] += 123456
    torch.testing.assert_close(actual, math.potential_difference(before, changed, terminated, timeouts))


@pytest.mark.parametrize("reason", ["shape", "dtype", "flag_type", "flag_shape", "nan", "gamma"])
def test_invalid_progress_input_rejected(reason):
    before, after = torch.zeros(4), torch.ones(4)
    terminated, timeouts = flags()
    gamma = 0.99
    if reason == "shape":
        after = after[:3]
    elif reason == "dtype":
        after = after.double()
    elif reason == "flag_type":
        timeouts = timeouts.float()
    elif reason == "flag_shape":
        terminated = terminated[:3]
    elif reason == "nan":
        before[0] = float("nan")
    else:
        gamma = 1.0
    with pytest.raises(ValueError):
        math.potential_difference(before, after, terminated, timeouts, gamma)


def test_complete_episode_progress_telescopes_to_action_independent_initial_shift():
    gamma = math.GAMMA
    phi = torch.tensor([-3, -1.2, -5, -0.5, -12345], dtype=torch.float64)
    terminated = torch.tensor([False, False, False, True])
    shaping = math.potential_difference(phi[:-1], phi[1:], terminated, torch.zeros(4, dtype=torch.bool))
    assert float((shaping * gamma ** torch.arange(4)).sum()) == pytest.approx(3, abs=3e-7)


def test_stock_rsl_timeout_bootstrap_and_shifted_value_have_identical_td_errors():
    from rsl_rl.algorithms import PPO

    terminated, timeouts = flags()
    done = terminated | timeouts
    before = torch.tensor([-2.0, -3, -4, -5], dtype=torch.float64)
    after = torch.tensor([-1.0, -9000, -7000, -8000], dtype=torch.float64)
    base = torch.tensor([1.0, -100, 2, -101], dtype=torch.float64)
    value, next_value = torch.tensor([5.0, 4, 3, 2], dtype=torch.float64), torch.ones(4, dtype=torch.float64)
    shaping = math.potential_difference(before, after, terminated, timeouts)

    def stored(reward, values):
        captured = []
        transition = SimpleNamespace(values=values[:, None], clear=lambda: None)
        model = SimpleNamespace(update_normalization=lambda _: None, reset=lambda _: None)
        alg = SimpleNamespace(
            actor=model,
            critic=model,
            rnd=None,
            gamma=math.GAMMA,
            device="cpu",
            transition=transition,
            storage=SimpleNamespace(add_transition=lambda t: captured.append(t.rewards.clone())),
        )
        PPO.process_env_step(alg, {}, reward, done, {"time_outs": timeouts})
        return captured[0]

    old_stored = stored(base, value)
    new_stored = stored(base + shaping, value - before)
    mask = (~done).to(value.dtype)
    old_td = old_stored + math.GAMMA * mask * next_value - value
    new_td = new_stored + math.GAMMA * mask * (next_value - after) - (value - before)
    torch.testing.assert_close(new_td, old_td, rtol=1e-12, atol=1e-12)
    # The tempting standard next-state formula is wrong for this installed
    # algorithm's self-bootstrap. It is deliberately NOT used on timeouts.
    wrong = math.GAMMA * torch.where(done, 0, after) - before
    assert not torch.allclose(wrong[timeouts], shaping[timeouts])


def test_base_bounds_include_failure_but_not_timeout_as_failure():
    terminated, _ = flags()
    math.assert_base_bounds(torch.tensor([0.1, -102.206, 2.406, -99.9]), terminated)
    for values in ([0.09, -100, 1, -100], [3, -100, 1, -100], [1, -99, 1, -100]):
        with pytest.raises(ValueError):
            math.assert_base_bounds(torch.tensor(values), terminated)


def test_profile_has_no_deployment_or_old_objective_equivalence_claim():
    c = math.reward_contract()
    assert c["alive_weight"] == pytest.approx(115.3)
    assert sum(-x for x in c["negative_weights"].values()) == pytest.approx(110.3)
    assert c["same_objective_as_unbounded_predecessor"] is False
    assert c["hardware_authorized"] is False and c["deployment_ready"] is False
    c["negative_weights"]["joint_limit"] = 0
    assert math.reward_contract()["negative_weights"]["joint_limit"] == -20


def raw_cfg():
    from gear_sonic.envs.mjlab.sonic_true23_bounded_progress import expected_functions

    weights = {**math.NEGATIVE_WEIGHTS, **math.POSITIVE_WEIGHTS, "alive": 5, "non_timeout_termination": -5000}
    return SimpleNamespace(
        scale_rewards_by_dt=True,
        rewards={
            n: SimpleNamespace(func=f, weight=weights[n], params={}) for n, f in expected_functions().items()
        },
        actions={"physical_axes": 23},
        terminations={"height": 0.25, "physical_caps": True},
        sim={"dt": 0.002},
        observations={"original29": True},
        commands={"source": "unchanged"},
    )


def test_configuration_changes_only_declared_reward_fields_and_retains_top_level_params():
    from gear_sonic.envs.mjlab.sonic_true23_bounded_progress import configure_environment, transformed_cost

    cfg = raw_cfg()
    cfg.rewards["joint_limit"].params["asset_cfg"] = "sentinel_original_entity_config"
    before = deepcopy(cfg)
    actual = configure_environment(cfg)
    assert cfg == before
    for name in ("actions", "terminations", "sim", "observations", "commands"):
        assert getattr(actual, name) == getattr(cfg, name)
    assert actual.rewards["joint_limit"].params["asset_cfg"] == "sentinel_original_entity_config"
    for name in math.NEGATIVE_WEIGHTS:
        term = actual.rewards[name]
        assert term.func is transformed_cost
        assert term.params.pop("raw_cost_function") is cfg.rewards[name].func
        assert term.params.pop("reward_term_name") == name
        term.func = cfg.rewards[name].func
        assert term == cfg.rewards[name]
    actual.rewards["alive"].weight = 5
    assert actual == cfg


@pytest.mark.parametrize("reason", ["dt_scale", "missing", "function", "weight", "reserved"])
def test_unknown_predecessor_rejected(reason):
    from gear_sonic.envs.mjlab.sonic_true23_bounded_progress import configure_environment

    cfg = raw_cfg()
    if reason == "dt_scale":
        cfg.scale_rewards_by_dt = False
    elif reason == "missing":
        del cfg.rewards["joint_limit"]
    elif reason == "function":
        cfg.rewards["joint_limit"].func = lambda _: None
    elif reason == "weight":
        cfg.rewards["joint_limit"].weight = -1
    else:
        cfg.rewards["joint_limit"].params["reward_term_name"] = "wrong"
    with pytest.raises(ValueError):
        configure_environment(cfg)


def test_step_overlay_one_original_step_same_actions_observations_flags_and_no_reset_phi_leak():
    from gear_sonic.envs.mjlab.sonic_true23_bounded_progress import ProgressRewardStep

    terminated, timeouts = flags()
    base = torch.tensor([1.0, -100, 1, -100])
    original_base = base.clone()
    observations, actions, calls = {"untouched": object()}, torch.ones(4, 23), []
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        max_episode_length_s=30,
        phi=torch.tensor([-2.0, -3, -4, -5]),
        reward_manager=SimpleNamespace(_step_reward=torch.zeros(4, 21)),
    )

    def original_step(received):
        calls.append(received)
        env.phi = torch.tensor([-1.0, -9000, -7000, -8000])
        env._bounded_progress_raw_costs.update({n: torch.zeros(4) for n in math.NEGATIVE_WEIGHTS})
        return observations, base, terminated, timeouts, {"log": {}}

    env.step = original_step
    wrapper = ProgressRewardStep(env, {}, potential=lambda e, _: (e.phi, torch.zeros(4, 3)))
    result = wrapper(actions)
    assert len(calls) == 1 and calls[0] is actions
    assert result[0] is observations and result[2] is terminated and result[3] is timeouts
    torch.testing.assert_close(result[1], torch.tensor([2.01, -97, 1.04, -99.95]))
    torch.testing.assert_close(base, original_base, rtol=0, atol=0)
    assert not result[1].data_ptr() == base.data_ptr()
    assert wrapper.capture()["returned_reward"].shape == (1, 4)
    torch.testing.assert_close(wrapper.episode_shaping[terminated | timeouts], torch.zeros(3))
    # An external new initial state is sampled directly, not a cached previous
    # episode potential. The production runner uses the normal internal reset.
    env.phi = torch.zeros(4)
    result2 = wrapper(actions)
    torch.testing.assert_close(result2[1], torch.tensor([0.01, -100, 1, -100]), atol=1e-6, rtol=1e-6)


def test_distinct_header_and_reward_contract_required_before_common_schema_validation():
    from gear_sonic.trl.mjlab.native23_bounded_progress_runner import CHECKPOINT_HEADER, decoder_schema_view
    from gear_sonic.trl.mjlab.native23_decoder_lora_runner import CHECKPOINT_HEADER as OLD

    checkpoint = {
        "header": deepcopy(CHECKPOINT_HEADER),
        "lineage": {
            "materials": {
                "resolved_config": {
                    "payload": {
                        "native23_bounded_progress": math.reward_contract(),
                        "agent": {"algorithm": {"gamma": 0.99}},
                    }
                }
            }
        },
    }
    view = decoder_schema_view(checkpoint)
    assert view["header"] == OLD and checkpoint["header"] == CHECKPOINT_HEADER
    with pytest.raises(ValueError):
        decoder_schema_view(view)
    checkpoint["lineage"]["materials"]["resolved_config"]["payload"]["native23_bounded_progress"][
        "deployment_ready"
    ] = True
    with pytest.raises(ValueError):
        decoder_schema_view(checkpoint)
