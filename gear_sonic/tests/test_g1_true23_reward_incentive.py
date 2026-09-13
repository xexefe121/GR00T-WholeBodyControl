"""Independent finite sums and conservative interpretation of return bounds."""

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_reward_incentive import (
    optimistic_suffix_returns,
    summarize_component_dominance,
)


def test_zero_cost_has_maximum_infinite_value():
    actual = optimistic_suffix_returns(np.zeros(8), gamma=0.99, positive_reward_cap=0.2)
    np.testing.assert_allclose(actual, np.full(8, 20.0), atol=1e-12, rtol=0)


def test_recurrence_matches_independent_explicit_sums_with_positive_tail():
    costs = np.array([0.0, 2.3, 0.5, 19.0, 1.0])
    gamma, cap = 0.91, 0.2
    actual = optimistic_suffix_returns(costs, gamma=gamma, positive_reward_cap=cap)
    expected = [cap / (1 - gamma) - np.dot(gamma ** np.arange(len(costs) - i), costs[i:]) for i in range(5)]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_nonpositive_omitted_rewards_cannot_raise_return():
    costs = np.arange(10, dtype=float)
    loose = optimistic_suffix_returns(costs, gamma=0.99, positive_reward_cap=0.2)
    strict = optimistic_suffix_returns(costs + 0.5, gamma=0.99, positive_reward_cap=0.2)
    assert np.all(strict < loose)


def test_report_does_not_claim_feasible_failure_or_hardware_causality():
    report = summarize_component_dominance(
        np.full(300, 5.0), gamma=0.99, positive_reward_cap=0.2, termination_penalty=-100.0
    )
    assert report["below_fixed_termination_component_count"] > 200
    assert not report["feasible_failure_counterfactual_evaluated"]
    assert not report["agent_failure_preference_proven"]
    assert not report["hardware_damping_cause_proven"]
    assert not report["training_termination_predicates_applied"]


def test_zero_discount_ignores_future():
    costs = np.array([1.0, 200.0, 0.0])
    np.testing.assert_array_equal(
        optimistic_suffix_returns(costs, gamma=0.0, positive_reward_cap=0.25), 0.25 - costs
    )


@pytest.mark.parametrize("costs", ([], [np.nan], [np.inf], [-0.01], [[1.0]], ["x"]))
def test_invalid_costs_rejected(costs):
    with pytest.raises(ValueError):
        optimistic_suffix_returns(costs, gamma=0.99, positive_reward_cap=0.2)


@pytest.mark.parametrize("gamma", (-0.1, 1.0, np.inf, np.nan))
def test_invalid_discount_rejected(gamma):
    with pytest.raises(ValueError):
        optimistic_suffix_returns([1.0], gamma=gamma, positive_reward_cap=0.2)


@pytest.mark.parametrize("cap", (-0.1, np.inf, np.nan))
def test_invalid_positive_cap_rejected(cap):
    with pytest.raises(ValueError):
        optimistic_suffix_returns([1.0], gamma=0.99, positive_reward_cap=cap)


@pytest.mark.parametrize("penalty", (0.0, 1.0, np.inf, np.nan))
def test_invalid_termination_component_rejected(penalty):
    with pytest.raises(ValueError):
        summarize_component_dominance([1.0], gamma=0.99, positive_reward_cap=0.2, termination_penalty=penalty)
