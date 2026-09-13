"""Return-bound diagnostics for saved native23 trajectories, never controllers.

These calculations compare known nonnegative tracking costs with a fixed
termination-penalty component. They do not construct a feasible falling action,
prove that an agent intentionally fails, or establish the cause of robot damping.
"""

import math

import numpy as np


def optimistic_suffix_returns(costs, *, gamma, positive_reward_cap):
    """Upper-bound a saved continuation, including an optimistic infinite tail.

    The supplied costs already include timestep scaling. During the recorded
    suffix every remaining reward term must be nonpositive or included in the
    positive cap. After the suffix, allow the maximum positive reward forever.
    This deliberately overestimates continuation value, with no critic needed.
    """
    value = np.asarray(costs)
    if (
        value.ndim != 1
        or not len(value)
        or not np.issubdtype(value.dtype, np.number)
        or not np.isfinite(value).all()
        or (value < 0).any()
    ):
        raise ValueError("return bound requires a nonempty finite nonnegative cost vector")
    if not math.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("discount must be finite in [0,1)")
    if not math.isfinite(positive_reward_cap) or positive_reward_cap < 0:
        raise ValueError("positive reward cap must be finite and nonnegative")
    result = np.empty(len(value), dtype=np.float64)
    future = float(positive_reward_cap) / (1 - gamma)
    for index in range(len(value) - 1, -1, -1):
        future = positive_reward_cap - float(value[index]) + gamma * future
        result[index] = future
    return result


def summarize_component_dominance(costs, *, gamma, positive_reward_cap, termination_penalty):
    """Report component dominance, explicitly not a realizable policy comparison."""
    if not math.isfinite(termination_penalty) or termination_penalty >= 0:
        raise ValueError("termination penalty must be finite and negative")
    bound = optimistic_suffix_returns(costs, gamma=gamma, positive_reward_cap=positive_reward_cap)
    indices = np.flatnonzero(bound < termination_penalty)
    return {
        "samples": len(bound),
        "minimum_optimistic_suffix_return": float(bound.min()),
        "maximum_optimistic_suffix_return": float(bound.max()),
        "below_fixed_termination_component_count": len(indices),
        "below_fixed_termination_component_fraction": float(len(indices) / len(bound)),
        "first_below_fixed_termination_component_index": int(indices[0]) if len(indices) else None,
        "positive_reward_cap_per_control": float(positive_reward_cap),
        "fixed_termination_penalty_component": float(termination_penalty),
        "gamma": float(gamma),
        "infinite_post_recording_tail_at_positive_cap": True,
        "omitted_nonpositive_costs_make_bound_optimistic": True,
        "training_termination_predicates_applied": False,
        "feasible_failure_counterfactual_evaluated": False,
        "agent_failure_preference_proven": False,
        "hardware_damping_cause_proven": False,
        "deployment_ready": False,
    }
