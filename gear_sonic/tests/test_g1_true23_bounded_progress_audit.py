"""Independent reconstruction must reject altered actual PPO reward records."""

from copy import deepcopy

import numpy as np
import pytest

from gear_sonic.scripts.verify_g1_true23_bounded_progress_update import audit_reward_arrays
from gear_sonic.utils.g1_true23_bounded_progress import ALIVE_WEIGHT, NEGATIVE_WEIGHTS, POSITIVE_WEIGHTS


def capture():
    rng = np.random.default_rng(807)
    shape = (2, 4)
    terminated = np.tile([False, True, False, True], (2, 1))
    timeouts = np.tile([False, False, True, True], (2, 1))
    raw = rng.lognormal(size=(*shape, 13)).astype(np.float32)
    names = [*NEGATIVE_WEIGHTS, *POSITIVE_WEIGHTS, "alive", "non_timeout_termination"]
    weights = np.array(list(NEGATIVE_WEIGHTS.values()), dtype=np.float64)
    components = np.zeros((*shape, 21), dtype=np.float32)
    components[..., :13] = 0.02 * weights * raw.astype(np.float64) / (1 + raw)
    for n, weight in POSITIVE_WEIGHTS.items():
        components[..., names.index(n)] = 0.01 * weight
    components[..., names.index("alive")] = (~terminated) * ALIVE_WEIGHT * 0.02
    components[..., names.index("non_timeout_termination")] = terminated * -100
    parts0, parts1 = [rng.random((*shape, 3)).astype(np.float32) for _ in range(2)]
    phi0, phi1 = -np.log1p(parts0.sum(-1)), -np.log1p(parts1.sum(-1))
    delta = 0.99 * phi1 - phi0
    delta[terminated] = -phi0[terminated]
    delta[timeouts] = -0.01 * phi0[timeouts]
    base = components.sum(-1)
    reward = base + delta
    value = rng.random(shape).astype(np.float32)
    return {
        "phi_before": phi0,
        "phi_after_including_reset_states": phi1,
        "world_cost_parts_before": parts0,
        "world_cost_parts_after_including_reset_states": parts1,
        "base_reward": base,
        "shaping_reward": delta,
        "returned_reward": reward,
        "critic_value_before_bootstrap": value,
        "stored_reward_with_timeout_bootstrap": reward + 0.99 * value * timeouts,
        "terminated": terminated,
        "timeouts": timeouts,
        "stored_done": terminated | timeouts,
        "ppo_recorded": np.ones(shape, dtype=np.bool_),
        "raw_costs": raw,
        "weighted_base_components": components,
    }, {"base_component_names": names, "raw_cost_names": list(NEGATIVE_WEIGHTS)}


def test_four_termination_cases_all_reconstruct():
    arrays, metadata = capture()
    result = audit_reward_arrays(arrays, metadata, 1, 2, 4)
    assert result["all_reward_and_PPO_storage_equations_reconstructed"]
    assert result["actual_transitions"] == 8
    assert result["true_terminal_transitions"] == 4
    assert result["timeout_transitions"] == 4
    assert result["simultaneous_failure_and_timeout_transitions"] == 2


@pytest.mark.parametrize(
    "reason",
    ["raw", "component", "potential", "shaping", "reward", "bootstrap", "done", "ppo", "missing", "order", "nan"],
)
def test_corrupt_actual_capture_rejected(reason):
    arrays, metadata = deepcopy(capture())
    names = {
        "raw": "raw_costs",
        "component": "weighted_base_components",
        "potential": "phi_before",
        "shaping": "shaping_reward",
        "reward": "returned_reward",
        "bootstrap": "stored_reward_with_timeout_bootstrap",
    }
    if reason in names:
        arrays[names[reason]].flat[0] += 0.1
    elif reason in ("done", "ppo"):
        arrays["stored_done" if reason == "done" else "ppo_recorded"].flat[0] ^= True
    elif reason == "missing":
        del arrays["base_reward"]
    elif reason == "order":
        metadata["raw_cost_names"].reverse()
    else:
        arrays["raw_costs"].flat[0] = np.nan
    with pytest.raises(ValueError):
        audit_reward_arrays(arrays, metadata, 1, 2, 4)
