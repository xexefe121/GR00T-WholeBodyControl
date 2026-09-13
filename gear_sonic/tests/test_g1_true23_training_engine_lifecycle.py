import numpy as np
import pytest
import torch

from gear_sonic.scripts.record_g1_true23_training_engine_lifecycle import actor_observations, fail, verify_replicas


def test_route_index_removed_once_without_mutating_other_groups():
    obs = {
        "tokenizer": torch.arange(2 * 268).reshape(2, 268).float(),
        "policy": torch.zeros(2, 930),
        "root_feedback": torch.ones(2, 9),
    }
    result = actor_observations(obs)
    torch.testing.assert_close(result["tokenizer"], obs["tokenizer"][:, 1:], atol=0, rtol=0)
    assert result["tokenizer"].shape == (2, 267)
    assert obs["tokenizer"].shape == (2, 268)
    assert result["policy"] is obs["policy"] and result["root_feedback"] is obs["root_feedback"]


@pytest.mark.parametrize("value", [torch.zeros(2, 267), torch.zeros(268), torch.full((2, 268), float("nan"))])
def test_invalid_semantic_shape_or_values_rejected(value):
    with pytest.raises(ValueError):
        actor_observations({"tokenizer": value})


def test_replicas_must_be_exact_for_compressed_trace():
    values = np.repeat(np.arange(4)[:, None], 8, axis=0).astype(float)
    for index in range(4):
        assert verify_replicas(values, index)[0] == index
    values[3, 0] += 1e-12
    with pytest.raises(ValueError):
        verify_replicas(values, 0)


def test_stop_latches_original_failure_and_actual_substep_count():
    case = dict(active=True, failure=None, completed=2, trace={"physics_post_qpos": [0] * 23})
    fail(case, "range", stage="substep_2")
    original = case["failure"].copy()
    fail(case, "later cleanup", stage="cleanup")
    assert not case["active"] and case["failure"] == original
    assert original["physics_steps"] == 23 and original["completed_controls"] == 2
