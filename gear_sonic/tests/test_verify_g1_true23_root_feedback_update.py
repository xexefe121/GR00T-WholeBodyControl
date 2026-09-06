"""Update mechanics and sensitivity never imply physical root tracking."""

from copy import deepcopy

import pytest
import torch

from gear_sonic.scripts.verify_g1_true23_root_feedback_update import (
    summarize_state_changes,
    root_action_sensitivity,
    ENCODER_PREFIX,
    DECODER_PREFIX,
)
from gear_sonic.tests.test_native23_root_feedback_actor import actor as actor


def states():
    before = {f"{ENCODER_PREFIX}{index}": torch.zeros(2) for index in range(10)}
    before.update({f"{DECODER_PREFIX}{index}": torch.zeros(2) for index in range(18)})
    before["root_conditioner.weight"] = torch.zeros(4, 9)
    before["distribution.raw_std"] = torch.zeros(23)
    after = deepcopy(before)
    for key in after:
        if not key.startswith(ENCODER_PREFIX):
            after[key] += 0.01
    return before, after


def test_complete_update_partition_and_bounds():
    before, after = states()
    result = summarize_state_changes(before, after, {"std_min": 0.02, "std_max": 0.5})
    assert all(result["checks"].values())
    assert result["conditioner_changed_elements"] == result["conditioner_total_elements"] == 36
    assert 0.02 <= result["std_min_observed"] <= result["std_max_observed"] <= 0.5


@pytest.mark.parametrize(
    "defect", ["unchanged_decoder", "changed_encoder", "unchanged_conditioner", "nonzero_initial_conditioner"]
)
def test_missing_training_checks_stay_failed(defect):
    before, after = states()
    if defect == "unchanged_decoder":
        after[f"{DECODER_PREFIX}0"] = before[f"{DECODER_PREFIX}0"].clone()
    elif defect == "changed_encoder":
        after[f"{ENCODER_PREFIX}0"] += 1
    elif defect == "unchanged_conditioner":
        after["root_conditioner.weight"] = before["root_conditioner.weight"].clone()
    else:
        before["root_conditioner.weight"] += 1
    result = summarize_state_changes(before, after, {"std_min": 0.02, "std_max": 0.5})
    assert not all(result["checks"].values())


def test_unknown_or_nonfinite_state_rejected():
    before, after = states()
    after["root_conditioner.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        summarize_state_changes(before, after, {"std_min": 0.02, "std_max": 0.5})
    before, after = states()
    after["unknown"] = torch.zeros(1)
    with pytest.raises(ValueError, match="partition"):
        summarize_state_changes(before, after, {"std_min": 0.02, "std_max": 0.5})


def test_real_actor_sensitivity_zero_then_nonzero_without_qualification(actor):
    saved = actor.root_conditioner.weight.detach().clone()
    try:
        zero = root_action_sensitivity(actor)
        assert not zero["any_root_feature_changes_action"]
        with torch.no_grad():
            actor.root_conditioner.weight.fill_(0.001)
        changed = root_action_sensitivity(actor)
        assert changed["all_nine_root_features_change_action"]
        assert not changed["root_tracking_quality_established"]
    finally:
        with torch.no_grad():
            actor.root_conditioner.weight.copy_(saved)


@pytest.mark.parametrize("component", ["critic", "optimizer_tensor", "optimizer_scalar"])
def test_nonfinite_critic_or_optimizer_rejected(component):
    from gear_sonic.scripts.verify_g1_true23_root_feedback_update import finite_training_state_summary

    value = {
        "critic_state_dict": {"weight": torch.ones(2, 3)},
        "optimizer_state_dict": {
            "state": {0: {"exp_avg": torch.zeros(2, 3)}},
            "param_groups": [{"name": "critic", "params": [0], "lr": 0.001}],
        },
    }
    if component == "critic":
        value["critic_state_dict"]["weight"][0, 0] = float("nan")
    elif component == "optimizer_tensor":
        value["optimizer_state_dict"]["state"][0]["exp_avg"][0, 0] = float("inf")
    else:
        value["optimizer_state_dict"]["param_groups"][0]["lr"] = float("nan")
    with pytest.raises(ValueError):
        finite_training_state_summary(value)
