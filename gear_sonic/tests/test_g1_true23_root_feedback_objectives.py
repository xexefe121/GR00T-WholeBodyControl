from types import SimpleNamespace

import pytest

from gear_sonic.utils.g1_true23_root_feedback_objectives import (
    objective_profile_contract,
    objective_transition_contract,
)


def test_legacy_has_no_added_posture_reward():
    contract = objective_profile_contract("legacy_root_tracking")
    assert contract["measured_joint_position_l2_weight"] == 0
    assert not contract["legacy_rewards_replaced"]


def test_posture_profile_is_versioned_without_changing_actuators_or_gates():
    contract = objective_profile_contract("root_and_posture_v1")
    assert contract["measured_joint_position_l2_weight"] == -10
    assert not contract["actuator_gains_or_acceptance_limits_changed"]
    with pytest.raises(ValueError, match="unknown"):
        objective_profile_contract("automatic")


def test_old_evaluated_checkpoint_is_legacy_and_requires_explicit_objective_transition():
    args = SimpleNamespace(objective_profile="root_and_posture_v1", allow_objective_transition=False)
    with pytest.raises(ValueError, match="explicit"):
        objective_transition_contract({}, args)
    args.allow_objective_transition = True
    contract = objective_transition_contract({}, args)
    assert contract["old"]["name"] == "legacy_root_tracking"
    assert contract["reward_objective_changed"]
    assert not contract["same_objective_resume_claimed"]
    assert not contract["new_objective_tracking_quality_qualified"]


def test_unchanged_objective_continuation_stays_compatible():
    contract = objective_transition_contract({}, SimpleNamespace())
    assert not contract["reward_objective_changed"]
    assert contract["same_objective_resume_claimed"]
