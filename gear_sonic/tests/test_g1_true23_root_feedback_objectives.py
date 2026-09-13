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


def test_upper_posture_keeps_existing_root_and_whole_body_objectives():
    old = objective_profile_contract("root_and_posture_v1")
    new = objective_profile_contract("root_and_upper_posture_v2")
    for key, value in old.items():
        if key != "name":
            assert new[key] == value
    upper = new["upper_body_posture"]
    assert upper["weight"] == -20
    assert upper["hardware_joint_indices"] == list(range(13, 23))
    assert len(upper["joint_names"]) == 10
    assert "elbow" in upper["joint_names"][3]
    assert all("hip" not in name and "waist" not in name for name in upper["joint_names"])
    assert "upper_body_posture" not in old
    args = SimpleNamespace(objective_profile=new["name"], allow_objective_transition=False)
    with pytest.raises(ValueError, match="explicit"):
        objective_transition_contract({"objective_profile": old["name"]}, args)
    args.allow_objective_transition = True
    assert objective_transition_contract({"objective_profile": old["name"]}, args)["reward_objective_changed"]


def test_world_priority_changes_only_declared_root_weight_and_requires_explicit_transition():
    old = objective_profile_contract("root_and_upper_posture_v2")
    new = objective_profile_contract("root_and_upper_world_priority_v3")
    assert new["root_world_tracking_error_weight"] == -30
    assert "root_world_tracking_error_weight" not in old
    for key, value in old.items():
        if key not in ("name", "upper_body_posture"):
            assert new[key] == value
    for key, value in old["upper_body_posture"].items():
        if key != "existing_root_leg_rewards_or_policy_architecture_changed":
            assert new["upper_body_posture"][key] == value
    assert not new["actuator_gains_or_acceptance_limits_changed"]
    args = SimpleNamespace(objective_profile=new["name"], allow_objective_transition=False)
    with pytest.raises(ValueError, match="explicit"):
        objective_transition_contract({"objective_profile": old["name"]}, args)
    args.allow_objective_transition = True
    transition = objective_transition_contract({"objective_profile": old["name"]}, args)
    assert transition["reward_objective_changed"]
    assert transition["critic_and_optimizer_preserved_not_reinitialized"]
    assert not transition["new_objective_tracking_quality_qualified"]


def test_feet_world_profile_only_adds_declared_measured_world_objective():
    old = objective_profile_contract("root_and_upper_posture_v2")
    new = objective_profile_contract("root_and_upper_feet_world_v4")
    assert {k: v for k, v in new.items() if k not in ("name", "feet_world_position")} == {
        k: v for k, v in old.items() if k != "name"
    }
    feet = new["feet_world_position"]
    assert feet["weight"] == -1 and feet["normalization_m"] == 0.05
    assert not feet["world_target_reanchored_to_measured_state"]
    assert not feet["clipping_or_exponential_saturation"]
    assert not feet["foot_contact_or_slip_qualification"]
    args = SimpleNamespace(objective_profile=new["name"], allow_objective_transition=False)
    with pytest.raises(ValueError, match="explicit"):
        objective_transition_contract({"objective_profile": old["name"]}, args)
    args.allow_objective_transition = True
    assert objective_transition_contract({"objective_profile": old["name"]}, args)["reward_objective_changed"]
