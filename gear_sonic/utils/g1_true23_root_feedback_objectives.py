"""Versioned simulation objectives; no actuator or acceptance-limit changes."""

from __future__ import annotations


OBJECTIVE_PROFILES = {"legacy_root_tracking": 0.0, "root_and_posture_v1": -10.0}


def objective_profile_contract(name):
    if name not in OBJECTIVE_PROFILES:
        raise ValueError("unknown root-feedback objective profile")
    return {
        "name": name,
        "measured_joint_position_l2_weight": OBJECTIVE_PROFILES[name],
        "measured_joint_position_source": "actual_native23_joint_state_not_requested_pd_target",
        "reference": "q10_current_received_native23_joint_position",
        "normalization": "mean_squared_error_divided_by_HARDWARE_23_ACTION_SCALE_squared",
        "legacy_rewards_replaced": False,
        "actuator_gains_or_acceptance_limits_changed": False,
    }


def objective_transition_contract(previous_feedback, args):
    before = previous_feedback.get("objective_profile", "legacy_root_tracking")
    after = getattr(args, "objective_profile", "legacy_root_tracking")
    old, new = objective_profile_contract(before), objective_profile_contract(after)
    changed = before != after
    if changed and not getattr(args, "allow_objective_transition", False):
        raise ValueError("objective transition requires explicit --allow-objective-transition")
    return {
        "old": old,
        "new": new,
        "reward_objective_changed": changed,
        "same_objective_resume_claimed": not changed,
        "critic_and_optimizer_preserved_not_reinitialized": True,
        "first_new_rollouts_use_fresh_environment_reset": True,
        "new_objective_tracking_quality_qualified": False,
    }
