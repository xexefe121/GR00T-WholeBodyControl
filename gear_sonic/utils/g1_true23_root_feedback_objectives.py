"""Versioned simulation objectives; no actuator or acceptance-limit changes."""

from __future__ import annotations

OBJECTIVE_PROFILES = {
    "legacy_root_tracking": 0.0,
    "root_and_posture_v1": -10.0,
    "root_and_upper_posture_v2": -10.0,
    "root_and_upper_world_priority_v3": -10.0,
    "root_and_upper_feet_world_v4": -10.0,
    "root_and_upper_feet_sole_world_v5": -10.0,
    "root_and_upper_feet_swing_load_v6": -10.0,
}


def objective_profile_contract(name):
    if name not in OBJECTIVE_PROFILES:
        raise ValueError("unknown root-feedback objective profile")
    result = {
        "name": name,
        "measured_joint_position_l2_weight": OBJECTIVE_PROFILES[name],
        "measured_joint_position_source": "actual_native23_joint_state_not_requested_pd_target",
        "reference": "q10_current_received_native23_joint_position",
        "normalization": "mean_squared_error_divided_by_HARDWARE_23_ACTION_SCALE_squared",
        "legacy_rewards_replaced": False,
        "actuator_gains_or_acceptance_limits_changed": False,
    }
    if name in (
        "root_and_upper_posture_v2",
        "root_and_upper_world_priority_v3",
        "root_and_upper_feet_world_v4",
        "root_and_upper_feet_sole_world_v5",
        "root_and_upper_feet_swing_load_v6",
    ):
        from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES

        result["upper_body_posture"] = {
            "weight": -20.0,
            "hardware_joint_indices": list(range(13, 23)),
            "joint_names": list(HARDWARE_23_JOINT_NAMES[13:]),
            "normalization": "mean_over_10_upper_joints_of_squared_error_divided_by_action_scale_squared",
            "measured_physical_joint_state_not_pd_target": True,
            "applies_to_all_reference_phases": True,
            "existing_root_leg_rewards_or_policy_architecture_changed": False,
            "tracking_improvement_proven": False,
        }
    if name == "root_and_upper_world_priority_v3":
        result["root_world_tracking_error_weight"] = -30.0
        result["upper_body_posture"]["existing_root_leg_rewards_or_policy_architecture_changed"] = True
        result["world_priority_transition"] = {
            "only_existing_root_world_tracking_error_weight_changed": True,
            "previous_weight": -10.0,
            "new_weight": -30.0,
            "position_velocity_formula_unchanged": True,
            "other_reward_weights_functions_and_parameters_unchanged": True,
            "tracking_improvement_proven": False,
        }
    if name in (
        "root_and_upper_feet_world_v4",
        "root_and_upper_feet_sole_world_v5",
        "root_and_upper_feet_swing_load_v6",
    ):
        result["feet_world_position"] = {
            "weight": -1.0,
            "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "normalization_m": 0.05,
            "formula": "mean_over_two_ankles_of_squared_world_xyz_distance_divided_by_0p05m_squared",
            "reference": "same_held_proof_sample_as_existing_root_world_rewards",
            "measured_source": "actual_physical_ankle_link_origins_not_pd_targets",
            "world_target_reanchored_to_measured_state": False,
            "clipping_or_exponential_saturation": False,
            "all_reference_phases_including_acquisition_and_return": True,
            "existing_reward_terms_unchanged": True,
            "foot_contact_or_slip_qualification": False,
            "tracking_improvement_proven": False,
        }
    if name == "root_and_upper_feet_sole_world_v5":
        from gear_sonic.utils.g1_true23_sole_tracking import (
            SOLE_SPHERE_CENTERS_M,
            SOLE_SPHERE_RADIUS_M,
            SOLE_TRACKING_NORMALIZATION_M,
        )

        result["sole_world_position"] = {
            "weight": -1.0,
            "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
            "sphere_centers_in_ankle_body_m": [list(point) for point in SOLE_SPHERE_CENTERS_M],
            "sphere_radius_m": SOLE_SPHERE_RADIUS_M,
            "point": "world_sphere_center_minus_radius_on_world_Z",
            "normalization_m": SOLE_TRACKING_NORMALIZATION_M,
            "formula": (
                "mean_over_two_feet_and_four_sole_points_of_squared_world_xyz_error_divided_by_0p02m_squared"
            ),
            "reference": "same_held_received_q10_position_and_quaternion_as_existing_root_world_rewards",
            "measured_source": "actual_physical_ankle_body_poses_not_requested_joint_targets",
            "all_reference_phases_including_standing_entry_source_and_return": True,
            "world_target_reanchored_to_measured_state": False,
            "clipping_or_exponential_saturation": False,
            "contact_schedule_or_reference_airborne_gate": False,
            "existing_reward_terms_changed": False,
            "actor_inputs_or_architecture_changed": False,
            "physical_geometry_gains_limits_or_acceptance_thresholds_changed": False,
            "contact_force_slip_or_dynamic_feasibility_proven": False,
            "tracking_improvement_proven": False,
        }
    if name == "root_and_upper_feet_swing_load_v6":
        from gear_sonic.utils.g1_true23_swing_load import swing_load_contract

        result["swing_foot_load"] = swing_load_contract()
    return result


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
