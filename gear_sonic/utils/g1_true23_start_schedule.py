"""Versioned rollout scheduling; reference content and evaluation stay unchanged."""

from __future__ import annotations

REFERENCE_RESET_SCHEDULES = ("mixed_reference_reset_v1", "phase_balanced_reference_reset_v1")
START_SCHEDULES = ("synchronous", "staggered_standing_start_v1", *REFERENCE_RESET_SCHEDULES)


def start_schedule_contract(name="synchronous"):
    if name not in START_SCHEDULES:
        raise ValueError("unknown standing-start schedule")
    if name == "phase_balanced_reference_reset_v1":
        return {
            "name": name,
            "environment_allocation": {
                "env_id_modulo_4_equals_0": "configured_standing_full_lifecycle",
                "env_id_modulo_4_equals_1": "uniform_q10_frame_in_acquisition_ramp",
                "env_id_modulo_4_equals_2": "uniform_q10_frame_in_source_motion",
                "env_id_modulo_4_equals_3": "uniform_q10_frame_in_return_ramp",
            },
            "physical_state_initialization": "reference_q10_pose_and_velocity_at_environment_reset_only",
            "reference_after_reset": "every_remaining_frame_through_final_standing_without_skips",
            "sampled_suffix_completion_is_full_lifecycle_completion": False,
            "sampled_reference_state_dynamically_qualified": False,
            "mid_episode_pose_writes": False,
            "reference_arrays_changed": False,
            "evaluation_schedule_changed": False,
            "evaluation_requires_full_standing_start_lifecycle": True,
            "deployment_ready": False,
        }
    if name == "mixed_reference_reset_v1":
        return {
            "name": name,
            "standing_environments": "env_id_modulo_4_equals_0",
            "other_environments": "uniform_q10_frame_within_selected_complete_source_motion",
            "physical_state_initialization": "reference_q10_pose_and_velocity_at_environment_reset_only",
            "reference_after_reset": "every_remaining_frame_through_final_standing_without_skips",
            "sampled_suffix_completion_is_full_lifecycle_completion": False,
            "sampled_reference_state_dynamically_qualified": False,
            "mid_episode_pose_writes": False,
            "evaluation_schedule_changed": False,
            "evaluation_requires_full_standing_start_lifecycle": True,
            "deployment_ready": False,
        }
    staggered = name != "synchronous"
    return {
        "name": name,
        "initial_hold_controls": "floor(env_id * selected_lifecycle_controls / num_envs)" if staggered else "0",
        "hold_applied": "first_environment_reset_only" if staggered else "none",
        "held_reference": "stationary_standing_q9_and_q10_only",
        "measured_state_and_history_continue_during_hold": True,
        "source_frames_skipped": 0,
        "mid_episode_pose_writes": False,
        "evaluation_schedule_changed": False,
        "deployment_ready": False,
    }


def start_schedule_transition_contract(previous_feedback, args):
    before = previous_feedback.get("start_schedule", start_schedule_contract())
    if before != start_schedule_contract(before["name"]):
        raise ValueError("parent standing-start schedule contract is invalid")
    after = start_schedule_contract(getattr(args, "start_schedule", "synchronous"))
    changed = before != after
    if changed and not getattr(args, "allow_start_schedule_transition", False):
        raise ValueError("start schedule transition requires explicit --allow-start-schedule-transition")
    return {
        "old": before,
        "new": after,
        "rollout_schedule_changed": changed,
        "same_schedule_resume_claimed": not changed,
        "actor_critic_optimizer_and_counters_preserved": True,
        "fresh_standing_environment_reset": after["name"] not in REFERENCE_RESET_SCHEDULES,
        "reference_arrays_and_evaluation_unchanged": True,
    }
