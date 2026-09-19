"""Explicit SIM reward transform; neither a controller nor a hardware fix.

The bounded base intentionally changes the old unbounded objective. Potential
shaping is relative to this NEW base, with the installed RSL self-bootstrap on
timeouts accounted for. No invariance to the predecessor reward is claimed.
"""

import math

import torch

KIND = "native23_bounded_costs_discounted_world_progress_v1"
GAMMA = 0.99
STEP_DT = 0.02
NEGATIVE_WEIGHTS = {
    "action_rate_l2": -0.2,
    "joint_limit": -20.0,
    "self_collisions": -10.0,
    "action_target_reference_l2": -2.0,
    "action_target_soft_limit_barrier": -10.0,
    "worst_ee_z_normalized_squared": -20.0,
    "right_wrist_prethreshold_barrier": -25.0,
    "requested_effort_excess": -0.1,
    "root_world_tracking_error": -10.0,
    "measured_joint_posture_l2": -10.0,
    "measured_feet_world_position_l2": -1.0,
    "original_task_position_l2": -1.0,
    "original_task_orientation_l2": -1.0,
}
POSITIVE_WEIGHTS = {
    "motion_global_root_pos": 0.5,
    "motion_global_root_ori": 0.5,
    "motion_body_pos": 1.0,
    "motion_body_ori": 1.0,
    "motion_body_lin_vel": 1.0,
    "motion_body_ang_vel": 1.0,
}
COST_RATE_CAP = math.fsum(-w for w in NEGATIVE_WEIGHTS.values())
ALIVE_WEIGHT = 5.0 + COST_RATE_CAP
FAILURE_WEIGHT = -5000.0


def reward_contract():
    return {
        "kind": KIND,
        "negative_weights": dict(NEGATIVE_WEIGHTS),
        "positive_weights": dict(POSITIVE_WEIGHTS),
        "raw_nonnegative_cost_transform": "x/(1+x)",
        "near_zero_cost_derivative": 1.0,
        "alive_weight": ALIVE_WEIGHT,
        "non_timeout_termination_weight": FAILURE_WEIGHT,
        "nonterminal_base_reward_bounds_per_control": [0.1, 2.406],
        "true_terminal_base_reward_bounds_per_control": [-102.206, -99.9],
        "step_dt": STEP_DT,
        "gamma": GAMMA,
        "potential": "-log1p(10*root_position_error_squared+feet_world_cost+original_task_world_cost)",
        "potential_sampling": "synchronized_actor_state_and_current_received_q1_before_and_after_env_step",
        "normal_shaping": "gamma*phi_next-phi_previous",
        "true_terminal_shaping_without_timeout": "-phi_previous",
        "timeout_shaping_including_simultaneous_termination": "(gamma-1)*phi_previous",
        "timeout_reason": "installed_RSL_process_env_step_bootstraps_V(previous)_not_V(terminal_next)",
        "post_reset_potential_used_for_done_transition": False,
        "additional_reward_scaled_by_dt_again": False,
        "old_original_intent_spec_role": "raw_reference_and_cost_definitions_before_this_explicit_transform",
        "same_objective_as_unbounded_predecessor": False,
        "potential_changes_optimal_policy_relative_to_new_base_claimed": False,
        "termination_flags_or_physics_changed": False,
        "actor_inputs_or_action_transform_changed": False,
        "training_resume_supported": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def finite_vector(value, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or not value.numel()
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} requires a nonempty finite real floating vector")
    return value


def bounded_cost(value):
    finite_vector(value, "raw cost")
    if (value < 0).any():
        raise ValueError("raw cost must be nonnegative")
    return value / (1 + value)


def potential_difference(before, after, terminated, timeouts, gamma=GAMMA):
    finite_vector(before, "previous potential")
    finite_vector(after, "next potential")
    if before.shape != after.shape or before.dtype != after.dtype or before.device != after.device:
        raise ValueError("potential vector shape/dtype/device mismatch")
    for flags in (terminated, timeouts):
        if (
            not isinstance(flags, torch.Tensor)
            or flags.shape != before.shape
            or flags.dtype != torch.bool
            or flags.device != before.device
        ):
            raise ValueError("termination and timeout must be matching boolean vectors")
    if type(gamma) not in (float, int) or not math.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be finite in [0,1)")
    normal_or_terminal = gamma * torch.where(terminated, torch.zeros_like(after), after) - before
    # The installed algorithm adds gamma*V(previous) on timeouts, even when
    # failure and timeout coincide. Match that algebra; do not change the flags.
    return torch.where(timeouts, (gamma - 1) * before, normal_or_terminal)


def assert_base_bounds(rewards, terminated):
    finite_vector(rewards, "bounded base reward")
    if terminated.dtype != torch.bool or terminated.shape != rewards.shape:
        raise ValueError("invalid true-terminal flags")
    lower = torch.where(terminated, -102.206, 0.1)
    upper = torch.where(terminated, -99.9, 2.406)
    if ((rewards < lower - 5e-5) | (rewards > upper + 5e-5)).any():
        raise ValueError("executed bounded base reward exceeds its declared bounds")
