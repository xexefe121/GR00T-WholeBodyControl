"""Simulation root-conditioned lifecycle task, with fixed-world q10 targets."""

from __future__ import annotations

import torch

from gear_sonic.envs.mjlab.sonic_true23_generalist_curriculum import (
    GeneralistCurriculumMotionCommand,
    configure_curriculum_environment,
)
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract, root_feedback_torch


def _causal_indices(command):
    anchor = command.time_steps
    proof = anchor + 1
    if torch.any(anchor < 0) or torch.any(proof >= command.motion.time_step_total):
        raise ValueError("root feedback requires available q9 and q10 source frames")
    return anchor, proof


def current_root_reference(command):
    anchor, proof = _causal_indices(command)
    index = command.motion_anchor_body_index
    position = command.motion.body_pos_w[proof, index]
    previous = command.motion.body_pos_w[anchor, index]
    return position + command._env.scene.env_origins, (position - previous) / 0.02


def root_feedback_observation(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    desired, desired_velocity = current_root_reference(command)
    return root_feedback_torch(
        desired,
        command.robot_anchor_pos_w,
        desired_velocity,
        command.robot_anchor_lin_vel_w,
        command.robot_anchor_quat_w,
    )


def root_tracking_squared_error(env, command_name="motion"):
    features = root_feedback_observation(env, command_name)
    return torch.sum(features[:, :3] ** 2, dim=-1) + 0.2 * torch.sum(
        (features[:, 3:6] - features[:, 6:9]) ** 2, dim=-1
    )


def refresh_world_targets(command):
    _, proof = _causal_indices(command)
    command.body_pos_relative_w.copy_(
        command.motion.body_pos_w[proof] + command._env.scene.env_origins[:, None, :]
    )
    command.body_quat_relative_w.copy_(command.motion.body_quat_w[proof])


def _q10_body_position(command):
    _, proof = _causal_indices(command)
    return command.motion.body_pos_w[proof] + command._env.scene.env_origins[:, None, :]


def _q10_body_quaternion(command):
    return command.motion.body_quat_w[_causal_indices(command)[1]]


def _body_indices(command, body_names):
    configured = tuple(command.cfg.body_names)
    if body_names is None:
        return list(range(len(configured)))
    if (
        not body_names
        or len(set(body_names)) != len(body_names)
        or any(name not in configured for name in body_names)
    ):
        raise ValueError("q10 objective requires unique known body names")
    return [i for i, name in enumerate(configured) if name in body_names]


def q10_root_position_reward(env, command_name, std):
    command = env.command_manager.get_term(command_name)
    desired, _ = current_root_reference(command)
    return torch.exp(-torch.sum((desired - command.robot_anchor_pos_w) ** 2, dim=-1) / std**2)


def q10_root_orientation_reward(env, command_name, std):
    from mjlab.utils.lab_api.math import quat_error_magnitude

    command = env.command_manager.get_term(command_name)
    desired = _q10_body_quaternion(command)[:, command.motion_anchor_body_index]
    return torch.exp(-(quat_error_magnitude(desired, command.robot_anchor_quat_w) ** 2) / std**2)


def q10_body_position_reward(env, command_name, std, body_names=None):
    command = env.command_manager.get_term(command_name)
    indices = _body_indices(command, body_names)
    error = ((_q10_body_position(command)[:, indices] - command.robot_body_pos_w[:, indices]) ** 2).sum(-1)
    return torch.exp(-error.mean(-1) / std**2)


def q10_body_orientation_reward(env, command_name, std, body_names=None):
    from mjlab.utils.lab_api.math import quat_error_magnitude

    command = env.command_manager.get_term(command_name)
    indices = _body_indices(command, body_names)
    error = (
        quat_error_magnitude(_q10_body_quaternion(command)[:, indices], command.robot_body_quat_w[:, indices]) ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def q10_body_linear_velocity_reward(env, command_name, std, body_names=None):
    command = env.command_manager.get_term(command_name)
    anchor, proof = _causal_indices(command)
    indices = _body_indices(command, body_names)
    desired = (command.motion.body_pos_w[proof] - command.motion.body_pos_w[anchor]) / 0.02
    error = ((desired[:, indices] - command.robot_body_lin_vel_w[:, indices]) ** 2).sum(-1)
    return torch.exp(-error.mean(-1) / std**2)


def current_body_angular_velocity_reference(command):
    """World-frame shortest-arc rotation q9 to q10; no NPZ derivative or q11."""
    from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul

    anchor, proof = _causal_indices(command)
    current, previous = command.motion.body_quat_w[proof], command.motion.body_quat_w[anchor]
    return axis_angle_from_quat(quat_mul(current, quat_conjugate(previous))) / 0.02


def q10_body_angular_velocity_reward(env, command_name, std, body_names=None):
    command = env.command_manager.get_term(command_name)
    indices = _body_indices(command, body_names)
    desired = current_body_angular_velocity_reference(command)
    error = ((desired[:, indices] - command.robot_body_ang_vel_w[:, indices]) ** 2).sum(-1)
    return torch.exp(-error.mean(-1) / std**2)


def q10_joint_position_reward(env, command_name, std, joint_indices=None):
    command = env.command_manager.get_term(command_name)
    desired = command.motion.joint_pos[_causal_indices(command)[1]]
    indices = joint_indices if joint_indices is not None else tuple(range(desired.shape[1]))
    return torch.exp(-((desired[:, indices] - command.robot_joint_pos[:, indices]) ** 2).mean(-1) / std**2)


def q10_joint_velocity_reward(env, command_name, std, joint_indices=None):
    command = env.command_manager.get_term(command_name)
    anchor, proof = _causal_indices(command)
    desired = (command.motion.joint_pos[proof] - command.motion.joint_pos[anchor]) / 0.02
    indices = joint_indices if joint_indices is not None else tuple(range(desired.shape[1]))
    return torch.exp(-((desired[:, indices] - command.robot_joint_vel[:, indices]) ** 2).mean(-1) / std**2)


def q10_measured_joint_position_l2(env, command_name="motion"):
    """Dense posture cost on actual physical joints, independent of PD request."""
    from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_ACTION_SCALE

    command = env.command_manager.get_term(command_name)
    desired = command.motion.joint_pos[_causal_indices(command)[1]]
    measured = command.robot_joint_pos
    if desired.shape != measured.shape or measured.ndim != 2 or measured.shape[1] != 23:
        raise ValueError("measured posture objective requires matching native23 joint states")
    if not torch.isfinite(desired).all() or not torch.isfinite(measured).all():
        raise ValueError("measured posture objective requires finite states")
    scale = torch.as_tensor(HARDWARE_23_ACTION_SCALE, dtype=measured.dtype, device=measured.device)
    return ((measured - desired) / scale).square().mean(-1)


def q10_action_target_reference_l2(env, command_name="motion", action_name="joint_pos"):
    from gear_sonic.envs.mjlab.sonic_true23_low_latency_recovery import _processed_target
    from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_ACTION_SCALE

    command = env.command_manager.get_term(command_name)
    reference = command.motion.joint_pos[_causal_indices(command)[1]]
    target = _processed_target(env, action_name)
    if reference.shape != target.shape or not torch.isfinite(reference).all():
        raise ValueError("q10 reference joint target must be finite [env,23]")
    scale = torch.as_tensor(HARDWARE_23_ACTION_SCALE, dtype=target.dtype, device=target.device)
    return ((target - reference) / scale).square().mean(-1)


def q10_bad_root_position(env, command_name, threshold):
    command = env.command_manager.get_term(command_name)
    return (
        torch.linalg.vector_norm(current_root_reference(command)[0] - command.robot_anchor_pos_w, dim=-1)
        > threshold
    )


def q10_bad_root_height(env, command_name, threshold):
    command = env.command_manager.get_term(command_name)
    return (current_root_reference(command)[0][:, 2] - command.robot_anchor_pos_w[:, 2]).abs() > threshold


def q10_bad_root_orientation(env, asset_cfg, command_name, threshold):
    from mjlab.utils.lab_api.math import quat_apply_inverse

    command = env.command_manager.get_term(command_name)
    gravity = env.scene[asset_cfg.name].data.gravity_vec_w
    desired = _q10_body_quaternion(command)[:, command.motion_anchor_body_index]
    reference = quat_apply_inverse(desired, gravity)
    measured = quat_apply_inverse(command.robot_anchor_quat_w, gravity)
    return (reference[:, 2] - measured[:, 2]).abs() > threshold


def q10_bad_body_position(env, command_name, threshold, body_names=None):
    command = env.command_manager.get_term(command_name)
    indices = _body_indices(command, body_names)
    error = _q10_body_position(command)[:, indices] - command.robot_body_pos_w[:, indices]
    return (torch.linalg.vector_norm(error, dim=-1) > threshold).any(-1)


def q10_bad_body_height(env, command_name, threshold, body_names=None):
    command = env.command_manager.get_term(command_name)
    indices = _body_indices(command, body_names)
    error = _q10_body_position(command)[:, indices, 2] - command.robot_body_pos_w[:, indices, 2]
    return (error.abs() > threshold).any(-1)


def configure_q10_objectives(cfg):
    """Replace known target functions only; leave weights, params and q9 properties intact."""
    from src.tasks.tracking.mdp import rewards, terminations
    from gear_sonic.envs.mjlab.sonic_true23_low_latency_recovery import action_target_reference_l2

    reward_functions = {
        rewards.motion_global_anchor_position_error_exp: q10_root_position_reward,
        rewards.motion_global_anchor_orientation_error_exp: q10_root_orientation_reward,
        rewards.motion_relative_body_position_error_exp: q10_body_position_reward,
        rewards.motion_relative_body_orientation_error_exp: q10_body_orientation_reward,
        rewards.motion_global_body_linear_velocity_error_exp: q10_body_linear_velocity_reward,
        rewards.motion_global_body_angular_velocity_error_exp: q10_body_angular_velocity_reward,
        rewards.motion_joint_position_error_exp: q10_joint_position_reward,
        rewards.motion_joint_velocity_error_exp: q10_joint_velocity_reward,
        action_target_reference_l2: q10_action_target_reference_l2,
    }
    termination_functions = {
        terminations.bad_anchor_pos: q10_bad_root_position,
        terminations.bad_anchor_pos_z_only: q10_bad_root_height,
        terminations.bad_anchor_ori: q10_bad_root_orientation,
        terminations.bad_motion_body_pos: q10_bad_body_position,
        terminations.bad_motion_body_pos_z_only: q10_bad_body_height,
    }
    for terms, functions in ((cfg.rewards, reward_functions), (cfg.terminations, termination_functions)):
        for name, term in terms.items():
            if term is None:
                continue
            if term.func in functions:
                term.func = functions[term.func]
            elif getattr(term.func, "__name__", "").startswith(("motion_", "bad_anchor_", "bad_motion_")):
                if term.func not in functions.values():
                    raise ValueError(f"unknown reference objective requires explicit q10 binding: {name}")
    return cfg


def root_objective_contract():
    return {
        "kind": "g1_native23_root_feedback_q10_objectives_v1",
        "reference_positions_orientations_and_joint_positions": "q10_current_received_sample_fixed_world",
        "linear_and_joint_velocity_reference": "backward_difference_q10_minus_q9_over_0p02s",
        "angular_velocity_reference": "world_shortest_arc_log_quat_q10_times_inverse_q9_over_0p02s",
        "tokenizer_and_original_command_properties": "unchanged_q9",
        "reward_and_termination_phase": "post_physics_before_command_advance_against_held_received_q10",
        "future_q11_reference_read": False,
        "derived_measured_reward_state_phase": "mjlab_last_preintegration_substep_2ms_stale_no_engine_change",
        "actor_observation_phase": "post_forward_synchronized_current_state",
        "existing_ee_barriers_reference": "q10_fixed_world_body_pos_relative_w_buffer",
        "physical_only_penalties_and_timeouts": "unchanged",
    }


class RootFeedbackCurriculumCommand(GeneralistCurriculumMotionCommand):
    def _refresh_targets_from_causal_anchor(self):
        # Only reward/termination targets change. Frozen SONIC observations
        # still use their original q9 reference and buffered orientation.
        refresh_world_targets(self)

    def curriculum_receipt(self):
        return {
            **super().curriculum_receipt(),
            "root_feedback": root_feedback_contract(),
            "reward_task_target_frame": "q10_fixed_world",
            "objective_contract": root_objective_contract(),
            "measured_root_source": "simulator_ground_truth",
            "physical_estimator_qualified": False,
        }


def install_root_feedback_command(spans):
    from gear_sonic.envs.mjlab import sonic_true23_causal_history as task

    def build(cfg, env):
        return RootFeedbackCurriculumCommand(cfg, env, spans=spans)

    task.CausalHistoryMotionCommandCfg.build = build


def configure_root_feedback_environment(
    cfg,
    spans,
    *,
    reset_position_range_m=0.0,
    reset_velocity_range_m_s=0.0,
    objective_profile="legacy_root_tracking",
):
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from gear_sonic.utils.g1_true23_root_feedback_objectives import objective_profile_contract

    objectives = objective_profile_contract(objective_profile)
    if not 0 <= reset_position_range_m <= 0.15 or not 0 <= reset_velocity_range_m_s <= 0.15:
        raise ValueError("root reset perturbation exceeds bounded nominal acquisition range")
    configure_curriculum_environment(cfg, spans)
    configure_q10_objectives(cfg)
    cfg.observations["root_feedback"] = ObservationGroupCfg(
        terms={"root_feedback": ObservationTermCfg(func=root_feedback_observation)},
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="error",
    )
    cfg.rewards["root_world_tracking_error"] = RewardTermCfg(func=root_tracking_squared_error, weight=-10.0)
    if objectives["measured_joint_position_l2_weight"]:
        cfg.rewards["measured_joint_posture_l2"] = RewardTermCfg(
            func=q10_measured_joint_position_l2,
            weight=objectives["measured_joint_position_l2_weight"],
        )
    # Applied only by the inherited environment-reset routine, never mid-cycle.
    cfg.commands["motion"].pose_range = {
        key: (-reset_position_range_m, reset_position_range_m) for key in ("x", "y")
    }
    cfg.commands["motion"].velocity_range = {
        key: (-reset_velocity_range_m_s, reset_velocity_range_m_s) for key in ("x", "y")
    }
    return cfg
