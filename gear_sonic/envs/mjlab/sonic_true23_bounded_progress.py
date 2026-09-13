"""Fresh SIM-only bounded base reward and correctly phased world progress."""

import copy

import torch

from gear_sonic.envs.mjlab import sonic_true23_original_intent as intent, sonic_true23_root_feedback as root
from gear_sonic.utils.g1_true23_bounded_progress import (
    ALIVE_WEIGHT,
    FAILURE_WEIGHT,
    NEGATIVE_WEIGHTS,
    POSITIVE_WEIGHTS,
    STEP_DT,
    assert_base_bounds,
    bounded_cost,
    finite_vector,
    potential_difference,
    reward_contract,
)


def expected_functions():
    from mjlab.envs.mdp import rewards as mdp
    from src.tasks.tracking.mdp.rewards import self_collision_cost

    from gear_sonic.envs.mjlab.sonic_true23_low_latency_recovery import action_target_soft_limit_barrier
    from gear_sonic.envs.mjlab.sonic_true23_native_model_actuation import requested_effort_excess

    return {
        "action_rate_l2": mdp.action_rate_l2,
        "joint_limit": mdp.joint_pos_limits,
        "self_collisions": self_collision_cost,
        "action_target_reference_l2": intent.lower_posture_l2,
        "action_target_soft_limit_barrier": action_target_soft_limit_barrier,
        "worst_ee_z_normalized_squared": intent.worst_ee_height_cost,
        "right_wrist_prethreshold_barrier": intent.right_hand_height_barrier,
        "requested_effort_excess": requested_effort_excess,
        "root_world_tracking_error": root.root_tracking_squared_error,
        "measured_joint_posture_l2": intent.lower_posture_l2,
        "measured_feet_world_position_l2": root.q10_measured_feet_world_position_l2,
        "original_task_position_l2": intent.original_task_position_l2,
        "original_task_orientation_l2": intent.original_task_orientation_l2,
        "motion_global_root_pos": root.q10_root_position_reward,
        "motion_global_root_ori": root.q10_root_orientation_reward,
        "motion_body_pos": root.q10_body_position_reward,
        "motion_body_ori": root.q10_body_orientation_reward,
        "motion_body_lin_vel": root.q10_body_linear_velocity_reward,
        "motion_body_ang_vel": root.q10_body_angular_velocity_reward,
        "alive": mdp.is_alive,
        "non_timeout_termination": mdp.is_terminated,
    }


def transformed_cost(env, raw_cost_function, reward_term_name, **params):
    raw = raw_cost_function(env, **params)
    value = bounded_cost(raw)
    capture = getattr(env, "_bounded_progress_raw_costs", None)
    if capture is not None:
        if reward_term_name in capture:
            raise ValueError("raw cost evaluated twice in a bounded reward step")
        capture[reward_term_name] = raw.detach().clone()
    return value


def configure_environment(cfg):
    if cfg.scale_rewards_by_dt is not True:
        raise ValueError("bounded reward requires the known dt-scaled predecessor")
    result = copy.deepcopy(cfg)
    functions = expected_functions()
    weights = {**NEGATIVE_WEIGHTS, **POSITIVE_WEIGHTS, "alive": 5.0, "non_timeout_termination": FAILURE_WEIGHT}
    if set(result.rewards) != set(weights):
        raise ValueError("bounded reward requires exactly the known21 raw terms")
    for name, weight in weights.items():
        term = result.rewards[name]
        if term is None or term.func is not functions[name] or term.weight != weight:
            raise ValueError(f"unrecognized bounded reward predecessor: {name}")
        if name in NEGATIVE_WEIGHTS:
            if {"raw_cost_function", "reward_term_name"} & set(term.params):
                raise ValueError("reserved bounded reward parameters already present")
            term.func = transformed_cost
            # Keep SceneEntityCfg parameters at the top level so the actual
            # RewardManager resolves precisely the same entity/joint indices.
            term.params.update(raw_cost_function=functions[name], reward_term_name=name)
    result.rewards["alive"].weight = ALIVE_WEIGHT
    return result


def verify_runtime(env):
    if abs(float(env.step_dt) - STEP_DT) > 1e-12 or env.reward_manager._scale_by_dt is not True:
        raise ValueError("bounded reward requires20ms dt-scaled controls")
    functions = expected_functions()
    weights = {
        **NEGATIVE_WEIGHTS,
        **POSITIVE_WEIGHTS,
        "alive": ALIVE_WEIGHT,
        "non_timeout_termination": FAILURE_WEIGHT,
    }
    if set(env.reward_manager.active_terms) != set(weights):
        raise ValueError("executed bounded reward inventory differs")
    for name, weight in weights.items():
        term = env.reward_manager.get_term_cfg(name)
        expected = transformed_cost if name in NEGATIVE_WEIGHTS else functions[name]
        if term.func is not expected or term.weight != weight:
            raise ValueError(f"executed bounded reward differs: {name}")
        if name in NEGATIVE_WEIGHTS and (
            term.params.get("raw_cost_function") is not functions[name]
            or term.params.get("reward_term_name") != name
        ):
            raise ValueError(f"executed raw cost binding differs: {name}")
    return reward_contract()


def world_potential(env, spec):
    command = env.command_manager.get_term("motion")
    desired, _ = root.current_root_reference(command)
    parts = torch.stack(
        (
            10 * (desired - command.robot_anchor_pos_w).square().sum(-1),
            root.q10_measured_feet_world_position_l2(env),
            intent.original_task_position_l2(env, spec),
        ),
        -1,
    )
    total = finite_vector(parts.sum(-1), "world potential cost")
    if (parts < 0).any():
        raise ValueError("world potential requires nonnegative costs")
    return -torch.log1p(total), parts


class ProgressRewardStep:
    """Read-only reward overlay around the original five-result SIM step.

    No pose writes, additional simulation/command/observation calls, cached
    previous potential, terminal-state substitution, or deployment hook.
    Every reward transition is retained for an independent arithmetic audit.
    """

    def __init__(self, env, spec, *, potential=world_potential):
        self.env, self.spec, self.potential = env, copy.deepcopy(spec), potential
        self.original_step = env.step
        self.rows = []
        self.episode_shaping = torch.zeros(env.num_envs, device=env.device)

    def __call__(self, actions):
        env = self.env
        before, parts_before = self.potential(env, self.spec)
        before = before.detach().clone()
        env._bounded_progress_raw_costs = {}
        obs, base, terminated, timeouts, extras = self.original_step(actions)
        after, parts_after = self.potential(env, self.spec)
        delta = potential_difference(before, after, terminated, timeouts)
        assert_base_bounds(base, terminated)
        reward = base + delta
        finite_vector(reward, "shaped reward")
        raw_costs = env._bounded_progress_raw_costs
        env._bounded_progress_raw_costs = None
        if set(raw_costs) != set(NEGATIVE_WEIGHTS):
            raise ValueError("incomplete actual raw-cost reward capture")
        self.rows.append(
            {
                name: value.detach().cpu().clone()
                for name, value in {
                    "phi_before": before,
                    "phi_after_including_reset_states": after,
                    "world_cost_parts_before": parts_before,
                    "world_cost_parts_after_including_reset_states": parts_after,
                    "base_reward": base,
                    "shaping_reward": delta,
                    "returned_reward": reward,
                    "terminated": terminated,
                    "timeouts": timeouts,
                    "raw_costs": torch.stack([raw_costs[n] for n in NEGATIVE_WEIGHTS], -1),
                    "weighted_base_components": env.reward_manager._step_reward * STEP_DT,
                    "critic_value_before_bootstrap": torch.zeros_like(before),
                    "stored_reward_with_timeout_bootstrap": torch.zeros_like(before),
                    "stored_done": torch.zeros_like(terminated),
                    "ppo_recorded": torch.zeros_like(terminated),
                }.items()
            }
        )
        self.episode_shaping += delta
        done = terminated | timeouts
        if done.any():
            extras.setdefault("log", {})["Episode_Reward/world_potential_shaping"] = (
                self.episode_shaping[done].mean() / env.max_episode_length_s
            )
            self.episode_shaping[done] = 0
        return obs, reward, terminated, timeouts, extras

    def capture(self):
        if not self.rows:
            return {}
        return {name: torch.stack([row[name] for row in self.rows]).numpy() for name in self.rows[0]}


def install_progress_step(env, spec):
    verify_runtime(env)
    if hasattr(env, "_bounded_progress_step"):
        raise ValueError("bounded progress step already installed")
    step = ProgressRewardStep(env, spec)
    env.step = step
    env._bounded_progress_step = step
    return step
