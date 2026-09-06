"""World targets must not follow displaced robot or use unreceived frames."""

from types import SimpleNamespace

import pytest
import torch

from gear_sonic.envs.mjlab import sonic_true23_root_feedback as task
from gear_sonic.envs.mjlab.sonic_true23_root_feedback import (
    current_root_reference,
    refresh_world_targets,
    root_feedback_observation,
    root_tracking_squared_error,
)


def fixture():
    positions = torch.zeros(12, 2, 3)
    positions[:, :, 2] = 0.8
    positions[10, :, 0] = 0.02
    positions[11, :, 0] = 400.0  # This is an unavailable future sample.
    quaternions = torch.zeros(12, 2, 4)
    quaternions[..., 0] = 1
    command = SimpleNamespace(
        time_steps=torch.tensor([9, 9]),
        motion_anchor_body_index=0,
        motion=SimpleNamespace(body_pos_w=positions, body_quat_w=quaternions, time_step_total=12),
        _env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.tensor([[0.0, 0, 0], [4.0, 5, 0]]))),
        robot_anchor_pos_w=torch.tensor([[0.0, 0, 0.8], [4.0, 5, 0.8]]),
        robot_anchor_quat_w=torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
        robot_anchor_lin_vel_w=torch.zeros(2, 3),
        body_pos_relative_w=torch.zeros(2, 2, 3),
        body_quat_relative_w=torch.zeros(2, 2, 4),
    )
    return command, SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))


def test_world_reference_uses_current_proof_and_backward_velocity_only():
    command, env = fixture()
    position, velocity = current_root_reference(command)
    torch.testing.assert_close(velocity, torch.tensor([[1.0, 0, 0], [1.0, 0, 0]]))
    torch.testing.assert_close(position[:, 0], torch.tensor([0.02, 4.02]))
    features = root_feedback_observation(env)
    command.motion.body_pos_w[11] = -500
    torch.testing.assert_close(root_feedback_observation(env), features, atol=0, rtol=0)


def test_targets_stay_world_fixed_and_displacement_changes_observation_and_cost():
    command, env = fixture()
    refresh_world_targets(command)
    reference = command.body_pos_relative_w.clone()
    before = root_feedback_observation(env)
    cost = root_tracking_squared_error(env)
    command.robot_anchor_pos_w[:, :2] += torch.tensor([8.0, -3.0])
    refresh_world_targets(command)
    torch.testing.assert_close(command.body_pos_relative_w, reference, atol=0, rtol=0)
    torch.testing.assert_close(
        root_feedback_observation(env)[:, :2] - before[:, :2], torch.tensor([[-8.0, 3.0], [-8.0, 3.0]])
    )
    assert torch.all(root_tracking_squared_error(env) > cost)


def test_missing_proof_rejected():
    command, _ = fixture()
    command.time_steps[:] = 11
    with pytest.raises(ValueError):
        current_root_reference(command)
    with pytest.raises(ValueError):
        refresh_world_targets(command)


def objective_fixture():
    command, env = fixture()
    command.cfg = SimpleNamespace(body_names=("pelvis", "hand"))
    command.motion.body_pos_w[10, :, 2] = 0.9
    command.motion.body_quat_w[10, :, 0] = torch.cos(torch.tensor(0.2))
    command.motion.body_quat_w[10, :, 1] = torch.sin(torch.tensor(0.2))
    command.motion.joint_pos = torch.zeros(12, 23)
    command.motion.joint_pos[10] = 0.02
    # Precomputed NPZ derivatives are deliberately wrong: never use them.
    command.motion.joint_vel = torch.full((12, 23), 777.0)
    command.motion.body_lin_vel_w = torch.full((12, 2, 3), 888.0)
    command.motion.body_ang_vel_w = torch.full((12, 2, 3), 999.0)
    command.robot_body_pos_w = task._q10_body_position(command).clone()
    command.robot_body_quat_w = task._q10_body_quaternion(command).clone()
    command.robot_body_lin_vel_w = (command.motion.body_pos_w[10] - command.motion.body_pos_w[9])[None].repeat(
        2, 1, 1
    ) / 0.02
    command.robot_body_ang_vel_w = task.current_body_angular_velocity_reference(command).clone()
    command.robot_anchor_pos_w = command.robot_body_pos_w[:, 0].clone()
    command.robot_anchor_quat_w = command.robot_body_quat_w[:, 0].clone()
    command.robot_joint_pos = command.motion.joint_pos[10:11].repeat(2, 1)
    command.robot_joint_vel = torch.ones(2, 23)
    env.num_envs = 2
    env.scene = {
        "robot": SimpleNamespace(data=SimpleNamespace(gravity_vec_w=torch.tensor([[0.0, 0, -1.0]]).repeat(2, 1)))
    }
    env.action_manager = SimpleNamespace(
        get_term=lambda name: SimpleNamespace(processed_action=command.robot_joint_pos.clone())
    )
    return command, env


REWARDS = (
    task.q10_root_position_reward,
    task.q10_root_orientation_reward,
    task.q10_body_position_reward,
    task.q10_body_orientation_reward,
    task.q10_body_linear_velocity_reward,
    task.q10_body_angular_velocity_reward,
    task.q10_joint_position_reward,
    task.q10_joint_velocity_reward,
)


@pytest.mark.parametrize("reward", REWARDS)
def test_every_tracking_reward_uses_q10_and_ignores_future_and_npz_velocity(reward):
    command, env = objective_fixture()
    before = reward(env, "motion", 0.3)
    torch.testing.assert_close(before, torch.ones(2))
    for value in vars(command.motion).values():
        if isinstance(value, torch.Tensor):
            value[11] = float("nan")
    torch.testing.assert_close(reward(env, "motion", 0.3), before, atol=0, rtol=0)


def test_action_target_and_termination_wrappers_preserve_q9_properties():
    from gear_sonic.envs.mjlab.sonic_true23 import SonicTrue23MotionCommand

    command, env = objective_fixture()
    before = {
        name: getattr(SonicTrue23MotionCommand, name).fget(command).clone()
        for name in ("joint_pos", "anchor_pos_w", "anchor_quat_w")
    }
    torch.testing.assert_close(task.q10_action_target_reference_l2(env), torch.zeros(2))
    for term in (
        task.q10_bad_root_position,
        task.q10_bad_root_height,
        task.q10_bad_body_position,
        task.q10_bad_body_height,
    ):
        assert not term(env, "motion", 0.01).any()
    assert not task.q10_bad_root_orientation(env, SimpleNamespace(name="robot"), "motion", 0.01).any()
    refresh_world_targets(command)
    for name, value in before.items():
        torch.testing.assert_close(getattr(SonicTrue23MotionCommand, name).fget(command), value, atol=0, rtol=0)
    assert not torch.equal(before["joint_pos"], command.robot_joint_pos)
    assert not torch.equal(before["anchor_pos_w"], command.robot_anchor_pos_w)
    # Measured q9 now violates current q10. Tests must not silently match q9.
    command.robot_anchor_pos_w = before["anchor_pos_w"]
    command.robot_anchor_quat_w = before["anchor_quat_w"]
    command.robot_body_pos_w[:, :, 2] -= 0.1
    for term in (
        task.q10_bad_root_position,
        task.q10_bad_root_height,
        task.q10_bad_body_position,
        task.q10_bad_body_height,
    ):
        assert term(env, "motion", 0.01).all()
    assert task.q10_bad_root_orientation(env, SimpleNamespace(name="robot"), "motion", 0.01).all()


def test_backward_angular_velocity_is_world_frame_and_quaternion_sign_invariant():
    from mjlab.utils.lab_api.math import quat_mul

    command, _ = objective_fixture()
    previous = torch.tensor([torch.cos(torch.tensor(0.4)), 0, torch.sin(torch.tensor(0.4)), 0])
    delta_world = torch.tensor([torch.cos(torch.tensor(0.02)), 0, 0, torch.sin(torch.tensor(0.02))])
    command.motion.body_quat_w[9] = previous
    command.motion.body_quat_w[10] = quat_mul(delta_world, previous)
    desired = task.current_body_angular_velocity_reference(command)
    # Two float32 quaternion products precede division by 20 ms.
    torch.testing.assert_close(desired, torch.tensor([0.0, 0, 2.0]).expand(2, 2, 3), atol=1e-5, rtol=0)
    command.motion.body_quat_w[10] *= -1
    torch.testing.assert_close(task.current_body_angular_velocity_reference(command), desired, atol=0, rtol=0)


def test_real_active_configuration_has_q10_targets_preserves_weights_and_physical_terms():
    from copy import deepcopy
    from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import make_causal_multimotion_v14_env_cfg

    cfg = make_causal_multimotion_v14_env_cfg(motion_file="unused_static_configuration.npz", num_envs=2)
    before = deepcopy(cfg)
    task.configure_root_feedback_environment(cfg, [{"timeline": {"total_requested_controls": 1850}}])
    expected = {
        "motion_global_root_pos": task.q10_root_position_reward,
        "motion_global_root_ori": task.q10_root_orientation_reward,
        "motion_body_pos": task.q10_body_position_reward,
        "motion_body_ori": task.q10_body_orientation_reward,
        "motion_body_lin_vel": task.q10_body_linear_velocity_reward,
        "motion_body_ang_vel": task.q10_body_angular_velocity_reward,
        "action_target_reference_l2": task.q10_action_target_reference_l2,
    }
    for name, old in before.rewards.items():
        assert cfg.rewards[name].weight == old.weight
        assert cfg.rewards[name].params == old.params
        assert cfg.rewards[name].func is expected.get(name, old.func)
    expected_terminations = {
        "anchor_pos": task.q10_bad_root_height,
        "anchor_ori": task.q10_bad_root_orientation,
        "ee_body_pos": task.q10_bad_body_height,
    }
    for name, old in before.terminations.items():
        assert cfg.terminations[name].params == old.params
        assert cfg.terminations[name].time_out == old.time_out
        assert cfg.terminations[name].func is expected_terminations.get(name, old.func)
    assert cfg.observations["tokenizer"] == before.observations["tokenizer"]
    assert cfg.observations["policy"] == before.observations["policy"]


def test_unknown_reference_objective_fails_closed():
    def motion_unknown_target():
        pass

    cfg = SimpleNamespace(rewards={"new_target": SimpleNamespace(func=motion_unknown_target)}, terminations={})
    with pytest.raises(ValueError, match="explicit q10"):
        task.configure_q10_objectives(cfg)


def test_objective_contract_distinguishes_held_reference_from_future_reward():
    contract = task.root_objective_contract()
    assert not contract["future_q11_reference_read"]
    assert "held_received_q10" in contract["reward_and_termination_phase"]
    assert "2ms_stale" in contract["derived_measured_reward_state_phase"]
    assert contract["tokenizer_and_original_command_properties"] == "unchanged_q9"
