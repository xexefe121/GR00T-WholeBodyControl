"""The added objective must see physical drift without changing existing terms."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from gear_sonic.envs.mjlab import sonic_true23_root_feedback as task


def fixture():
    reference = torch.zeros(12, 3, 3)
    reference[1, 0] = torch.tensor([0.1, 0.2, 0.3])
    reference[1, 2] = torch.tensor([-0.1, -0.2, 0.3])
    reference[2:] = float("nan")  # Never consume unreceived future target.
    command = SimpleNamespace(
        time_steps=torch.tensor([0, 0]),
        motion=SimpleNamespace(body_pos_w=reference, time_step_total=12),
        cfg=SimpleNamespace(body_names=("right_ankle_roll_link", "pelvis", "left_ankle_roll_link")),
        _env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.tensor([[0.0, 0, 0], [4.0, 5.0, 0]]))),
    )
    command.robot_body_pos_w = task._q10_body_position(command).clone()
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    return command, env


def test_feet_term_uses_named_ankles_and_held_proof_with_environment_origins():
    command, env = fixture()
    torch.testing.assert_close(task.q10_measured_feet_world_position_l2(env), torch.zeros(2))
    command.robot_body_pos_w[:, 1] = float("nan")  # Unselected pelvis is not an ankle.
    command.robot_body_pos_w[:, 0, 0] += 0.05
    torch.testing.assert_close(task.q10_measured_feet_world_position_l2(env), torch.full((2,), 0.5))


def test_world_drift_not_hidden_by_measured_root_or_requested_actions():
    command, env = fixture()
    command.robot_body_pos_w[:, :, 1] += 0.1
    command.robot_anchor_pos_w = torch.full((2, 3), 1000.0)
    env.action_manager = None
    torch.testing.assert_close(task.q10_measured_feet_world_position_l2(env), torch.full((2,), 4.0))
    command.robot_body_pos_w[:, :, 1] += 0.1
    torch.testing.assert_close(task.q10_measured_feet_world_position_l2(env), torch.full((2,), 16.0))


def test_ankle_error_is_xyz_norm_before_two_foot_average():
    command, env = fixture()
    command.robot_body_pos_w[:, 0] += torch.tensor([0.03, 0.04, 0.0])
    command.robot_body_pos_w[:, 2, 2] += 0.05
    torch.testing.assert_close(task.q10_measured_feet_world_position_l2(env), torch.ones(2))


@pytest.mark.parametrize("target", [True, False])
def test_nonfinite_selected_position_fails(target):
    command, env = fixture()
    if target:
        command.motion.body_pos_w[1, 0, 0] = float("inf")
    else:
        command.robot_body_pos_w[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        task.q10_measured_feet_world_position_l2(env)


def test_missing_ankle_and_missing_proof_fail():
    command, env = fixture()
    command.cfg.body_names = ("right_ankle_roll_link", "pelvis", "hand")
    with pytest.raises(ValueError, match="known body names"):
        task.q10_measured_feet_world_position_l2(env)
    command, env = fixture()
    command.time_steps[:] = 11
    with pytest.raises(ValueError, match="available"):
        task.q10_measured_feet_world_position_l2(env)


def test_actual_configuration_adds_only_foot_term_no_physics_or_interface_changes():
    from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import make_causal_multimotion_v14_env_cfg

    old = make_causal_multimotion_v14_env_cfg(motion_file="unused_static_configuration.npz", num_envs=2)
    new = deepcopy(old)
    spans = [{"timeline": {"total_requested_controls": 1850}}]
    task.configure_root_feedback_environment(old, spans, objective_profile="root_and_upper_posture_v2")
    task.configure_root_feedback_environment(new, spans, objective_profile="root_and_upper_feet_world_v4")
    assert set(new.rewards) - set(old.rewards) == {"measured_feet_world_position_l2"}
    for key in old.rewards:
        assert old.rewards[key] == new.rewards[key]
    assert new.rewards["measured_feet_world_position_l2"].weight == -1.0
    for key in ("actions", "terminations", "observations", "events", "commands", "sim", "scene"):
        assert getattr(new, key) == getattr(old, key)
