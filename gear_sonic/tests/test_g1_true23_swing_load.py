from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from gear_sonic.envs.mjlab import sonic_true23_root_feedback as task
from gear_sonic.tests.test_g1_true23_root_feedback_environment import objective_fixture
from gear_sonic.utils.g1_true23_root_feedback_objectives import (
    objective_profile_contract,
    objective_transition_contract,
)
from gear_sonic.utils.g1_true23_swing_load import (
    FOOT_SENSOR_NAMES,
    SWING_LOAD_PROFILE,
    configure_swing_load,
    received_swing_load_l2,
    swing_load_l2,
)


def values():
    position = torch.zeros(3, 2, 3, dtype=torch.float64)
    position[..., 2] = 0.035  # Physical sphere bottoms on the fixed floor.
    quaternion = torch.zeros(3, 2, 4, dtype=torch.float64)
    quaternion[..., 0] = 1
    forces = torch.zeros(3, 2, 10, 3, dtype=torch.float64)
    forces[..., 2] = 100
    return position, quaternion, forces, torch.zeros(3, dtype=torch.float64)


def test_stance_loading_is_free_and_reference_airborne_mask_is_graded():
    p, q, force, floor = values()
    p[1, 0, 2] += 0.02
    p[2, 0, 2] += 0.04
    torch.testing.assert_close(swing_load_l2(p, q, force, floor), torch.tensor([0, 0.25, 0.5]).double())
    force[:, 1] *= 50  # Stance foot can support weight without this cost.
    torch.testing.assert_close(swing_load_l2(p, q, force, floor), torch.tensor([0, 0.25, 0.5]).double())


def test_unloaded_swing_has_zero_cost_and_impacts_are_not_averaged_away():
    p, q, force, floor = values()
    p[..., 2] += 0.04
    force.zero_()
    torch.testing.assert_close(swing_load_l2(p, q, force, floor), floor)
    force[0, 0, 4, 2] = 1000
    force[1, 0, :, 2] = 100
    force[2, 0, :, 2] = -100
    torch.testing.assert_close(swing_load_l2(p, q, force, floor), torch.tensor([5, 0.5, 0.5]).double())


def test_world_floor_translation_and_force_sign_do_not_change_cost():
    p, q, force, floor = values()
    p[..., 2] += 0.03
    expected = swing_load_l2(p, q, force, floor)
    p += 4
    torch.testing.assert_close(swing_load_l2(p, q, -force, floor + 4), expected)


def test_tilted_foot_uses_lowest_sole_not_ankle_height():
    p, q, force, floor = values()
    p[..., 2] += 0.02
    before = swing_load_l2(p, q, force, floor)
    q[..., 0] = torch.cos(torch.tensor(0.2))
    q[..., 2] = torch.sin(torch.tensor(0.2))
    assert torch.all(swing_load_l2(p, q, force, floor) < before)


@pytest.mark.parametrize("bad", ["shape", "history", "nan", "dtype", "floor", "floor_nan"])
def test_incomplete_or_invalid_sensor_data_is_not_zero_load(bad):
    p, q, force, floor = values()
    if bad == "shape":
        force = force[:, :1]
    elif bad == "history":
        force = force[:, :, :9]
    elif bad == "nan":
        force[0, 0, 0, 2] = float("nan")
    elif bad == "dtype":
        force = force.float()
    elif bad == "floor":
        floor = floor[:, None]
    else:
        floor[0] = float("nan")
    with pytest.raises(ValueError):
        swing_load_l2(p, q, force, floor)


def sensor_fixture():
    command, env = objective_fixture()
    env.cfg = SimpleNamespace(decimation=10, sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=0.002)))
    # Deliberately reverse body order; named sensors must retain left/right order.
    command.cfg.body_names = ("right_ankle_roll_link", "left_ankle_roll_link")
    command.motion.body_quat_w[:] = 0
    command.motion.body_quat_w[..., 0] = 1
    command.motion.body_pos_w[:, :, 2] = 0.035
    command.motion.body_pos_w[10, 1, 2] += 0.04
    for name in FOOT_SENSOR_NAMES:
        history = torch.zeros(2, 1, 10, 3)
        history[..., 2] = 100
        env.scene[name] = SimpleNamespace(data=SimpleNamespace(force_history=history))
    return command, env


def test_received_reference_named_physical_sensors_not_future_or_action_targets():
    command, env = sensor_fixture()
    torch.testing.assert_close(received_swing_load_l2(env), torch.full((2,), 0.5))
    command.motion.body_pos_w[11] = float("nan")
    command.motion.body_quat_w[11] = float("nan")
    command.robot_body_pos_w[:] = float("nan")
    env.action_manager = None
    env.scene[FOOT_SENSOR_NAMES[1]].data.force_history[..., 2] *= 10
    torch.testing.assert_close(received_swing_load_l2(env), torch.full((2,), 0.5))
    env.scene[FOOT_SENSOR_NAMES[0]].data.force_history.zero_()
    torch.testing.assert_close(received_swing_load_l2(env), torch.zeros(2))


def test_missing_history_fails_closed():
    _, env = sensor_fixture()
    env.scene[FOOT_SENSOR_NAMES[0]].data.force_history = None
    with pytest.raises(ValueError, match="ten-substep"):
        received_swing_load_l2(env)


@pytest.mark.parametrize("decimation, timestep", [(4, 0.005), (10, 0.005), (4, 0.002)])
def test_finalized_runtime_requires_original_physics_clock(decimation, timestep):
    _, env = sensor_fixture()
    env.cfg.decimation = decimation
    env.cfg.sim.mujoco.timestep = timestep
    with pytest.raises(ValueError, match="ten 2ms"):
        received_swing_load_l2(env)


def test_training_cli_requires_explicit_pinned_nominal_physics():
    from gear_sonic.scripts.train_g1_true23_root_feedback import validate_bounds
    from gear_sonic.tests.test_train_g1_true23_root_feedback import arguments

    args = arguments("regression", "--objective-profile", SWING_LOAD_PROFILE)
    with pytest.raises(ValueError, match="pinned nominal physics"):
        validate_bounds(args)
    args.training_physics_profile = "pinned_cpu_referee_scene_v1"
    validate_bounds(args)


def test_new_profile_only_adds_named_sensors_and_one_cost_to_v4():
    from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import make_causal_multimotion_v14_env_cfg

    old = make_causal_multimotion_v14_env_cfg(motion_file="unused_configuration.npz", num_envs=2)
    new = deepcopy(old)
    spans = [{"timeline": {"total_requested_controls": 1850}}]
    task.configure_root_feedback_environment(old, spans, objective_profile="root_and_upper_feet_world_v4")
    task.configure_root_feedback_environment(new, spans, objective_profile=SWING_LOAD_PROFILE)
    assert set(new.rewards) - set(old.rewards) == {"measured_swing_load_l2"}
    assert all(new.rewards[name] == value for name, value in old.rewards.items())
    assert new.rewards["measured_swing_load_l2"].func is received_swing_load_l2
    assert tuple(new.scene.sensors[:-2]) == tuple(old.scene.sensors)
    for name, sensor in zip(FOOT_SENSOR_NAMES, new.scene.sensors[-2:], strict=True):
        assert sensor.name == name and sensor.history_length == 10
        assert sensor.reduce == "netforce" and sensor.num_slots == 1
        assert sensor.primary.mode == "body"
        assert sensor.secondary.mode == "geom" and sensor.secondary.pattern == "floor"
    for name in ("actions", "terminations", "observations", "events", "commands", "sim", "decimation"):
        assert getattr(new, name) == getattr(old, name)
    assert new.scene.entities == old.scene.entities
    with pytest.raises(ValueError, match="already configured"):
        configure_swing_load(new)


def test_profile_requires_explicit_changed_objective_not_deployment_claim():
    old = objective_profile_contract("root_and_upper_feet_world_v4")
    new = objective_profile_contract(SWING_LOAD_PROFILE)
    assert {k: v for k, v in new.items() if k not in ("name", "swing_foot_load")} == {
        k: v for k, v in old.items() if k != "name"
    }
    assert not new["swing_foot_load"]["contact_slip_balance_or_tracking_improvement_proven"]
    args = SimpleNamespace(objective_profile=SWING_LOAD_PROFILE, allow_objective_transition=False)
    with pytest.raises(ValueError, match="explicit"):
        objective_transition_contract({"objective_profile": old["name"]}, args)
    args.allow_objective_transition = True
    assert objective_transition_contract({"objective_profile": old["name"]}, args)["reward_objective_changed"]
