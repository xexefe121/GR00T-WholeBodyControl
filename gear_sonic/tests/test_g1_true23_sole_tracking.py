from copy import deepcopy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab import sonic_true23_root_feedback as task
from gear_sonic.tests.test_g1_true23_contact_step_transition import native  # noqa: F401
from gear_sonic.tests.test_g1_true23_root_feedback_environment import objective_fixture
from gear_sonic.utils.g1_true23_root_feedback_objectives import (
    objective_profile_contract,
    objective_transition_contract,
)
from gear_sonic.utils.g1_true23_sole_tracking import (
    SOLE_SPHERE_CENTERS_M,
    SOLE_SPHERE_RADIUS_M,
    sole_points_world,
    sole_world_position_l2,
)


def poses():
    position = torch.zeros(3, 2, 3)
    quaternion = torch.zeros(3, 2, 4)
    quaternion[..., 0] = 1
    return position, quaternion


def test_sole_points_match_physical_four_spheres_under_full_fk(native):  # noqa: F811
    model, initial, _ = native
    data = mujoco.MjData(model)
    generator = np.random.default_rng(98)
    feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    geoms = []
    for foot in feet:
        ids = np.flatnonzero(
            (model.geom_bodyid == foot) & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        )
        assert len(ids) == 4
        assert np.all(model.geom_type[ids] == mujoco.mjtGeom.mjGEOM_SPHERE)
        np.testing.assert_array_equal(model.geom_pos[ids], SOLE_SPHERE_CENTERS_M)
        np.testing.assert_array_equal(model.geom_size[ids, 0], np.full(4, SOLE_SPHERE_RADIUS_M))
        geoms.append(ids)
    for _ in range(12):
        data.qpos[:] = initial
        data.qpos[:3] += generator.normal(0, 0.1, 3)
        data.qpos[3:7] = generator.normal(size=4)
        data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
        data.qpos[7:] += generator.normal(0, 0.06, 23)
        mujoco.mj_forward(model, data)
        predicted = sole_points_world(torch.tensor(data.xpos[feet][None]), torch.tensor(data.xquat[feet][None]))
        expected = data.geom_xpos[geoms].copy()
        expected[..., 2] -= SOLE_SPHERE_RADIUS_M
        np.testing.assert_allclose(predicted[0].numpy(), expected, atol=5e-15, rtol=0)


def test_stance_translation_and_swing_lift_are_same_coupled_world_cost():
    desired, quaternion = poses()
    measured = desired.clone()
    torch.testing.assert_close(sole_world_position_l2(desired, quaternion, measured, quaternion), torch.zeros(3))
    measured[0, 0, 0] += 0.02
    measured[1, 0, 2] -= 0.04
    measured[2, :, 1] += 0.04
    torch.testing.assert_close(
        sole_world_position_l2(desired, quaternion, measured, quaternion), torch.tensor([0.5, 2.0, 4.0])
    )
    # Isolate frame invariance from float32 quantization when adding 4 m to
    # centimetre-scale displacements; the executed float32 cost is tested above.
    desired, measured, quaternion = desired.double(), measured.double(), quaternion.double()
    original_cost = sole_world_position_l2(desired, quaternion, measured, quaternion)
    shifted_desired, shifted_measured = desired + 4, measured + 4
    torch.testing.assert_close(
        sole_world_position_l2(shifted_desired, quaternion, shifted_measured, quaternion),
        original_cost,
        atol=1e-10,
        rtol=0,
    )


def test_sole_orientation_error_observable_without_ankle_translation():
    position, desired = poses()
    measured = desired.clone()
    measured[..., 0] = torch.cos(torch.tensor(0.1))
    measured[..., 1] = torch.sin(torch.tensor(0.1))
    result = sole_world_position_l2(position, desired, position, measured)
    assert torch.all(result > 0)
    torch.testing.assert_close(
        sole_world_position_l2(position, -desired, position, -measured), result, atol=0, rtol=0
    )


@pytest.mark.parametrize("bad", ["shape", "nan_position", "nan_quaternion", "not_unit", "dtype"])
def test_bad_physical_state_rejected(bad):
    position, quaternion = poses()
    if bad == "shape":
        position = position[:, :1]
    elif bad == "nan_position":
        position[0, 0, 0] = float("nan")
    elif bad == "nan_quaternion":
        quaternion[0, 0, 0] = float("nan")
    elif bad == "not_unit":
        quaternion *= 2
    else:
        quaternion = quaternion.double()
    with pytest.raises(ValueError):
        sole_points_world(position, quaternion)


def test_reward_uses_actual_received_q10_not_actions_or_unreceived_future():
    command, env = objective_fixture()
    command.cfg.body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
    torch.testing.assert_close(task.q10_measured_sole_world_position_l2(env), torch.zeros(2))
    command.robot_body_pos_w[:, 0, 2] -= 0.04
    command.motion.body_pos_w[11] = float("nan")
    command.motion.body_quat_w[11] = float("nan")
    env.action_manager = None
    torch.testing.assert_close(task.q10_measured_sole_world_position_l2(env), torch.full((2,), 2.0))


def test_v5_only_adds_declared_sole_cost_preserving_all_existing_terms():
    from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import make_causal_multimotion_v14_env_cfg

    old = make_causal_multimotion_v14_env_cfg(motion_file="unused_static_configuration.npz", num_envs=2)
    new = deepcopy(old)
    spans = [{"timeline": {"total_requested_controls": 1850}}]
    task.configure_root_feedback_environment(old, spans, objective_profile="root_and_upper_feet_world_v4")
    task.configure_root_feedback_environment(new, spans, objective_profile="root_and_upper_feet_sole_world_v5")
    assert set(new.rewards) - set(old.rewards) == {"measured_sole_world_position_l2"}
    for name, value in old.rewards.items():
        assert new.rewards[name] == value
    assert new.rewards["measured_sole_world_position_l2"].func is task.q10_measured_sole_world_position_l2
    assert new.rewards["measured_sole_world_position_l2"].weight == -1
    for name in ("actions", "terminations", "observations", "events", "commands"):
        assert getattr(new, name) == getattr(old, name)


def test_new_objective_is_explicit_not_a_same_objective_resume_or_contact_proof():
    old = objective_profile_contract("root_and_upper_feet_world_v4")
    new = objective_profile_contract("root_and_upper_feet_sole_world_v5")
    assert {k: v for k, v in new.items() if k not in ("name", "sole_world_position")} == {
        k: v for k, v in old.items() if k != "name"
    }
    assert not new["sole_world_position"]["contact_force_slip_or_dynamic_feasibility_proven"]
    args = SimpleNamespace(objective_profile=new["name"], allow_objective_transition=False)
    with pytest.raises(ValueError, match="explicit"):
        objective_transition_contract({"objective_profile": old["name"]}, args)
    args.allow_objective_transition = True
    change = objective_transition_contract({"objective_profile": old["name"]}, args)
    assert change["reward_objective_changed"] and not change["same_objective_resume_claimed"]
