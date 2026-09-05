"""Small CPU fixtures; no policy checkpoint, external assets or robot runtime."""

import copy

import mujoco
import numpy as np
import pytest

from gear_sonic.scripts.audit_g1_true23_training_replay_parity import (
    ROOT_DOF_FIELDS,
    clip_sample_plan,
    compare_model_parameters,
    differences,
    model_counterfactuals,
    validate_native_layout,
)
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_step1b_mujoco import _body_velocity, world_angular_velocity_to_body


def small_robot(prefix=""):
    limbs = "".join(
        f'<body name="{prefix}{name}_body" pos="0 0 {0.05 * (i + 1)}">'
        f'<joint name="{prefix}{name}" axis="0 1 0"/>'
        '<geom type="sphere" size=".01" mass=".1" contype="0" conaffinity="0"/></body>'
        for i, name in enumerate(HARDWARE_23_JOINT_NAMES)
    )
    motors = "".join(f'<motor joint="{prefix}{name}"/>' for name in HARDWARE_23_JOINT_NAMES)
    return mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep=".002"/>'
        f'<worldbody><body name="{prefix}pelvis" pos="0 0 2"><freejoint/>'
        '<geom type="sphere" size=".1" mass="1"/>'
        f"{limbs}</body></worldbody><actuator>{motors}</actuator></mujoco>"
    )


def test_complete_sampling_includes_all_clips_and_history_complete_endpoints():
    rows = clip_sample_plan(
        [{"name": "short", "start": 0, "length": 16}, {"name": "long", "start": 16, "length": 100}], 116
    )
    assert len(rows) == 8
    assert rows[0]["phase"] == 12 and rows[3]["phase"] == 13
    assert rows[4]["phase"] == 28 and rows[7]["phase"] == 113
    assert {row["clip"] for row in rows} == {"short", "long"}


@pytest.mark.parametrize("start,length,total", [(1, 16, 16), (False, 16, 16), (0, 15, 15), (0, 16, 17)])
def test_sampling_rejects_incomplete_or_ambiguous_spans(start, length, total):
    with pytest.raises(ValueError, match="parity"):
        clip_sample_plan([{"name": "clip", "start": start, "length": length}], total)


@pytest.mark.parametrize("tolerance", [-1, np.nan, np.inf, True, [1e-5]])
def test_tolerance_is_not_a_bypass(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        differences([0], [0], tolerance)


@pytest.mark.parametrize("right", [[np.nan], [np.inf], [0, 0]])
def test_comparison_rejects_nonfinite_and_broadcasting(right):
    with pytest.raises(ValueError, match="equal-shape finite"):
        differences([0], right, 1e-5)


def test_comparison_uses_absolute_not_relative_tolerance():
    assert not differences([1e6], [1e6 + 0.1], 1e-3)["within_tolerance"]
    assert differences([0], [1e-5], 1e-5)["within_tolerance"]


def test_counterfactuals_isolate_root_and_options_without_changing_models():
    training, replay = small_robot("robot/"), small_robot()
    replay.dof_armature[:6], replay.dof_damping[:6], replay.dof_frictionloss[:6] = 0.01, 0.001, 0.1
    training.opt.iterations, training.opt.ls_iterations = 10, 20
    before = [compiled_model_sha256(model) for model in (training, replay)]
    models = model_counterfactuals(training, replay)
    for name, model in models.items():
        for field in ROOT_DOF_FIELDS:
            expected = replay if name == "options_only" else training
            np.testing.assert_array_equal(getattr(model, field)[:6], getattr(expected, field)[:6])
            np.testing.assert_array_equal(getattr(model, field)[6:], getattr(replay, field)[6:])
        assert model.opt.iterations == (100 if name == "root_only" else 10)
    assert before == [compiled_model_sha256(model) for model in (training, replay)]
    parameters = compare_model_parameters(training, replay)
    assert not parameters["free_root_dof_frictionloss"]["within_tolerance"]
    assert parameters["body_mass"]["within_tolerance"]
    assert parameters["actuator_gear"]["within_tolerance"]


def test_root_alignment_removes_free_flight_difference_in_synthetic_model():
    training, replay = small_robot(), small_robot()
    replay.dof_armature[:6], replay.dof_damping[:6], replay.dof_frictionloss[:6] = 0.01, 0.001, 0.1
    candidate = model_counterfactuals(training, replay)["root_only"]
    velocities = []
    for model in (training, replay, candidate):
        data = mujoco.MjData(model)
        data.qvel[:] = np.linspace(-1, 1, model.nv)
        data.ctrl[:] = np.linspace(-0.1, 0.1, model.nu)
        mujoco.mj_step(model, data)
        velocities.append(data.qvel.copy())
    assert np.max(np.abs(velocities[0] - velocities[1])) > 1e-4
    np.testing.assert_allclose(velocities[0], velocities[2], atol=1e-10, rtol=0)


def test_counterfactual_rejects_fixed_base():
    fixed = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="sphere" size="1"/></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="free joint"):
        model_counterfactuals(fixed, small_robot())


def test_native_layout_binds_actuator_order_not_just_dimensions():
    model = small_robot("robot/")
    validate_native_layout(model, "robot/")
    model.actuator_trnid[[0, 1]] = model.actuator_trnid[[1, 0]]
    with pytest.raises(ValueError, match="state layout"):
        validate_native_layout(model, "robot/")


@pytest.mark.parametrize("rotated", [False, True])
def test_policy_gyro_uses_post_step_state_without_integrating_or_resolving_contacts(rotated):
    controller = CleanTrue23MujocoController.__new__(CleanTrue23MujocoController)
    controller.module, controller.model = mujoco, small_robot()
    controller.data = mujoco.MjData(controller.model)
    controller.previous_safe_native = np.zeros(23, dtype=np.float32)
    if rotated:
        controller.data.qpos[3:7] = np.array([0.8, 0.2, -0.1, 0.3]) / np.linalg.norm([0.8, 0.2, -0.1, 0.3])
    controller.data.qvel[:] = np.linspace(-0.2, 0.6, 29)
    controller.data.xfrc_applied[1, 3:] = [0.1, -0.2, 0.3]
    mujoco.mj_step(controller.model, controller.data)
    data = controller.data
    _, old_world = _body_velocity(mujoco, controller.model, data, "pelvis", (0.0, 0.0, 0.0))
    old_local = world_angular_velocity_to_body(data.qpos[3:7], old_world)
    assert np.max(np.abs(old_local - data.qvel[3:6])) > 1e-4
    before = {
        field: np.array(getattr(data, field), copy=True)
        for field in ("qpos", "qvel", "time", "ctrl", "qacc", "qacc_warmstart", "qfrc_constraint", "xfrc_applied")
    }
    oracle = copy.copy(data)
    mujoco.mj_forward(controller.model, oracle)
    frame = controller._policy_frame()
    np.testing.assert_allclose(frame[:3], data.qvel[3:6], atol=1e-7, rtol=0)
    np.testing.assert_allclose(data.xpos, oracle.xpos, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data.cvel, oracle.cvel, atol=1e-12, rtol=0)
    for field, value in before.items():
        np.testing.assert_array_equal(getattr(data, field), value)
