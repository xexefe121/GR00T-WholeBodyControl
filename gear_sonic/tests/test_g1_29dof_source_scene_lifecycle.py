"""Source SIM layout/packing checks; synthetic plants are not qualification."""

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.g1_29dof_source_scene_lifecycle import SourceBodyLayout
from gear_sonic.utils.g1_true23_original29_reference import SOURCE_JOINT_NAMES


def plant(with_hands=False, timestep=0.002):
    names = list(SOURCE_JOINT_NAMES)
    if with_hands:
        names = names[:22] + [f"left_hand_{i}_joint" for i in range(7)] + names[22:]
        names += [f"right_hand_{i}_joint" for i in range(7)]
    bodies = "".join(
        f'<body name="body_{i}" pos="0 0 {i * 0.1}"><joint name="{name}"/>'
        '<geom type="sphere" size="0.01" mass="0.1" contype="0" conaffinity="0"/></body>'
        for i, name in enumerate(names)
    )
    motors = "".join(f'<motor name="{name.removesuffix("_joint")}" joint="{name}"/>' for name in names)
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><option timestep="{timestep}"/><worldbody><body><freejoint/>'
        '<geom type="sphere" size="0.1" mass="1" contype="0" conaffinity="0"/>'
        f"{bodies}</body></worldbody><actuator>{motors}</actuator></mujoco>"
    )


@pytest.mark.parametrize("with_hands", [False, True])
@pytest.mark.parametrize("dt,steps", [(0.002, 10), (0.005, 4)])
def test_body_layout_extracts_actual_named_states(with_hands, dt, steps):
    model = plant(with_hands, dt)
    layout = SourceBodyLayout(model)
    data = mujoco.MjData(model)
    data.qpos[7:] = np.arange(model.nu) / 100
    data.qvel[:] = np.arange(model.nv) / 20
    before_q, before_v = data.qpos.copy(), data.qvel.copy()
    canonical = layout.canonical(data)
    assert canonical.qpos.shape == (36,) and canonical.qvel.shape == (35,)
    np.testing.assert_array_equal(canonical.qpos[:7], data.qpos[:7])
    for i, name in enumerate(SOURCE_JOINT_NAMES):
        joint = model.joint(name).id
        assert canonical.qpos[i + 7] == data.qpos[model.jnt_qposadr[joint]]
        assert canonical.qvel[i + 6] == data.qvel[model.jnt_dofadr[joint]]
    canonical.qpos[:] = 999
    canonical.qvel[:] = 999
    np.testing.assert_array_equal(before_q, data.qpos)
    np.testing.assert_array_equal(before_v, data.qvel)
    assert layout.substeps == steps
    assert len(layout.finger_actuators) == (14 if with_hands else 0)
    assert not layout.qpos.flags.writeable
    assert not layout.descriptor()["hardware_authorized"]


def test_mapped_control_does_not_command_fingers():
    model = plant(True)
    layout = SourceBodyLayout(model)
    ctrl = np.zeros(model.nu)
    ctrl[layout.actuators] = np.arange(29)
    np.testing.assert_array_equal(ctrl[layout.finger_actuators], np.zeros(14))
    np.testing.assert_array_equal(ctrl[layout.actuators], np.arange(29))
    assert layout.actuators[22] == 29


@pytest.mark.parametrize("dt", [0.001, 0.003, 0.01])
def test_rejects_unapproved_integration_period(dt):
    with pytest.raises(ValueError, match="2ms or5ms"):
        SourceBodyLayout(plant(timestep=dt))


def test_rejects_changed_transmission():
    model = plant(True)
    model.actuator_gear[0, 0] = 2
    with pytest.raises(ValueError, match="unit-gear"):
        SourceBodyLayout(model)
