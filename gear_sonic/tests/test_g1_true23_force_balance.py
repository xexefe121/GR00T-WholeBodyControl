"""Force partition arithmetic and passive observation; no robot transport."""

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.g1_true23_force_balance import COMPONENTS, SolvedForceObserver, solve_components

XML = """
<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
<worldbody><geom name="floor" type="plane" size="3 3 .1"/>
<body pos="0 0 .27"><freejoint/><geom type="sphere" size=".05" mass="1"/>
<body name="left_ankle_roll_link" pos="0 .12 -.25">
<joint name="left" axis="1 0 0" range="-8 8"/>
<geom type="box" size=".08 .04 .03" mass=".2"/></body>
<body name="right_ankle_roll_link" pos="0 -.12 -.25">
<joint name="right" axis="1 0 0" range="-8 8"/>
<geom type="box" size=".08 .04 .03" mass=".2"/></body>
</body></worldbody><actuator><motor joint="left"/><motor joint="right"/></actuator></mujoco>
"""


def test_coupled_mass_is_not_diagonal_torque_division():
    mass = np.array([[2.0, 1.0], [1.0, 2.0]])
    force = np.array([[1.0, 0.0], [0.0, 2.0]])
    acc = np.linalg.solve(mass, force.sum(0))
    result = solve_components(mass, force, acc)
    np.testing.assert_allclose(result["acceleration_components"], [[2 / 3, -1 / 3], [-2 / 3, 4 / 3]])
    assert result["force_closure_error"] < 1e-12
    assert result["acceleration_closure_error"] < 1e-12


@pytest.mark.parametrize("mass", [np.ones((2, 2)), np.array([[1.0, 1.0], [0.0, 1.0]]), np.full((2, 2), np.nan)])
def test_invalid_inertia_rejected(mass):
    with pytest.raises(ValueError):
        solve_components(mass, np.ones((3, 2)), np.ones(2))


def test_passive_solved_contact_partition():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[7] = 0.2
    data.ctrl[:] = [0.7, -0.4]
    observer = SolvedForceObserver(model)
    for _ in range(3):
        mujoco.mj_step(model, data)
        state = observer.integration_state(data)
        row = observer.capture(data)
        np.testing.assert_array_equal(observer.integration_state(data), state)
        assert row["constraint_partition_error"] < 1e-9
        assert row["smooth_partition_error"] < 1e-9
        assert row["force_closure_error"] < 1e-9
        assert row["acceleration_closure_error"] < 1e-8
        assert row["foot_normal_load_n"].sum() > 0
        assert len(row["contacts"]) > 0


def test_active_joint_limit_partition_without_contact_unloading():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[2], data.qpos[7], data.qvel[6] = 1.0, 0.3, 2.0
    mujoco.mj_step(model, data)
    row = SolvedForceObserver(model).capture(data)
    assert row["constraint_partition_error"] < 1e-9
    assert np.linalg.norm(row["force_components"][COMPONENTS.index("joint_limit")]) > 0
    np.testing.assert_array_equal(row["foot_normal_load_n"], [0.0, 0.0])


def test_spatial_force_not_silently_omitted():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.xfrc_applied[1, 0] = 1
    mujoco.mj_step(model, data)
    with pytest.raises(ValueError, match="spatial"):
        SolvedForceObserver(model).capture(data)
