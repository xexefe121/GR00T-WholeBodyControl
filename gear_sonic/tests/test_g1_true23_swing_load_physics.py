"""Independent CPU checks of sensor forces and unmodified native dynamics."""

import json

import mujoco
import numpy as np
import pytest

from gear_sonic.envs.mjlab.sonic_true23_nominal_scene import configure_nominal_scene, verify_nominal_scene
from gear_sonic.tests.test_g1_true23_nominal_scene import MODEL, PHYSICS, configuration
from gear_sonic.utils.g1_true23_swing_load import FOOT_BODY_NAMES, FOOT_SENSOR_NAMES, configure_swing_load


@pytest.fixture(scope="module")
def models():
    from mjlab.scene import Scene

    result = []
    for add_sensor in (False, True):
        cfg = configuration()
        if add_sensor:
            configure_swing_load(cfg)
        cfg = configure_nominal_scene(cfg, MODEL, PHYSICS)
        scene = Scene(cfg.scene, device="cpu")
        model = scene.compile()
        cfg.sim.mujoco.apply(model)
        assert verify_nominal_scene(model, MODEL, PHYSICS)["parameter_parity_passed"]
        result.append(model)
    return result


def test_sensors_resolve_each_native_ankle_once_on_original_fixed_floor(models):
    old, new = models
    assert new.nsensor == old.nsensor + 2
    for sensor_name, body_name in zip(FOOT_SENSOR_NAMES, FOOT_BODY_NAMES, strict=True):
        ids = [i for i in range(new.nsensor) if new.sensor(i).name.startswith(sensor_name + "_")]
        assert len(ids) == 1
        sensor = new.sensor(ids[0])
        assert sensor.dim[0] == 3
        assert body_name in sensor.name


def test_measured_world_z_forces_match_independent_all_contact_sum_and_dynamics(models):
    old, new = models
    a, b = mujoco.MjData(old), mujoco.MjData(new)
    initial = json.loads(PHYSICS.read_text())["initial_state"]
    q0 = np.r_[initial["base_position_m"], initial["base_quaternion_wxyz"], initial["joint_position_hardware_rad"]]
    a.qpos[:] = b.qpos[:] = q0
    sensor_ids = [
        next(i for i in range(new.nsensor) if new.sensor(i).name.startswith(name + "_"))
        for name in FOOT_SENSOR_NAMES
    ]
    feet = [new.body("robot/" + name).id for name in FOOT_BODY_NAMES]
    floor = new.geom("floor").id
    wrench = np.empty(6)
    peak = np.zeros(2)
    for _ in range(100):
        mujoco.mj_step(old, a)
        mujoco.mj_step(new, b)
        np.testing.assert_array_equal(a.qpos, b.qpos)
        np.testing.assert_array_equal(a.qvel, b.qvel)
        np.testing.assert_array_equal(a.qacc_warmstart, b.qacc_warmstart)
        for side, body in enumerate(feet):
            total = np.zeros(3)
            for i, contact in enumerate(b.contact):
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if floor not in (g1, g2):
                    continue
                other = g2 if g1 == floor else g1
                if new.geom_bodyid[other] != body:
                    continue
                mujoco.mj_contactForce(new, b, i, wrench)
                world = contact.frame.reshape(3, 3).T @ wrench[:3]
                total += world if g1 == floor else -world
            sensor = new.sensor(sensor_ids[side])
            measured = b.sensordata[sensor.adr[0] : sensor.adr[0] + 3]
            np.testing.assert_allclose(abs(measured[2]), abs(total[2]), atol=1e-7, rtol=1e-12)
            peak[side] = max(peak[side], abs(measured[2]))
    assert np.all(peak > 20), "A zero-contact scene would not test sensor scaling or matching."
