"""Native23 actuator algebra for batched offline planning, without model substitution."""

import copy

import mujoco
import numpy as np


def position_servo_copy(native, kp, kd, effort):
    """Represent per-2ms clipped PD through affine actuator gain/bias.

    Euler integration is required. Other integrators can treat actuator damping
    implicitly and differ from the explicit external PD controller. Callers must
    validate torque and trajectory parity on their MuJoCo version.
    """
    if (native.nq, native.nv, native.nu) != (30, 29, 23):
        raise ValueError("position servo adapter requires native23 topology")
    if native.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER or native.opt.timestep != 0.002:
        raise ValueError("position servo parity requires native 500Hz Euler integration")
    kp, kd, effort = [np.asarray(value, dtype=np.float64) for value in (kp, kd, effort)]
    if any(value.shape != (23,) or not np.isfinite(value).all() for value in (kp, kd, effort)):
        raise ValueError("invalid PD actuator parameters")
    if np.any(kp <= 0) or np.any(kd < 0) or np.any(effort <= 0):
        raise ValueError("nonpositive PD/effort parameters")
    np.testing.assert_array_equal(native.actuator_trnid[:, 0], np.arange(1, 24))
    np.testing.assert_array_equal(native.actuator_gear, np.tile([1, 0, 0, 0, 0, 0], (23, 1)))
    servo = copy.deepcopy(native)
    servo.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    servo.actuator_gainprm[:] = 0
    servo.actuator_gainprm[:, 0] = kp
    servo.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    servo.actuator_biasprm[:] = 0
    servo.actuator_biasprm[:, 1] = -kp
    servo.actuator_biasprm[:, 2] = -kd
    servo.actuator_ctrllimited[:] = 1
    servo.actuator_ctrlrange[:] = native.jnt_range[1:]
    servo.actuator_forcelimited[:] = 1
    servo.actuator_forcerange[:] = np.column_stack((-effort, effort))
    # Native joint-level force limits remain intact as well.
    mujoco.mj_setConst(servo, mujoco.MjData(servo))
    return servo


def verify_servo_parity(native, servo, kp, kd, effort, qpos, qvel, targets):
    """Continuous independent manual-PD and affine-PD trajectory comparison."""
    explicit, implicit = mujoco.MjData(native), mujoco.MjData(servo)
    for model, data in ((native, explicit), (servo, implicit)):
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(model, data)
    errors = dict(qpos=0.0, qvel=0.0, actuator_torque=0.0)
    for target in targets:
        target = np.clip(target, native.jnt_range[1:, 0], native.jnt_range[1:, 1])
        for _ in range(10):
            explicit.ctrl[:] = np.clip(kp * (target - explicit.qpos[7:]) - kd * explicit.qvel[6:], -effort, effort)
            implicit.ctrl[:] = target
            mujoco.mj_step(native, explicit)
            mujoco.mj_step(servo, implicit)
            for name, a, b in (
                ("qpos", explicit.qpos, implicit.qpos),
                ("qvel", explicit.qvel, implicit.qvel),
                ("actuator_torque", explicit.qfrc_actuator, implicit.qfrc_actuator),
            ):
                errors[name] = max(errors[name], float(np.max(np.abs(a - b))))
    return {
        "mujoco": mujoco.__version__,
        "physics_steps": len(targets) * 10,
        "max_absolute_error": errors,
        "tolerance": {"qpos": 1e-8, "qvel": 1e-6, "actuator_torque": 1e-4},
        "passed": errors["qpos"] <= 1e-8 and errors["qvel"] <= 1e-6 and errors["actuator_torque"] <= 1e-4,
        "native_physics_and_effort_unchanged": True,
    }
