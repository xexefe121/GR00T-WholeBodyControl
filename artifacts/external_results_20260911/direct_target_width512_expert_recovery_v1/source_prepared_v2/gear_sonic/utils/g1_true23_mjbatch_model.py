"""Native23 actuator algebra for batched offline planning, without model substitution."""

import copy

import mujoco
import numpy as np


class Native23Feasibility:
    """Strict physical predicates, separate from the optimizer's soft costs."""

    def __init__(self, model, contract):
        if (model.nq, model.nv, model.nu) != (30, 29, 23) or model.opt.timestep != 0.002:
            raise ValueError("hard feasibility requires native23 at 500Hz")
        self.lower, self.upper = model.jnt_range[1:].T.copy()
        self.velocity = np.asarray(contract["native_velocity"], float)
        self.effort = np.asarray(contract["native_effort"], float)
        if any(v.shape != (23,) or not np.isfinite(v).all() or np.any(v <= 0)
               for v in (self.velocity, self.effort)):
            raise ValueError("invalid native feasibility contract")

    def contract(self):
        return dict(
            joint_excess_tolerance_rad=1e-6, maximum_speed_ratio=1.0,
            maximum_effort_ratio=1 + 1e-9, minimum_root_height_m=0.25,
            maximum_root_tilt_rad=1.2, maximum_warning_count=0,
            clock_tolerance_seconds=1e-10, sample_seconds=0.002,
            root_quaternion_norm_tolerance=1e-10,
        )

    def assess(self, qpos, qvel, *, force=None, warning=None, time=None, expected_time=None):
        """Vectorized post-step checks; arrays have one leading candidate axis."""
        qpos, qvel = np.atleast_2d(qpos), np.atleast_2d(qvel)
        if qpos.shape != (len(qpos), 30) or qvel.shape != (len(qpos), 29):
            raise ValueError("invalid native feasibility state shape")
        for name, value, shape in (("force", force, (len(qpos), 23)),
                                   ("warning", warning, (len(qpos), 8)),
                                   ("time", time, (len(qpos),))):
            if value is not None and np.asarray(value).shape != shape:
                raise ValueError("invalid native feasibility " + name + " shape")
        if time is not None and (
            expected_time is None or not np.isfinite(expected_time).all()
            or np.asarray(expected_time).shape not in ((), (len(qpos),))
        ):
            raise ValueError("expected clock must be finite scalar or one value per candidate")
        with np.errstate(invalid="ignore", over="ignore"):
            excess = np.max(np.maximum(0, np.maximum(self.lower - qpos[:, 7:], qpos[:, 7:] - self.upper)), axis=1)
            speed = np.max(np.abs(qvel[:, 6:]) / self.velocity, axis=1)
            tilt = np.arccos(np.clip(1 - 2 * np.sum(qpos[:, 4:6] ** 2, axis=1), -1, 1))
            effort = np.zeros(len(qpos)) if force is None else np.max(np.abs(force) / self.effort, axis=1)
        reasons = dict(
            nonfinite_state=~(np.isfinite(qpos).all(axis=1) & np.isfinite(qvel).all(axis=1)),
            invalid_root_quaternion=np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1) > 1e-10,
            joint_range=excess > 1e-6,
            joint_speed=speed > 1.0,
            actuator_effort=(effort > 1 + 1e-9) | ~np.isfinite(effort),
            fall=(qpos[:, 2] < 0.25) | (tilt > 1.2),
        )
        if warning is not None:
            reasons["engine_warning"] = np.any(np.asarray(warning) != 0, axis=1)
        if time is not None:
            clock = np.asarray(time)
            reasons["clock_reset_or_nonfinite"] = ~np.isfinite(clock) | (np.abs(clock - expected_time) > 1e-10)
        invalid = np.logical_or.reduce(list(reasons.values()))
        metrics = dict(range_excess_rad=excess, speed_ratio=speed, effort_ratio=effort,
                       root_height_m=qpos[:, 2], root_tilt_rad=tilt)
        return invalid, reasons, metrics

    @staticmethod
    def witness(index, reasons, metrics, **location):
        return dict(
            **location, reasons=[name for name, mask in reasons.items() if bool(mask[index])],
            **{name: float(values[index]) if np.isfinite(values[index]) else None
               for name, values in metrics.items()},
        )


def preview_native_control(native, data, target, contract, feasibility):
    """Check the imminent real target using a complete private copy of MjData.

    Rejection does not supply recovery or advance the physical simulation clock.
    No target clipping, state correction, warm-start reset, or root assistance.
    """
    target = np.asarray(target, float)
    if (np.any(native.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED)
            or np.any(native.actuator_gainprm[:, 0] != 1)
            or np.any(native.actuator_biastype != mujoco.mjtBias.mjBIAS_NONE)
            or native.na != 0):
        raise ValueError("imminent preview requires native unit-gain torque actuators")
    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("imminent preview does not permit external generalized or body forces")
    if (target.shape != (23,) or not np.isfinite(target).all()
            or np.any(target < feasibility.lower) or np.any(target > feasibility.upper)):
        raise ValueError("imminent target must already be finite and within native bounds")
    private = copy.deepcopy(data)
    kp, kd, caps = [np.asarray(contract[k]) for k in ("kp", "kd", "native_effort")]
    started = float(data.time)
    states = [np.r_[private.qpos, private.qvel]]
    forces = []
    first = None
    for substep in range(11):
        if substep:
            private.ctrl[:] = np.clip(kp * (target - private.qpos[7:]) - kd * private.qvel[6:], -caps, caps)
            mujoco.mj_step(native, private)
            states.append(np.r_[private.qpos, private.qvel])
            forces.append(private.qfrc_actuator[6:].copy())
        invalid, reasons, metrics = feasibility.assess(
            private.qpos, private.qvel, force=private.qfrc_actuator[None, 6:] if substep else None,
            warning=private.warning.number[None], time=np.array([private.time]),
            expected_time=started + substep * native.opt.timestep,
        )
        if invalid[0]:
            first = feasibility.witness(0, reasons, metrics, substep=substep)
            break
    return dict(feasible=first is None, first_violation=first, checked_physics_steps=len(forces),
                physical_clock_advanced=False), np.asarray(states), np.asarray(forces).reshape(-1, 23)


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
