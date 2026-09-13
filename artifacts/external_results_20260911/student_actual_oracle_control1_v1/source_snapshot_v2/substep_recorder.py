import mujoco
import numpy as np

def record_native_substep(native, data, target, contract, feasibility, trace,
                          expected_time, predicted_state, predicted_force):
    """Append raw actual evidence before reporting any physics/parity failure."""
    kp, kd, effort = [np.asarray(contract[name]) for name in ("kp", "kd", "native_effort")]
    data.ctrl[:] = np.clip(kp * (target - data.qpos[7:]) - kd * data.qvel[6:], -effort, effort)
    mujoco.mj_step(native, data)
    expected_time += native.opt.timestep
    for name, value in (("physics_qpos", data.qpos), ("physics_qvel", data.qvel),
                        ("physics_torque", data.ctrl), ("physics_actuator_force", data.qfrc_actuator[6:]),
                        ("physics_warning_number", data.warning.number),
                        ("physics_warning_lastinfo", data.warning.lastinfo)):
        trace[name].append(value.copy())
    trace["physics_time"].append(float(data.time))
    trace["physics_expected_time"].append(expected_time)
    invalid, reasons, metrics = feasibility.assess(
        data.qpos, data.qvel, force=data.qfrc_actuator[None, 6:], warning=data.warning.number[None],
        time=np.array([data.time]), expected_time=expected_time,
    )
    if invalid[0]:
        return expected_time, dict(kind="actual_physics_infeasible",
                                   witness=feasibility.witness(0, reasons, metrics))
    actual = np.r_[data.qpos, data.qvel]
    if not np.array_equal(actual, predicted_state) or not np.array_equal(data.qfrc_actuator[6:], predicted_force):
        return expected_time, dict(kind="private_forecast_mismatch",
                                   state_max_difference=float(np.max(np.abs(actual - predicted_state))),
                                   force_max_difference=float(np.max(
                                       np.abs(data.qfrc_actuator[6:] - predicted_force))))
    return expected_time, None
