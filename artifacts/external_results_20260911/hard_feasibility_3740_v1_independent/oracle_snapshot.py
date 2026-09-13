"""Independent manual-PD feasibility oracle for native23 simulation experiments.

This module deliberately imports no planner, optimizer, or production feasibility
helper. A successful private preview is evidence about that simulated segment,
not a recovery policy or permission to operate hardware.
"""

from __future__ import annotations

import copy
from typing import Mapping

import mujoco
import numpy as np


def _integration_state(model, data):
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, specification))
    mujoco.mj_getState(model, data, state, specification)
    return state


def inspect_native_segment(model, data, targets, contract: Mapping,
                           *, stop_on_failure=True, retain_trace=True):
    """Privately apply explicit targets with 500Hz native PD, 50Hz target hold.

    All source MjData integration state and warnings must remain unchanged.
    Caller-supplied targets are never clipped or repaired. Each 2ms post-step
    position, velocity, actuator force, warning counter, and clock is checked.
    The initial state is checked too. A failure remains a failure if continued
    for a separately labelled diagnostic via stop_on_failure=False.
    """
    if (model.nq, model.nv, model.nu, model.njnt) != (30, 29, 23, 24):
        raise ValueError("oracle requires the native23 free-base model")
    if abs(float(model.opt.timestep) - .002) > 1e-15:
        raise ValueError("oracle requires the native 2ms physics clock")
    np.testing.assert_array_equal(model.actuator_trnid[:, 0], np.arange(1, 24))
    np.testing.assert_array_equal(model.jnt_qposadr[1:], np.arange(7, 30))
    np.testing.assert_array_equal(model.jnt_dofadr[1:], np.arange(6, 29))
    if np.any(model.actuator_dyntype) or np.any(model.actuator_gaintype) or np.any(model.actuator_biastype):
        raise ValueError("oracle requires stateless fixed-gain torque actuators")
    np.testing.assert_array_equal(model.actuator_gainprm[:, 0], np.ones(23))
    np.testing.assert_array_equal(model.actuator_gear[:, 0], np.ones(23))
    np.testing.assert_array_equal(model.actuator_gear[:, 1:], np.zeros((23, 5)))
    gains = [np.asarray(contract[key], dtype=float) for key in
             ("kp", "kd", "native_effort", "native_velocity")]
    if any(a.shape != (23,) or not np.isfinite(a).all() or np.any(a <= 0) for a in gains):
        raise ValueError("invalid native PD, effort, or speed contract")
    kp, kd, effort, velocity = gains
    limits = np.asarray(model.jnt_range[1:]).copy()
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 23 or not len(targets) or not np.isfinite(targets).all():
        raise ValueError("targets must be a nonempty finite N-by-23 sequence")
    if np.any(targets < limits[:, 0]) or np.any(targets > limits[:, 1]):
        raise ValueError("targets must already respect native target bounds")
    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("oracle does not permit external or root assistance forces")

    original = _integration_state(model, data)
    original_warnings = data.warning.number.copy(), data.warning.lastinfo.copy()
    private = copy.deepcopy(data)
    np.testing.assert_array_equal(_integration_state(model, private), original)
    if np.shares_memory(private.qpos, data.qpos) or np.shares_memory(private.qvel, data.qvel):
        raise AssertionError("private preview aliases the physical state")
    start_time = float(data.time)
    positions, velocities, times = [private.qpos.copy()], [private.qvel.copy()], [start_time]
    torques, actuator_forces = [], []
    warnings = [private.warning.number.copy()]
    warnings_lastinfo = [private.warning.lastinfo.copy()]
    first_failure = None
    maximum = dict(joint_excess_rad=0., speed_ratio=0., effort_ratio=0.,
                   tilt_rad=0., clock_error_seconds=0.)
    minimum_height = float(private.qpos[2])

    def assess(step):
        nonlocal minimum_height
        q, dq = private.qpos.copy(), private.qvel.copy()
        finite = bool(np.isfinite(q).all() and np.isfinite(dq).all())
        if finite:
            excess_by_joint = np.maximum(np.maximum(limits[:, 0] - q[7:], q[7:] - limits[:, 1]), 0.)
            excess = float(excess_by_joint.max())
            speed = float(np.max(np.abs(dq[6:]) / velocity))
            tilt = float(np.arccos(np.clip(1 - 2 * (q[4] ** 2 + q[5] ** 2), -1, 1)))
        else:
            excess, speed, tilt = float("inf"), float("inf"), float("inf")
            excess_by_joint = np.full(23, np.inf)
        force = private.qfrc_actuator[6:]
        force_finite = bool(force.shape == (23,) and np.isfinite(force).all())
        effort_ratio = float(np.max(np.abs(force) / effort)) if force_finite and step else 0.
        clock_error = abs(float(private.time) - (start_time + step * .002))
        reasons = []
        if not finite: reasons.append("nonfinite_state")
        if finite and abs(float(np.linalg.norm(q[3:7])) - 1.) > 1e-10:
            reasons.append("invalid_root_quaternion")
        if excess > 1e-6: reasons.append("native_joint_bound")
        if speed > 1.: reasons.append("native_joint_speed")
        if step and (not force_finite or effort_ratio > 1 + 1e-9):
            reasons.append("native_actuator_effort")
        if q[2] < .25 or tilt > 1.2: reasons.append("fall")
        if np.any(private.warning.number): reasons.append("engine_warning")
        if not np.isfinite(private.time) or clock_error > 1e-10:
            reasons.append("physics_clock")
        current = dict(joint_excess_rad=excess, speed_ratio=speed, effort_ratio=effort_ratio,
                       tilt_rad=tilt, clock_error_seconds=clock_error)
        for key, value in current.items(): maximum[key] = max(maximum[key], value)
        minimum_height = min(minimum_height, float(q[2]))
        return None if not reasons else dict(
            reasons=reasons, physics_step=step,
            control_index=(step - 1) // 10 if step else None,
            substep=(step - 1) % 10 + 1 if step else 0,
            worst_joint_index=int(np.argmax(excess_by_joint)),
            **current,
        )

    try:
        first_failure = assess(0)
        if first_failure is None or not stop_on_failure:
            for target in targets:
                for _ in range(10):
                    command = np.minimum(np.maximum(kp * (target - private.qpos[7:]) - kd * private.qvel[6:], -effort), effort)
                    private.ctrl[:] = command
                    mujoco.mj_step(model, private)
                    positions.append(private.qpos.copy())
                    velocities.append(private.qvel.copy())
                    times.append(float(private.time))
                    torques.append(command.copy())
                    actuator_forces.append(private.qfrc_actuator[6:].copy())
                    warnings.append(private.warning.number.copy())
                    warnings_lastinfo.append(private.warning.lastinfo.copy())
                    failure = assess(len(torques))
                    if first_failure is None and failure is not None: first_failure = failure
                    if failure is not None and stop_on_failure: break
                if first_failure is not None and stop_on_failure: break
    finally:
        np.testing.assert_array_equal(_integration_state(model, data), original)
        np.testing.assert_array_equal(data.warning.number, original_warnings[0])
        np.testing.assert_array_equal(data.warning.lastinfo, original_warnings[1])

    report = dict(
        kind="independent_native23_manual_PD_private_segment", feasible=first_failure is None,
        first_failure=first_failure, requested_controls=len(targets),
        physics_steps=len(torques), complete_controls=len(torques) // 10,
        partial_substeps=len(torques) % 10, original_data_unchanged=True,
        root_forces=False, source_state_copy_after_initialization=False,
        initial_time=start_time, final_time=float(private.time),
        maximum=maximum, minimum_root_height_m=minimum_height,
        final_warning_counts=private.warning.number.tolist(),
        diagnostic_continued_after_failure=first_failure is not None and not stop_on_failure,
        live_or_hardware_qualified=False,
    )
    trace = None
    if retain_trace:
        trace = dict(physics_qpos=np.asarray(positions), physics_qvel=np.asarray(velocities),
                     physics_torque=np.asarray(torques).reshape(-1, 23),
                     physics_actuator_force=np.asarray(actuator_forces).reshape(-1, 23),
                     physics_time=np.asarray(times), warning_counts=np.asarray(warnings),
                     warning_lastinfo=np.asarray(warnings_lastinfo))
    return report, trace
