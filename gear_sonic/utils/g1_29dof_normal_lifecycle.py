"""Source29 NORMAL teleop control, preserving original29 diagnostic actuation.

Derived from the immutable low-latency public29 recorder. New source weights
and47-sample timing are explicit; all29 measured/command channels retained.
Original29 raw-action, finite-state, absolute-height/tilt stops and C++ target/
effort arithmetic stay unchanged. No native23 safety guard is edited or used
as a hardware bypass. This29 scene has different gains/physics; ranges and
actual efforts must be audited separately. Legacy landmark metrics retain
their earlier offsets and are not the matched original-intent task referee.
"""

from collections import deque

import mujoco
import numpy as np

from gear_sonic.scripts import simulate_g1_sonic_library_motions as stock
from gear_sonic.scripts.record_g1_sonic_public29_baseline import FLAGS, points_at
from gear_sonic.teleop.normal_source_horizon import ReceivedNormalSourceHorizon
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_JOINT_NAMES,
    REFERENCE_PROFILE_NORMAL,
    SOURCE_MJ29_KEEP_INDICES,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


def run_case(model, source_model, parameters, teacher, poses, phases, virtual_vr21):
    if teacher.reference_profile != REFERENCE_PROFILE_NORMAL:
        raise ValueError("normal29 lifecycle refuses a different source policy profile")
    if virtual_vr21.shape != (len(poses), 21) or virtual_vr21.dtype != np.float32:
        raise ValueError("normal29 requires the saved finite float32 original21-VR source")
    if not np.isfinite(virtual_vr21).all():
        raise ValueError("normal29 source VR contains nonfinite values")
    if (model.nq, model.nv, model.nu) != (36, 35, 29) or model.opt.timestep != 0.002:
        raise ValueError("requires original29 model at 500 Hz")
    if (source_model.nq, source_model.nv, source_model.nu) != (36, 35, 29):
        raise ValueError("reference geometry must contain all29 joints")
    if poses.ndim != 2 or poses.shape[1] != 36 or len(poses) < 22 or not np.isfinite(poses).all():
        raise ValueError("requires a finite original29 lifecycle")
    if not np.array_equal(poses[-10:], np.repeat(poses[-1:], 10, axis=0)):
        raise ValueError("only an explicit constant standing tail may extend")
    before = compiled_model_sha256(model)
    requested = len(poses) - 11
    extended = np.concatenate((poses, np.repeat(poses[-1:], 46, axis=0)))
    # Identical received-only normal47-sample stream as the native23 control.
    # This carrier's23 joint fields provide only the lower12 encoder reference;
    # the robot still has29 measured joints,29 actions and29 actuators.
    vr = np.concatenate((virtual_vr21, np.repeat(virtual_vr21[-1:], 46, axis=0)))
    source = dict(
        joint_pos=extended[:, 7:][:, SOURCE_MJ29_KEEP_INDICES].astype(np.float32),
        root_position_w=extended[:, :3].astype(np.float32),
        root_quaternion_wxyz=extended[:, 3:7].astype(np.float32),
        virtual_vr21=vr,
    )
    buffer = ReceivedNormalSourceHorizon()

    def emit(index):
        return buffer.push(
            source_timestamp_s=index * 0.02,
            arrival_timestamp_s=index * 0.02,
            joint_names=HARDWARE_23_JOINT_NAMES,
            joint_position23=source["joint_pos"][index],
            root_position_w=source["root_position_w"][index],
            root_quaternion_wxyz=source["root_quaternion_wxyz"][index],
            virtual_source_vr21=vr[index],
        )

    for index in range(55):
        emit(index)
    data, probe, reference = mujoco.MjData(model), mujoco.MjData(model), mujoco.MjData(source_model)
    data.qpos[:] = poses[10]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    previous = np.zeros(29, dtype=np.float32)

    def measured():
        value = stock._current_history_frame(data, previous)
        value["joint_position"] = (data.qpos[7:] - parameters.default_angles)[stock.ISAAC_TO_MUJOCO_INDEX]
        return value

    history = deque((measured() for _ in range(10)), maxlen=10)
    arrays = {
        name: []
        for name in (
            "attempt_encoder267",
            "attempt_history930",
            "attempt_raw29",
            "attempt_token64",
            "attempt_qpos",
            "attempt_qvel",
            "attempt_source_timestamps_s",
            "encoder267",
            "history930",
            "raw29",
            "target29",
            "physics_pre_qpos",
            "physics_post_qpos",
            "physics_pre_qvel",
            "physics_post_qvel",
            "requested_torque29",
            "applied_torque29",
            "engine_force29",
            "contact_count",
            "landmark_error_m",
            "relative_landmark_error_m",
            "root_error_m",
            "joint_error29",
            "source_timestamps_s",
        )
    }
    arrays.update(qpos=[data.qpos.copy()], qvel=[data.qvel.copy()])
    failure = None
    try:
        for control in range(requested):
            sample = emit(55 + control)
            semantic = sample.encoder267(data.qpos[3:7].astype(np.float32))
            history.append(measured())
            proprio = stock._policy_observation(history)
            for key, value in (
                ("attempt_encoder267", semantic),
                ("attempt_history930", proprio),
                ("attempt_qpos", data.qpos),
                ("attempt_qvel", data.qvel),
            ):
                arrays[key].append(value.copy())
            arrays["attempt_source_timestamps_s"].append(
                [
                    sample.emission_source_timestamp_s,
                    sample.encoder_anchor_timestamp_s,
                    sample.root_setpoint_timestamp_s,
                ]
            )
            action, token = teacher.infer_with_token(semantic, proprio)
            arrays["attempt_raw29"].append(action.copy())
            arrays["attempt_token64"].append(token.copy())
            if action.shape != (29,) or not np.isfinite(action).all() or np.max(np.abs(action)) >= 10:
                raise RuntimeError("invalid or raw-bound original29 action")
            target = parameters.target(action)
            for key, value in (
                ("encoder267", semantic),
                ("history930", proprio),
                ("raw29", action),
                ("target29", target),
            ):
                arrays[key].append(value.copy())
            arrays["source_timestamps_s"].append(
                [
                    sample.emission_source_timestamp_s,
                    sample.encoder_anchor_timestamp_s,
                    sample.root_setpoint_timestamp_s,
                ]
            )
            for _ in range(10):
                arrays["physics_pre_qpos"].append(data.qpos.copy())
                arrays["physics_pre_qvel"].append(data.qvel.copy())
                torque = parameters.kps * (target - data.qpos[7:]) - parameters.kds * data.qvel[6:]
                applied = np.clip(torque, -parameters.effort, parameters.effort)
                data.ctrl[:] = applied
                start_time = float(data.time)
                mujoco.mj_step(model, data)
                if abs(float(data.time) - start_time - 0.002) > 1e-10:
                    raise RuntimeError("original29 integration time discontinuity")
                for key, value in (
                    ("physics_post_qpos", data.qpos),
                    ("physics_post_qvel", data.qvel),
                    ("requested_torque29", torque),
                    ("applied_torque29", applied),
                    ("engine_force29", data.qfrc_actuator[6:]),
                ):
                    arrays[key].append(value.copy())
                arrays["contact_count"].append(int(data.ncon))
            arrays["qpos"].append(data.qpos.copy())
            arrays["qvel"].append(data.qvel.copy())
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("nonfinite original29 integrated state")
            probe.qpos[:], reference.qpos[:] = data.qpos, poses[11 + control]
            mujoco.mj_fwdPosition(model, probe)
            mujoco.mj_fwdPosition(source_model, reference)
            actual, desired = points_at(model, probe), points_at(source_model, reference)
            arrays["landmark_error_m"].append(np.linalg.norm(actual - desired, axis=1))
            arrays["relative_landmark_error_m"].append(
                np.linalg.norm((actual - data.qpos[:3]) - (desired - poses[11 + control, :3]), axis=1)
            )
            arrays["root_error_m"].append(float(np.linalg.norm(data.qpos[:3] - poses[11 + control, :3])))
            arrays["joint_error29"].append(data.qpos[7:] - poses[11 + control, 7:])
            previous = action.copy()
            gravity = stock._quat_rotate(stock._quat_conjugate(data.qpos[3:7]), np.array([0, 0, -1]))
            if data.qpos[2] < 0.12 or np.arccos(np.clip(-gravity[2], -1, 1)) > 2.2:
                raise RuntimeError("absolute height/tilt diagnostic stop")
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    completed = len(arrays["qpos"]) - 1
    phase = next(row for row in phases if row["name"] == "source_motion")
    stop = min(completed, phase["control_stop"], len(arrays["landmark_error_m"]))
    count = max(0, stop - phase["control_start"])
    section = slice(phase["control_start"], stop)
    errors = arrays["landmark_error_m"][section]
    metrics = dict(
        requested_controls=requested,
        completed_controls=completed,
        failure=failure,
        requested_source_controls=phase["control_stop"] - phase["control_start"],
        completed_source_controls=count,
        source_landmark_p95_m=np.percentile(errors, 95, axis=0).tolist() if count else None,
        source_root_position_p95_m=float(np.percentile(arrays["root_error_m"][section], 95)) if count else None,
        source_leg_joint_rmse_rad=float(np.sqrt(np.mean(arrays["joint_error29"][section, :12] ** 2)))
        if count
        else None,
        source_landmark_screen_passed=bool(
            count == phase["control_stop"] - phase["control_start"]
            and count
            and np.all(np.percentile(errors, 95, axis=0) <= [0.05, 0.05, 0.1, 0.1, 0.1])
        ),
        no_postinitial_state_rewrites=True,
        no_fallback_controller=True,
        **FLAGS,
    )
    if compiled_model_sha256(model) != before:
        raise ValueError("original29 simulation mutated its physical model")
    return arrays, source, metrics
