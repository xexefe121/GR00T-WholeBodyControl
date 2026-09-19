"""Isolated SIM-only normal29 body control on source29/source43 scenes.

All29 body axes remain measured and commanded. Optional14 physical simulator
finger axes retain their actual dynamics and zero applied control; they are
never synthetic body feedback. Only explicit2/5ms model timesteps are accepted.
This is deterministic direct physics, not the upstream DDS scheduling stack,
not native23, not a hardware entry point. Original lifecycle remains unchanged.
"""

from collections import deque
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.scripts import simulate_g1_sonic_library_motions as stock
from gear_sonic.scripts.record_g1_sonic_public29_baseline import FLAGS, points_at
from gear_sonic.teleop.normal_source_horizon import ReceivedNormalSourceHorizon
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_JOINT_NAMES,
    REFERENCE_PROFILE_NORMAL,
    SOURCE_MJ29_KEEP_INDICES,
)
from gear_sonic.utils.g1_true23_original29_reference import SOURCE_JOINT_NAMES, VR_BODIES, VR_OFFSETS
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


class SourceBodyLayout:
    """Validate explicit free-root29/43 scene and map canonical body motors."""

    def __init__(self, model):
        if (model.nq, model.nv, model.nu, model.njnt) not in ((36, 35, 29, 30), (50, 49, 43, 44)):
            raise ValueError("source body layout requires exactly29 or43 scalar actuators")
        if int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE) or model.jnt_qposadr[0] != 0:
            raise ValueError("source body layout requires one free root at zero")
        if np.any(model.jnt_type[1:] != int(mujoco.mjtJoint.mjJNT_HINGE)):
            raise ValueError("source body layout requires scalar hinges")
        names = [model.joint(i).name for i in range(1, model.njnt)]
        if any(name not in names for name in SOURCE_JOINT_NAMES):
            raise ValueError("source body layout is missing a canonical body joint")
        extras = set(names) - set(SOURCE_JOINT_NAMES)
        if len(extras) != model.nu - 29 or any("_hand_" not in name for name in extras):
            raise ValueError("only14 source hand joints may supplement canonical29")
        self.joints = np.array([model.joint(name).id for name in SOURCE_JOINT_NAMES])
        self.qpos = model.jnt_qposadr[self.joints].copy()
        self.qvel = model.jnt_dofadr[self.joints].copy()
        self.actuators = np.array([model.actuator(name.removesuffix("_joint")).id for name in SOURCE_JOINT_NAMES])
        if not np.array_equal(model.actuator_trnid[self.actuators, 0], self.joints):
            raise ValueError("canonical source motor transmission does not match joint")
        if np.any(model.actuator_gear[self.actuators, 0] != 1) or np.any(
            model.actuator_gear[self.actuators, 1:] != 0
        ):
            raise ValueError("source body layout requires direct unit-gear motors")
        if float(model.opt.timestep) not in (0.002, 0.005):
            raise ValueError("source body layout requires explicit2ms or5ms timestep")
        self.substeps = int(round(0.02 / float(model.opt.timestep)))
        self.timestep = float(model.opt.timestep)
        self.finger_actuators = np.array([i for i in range(model.nu) if i not in self.actuators], dtype=int)
        for value in (self.joints, self.qpos, self.qvel, self.actuators, self.finger_actuators):
            value.setflags(write=False)

    def canonical(self, data):
        return SimpleNamespace(
            qpos=np.r_[data.qpos[:7], data.qpos[self.qpos]],
            qvel=np.r_[data.qvel[:6], data.qvel[self.qvel]],
        )

    def descriptor(self):
        return dict(
            body_joint_ids=self.joints.tolist(),
            body_qpos_indices=self.qpos.tolist(),
            body_qvel_indices=self.qvel.tolist(),
            body_actuator_ids=self.actuators.tolist(),
            finger_actuator_ids=self.finger_actuators.tolist(),
            timestep_s=self.timestep,
            policy_period_s=0.02,
            physics_substeps_per_control=self.substeps,
            finger_control="zero_applied_control_free_dynamics",
            hardware_authorized=False,
            native23_qualified=False,
            deployment_ready=False,
        )


def run_case(model, source_model, parameters, teacher, poses, phases, virtual_vr21):
    if teacher.reference_profile != REFERENCE_PROFILE_NORMAL:
        raise ValueError("normal29 lifecycle refuses a different source policy profile")
    if virtual_vr21.shape != (len(poses), 21) or virtual_vr21.dtype != np.float32:
        raise ValueError("normal29 requires the saved finite float32 original21-VR source")
    if not np.isfinite(virtual_vr21).all():
        raise ValueError("normal29 source VR contains nonfinite values")
    layout = SourceBodyLayout(model)
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
    data.qpos[:7] = poses[10, :7]
    data.qpos[layout.qpos] = poses[10, 7:]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    previous = np.zeros(29, dtype=np.float32)

    def measured():
        body = layout.canonical(data)
        value = stock._current_history_frame(body, previous)
        value["joint_position"] = (body.qpos[7:] - parameters.default_angles)[stock.ISAAC_TO_MUJOCO_INDEX]
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
            "full_actuator_ctrl",
            "full_actuator_force",
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
            for _ in range(layout.substeps):
                arrays["physics_pre_qpos"].append(data.qpos.copy())
                arrays["physics_pre_qvel"].append(data.qvel.copy())
                torque = (
                    parameters.kps * (target - data.qpos[layout.qpos]) - parameters.kds * data.qvel[layout.qvel]
                )
                applied = np.clip(torque, -parameters.effort, parameters.effort)
                data.ctrl[:] = 0
                data.ctrl[layout.actuators] = applied
                start_time = float(data.time)
                mujoco.mj_step(model, data)
                if abs(float(data.time) - start_time - layout.timestep) > 1e-10:
                    raise RuntimeError("original29 integration time discontinuity")
                for key, value in (
                    ("physics_post_qpos", data.qpos),
                    ("physics_post_qvel", data.qvel),
                    ("requested_torque29", torque),
                    ("applied_torque29", applied),
                    ("engine_force29", data.qfrc_actuator[layout.qvel]),
                    ("full_actuator_ctrl", data.ctrl),
                    ("full_actuator_force", data.actuator_force),
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
            arrays["joint_error29"].append(data.qpos[layout.qpos] - poses[11 + control, 7:])
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
        layout=layout.descriptor(),
        no_postinitial_state_rewrites=True,
        no_fallback_controller=True,
        **FLAGS,
    )
    if compiled_model_sha256(model) != before:
        raise ValueError("original29 simulation mutated its physical model")
    return arrays, source, metrics


def matched_metrics(model, trace, reference, phase, completed=None):
    n = len(trace["qpos"]) - 1 if completed is None else completed
    start, stop = phase["control_start"], min(n, phase["control_stop"])
    count = max(0, stop - start)
    if not count:
        return dict(measured=False, source_controls_completed=0)
    poses = trace["qpos"][1 : n + 1]
    layout = SourceBodyLayout(model)
    probe = mujoco.MjData(model)
    positions, quaternions = [], []
    for pose in poses:
        probe.qpos[:] = pose
        mujoco.mj_fwdPosition(model, probe)
        positions.append(
            [
                probe.xpos[model.body(body).id]
                + stock._quat_rotate(probe.xquat[model.body(body).id], np.asarray(offset))
                for body, offset in zip(VR_BODIES, VR_OFFSETS, strict=True)
            ]
        )
        quaternions.append([probe.xquat[model.body(body).id].copy() for body in VR_BODIES])
    actual = SimpleNamespace(
        source_task_position_w=np.asarray(positions), source_task_quaternion_wxyz=np.asarray(quaternions)
    )
    wanted_p = reference.source_task_position_w[11 : 11 + n]
    wanted_q = reference.source_task_quaternion_wxyz[11 : 11 + n]
    section = slice(start, stop)
    relative = (actual.source_task_position_w - poses[:, None, :3]) - (
        wanted_p - reference.source_qpos29[11 : 11 + n, None, :3]
    )
    angle = (
        (
            Rotation.from_quat(actual.source_task_quaternion_wxyz.reshape(-1, 4)[:, [1, 2, 3, 0]]).inv()
            * Rotation.from_quat(wanted_q.reshape(-1, 4)[:, [1, 2, 3, 0]])
        )
        .magnitude()
        .reshape(n, 3)
    )
    return dict(
        measured=True,
        source_controls_completed=count,
        order=["left_hand", "right_hand", "head_proxy"],
        original_task_world_position_p95_m=np.percentile(
            np.linalg.norm(actual.source_task_position_w - wanted_p, axis=-1)[section], 95, axis=0
        ).tolist(),
        original_task_pelvis_centered_world_axes_position_p95_m=np.percentile(
            np.linalg.norm(relative, axis=-1)[section], 95, axis=0
        ).tolist(),
        original_task_world_orientation_p95_rad=np.percentile(angle[section], 95, axis=0).tolist(),
        root_world_position_p95_m=float(
            np.percentile(
                np.linalg.norm(poses[:, :3] - reference.source_qpos29[11 : 11 + n, :3], axis=-1)[section], 95
            )
        ),
        leg_joint_rmse_rad=float(
            np.sqrt(
                np.mean((poses[:, layout.qpos[:12]] - reference.source_qpos29[11 : 11 + n, 7:19])[section] ** 2)
            )
        ),
        foot_body_origin_world_position_p95_m=np.percentile(
            trace["landmark_error_m"][section, :2], 95, axis=0
        ).tolist(),
        scoring_phase="unchanged_referee_post_control_q2",
        same_signed_hand_offsets_as_native23_original_intent=True,
    )
