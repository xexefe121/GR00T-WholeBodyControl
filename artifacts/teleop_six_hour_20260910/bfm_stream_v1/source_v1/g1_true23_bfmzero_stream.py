"""Recorded-input BFM stream simulator; no transport or hardware control API.

Source packets and the independently clocked simulator have separate clocks.
Only received samples enter a source goal. A fault switches to a generated
standing reference, with the same learned BFM policy continuing to balance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import corrected_goal
from gear_sonic.utils.g1_true23_bfmzero_inference import BFMHistory, reference_features, state_and_terms


DT = 0.02
HORIZON = 8
SOURCE_BUFFER_SECONDS = (HORIZON - 1) * DT
FIELD_SHAPES = {
    "joint_pos": (23,),
    "joint_vel": (23,),
    "body_pos_w": (24, 3),
    "body_quat_w": (24, 4),
    "body_lin_vel_w": (24, 3),
    "body_ang_vel_w": (24, 3),
}


class StreamMode(str, Enum):
    BUFFERING = "buffering"
    SOURCE = "source"
    STOPPING = "stopping"
    LATCHED_STANDING = "latched_standing"
    COMPLETE = "complete"


class PacketRejected(ValueError):
    pass


@dataclass(frozen=True)
class Packet:
    epoch: int
    sequence: int
    source_time: float
    fields: dict[str, np.ndarray]
    final: bool = False


@dataclass
class ReceivedSample:
    packet: Packet
    received_at: float
    state: np.ndarray
    privileged: np.ndarray


def yaw(quaternion_wxyz):
    direction = Rotation.from_quat(np.asarray(quaternion_wxyz)[[1, 2, 3, 0]]).apply([1, 0, 0])
    return math.atan2(direction[1], direction[0])


def rotation_wxyz(rotation):
    return rotation.as_quat()[[3, 0, 1, 2]]


def packet_fields(motion, index):
    return {name: np.asarray(motion[name][index]).copy() for name in FIELD_SHAPES}


def validate_fields(fields):
    if set(fields) != set(FIELD_SHAPES):
        raise PacketRejected("packet_fields")
    for name, shape in FIELD_SHAPES.items():
        array = np.asarray(fields[name])
        if array.shape != shape or not np.issubdtype(array.dtype, np.number):
            raise PacketRejected("packet_shape:" + name)
        if not np.isfinite(array).all():
            raise PacketRejected("packet_nonfinite:" + name)
    norms = np.linalg.norm(fields["body_quat_w"], axis=-1)
    if np.max(np.abs(norms - 1)) > 1e-5:
        raise PacketRejected("packet_quaternion_norm")


def stack_samples(samples):
    motion = {name: np.stack([sample.packet.fields[name] for sample in samples]) for name in FIELD_SHAPES}
    return (
        motion,
        np.stack([sample.state for sample in samples]),
        np.stack([sample.privileged for sample in samples]),
    )


class ReferenceAnchor:
    """One declared rigid frame calibration per explicit rearm, never a fitted path."""

    def __init__(self, first_fields, measured_qpos):
        self.original_origin = first_fields["body_pos_w"][0].copy()
        self.measured_origin = np.asarray(measured_qpos[:3]).copy()
        self.rotation = Rotation.from_euler("z", yaw(measured_qpos[3:7]) - yaw(first_fields["body_quat_w"][0]))

    def apply(self, fields):
        result = {name: np.asarray(value).copy() for name, value in fields.items()}
        position = fields["body_pos_w"] - self.original_origin
        result["body_pos_w"] = self.rotation.apply(position) + self.measured_origin
        # Preserve original absolute height; calibration affects only world XY/yaw.
        result["body_pos_w"][:, 2] = fields["body_pos_w"][:, 2]
        rotations = Rotation.from_quat(fields["body_quat_w"][:, [1, 2, 3, 0]])
        result["body_quat_w"] = (self.rotation * rotations).as_quat()[:, [3, 0, 1, 2]]
        result["body_lin_vel_w"] = self.rotation.apply(fields["body_lin_vel_w"])
        result["body_ang_vel_w"] = self.rotation.apply(fields["body_ang_vel_w"])
        return result

    def report(self):
        return {
            "kind": "constant_source_xy_yaw_calibration_at_explicit_rearm",
            "source_origin": self.original_origin.tolist(),
            "measured_origin": self.measured_origin.tolist(),
            "rotation_wxyz": rotation_wxyz(self.rotation).tolist(),
            "source_joint_values_and_timestamps_unchanged": True,
        }


class PacketGate:
    """Strict admission and fault latch, independent of packet delivery cadence."""

    def __init__(self, contract, *, stale_seconds=0.1, now=0.0):
        if not math.isfinite(stale_seconds) or stale_seconds < DT:
            raise ValueError("invalid packet freshness bound")
        self.contract = contract
        self.stale_seconds = stale_seconds
        self.epoch = 0
        self.epoch_started_at = now
        self.last_received_at = now
        self.last_sequence = -1
        self.last_source_time = None
        self.received = 0
        self.consumed = 0
        self.ignored = 0
        self.rejected = 0
        self.final_received = False
        self.final_consumed = False
        self.fault = None
        self.queue = deque()
        self.events: list[dict[str, Any]] = []
        self.epochs = []
        self.anchor = None
        self.anchor_pose = None

    def latch(self, reason, now):
        if self.fault is not None:
            return False
        self.fault = {
            "reason": reason,
            "time": float(now),
            "epoch": self.epoch,
            "last_sequence": self.last_sequence,
            "consumed": self.consumed,
        }
        self.events.append({"kind": "latched_fault", **self.fault})
        self.queue.clear()
        return True

    def receive(self, packet, now):
        if self.fault is not None or packet.epoch != self.epoch:
            self.ignored += 1
            return False
        try:
            if type(packet.epoch) is not int or type(packet.sequence) is not int or packet.sequence < 0:
                raise PacketRejected("packet_integer_identity")
            if type(packet.final) is not bool:
                raise PacketRejected("packet_final_flag")
            if not math.isfinite(now) or now < self.last_received_at:
                raise PacketRejected("receive_clock_regression")
            if not math.isfinite(packet.source_time) or packet.source_time < 0:
                raise PacketRejected("packet_timestamp_nonfinite_or_negative")
            if self.final_received:
                raise PacketRejected("packet_after_final")
            if packet.sequence != self.last_sequence + 1:
                raise PacketRejected("packet_sequence_gap_or_reorder")
            expected_time = packet.sequence * DT
            if abs(packet.source_time - expected_time) > 1e-7:
                raise PacketRejected("packet_source_clock_not_50hz")
            source_age = now - self.epoch_started_at - packet.source_time
            if source_age < -1e-7:
                raise PacketRejected("packet_from_future")
            if source_age > self.stale_seconds + 1e-7:
                raise PacketRejected("packet_arrived_stale")
            validate_fields(packet.fields)
            fields = {name: np.asarray(value, dtype=np.float64).copy() for name, value in packet.fields.items()}
            if self.anchor_pose is not None and self.anchor is None:
                self.anchor = ReferenceAnchor(fields, self.anchor_pose)
                self.events.append({"kind": "reference_calibration", "epoch": self.epoch, **self.anchor.report()})
            if self.anchor is not None:
                fields = self.anchor.apply(fields)
            bank = {name: value[None] for name, value in fields.items()}
            state, privileged = reference_features(bank, self.contract)
            accepted = Packet(packet.epoch, packet.sequence, packet.source_time, fields, packet.final)
            self.queue.append(ReceivedSample(accepted, now, state[0], privileged[0]))
            self.last_received_at = now
            self.last_sequence = packet.sequence
            self.last_source_time = packet.source_time
            self.received += 1
            self.final_received = packet.final
            return True
        except (PacketRejected, ValueError, TypeError) as exc:
            self.rejected += 1
            self.latch(str(exc), now)
            return False

    def check_freshness(self, now):
        if (
            self.fault is None
            and not self.final_received
            and now - self.last_received_at > self.stale_seconds + 1e-7
        ):
            return self.latch("packet_timeout", now)
        return False

    def ready(self):
        return self.fault is None and (len(self.queue) >= HORIZON or (self.final_received and bool(self.queue)))

    def window(self):
        return list(self.queue)[:HORIZON]

    def consume(self):
        sample = self.queue.popleft()
        self.consumed += 1
        self.final_consumed = sample.packet.final
        return sample

    def epoch_report(self):
        return {
            "epoch": self.epoch,
            "received": self.received,
            "consumed": self.consumed,
            "ignored": self.ignored,
            "rejected": self.rejected,
            "final_received": self.final_received,
            "full_source_consumed": self.final_consumed and self.fault is None,
            "fault": self.fault,
            "calibration": None if self.anchor is None else self.anchor.report(),
        }

    def rearm(self, now, measured_qpos):
        self.epochs.append(self.epoch_report())
        self.epoch += 1
        self.epoch_started_at = now
        self.last_received_at = now
        self.last_sequence = -1
        self.last_source_time = None
        self.received = self.consumed = self.ignored = self.rejected = 0
        self.final_received = self.final_consumed = False
        self.fault = None
        self.queue.clear()
        self.anchor = None
        self.anchor_pose = measured_qpos.copy()
        self.events.append({"kind": "explicit_simulated_operator_rearm", "time": now, "epoch": self.epoch})


class StandingReference:
    """Lazy quintic pose transition; FK operates on separate reference-only data."""

    def __init__(self, model, contract, measured_qpos, standing_q, standing_height, *, duration=2.0):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("standing transition duration must be positive")
        self.model = model
        self.contract = contract
        self.duration = duration
        self.start = np.asarray(measured_qpos, dtype=np.float64).copy()
        self.target = self.start.copy()
        self.target[2] = standing_height
        self.target[3:7] = rotation_wxyz(Rotation.from_euler("z", yaw(self.start[3:7])))
        self.target[7:] = standing_q
        self.start_rotation = Rotation.from_quat(self.start[[4, 5, 6, 3]])
        end_rotation = Rotation.from_quat(self.target[[4, 5, 6, 3]])
        self.rotation_delta = (self.start_rotation.inv() * end_rotation).as_rotvec()
        self.probe = mujoco.MjData(model)
        self.cache = {}

    def pose(self, index):
        t = min(max(index * DT / self.duration, 0.0), 1.0)
        blend = 10 * t**3 - 15 * t**4 + 6 * t**5
        derivative = (30 * t**2 - 60 * t**3 + 30 * t**4) / self.duration
        qpos = self.start + blend * (self.target - self.start)
        qpos[3:7] = rotation_wxyz(self.start_rotation * Rotation.from_rotvec(blend * self.rotation_delta))
        qvel = np.r_[
            derivative * (self.target[:3] - self.start[:3]),
            derivative * self.rotation_delta,
            derivative * (self.target[7:] - self.start[7:]),
        ]
        return qpos, qvel

    def sample(self, index):
        key = min(index, math.ceil(self.duration / DT))
        if key not in self.cache:
            qpos, qvel = self.pose(key)
            self.probe.qpos[:] = qpos
            self.probe.qvel[:] = qvel
            mujoco.mj_forward(self.model, self.probe)
            velocity = np.empty((24, 6), dtype=np.float64)
            for body in range(1, 25):
                mujoco.mj_objectVelocity(
                    self.model, self.probe, mujoco.mjtObj.mjOBJ_BODY, body, velocity[body - 1], 0
                )
            fields = {
                "joint_pos": qpos[7:].copy(),
                "joint_vel": qvel[6:].copy(),
                "body_pos_w": self.probe.xpos[1:].copy(),
                "body_quat_w": self.probe.xquat[1:].copy(),
                "body_lin_vel_w": velocity[:, 3:].copy(),
                "body_ang_vel_w": velocity[:, :3].copy(),
            }
            state, privileged = reference_features(
                {name: value[None] for name, value in fields.items()}, self.contract
            )
            packet = Packet(-1, key, key * DT, fields)
            self.cache[key] = ReceivedSample(packet, float("nan"), state[0], privileged[0])
        return self.cache[key]

    def window(self, first):
        # Cache is bounded to the live horizon plus one shared terminal pose.
        result = [self.sample(index) for index in range(first, first + HORIZON)]
        for key in tuple(self.cache):
            if key < first and key < math.ceil(self.duration / DT):
                del self.cache[key]
        return result


def foot_support(model, data):
    feet = [model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    normal_load = np.zeros(2)
    force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        bodies = [int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])]
        if 0 not in bodies:
            continue
        mujoco.mj_contactForce(model, data, index, force)
        for foot, body in enumerate(feet):
            if body in bodies:
                normal_load[foot] += max(0.0, float(force[0]))
    return normal_load


class BFMStreamSimulator:
    """One method call per fixed control tick; packet receipt never advances physics."""

    def __init__(
        self,
        model,
        physics,
        contract,
        policy,
        initial_qpos,
        initial_qvel,
        standing_q,
        standing_height,
        velocity_limits,
        *,
        position_gain=1.0,
        yaw_gain=2.0,
        stale_seconds=0.1,
        stop_seconds=2.0,
        stable_controls=50,
    ):
        if (model.nq, model.nv, model.nu, model.nbody) != (30, 29, 23, 25):
            raise ValueError("stream simulator requires the physical native23 topology")
        if abs(model.opt.timestep - 0.002) > 1e-12 or physics.decimation != 10:
            raise ValueError("stream simulator requires 500Hz physics and 50Hz control")
        self.model, self.physics, self.contract, self.policy = model, physics, contract, policy
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = initial_qpos
        self.data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, self.data)
        self.standing_q = np.asarray(standing_q).copy()
        self.standing_height = standing_height
        self.velocity_limits = np.asarray(velocity_limits)
        self.position_gain, self.yaw_gain = position_gain, yaw_gain
        self.stop_seconds = stop_seconds
        self.stable_controls_required = stable_controls
        self.stable_controls = 0
        self.gate = PacketGate(contract, stale_seconds=stale_seconds)
        self.mode = StreamMode.BUFFERING
        self.safety_reference = self._standing_reference()
        self.safety_control = 1
        self.stop_reason = None
        self.history = BFMHistory()
        self.action = np.zeros(23, np.float32)
        self.controls = 0
        self.physical_failure = None
        self.events = []
        self.epoch_any_fault = False
        self.arrivals = []
        self.trace = {
            name: []
            for name in (
                "qpos",
                "qvel",
                "state",
                "history",
                "goal",
                "action",
                "target",
                "unclipped_target",
                "mode",
                "source_sequence",
                "source_timestamp",
                "source_epoch",
                "source_buffer_age",
                "source_latest_timestamp_used",
                "goal_window_samples",
                "reference_qpos",
                "physics_qpos",
                "physics_qvel",
                "physics_torque",
                "physics_requested_torque",
                "range_excess",
                "velocity_ratio",
                "effort_ratio",
                "foot_support",
                "standing_stable",
            )
        }
        for name, value in (
            ("qpos", self.data.qpos),
            ("qvel", self.data.qvel),
            ("physics_qpos", self.data.qpos),
            ("physics_qvel", self.data.qvel),
        ):
            self.trace[name].append(value.copy())

    def _standing_reference(self):
        return StandingReference(
            self.model,
            self.contract,
            self.data.qpos,
            self.standing_q,
            self.standing_height,
            duration=self.stop_seconds,
        )

    def receive(self, packet, now):
        accepted = self.gate.receive(packet, now)
        self.arrivals.append([now, packet.epoch, packet.sequence, packet.source_time, int(accepted)])
        return accepted

    def _start_stop(self, reason, now):
        self.mode = StreamMode.STOPPING
        self.stop_reason = reason
        self.safety_reference = self._standing_reference()
        self.safety_control = 1
        self.stable_controls = 0
        self.events.append(
            {
                "kind": "active_bfm_standing_transition",
                "reason": reason,
                "time": now,
                "control": self.controls,
                "target_qpos": self.safety_reference.target.tolist(),
                "simulator_state_rewritten": False,
                "root_force_applied": False,
            }
        )

    def rearm(self, now):
        if (
            self.mode not in (StreamMode.LATCHED_STANDING, StreamMode.COMPLETE)
            or self.stable_controls < self.stable_controls_required
        ):
            self.events.append(
                {
                    "kind": "rearm_rejected_not_stable_standing",
                    "time": now,
                    "mode": self.mode.value,
                    "stable_controls": self.stable_controls,
                }
            )
            return False
        self.gate.rearm(now, self.data.qpos)
        self.mode = StreamMode.BUFFERING
        self.stop_reason = None
        self.safety_reference = self._standing_reference()
        self.safety_control = 1
        self.stable_controls = 0
        # The controller's physical action/state history continues across rearm.
        return True

    def step(self, now):
        if self.physical_failure is not None:
            raise RuntimeError("physical diagnostic failure already ended this simulation")
        self.gate.check_freshness(now)
        if self.gate.fault is not None and self.mode in (StreamMode.SOURCE, StreamMode.BUFFERING):
            self.epoch_any_fault = True
            self._start_stop(self.gate.fault["reason"], now)
        if self.mode == StreamMode.BUFFERING and self.gate.ready():
            self.mode = StreamMode.SOURCE
            self.events.append(
                {
                    "kind": "source_control_started",
                    "time": now,
                    "epoch": self.gate.epoch,
                    "received_window": len(self.gate.queue),
                }
            )
        if self.mode == StreamMode.SOURCE and not self.gate.queue:
            if self.gate.final_consumed:
                self._start_stop("normal_received_end_of_stream", now)
            else:
                self.gate.latch("source_buffer_underrun", now)
                self.epoch_any_fault = True
                self._start_stop("source_buffer_underrun", now)
        source_sample = None
        if self.mode == StreamMode.SOURCE:
            samples = self.gate.window()
            source_sample = samples[0]
        else:
            samples = self.safety_reference.window(self.safety_control)
        motion, states, privileged = stack_samples(samples)
        sensed, terms = state_and_terms(
            self.data.qpos[7:],
            self.data.qvel[6:],
            self.data.qpos[3:7],
            self.data.qvel[3:6],
            self.action,
            self.contract["default_q"],
        )
        history = self.history.before_update(terms)
        if self.position_gain or self.yaw_gain:
            goal = corrected_goal(
                self.policy,
                states,
                privileged,
                motion,
                0,
                self.data.qpos,
                len(samples),
                self.position_gain,
                self.yaw_gain,
            )
        else:
            latent = self.policy.backward(torch.from_numpy(states), torch.from_numpy(privileged)).mean(
                0, keepdim=True
            )
            goal = 16 * F.normalize(latent, dim=-1)
        raw = self.policy.actor(
            torch.from_numpy(sensed[None]),
            torch.from_numpy(self.action[None]),
            torch.from_numpy(history[None]),
            goal,
        )[0].numpy()
        self.action = raw * 5.0
        target = (
            self.contract["default_q"]
            + self.action * 0.25 * self.contract["training_effort"] / self.contract["kp"]
        )
        limits = self.model.jnt_range[1:]
        clipped = np.clip(target, limits[:, 0], limits[:, 1])
        excess = effort = velocity = 0.0
        for _ in range(self.physics.decimation):
            torque = (
                self.contract["kp"] * (clipped - self.data.qpos[7:]) - self.contract["kd"] * self.data.qvel[6:]
            )
            self.data.ctrl[:] = np.clip(torque, -self.physics.effort, self.physics.effort)
            mujoco.mj_step(self.model, self.data)
            self.trace["physics_qpos"].append(self.data.qpos.copy())
            self.trace["physics_qvel"].append(self.data.qvel.copy())
            self.trace["physics_torque"].append(self.data.ctrl.copy())
            self.trace["physics_requested_torque"].append(torque.copy())
            excess = max(
                excess,
                float(np.maximum(limits[:, 0] - self.data.qpos[7:], self.data.qpos[7:] - limits[:, 1]).max()),
            )
            velocity = max(velocity, float((np.abs(self.data.qvel[6:]) / self.velocity_limits).max()))
            effort = max(effort, float((np.abs(self.data.qfrc_actuator[6:]) / self.physics.effort).max()))
        self.controls += 1
        support = foot_support(self.model, self.data)
        tilt = math.acos(np.clip(Rotation.from_quat(self.data.qpos[[4, 5, 6, 3]]).as_matrix()[2, 2], -1, 1))
        stable = (
            abs(self.data.qpos[2] - self.standing_height) < 0.10
            and tilt < 0.15
            and np.sqrt(np.mean((self.data.qpos[7:] - self.standing_q) ** 2)) < 0.18
            and np.linalg.norm(self.data.qvel[:3]) < 0.15
            and np.linalg.norm(self.data.qvel[3:6]) < 0.3
            and np.sqrt(np.mean(self.data.qvel[6:] ** 2)) < 0.4
            and np.min(support) > 5
            and np.sum(support) > 0.6 * np.sum(self.model.body_mass) * 9.81
        )
        if self.mode == StreamMode.STOPPING:
            reached_endpoint = self.safety_control * DT >= self.stop_seconds
            self.stable_controls = self.stable_controls + 1 if reached_endpoint and stable else 0
            if self.stable_controls >= self.stable_controls_required:
                self.mode = (
                    StreamMode.COMPLETE
                    if self.stop_reason == "normal_received_end_of_stream"
                    else StreamMode.LATCHED_STANDING
                )
                self.events.append(
                    {
                        "kind": "measured_standing_verified",
                        "time": now + DT,
                        "mode": self.mode.value,
                        "consecutive_controls": self.stable_controls,
                    }
                )
        elif self.mode in (StreamMode.LATCHED_STANDING, StreamMode.COMPLETE):
            self.stable_controls = self.stable_controls + 1 if stable else 0
        if source_sample is not None:
            self.gate.consume()
        else:
            self.safety_control += 1
        values = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "state": sensed,
            "history": history,
            "goal": goal[0].numpy().copy(),
            "action": self.action.copy(),
            "target": clipped.copy(),
            "unclipped_target": target.copy(),
            "mode": self.mode.value,
            "source_sequence": -1 if source_sample is None else source_sample.packet.sequence,
            "source_timestamp": -1.0 if source_sample is None else source_sample.packet.source_time,
            "source_epoch": self.gate.epoch,
            "source_buffer_age": -1.0 if source_sample is None else now - source_sample.received_at,
            "source_latest_timestamp_used": -1.0 if source_sample is None else samples[-1].packet.source_time,
            "goal_window_samples": len(samples),
            "reference_qpos": np.r_[
                motion["body_pos_w"][0, 0], motion["body_quat_w"][0, 0], motion["joint_pos"][0]
            ],
            "range_excess": excess,
            "velocity_ratio": velocity,
            "effort_ratio": effort,
            "foot_support": support,
            "standing_stable": stable,
        }
        for name, value in values.items():
            self.trace[name].append(value)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            self.physical_failure = {"reason": "nonfinite_physics", "control": self.controls}
        elif self.data.qpos[2] < 0.25 or tilt > 1.2:
            self.physical_failure = {
                "reason": "fall",
                "control": self.controls,
                "height": float(self.data.qpos[2]),
                "tilt": tilt,
            }
        elif excess > 0.01 or velocity > 1.0:
            self.physical_failure = {
                "reason": "physical_limit_diagnostic_stop",
                "control": self.controls,
                "range_excess": excess,
                "velocity_ratio": velocity,
            }
        return values

    def arrays(self):
        return {name: np.asarray(values) for name, values in self.trace.items()}

    def report(self):
        arrays = self.arrays()
        return {
            "kind": "bfmzero_received_stream_simulation_v1",
            "mode": self.mode.value,
            "controls": self.controls,
            "physics_steps": len(self.trace["physics_torque"]),
            "physics_steps_per_control": self.physics.decimation,
            "control_hz": 50,
            "physics_hz": 500,
            "received_source_buffer_seconds": SOURCE_BUFFER_SECONDS,
            "stale_seconds": self.gate.stale_seconds,
            "epochs": [*self.gate.epochs, self.gate.epoch_report()],
            "any_fault_latched": self.epoch_any_fault,
            "physical_failure": self.physical_failure,
            "full_uninterrupted_source_consumed": self.gate.epoch == 0
            and not self.epoch_any_fault
            and self.gate.final_consumed,
            "standing_return_verified": (
                self.mode in (StreamMode.COMPLETE, StreamMode.LATCHED_STANDING)
                and self.stable_controls >= self.stable_controls_required
            ),
            "range_excess_max_rad": float(arrays["range_excess"].max()) if self.controls else None,
            "range_excess_controls": int(np.count_nonzero(arrays["range_excess"] > 0)),
            "effort_ratio_max": float(arrays["effort_ratio"].max()) if self.controls else None,
            "velocity_ratio_max": float(arrays["velocity_ratio"].max()) if self.controls else None,
            "events": [*self.events, *self.gate.events],
            "ground_truth_pose_feedback": True,
            "simulator_pose_writes_after_initialization": 0,
            "root_assistance_forces": 0,
            "controller_fallback": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
