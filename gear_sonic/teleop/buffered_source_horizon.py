"""Received-only SONIC source horizon with an explicit 200-ms anchor delay.

This is a reference buffer, not a controller, predictor or robot connection.
Eleven received 50-Hz samples supply ten positions plus forward differences.
The source encoder's anchor is the oldest sample, not the newest past sample.
No terminal padding/draining, clock repair or implicit reinitialization exists.
The existing q9-anchored causal-policy artifacts must not be relabeled as this
different interface; integration and retraining remain separate requirements.
"""

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_numpy

PERIOD_S = 0.02
SAMPLE_COUNT = 11


def buffered_horizon_contract():
    payload = {
        "kind": "sonic_received_source_horizon_200ms_v1",
        "source_period_s": PERIOD_S,
        "received_samples_required": SAMPLE_COUNT,
        "encoder_anchor_age_s": 0.2,
        "next_root_setpoint_age_s": 0.18,
        "positions": "received_q0_to_q9_left_six_then_right_six",
        "velocities": "received_forward_differences_q1_to_q10_minus_q0_to_q9",
        "vr_reference": "received_q0_original_source_geometry",
        "relative_orientation": "received_q0_pelvis_relative_to_current_measured_pelvis",
        "root_feedback_setpoint": "received_q1_with_velocity_q1_minus_q0",
        "measured_state": "current_control_time_not_delayed",
        "unreceived_samples_or_prediction": False,
        "terminal_padding_or_automatic_hold": False,
        "old_causal_artifacts_may_be_relabelled": False,
        "robot_transport_or_motor_commands": False,
        "integration_and_policy_retraining_required": True,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "contract_sha256": digest}


def _finite_vector(value, size, name):
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"{name} requires a finite float32 vector of size {size}")
    return array.copy()


def _quaternion(value, name):
    array = _finite_vector(value, 4, name)
    if abs(float(np.linalg.norm(array)) - 1) > 1e-4:
        raise ValueError(f"{name} requires normalized WXYZ")
    return array


def _relative_orientation_6d(measured, reference):
    w, x, y, z = measured.astype(np.float64) * np.array([1, -1, -1, -1])
    a, b, c, d = reference.astype(np.float64)
    q = np.array(
        [
            w * a - x * b - y * c - z * d,
            w * b + x * a + y * d - z * c,
            w * c - x * d + y * a + z * b,
            w * d + x * c - y * b + z * a,
        ]
    )
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class BufferedReference:
    emission_source_timestamp_s: float
    encoder_anchor_timestamp_s: float
    root_setpoint_timestamp_s: float
    lower_body240: np.ndarray
    virtual_vr21: np.ndarray
    anchor_root_position_w: np.ndarray
    anchor_root_quaternion_wxyz: np.ndarray
    next_root_position_w: np.ndarray

    def encoder267(self, current_measured_pelvis_quaternion_wxyz):
        measured = _quaternion(current_measured_pelvis_quaternion_wxyz, "measured pelvis")
        relative = _relative_orientation_6d(measured, self.anchor_root_quaternion_wxyz)
        return np.concatenate((self.lower_body240, self.virtual_vr21, relative)).astype(np.float32)

    def root_feedback9(self, current_position_w, current_velocity_w, current_quaternion_wxyz):
        position = _finite_vector(current_position_w, 3, "current position")
        velocity = _finite_vector(current_velocity_w, 3, "current velocity")
        quaternion = _quaternion(current_quaternion_wxyz, "current quaternion")
        desired_velocity = (self.next_root_position_w - self.anchor_root_position_w) / np.float32(PERIOD_S)
        return root_feedback_numpy(self.next_root_position_w, position, desired_velocity, velocity, quaternion)


class ReceivedSourceHorizon:
    """One same-clock source sample per push; no output until 11 samples arrive."""

    def __init__(self):
        self._samples = deque(maxlen=SAMPLE_COUNT)
        self._failed = False
        self._last_arrival = None

    def reset(self):
        """Discard reference history only. This cannot reset or arm a robot."""
        self._samples.clear()
        self._failed = False
        self._last_arrival = None

    def push(
        self,
        *,
        source_timestamp_s,
        arrival_timestamp_s,
        joint_names,
        joint_position23,
        root_position_w,
        root_quaternion_wxyz,
        virtual_source_vr21,
    ):
        if self._failed:
            raise RuntimeError("reference buffer requires explicit reset after rejection")
        try:
            timestamps = (source_timestamp_s, arrival_timestamp_s)
            if any(
                isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in timestamps
            ):
                raise ValueError("source and arrival timestamps must be finite same-clock seconds")
            age = arrival_timestamp_s - source_timestamp_s
            if age < -0.002 or age > 0.04:
                raise ValueError("newest received source is stale or future-dated")
            if self._last_arrival is not None and arrival_timestamp_s < self._last_arrival:
                raise ValueError("arrival clock moved backwards")
            if self._samples and abs(source_timestamp_s - self._samples[-1][0] - PERIOD_S) > 1e-6:
                raise ValueError("source samples must be strictly contiguous at 50 Hz")
            if tuple(joint_names) != tuple(HARDWARE_23_JOINT_NAMES):
                raise ValueError("reference must explicitly name native23 hardware order")
            q = _finite_vector(joint_position23, 23, "reference joints")
            p = _finite_vector(root_position_w, 3, "reference root position")
            quat = _quaternion(root_quaternion_wxyz, "reference root quaternion")
            vr = _finite_vector(virtual_source_vr21, 21, "reference VR")
            for i in range(3):
                _quaternion(vr[9 + 4 * i : 13 + 4 * i], "reference VR quaternion")
            self._samples.append((float(source_timestamp_s), q, p, quat, vr))
            self._last_arrival = float(arrival_timestamp_s)
        except (TypeError, ValueError):
            self._failed = True
            self._samples.clear()
            raise
        if len(self._samples) < SAMPLE_COUNT:
            return None
        samples = tuple(self._samples)
        joints = np.stack([row[1][:12] for row in samples])
        lower = np.concatenate(
            (joints[:-1].reshape(-1), ((joints[1:] - joints[:-1]) / np.float32(PERIOD_S)).reshape(-1))
        )
        if not np.isfinite(lower).all():
            self._failed = True
            self._samples.clear()
            raise ValueError("received joint differences overflowed the finite encoder boundary")
        return BufferedReference(
            emission_source_timestamp_s=samples[-1][0],
            encoder_anchor_timestamp_s=samples[0][0],
            root_setpoint_timestamp_s=samples[1][0],
            lower_body240=lower,
            virtual_vr21=samples[0][4].copy(),
            anchor_root_position_w=samples[0][2].copy(),
            anchor_root_quaternion_wxyz=samples[0][3].copy(),
            next_root_position_w=samples[1][2].copy(),
        )
