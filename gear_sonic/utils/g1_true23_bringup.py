"""Fail-closed G1 true23 command-path model.

This module deliberately has no DDS, Unitree SDK, or robot-network dependency.
It is the single command calculation and safety state machine exercised by the
offline proof.  The native publisher may only consume its ``LowCommand`` values
after its independently checked arming gate has succeeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import math
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np


JOINTS = 23
HARDWARE_SLOTS = np.asarray((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26))
RECOGNISED_MODE_MACHINES = frozenset((4,))
LOOPBACK_DOMAINS = frozenset((232,))
LOOPBACK_INTERFACES = frozenset(("lo", "lo0"))

# Bounds are intentionally conservative relative to the qualified 2 ms loop
# and to the existing 50 Hz BFM brake.  They are fixed here, not CLI knobs.
CONTROL_HZ = 500
DT = 1.0 / CONTROL_HZ
PREFLIGHT_STATE_MAX_AGE_S = 0.020
POLICY_TARGET_MAX_AGE_S = 0.100
OPERATOR_TOKEN_MAX_AGE_S = 60.0
OPERATOR_LIVENESS_MAX_AGE_S = 1.0
QUATERNION_NORM_TOLERANCE = 0.01
ZERO_TORQUE_MOTION_THRESHOLD_RAD = 0.002
DAMPING_KD = 1.0
OPERATING_KP = 20.0
OPERATING_KD = 1.0
HOLD_RAMP_S = 3.0
DEFAULT_POSE_RATE_RAD_S = 0.20
# This is the existing BFM bounded-brake increment; do not relax it here.
BRAKE_STEP_RAD = 0.100
MEASURED_POSITION_ERROR_RAD = 0.35
MEASURED_VELOCITY_RAD_S = 6.0
TILT_LIMIT_RAD = 0.35
ABORT_DAMPING_RAMP_S = 0.050
ABORT_ZERO_TORQUE_AFTER_S = 0.250
MAX_CONSECUTIVE_DEADLINE_MISSES = 1


class Stage(str, Enum):
    DISARMED = "disarmed"
    OBSERVE = "observe"
    ZERO_TORQUE = "zero_torque"
    DAMPING = "damping"
    POSITION_HOLD = "position_hold"
    DEFAULT_POSE = "default_pose"
    POLICY = "policy"
    ABORT_DAMPING = "abort_damping"
    ABORT_ZERO_TORQUE = "abort_zero_torque"
    ABORTED = "aborted"


@dataclass(frozen=True)
class LiveState:
    received_monotonic_s: float
    q: np.ndarray
    dq: np.ndarray
    quaternion_wxyz: np.ndarray
    mode_machine: int
    motor_slots: int = 35

    def __post_init__(self):
        # Retain malformed telemetry long enough for the safety gate to name
        # it as an abort reason; never pass it to a LowCommand constructor.
        object.__setattr__(self, "q", _state_vector(self.q, "q"))
        object.__setattr__(self, "dq", _state_vector(self.dq, "dq"))
        quat = np.asarray(self.quaternion_wxyz, dtype=np.float64)
        if quat.shape != (4,):
            raise ValueError("quaternion_wxyz must have shape [4]")
        object.__setattr__(self, "quaternion_wxyz", quat.copy())


@dataclass(frozen=True)
class LowCommand:
    q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray
    mode_pr: int
    mode_machine: int
    stage: Stage

    def __post_init__(self):
        for name in ("q", "kp", "kd", "tau"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))


@dataclass(frozen=True)
class Event:
    timestamp_s: float
    kind: str
    detail: str


def _vector(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (JOINTS,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite [23]")
    return result.copy()


def _state_vector(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (JOINTS,):
        raise ValueError(f"{name} must have shape [23]")
    return result.copy()


def is_loopback_endpoint(domain: int, interface: str) -> bool:
    return domain in LOOPBACK_DOMAINS and interface in LOOPBACK_INTERFACES


def validate_real_arm_request(
    *, arm_flag: bool, domain_was_explicit: bool, interface_was_explicit: bool,
    dds_domain: int, dds_interface: str, token_file: str | Path | None,
    supplied_token: str | None, state: LiveState | None, lower: Sequence[float],
    upper: Sequence[float], now_s: float | None = None,
) -> list[str]:
    """Return every independent refusal, never silently downgrade a request."""

    failures: list[str] = []
    now_s = time.time() if now_s is None else now_s
    if not arm_flag:
        failures.append("missing --arm")
    if not domain_was_explicit:
        failures.append("--dds-domain must be explicit")
    if not interface_was_explicit:
        failures.append("--dds-interface must be explicit")
    if is_loopback_endpoint(dds_domain, dds_interface):
        failures.append("real arming requires a non-loopback endpoint")
    if token_file is None:
        failures.append("missing operator token file")
    elif supplied_token is None:
        failures.append("missing operator token")
    else:
        try:
            path = Path(token_file)
            contents = path.read_text(encoding="utf-8").strip()
            age = now_s - path.stat().st_mtime
            if not (math.isfinite(age) and 0.0 <= age <= OPERATOR_TOKEN_MAX_AGE_S):
                failures.append("operator token file is older than 60 s")
            if not contents or not hmac.compare_digest(contents, supplied_token):
                failures.append("operator token does not match token file")
        except OSError:
            failures.append("operator token file cannot be read")
    failures.extend(preflight_failures(state, lower, upper, now_s))
    return failures


def preflight_failures(state: LiveState | None, lower: Sequence[float], upper: Sequence[float], now_s: float) -> list[str]:
    lower = _vector(lower, "lower")
    upper = _vector(upper, "upper")
    failures: list[str] = []
    if np.any(lower >= upper):
        return ["invalid model joint limits"]
    if state is None:
        return ["no live LowState"]
    if state.motor_slots < int(HARDWARE_SLOTS.max()) + 1:
        failures.append("LowState joint layout is incomplete")
    if state.mode_machine not in RECOGNISED_MODE_MACHINES:
        failures.append(f"unrecognised mode_machine {state.mode_machine}")
    if not np.isfinite(state.quaternion_wxyz).all() or abs(np.linalg.norm(state.quaternion_wxyz) - 1.0) > QUATERNION_NORM_TOLERANCE:
        failures.append("IMU quaternion is non-finite or not unit length")
    if not np.isfinite(state.q).all() or np.any(state.q < lower) or np.any(state.q > upper):
        failures.append("measured joint is outside model limits")
    age = now_s - state.received_monotonic_s
    if not math.isfinite(age) or age < 0.0 or age > PREFLIGHT_STATE_MAX_AGE_S:
        failures.append("LowState is older than 20 ms")
    return failures


def crc32_unitree_words(raw: bytes) -> int:
    """Unitree HG CRC core, over message words excluding the CRC word."""
    if len(raw) % 4:
        raise ValueError("CRC input must contain complete words")
    crc = 0xFFFFFFFF
    for offset in range(0, len(raw), 4):
        data = int.from_bytes(raw[offset:offset + 4], "little")
        bit = 1 << 31
        for _ in range(32):
            crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
            if data & bit:
                crc ^= 0x04C11DB7
            bit >>= 1
    return crc


@dataclass
class BringupLadder:
    lower: Sequence[float]
    upper: Sequence[float]
    default_q: Sequence[float]
    operating_kp: Sequence[float] | float = OPERATING_KP
    operating_kd: Sequence[float] | float = OPERATING_KD
    stage: Stage = Stage.DISARMED
    events: list[Event] = field(default_factory=list)
    _held_q: np.ndarray | None = field(default=None, init=False, repr=False)
    _previous_q: np.ndarray | None = field(default=None, init=False, repr=False)
    _stage_started_s: float = field(default=0.0, init=False)
    _abort_reason: str | None = field(default=None, init=False)
    _deadline_misses: int = field(default=0, init=False)

    def __post_init__(self):
        self.lower = _vector(self.lower, "lower")
        self.upper = _vector(self.upper, "upper")
        self.default_q = _vector(self.default_q, "default_q")
        self.operating_kp = _gain_vector(self.operating_kp, "operating_kp")
        self.operating_kd = _gain_vector(self.operating_kd, "operating_kd")
        if np.any(self.lower >= self.upper) or np.any(self.default_q < self.lower) or np.any(self.default_q > self.upper):
            raise ValueError("invalid model joint limits or default pose")

    @property
    def aborted(self) -> bool:
        return self.stage in (Stage.ABORT_DAMPING, Stage.ABORT_ZERO_TORQUE, Stage.ABORTED)

    def arm(self, state: LiveState, now_s: float) -> None:
        if self.stage is not Stage.DISARMED:
            raise RuntimeError("operator must start a new process to arm again")
        failures = preflight_failures(state, self.lower, self.upper, now_s)
        if failures:
            raise RuntimeError("pre-flight refused: " + "; ".join(failures))
        self._enter(Stage.OBSERVE, now_s, "operator armed; observe only")

    def advance(self, state: LiveState, now_s: float) -> None:
        if self.aborted:
            raise RuntimeError("abort is latched; operator must arm again from the beginning")
        next_stage = {
            Stage.OBSERVE: Stage.ZERO_TORQUE,
            Stage.ZERO_TORQUE: Stage.DAMPING,
            Stage.DAMPING: Stage.POSITION_HOLD,
            Stage.POSITION_HOLD: Stage.DEFAULT_POSE,
            Stage.DEFAULT_POSE: Stage.POLICY,
        }.get(self.stage)
        if next_stage is None:
            raise RuntimeError("no higher stage is available")
        failures = preflight_failures(state, self.lower, self.upper, now_s)
        if failures:
            raise RuntimeError("advance refused: " + "; ".join(failures))
        if next_stage is Stage.POSITION_HOLD:
            self._held_q = state.q.copy()  # sampled exactly once at entry
            self._previous_q = state.q.copy()
        self._enter(next_stage, now_s, "fresh explicit operator advance")

    def abort(self, reason: str, now_s: float) -> None:
        if not self.aborted:
            self._abort_reason = reason
            self._enter(Stage.ABORT_DAMPING, now_s, "abort: " + reason)

    def command(self, state: LiveState, now_s: float, policy_q: Sequence[float] | None = None,
                policy_received_s: float | None = None, deadline_missed: bool = False,
                operator_liveness_s: float | None = None) -> LowCommand | None:
        if self.stage is Stage.DISARMED:
            return None
        self._check_abort(state, now_s, policy_q, policy_received_s, deadline_missed, operator_liveness_s)
        if self.stage is Stage.ABORT_DAMPING and now_s - self._stage_started_s >= ABORT_DAMPING_RAMP_S:
            self._enter(Stage.ABORT_ZERO_TORQUE, now_s, "damping ramp complete")
        if self.stage is Stage.ABORT_ZERO_TORQUE and now_s - self._stage_started_s >= ABORT_ZERO_TORQUE_AFTER_S:
            self._enter(Stage.ABORTED, now_s, "zero torque continuing; re-arm required")
        mode_machine = state.mode_machine
        q = state.q.copy() if self._previous_q is None else self._previous_q.copy()
        kp = np.zeros(JOINTS); kd = np.zeros(JOINTS); tau = np.zeros(JOINTS)
        if self.stage is Stage.OBSERVE:
            return None
        if self.stage in (Stage.ZERO_TORQUE, Stage.ABORT_ZERO_TORQUE, Stage.ABORTED):
            pass
        elif self.stage in (Stage.DAMPING, Stage.ABORT_DAMPING):
            kd.fill(DAMPING_KD * self._abort_damping_alpha(now_s))
        elif self.stage is Stage.POSITION_HOLD:
            assert self._held_q is not None
            q = self._held_q.copy(); kp = self.operating_kp * min((now_s - self._stage_started_s) / HOLD_RAMP_S, 1.0); kd = self.operating_kd.copy()
        elif self.stage is Stage.DEFAULT_POSE:
            assert self._held_q is not None
            q = self._slew(self.default_q, DEFAULT_POSE_RATE_RAD_S * DT); kp = self.operating_kp.copy(); kd = self.operating_kd.copy()
        elif self.stage is Stage.POLICY:
            if policy_q is None:
                self.abort("missing policy target", now_s)
                return self.command(state, now_s)
            q = self._slew(_vector(policy_q, "policy_q"), BRAKE_STEP_RAD); kp = self.operating_kp.copy(); kd = self.operating_kd.copy()
        self._previous_q = q.copy()
        return LowCommand(q, kp, kd, tau, mode_pr=0, mode_machine=mode_machine, stage=self.stage)

    def _check_abort(self, state: LiveState, now_s: float, policy_q: Sequence[float] | None,
                     policy_received_s: float | None, deadline_missed: bool, operator_liveness_s: float | None) -> None:
        if self.aborted:
            return
        failures = preflight_failures(state, self.lower, self.upper, now_s)
        if failures:
            self.abort(failures[0], now_s); return
        values = [state.q, state.dq, state.quaternion_wxyz]
        if any(not np.isfinite(value).all() for value in values):
            self.abort("non-finite value in command path", now_s); return
        if deadline_missed: self._deadline_misses += 1
        else: self._deadline_misses = 0
        if self._deadline_misses > MAX_CONSECUTIVE_DEADLINE_MISSES:
            self.abort("more than one consecutive deadline miss", now_s); return
        if operator_liveness_s is None or now_s - operator_liveness_s > OPERATOR_LIVENESS_MAX_AGE_S:
            self.abort("operator liveness lost", now_s); return
        tilt = math.acos(float(np.clip(1.0 - 2.0 * (state.quaternion_wxyz[1] ** 2 + state.quaternion_wxyz[2] ** 2), -1.0, 1.0)))
        if tilt > TILT_LIMIT_RAD: self.abort("estimated tilt exceeds limit", now_s); return
        if np.max(np.abs(state.dq)) > MEASURED_VELOCITY_RAD_S:
            self.abort("measured joint velocity exceeds limit", now_s); return
        if self._previous_q is not None and np.max(np.abs(state.q - self._previous_q)) > MEASURED_POSITION_ERROR_RAD:
            self.abort("measured joint position error exceeds limit", now_s); return
        if self.stage is Stage.POLICY:
            if policy_q is None or policy_received_s is None or now_s - policy_received_s > POLICY_TARGET_MAX_AGE_S:
                self.abort("policy target stale beyond 100 ms", now_s); return
            target = _vector(policy_q, "policy_q")
            if np.any(target < self.lower) or np.any(target > self.upper):
                self.abort("commanded joint outside model limits", now_s); return
            if self._previous_q is not None and np.max(np.abs(target - self._previous_q)) > BRAKE_STEP_RAD + 1e-12:
                # The brake would slew it, but this is an input safety fault, not a clamp.
                self.abort("commanded step exceeds existing brake bound", now_s)

    def _slew(self, requested: np.ndarray, maximum_step: float) -> np.ndarray:
        previous = self._previous_q if self._previous_q is not None else requested
        return np.clip(previous + np.clip(requested - previous, -maximum_step, maximum_step), self.lower, self.upper)

    def _abort_damping_alpha(self, now_s: float) -> float:
        if self.stage is not Stage.ABORT_DAMPING:
            return 1.0
        return min(max((now_s - self._stage_started_s) / ABORT_DAMPING_RAMP_S, 0.0), 1.0)

    def _enter(self, stage: Stage, now_s: float, detail: str) -> None:
        self.stage = stage; self._stage_started_s = now_s; self.events.append(Event(now_s, stage.value, detail))


def command_digest(command: LowCommand) -> str:
    """Stable evidence digest; native code separately computes DDS-object CRC."""
    payload = np.concatenate((command.q, command.kp, command.kd, command.tau)).astype("<f4").tobytes()
    return hashlib.sha256(payload).hexdigest()


def _gain_vector(value: Sequence[float] | float, name: str) -> np.ndarray:
    result = np.full(JOINTS, value, dtype=np.float64) if np.isscalar(value) else _vector(value, name)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative finite [23]")
    return result
