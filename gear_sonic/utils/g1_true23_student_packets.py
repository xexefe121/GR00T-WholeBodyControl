"""Received-sample buffer for native23 student features, without a controller.

This module admits canonical float64 prepared samples; it does not certify upstream retargeting or
raw-pose derivative support. No model, transport, physics or hardware API lives
here. A controller must independently keep physics running and verify standing
before explicitly rearming this input gate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

DT = .02
HORIZON = 38
CLOCK_TOLERANCE = 1e-7
NATIVE_SHAPES = {
    'joint_pos': (23,), 'joint_vel': (23,), 'body_pos_w': (24, 3),
    'body_quat_w': (24, 4), 'body_lin_vel_w': (24, 3), 'body_ang_vel_w': (24, 3),
}


@dataclass(frozen=True)
class StudentPacket:
    epoch: int
    sequence: int
    source_time: float
    original_native: Mapping[str, np.ndarray]
    retargeted_native: Mapping[str, np.ndarray]
    task_position_w: np.ndarray
    task_quaternion_wxyz: np.ndarray
    final: bool = False


@dataclass(frozen=True)
class ReceivedStudentSample:
    packet: StudentPacket
    received_at: float


@dataclass(frozen=True)
class ReceivedStudentWindow:
    samples: tuple[ReceivedStudentSample, ...]
    previous: ReceivedStudentSample | None
    consumed_at: float

    @property
    def feature_frame(self):
        return int(self.previous is not None)

    @property
    def playback_latency(self):
        return self.consumed_at - self.samples[0].received_at

    def arrays(self):
        """Return local arrays plus the true previous sample for task derivatives.

        Feed ``feature_frame`` to GoalFeatures. A short last window contains its
        actual final sample; GoalFeatures' existing index clipping implements
        padding only because an explicit final packet was already received.
        """
        rows = ((self.previous,) if self.previous is not None else ()) + self.samples
        packets = [row.packet for row in rows]
        native = [
            {name: np.stack([getattr(packet, source)[name] for packet in packets]) for name in NATIVE_SHAPES}
            for source in ('original_native', 'retargeted_native')
        ]
        original_tasks = {
            'source_task_position_w': np.stack([packet.task_position_w for packet in packets]),
            'source_task_quaternion_wxyz': np.stack([packet.task_quaternion_wxyz for packet in packets]),
        }
        return *native, original_tasks


def _array(value, shape, name, *, quaternion=False):
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError('packet_shape:' + name)
    if array.dtype != np.dtype(np.float64):
        raise ValueError('packet_dtype_requires_float64:' + name)
    if not np.isfinite(array).all():
        raise ValueError('packet_nonfinite:' + name)
    if quaternion and np.max(np.abs(np.linalg.norm(array, axis=-1) - 1)) > 1e-5:
        raise ValueError('packet_quaternion_norm:' + name)
    # A bytes-backed array cannot regain write access through setflags.
    return np.frombuffer(np.asarray(array, dtype=np.float64).tobytes(), dtype=np.float64).reshape(shape)


def _native(fields, source):
    if not isinstance(fields, Mapping) or set(fields) != set(NATIVE_SHAPES):
        raise ValueError('packet_fields:' + source)
    return MappingProxyType({name: _array(fields[name], shape, source + '.' + name,
        quaternion=name == 'body_quat_w') for name, shape in NATIVE_SHAPES.items()})


def _finite_scalar(value):
    try:
        return (not isinstance(value, (bool, np.bool_)) and
            isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(value))
    except (OverflowError, ValueError, TypeError):
        return False


class StudentPacketGate:
    """Strict FIFO admission, bounded memory and a persistent input fault latch."""

    def __init__(self, *, now=0., stale_seconds=.1, max_buffer=76):
        if not _finite_scalar(now) or now < 0:
            raise ValueError('invalid initial receive clock')
        if not _finite_scalar(stale_seconds) or stale_seconds < DT:
            raise ValueError('invalid freshness bound')
        if type(max_buffer) is not int or max_buffer < HORIZON:
            raise ValueError('buffer must hold the full38-sample window')
        self.stale_seconds, self.max_buffer = float(stale_seconds), max_buffer
        self.epoch = 0
        self.events = []
        self._reset(now)

    def _reset(self, now):
        self.epoch_started_at = self.last_clock = self.last_received_at = float(now)
        self.last_sequence = -1
        self.received = self.consumed = self.ignored = self.rejected = 0
        self.final_received = self.final_consumed = False
        self.fault = None
        self._queue = deque()
        self._previous = None

    def _latch(self, reason, now):
        if self.fault is None:
            self.fault = dict(reason=reason, time=float(now), epoch=self.epoch,
                last_sequence=self.last_sequence, consumed=self.consumed)
            self.events.append(dict(kind='latched_input_fault', **self.fault))
            self._queue.clear()
            self._previous = None

    def _clock(self, now):
        if not _finite_scalar(now) or now < self.last_clock:
            self._latch('receive_clock_invalid_or_regressed', self.last_clock)
            return False
        self.last_clock = float(now)
        return True

    def receive(self, packet, now):
        if self.fault is not None:
            self.ignored += 1
            return False
        if not self._clock(now):
            self.rejected += 1
            return False
        try:
            if not isinstance(packet, StudentPacket):
                raise ValueError('packet_type')
            if type(packet.epoch) is not int or type(packet.sequence) is not int or packet.sequence < 0:
                raise ValueError('packet_integer_identity')
            if packet.epoch < self.epoch:
                self.ignored += 1
                return False
            if packet.epoch != self.epoch:
                raise ValueError('packet_future_epoch')
            if type(packet.final) is not bool:
                raise ValueError('packet_final_flag')
            if self.final_received:
                raise ValueError('packet_after_final')
            if packet.sequence != self.last_sequence + 1:
                raise ValueError('packet_sequence_gap_or_reorder')
            if not _finite_scalar(packet.source_time) or packet.source_time < 0:
                raise ValueError('packet_source_clock_invalid')
            if abs(packet.source_time - packet.sequence * DT) > CLOCK_TOLERANCE:
                raise ValueError('packet_source_clock_not_50hz')
            age = now - self.epoch_started_at - packet.source_time
            if age < -CLOCK_TOLERANCE:
                raise ValueError('packet_from_future')
            if age > self.stale_seconds + CLOCK_TOLERANCE:
                raise ValueError('packet_arrived_stale')
            if len(self._queue) >= self.max_buffer:
                raise ValueError('packet_buffer_overflow')
            accepted = StudentPacket(packet.epoch, packet.sequence, float(packet.source_time),
                _native(packet.original_native, 'original_native'),
                _native(packet.retargeted_native, 'retargeted_native'),
                _array(packet.task_position_w, (3, 3), 'task_position_w'),
                _array(packet.task_quaternion_wxyz, (3, 4), 'task_quaternion_wxyz', quaternion=True),
                packet.final)
            self._queue.append(ReceivedStudentSample(accepted, float(now)))
            self.last_sequence = packet.sequence
            self.last_received_at = float(now)
            self.received += 1
            self.final_received = packet.final
            return True
        except (ValueError, TypeError, OverflowError) as exc:
            self.rejected += 1
            self._latch(str(exc), now)
            return False

    def consume(self, now):
        """Atomically return one admitted window and advance exactly one sample."""
        if self.fault is not None or not self._clock(now):
            return None
        if not self.final_received and now - self.last_received_at > self.stale_seconds + CLOCK_TOLERANCE:
            self._latch('packet_timeout', now)
            return None
        if not self._queue or (len(self._queue) < HORIZON and not self.final_received):
            return None
        samples = tuple(list(self._queue)[:HORIZON])
        result = ReceivedStudentWindow(samples, self._previous, float(now))
        self._previous = self._queue.popleft()
        self.consumed += 1
        self.final_consumed = self._previous.packet.final
        return result

    def rearm(self, now):
        """Explicit input-epoch reset; caller must first verify physical standing.

        No plant/action history or coordinate calibration is modified here.
        """
        if self.fault is None:
            raise ValueError('rearm requires a latched input fault')
        if not _finite_scalar(now) or now < self.last_clock:
            raise ValueError('invalid rearm clock')
        self.epoch += 1
        self._reset(now)
        self.events.append(dict(kind='explicit_input_rearm', epoch=self.epoch, time=float(now)))
