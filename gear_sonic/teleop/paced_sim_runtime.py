"""Wall-clock-paced, localhost-only SIM input; never a robot control interface.

Every admitted control integrates one unchanged 20-ms controller step. Input
waiting cannot stall the plant. A missed compute deadline latches balance and
rebases the wall schedule; it is reported as a timing FAILURE, never hidden by
catch-up steps or by changing MuJoCo's timestep. No automatic SONIC re-entry.
"""

from collections import deque
import hashlib
import json
import time

import zmq

from gear_sonic.teleop.clocked_sim_session import ClockedSimSession
from gear_sonic.utils.g1_true23_frozen_lora_live_teleop import (
    CONTROL_PERIOD_NS,
    LiveTransportFault,
    validate_live_packet,
)


class PacedSimSession:
    """Add actual-use freshness and measured execution deadlines to virtual SIM."""

    def __init__(self, controller, *, start_ns, now_ns=time.monotonic_ns):
        self.session = ClockedSimSession(controller, start_ns=start_ns)
        self.now_ns = now_ns
        self.next_deadline_ns = start_ns
        self.runtime_fault = None
        self.runtime_fault_tick = None
        self.rows = []

    def _timing_fault(self, reason):
        if self.runtime_fault is None:
            self.runtime_fault, self.runtime_fault_tick = reason, len(self.rows)
        if not self.session.controller.fallback_active:
            # Use the controller's already reviewed timeout activation. Keep the
            # distinct scheduler cause in runtime_fault, not a fabricated input
            # fault or a new unreviewed trigger added to the controller module.
            self.session.controller.activate_fallback("transport_timeout")

    def _input_fault(self, error):
        session = self.session
        if not session.controller.fallback_active:
            session.fault, session.fault_tick, session.fault_detail = error.fault, session.ticks, str(error)
            session.controller.activate_fallback(error.trigger)

    def tick(self, packet, *, input_fault=None, capture=lambda: None):
        session = self.session
        if session.stopped:
            raise RuntimeError("failed simulation cannot be resumed implicitly")
        deadline, started = self.next_deadline_ns, self.now_ns()
        if type(started) is not int or started < deadline:
            raise ValueError("caller must wait until the actual control deadline")
        if started - deadline >= CONTROL_PERIOD_NS:
            self._timing_fault("wake_missed_entire_control_period")
        if input_fault is not None:
            if not isinstance(input_fault, LiveTransportFault):
                raise TypeError("input fault must retain its explicit transport classification")
            self._input_fault(input_fault)
        if packet is not None and not session.controller.fallback_active:
            try:
                # Virtual deadlines alone understate actual age when the process
                # wakes late. Validate again here using this host's real clock.
                validate_live_packet(packet, previous=session.previous, maximum_age_ns=100_000_000, now_ns=started)
            except LiveTransportFault as error:
                self._input_fault(error)
        virtual = session.start_ns + session.ticks * CONTROL_PERIOD_NS
        try:
            return session.tick(packet, deadline_ns=virtual)
        except Exception:
            session.stopped = True
            raise
        finally:
            try:
                capture()  # Include recording overhead and failing physical states.
            except Exception:
                session.stopped = True
                raise
            finished = self.now_ns()
            if type(finished) is not int or finished < started:
                session.stopped = True
                raise RuntimeError("monotonic clock moved backwards during simulation")
            overrun = finished > deadline + CONTROL_PERIOD_NS
            if overrun and not session.stopped:
                self._timing_fault("control_finished_after_next_deadline")
            self.rows.append(
                dict(
                    deadline_ns=deadline,
                    started_ns=started,
                    finished_ns=finished,
                    wake_lateness_ns=started - deadline,
                    execution_ns=finished - started,
                    missed_compute_deadline=overrun,
                )
            )
            # On failure do not run rapid successive controls to erase lost wall
            # time. A rebased interval is visible and disqualifies timing.
            self.next_deadline_ns = finished + CONTROL_PERIOD_NS if overrun else deadline + CONTROL_PERIOD_NS


class LoopbackSubInbox:
    """Bounded FIFO; receive is always nonblocking after socket creation.

    Transport evidence retains message hashes and actual receive timestamps,
    not an unbounded second copy of each JSON body. Source packet validation
    happens on use; malformed JSON occupies its FIFO slot as a typed fault.
    """

    def __init__(self, context, endpoint, *, now_ns=time.monotonic_ns, sleep=time.sleep):
        prefix = "tcp://127.0.0.1:"
        port = endpoint.removeprefix(prefix)
        if not endpoint.startswith(prefix) or not port.isdecimal() or not 1 <= int(port) <= 65535:
            raise ValueError("paced SIM accepts only an explicit localhost TCP port")
        self.now_ns, self.sleep = now_ns, sleep
        self.queue, self.receipts = deque(), []
        self.overflow = None
        self.maximum_depth = 0
        self.socket = context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 8)
        self.socket.setsockopt(zmq.MAXMSGSIZE, 131072)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect(endpoint)

    def pump(self):
        # Bounded work even if the publisher floods the local socket.
        for _ in range(9):
            try:
                raw = self.socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            received = self.now_ns()
            self.receipts.append(dict(received_ns=received, sha256=hashlib.sha256(raw).hexdigest()))
            try:
                value = json.loads(raw)
            except (UnicodeError, ValueError) as error:
                value = LiveTransportFault("payload", f"invalid JSON from local publisher: {error}")
            if len(self.queue) == 8:
                self.overflow = LiveTransportFault("gap", "bounded input FIFO overflow; no silent packet dropping")
                self.queue.clear()
            self.queue.append(value)
            self.maximum_depth = max(self.maximum_depth, len(self.queue))

    def pop(self):
        if self.overflow is not None:
            return self.overflow
        return self.queue.popleft() if self.queue else None

    def wait_until(self, deadline_ns):
        while True:
            self.pump()
            remaining = deadline_ns - self.now_ns()
            if remaining <= 0:
                return
            # Never wait for an input packet. A missing source cannot postpone
            # the next simulation tick. No busy spin or OS real-time claim.
            self.sleep(min(remaining, 1_000_000) / 1e9)

    def startup_pair(self, *, timeout_ns=5_000_000_000):
        end = self.now_ns() + timeout_ns
        while len(self.queue) < 2 and self.overflow is None:
            if self.now_ns() >= end:
                raise LiveTransportFault("timeout", "startup ended before two source packets arrived")
            self.wait_until(min(end, self.now_ns() + 1_000_000))
        result = [self.pop(), self.pop()]
        for item in result:
            if isinstance(item, LiveTransportFault):
                raise item
        return result

    def close(self):
        self.socket.close()
