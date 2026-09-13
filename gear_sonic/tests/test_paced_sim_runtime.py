"""Deterministic scheduling faults, actual-age admission and bounded input."""

from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.teleop import clocked_sim_session as clocked
from gear_sonic.teleop import paced_sim_runtime as paced
from gear_sonic.scripts import run_g1_true23_paced_sim as driver
from gear_sonic.utils.g1_true23_frozen_lora_live_teleop import LiveTransportFault


def rig(monkeypatch, *, age=40_000_000):
    clock = SimpleNamespace(now=200_000_000, step_ns=2_000_000)
    c = SimpleNamespace(completed=0, data=SimpleNamespace(time=0.0), fallback_active=False, triggers=[], modes=[])

    def activate(trigger):
        c.fallback_active = True
        c.triggers.append(trigger)

    def step(_):
        clock.now += clock.step_ns
        c.modes.append(c.fallback_active)
        c.completed += 1
        c.data.time += 0.02
        return {}

    def validate(packet, *, previous, maximum_age_ns, now_ns):
        if not isinstance(packet, dict):
            raise LiveTransportFault("payload", "test payload")
        if previous is not None and packet["i"] != previous["i"] + 1:
            raise LiveTransportFault("gap", "test gap")
        if not 0 <= now_ns - packet["time"] <= maximum_age_ns:
            raise LiveTransportFault("stale", "test age")
        return packet.copy()

    c.activate_fallback, c.step = activate, step
    monkeypatch.setattr(clocked, "validate_live_packet", validate)
    monkeypatch.setattr(paced, "validate_live_packet", validate)
    monkeypatch.setattr(clocked, "step_live_packet", lambda controller, packet: step(packet))
    session = paced.PacedSimSession(c, start_ns=clock.now, now_ns=lambda: clock.now)

    def packet(i):
        return dict(i=i, time=200_000_000 + i * 20_000_000 - age)

    return clock, c, session, packet


def test_complete_stream_then_disconnect_keeps_physics(monkeypatch):
    clock, controller, runtime, packet = rig(monkeypatch)
    for i in range(12):
        clock.now = runtime.next_deadline_ns
        runtime.tick(packet(i) if i < 5 or i > 8 else None)
    assert controller.completed == 12
    assert controller.data.time == pytest.approx(0.24)
    assert controller.modes == [False] * 5 + [True] * 7
    assert controller.triggers == ["transport_timeout"]
    assert runtime.session.packets_ignored_after_latch == 3
    assert runtime.runtime_fault is None
    assert all(r["execution_ns"] == 2_000_000 for r in runtime.rows)


def test_actual_age_not_virtual_deadline_rejects_stale_input(monkeypatch):
    clock, controller, runtime, packet = rig(monkeypatch, age=99_000_000)
    clock.now += 2_000_000
    runtime.tick(packet(0))
    assert runtime.session.fault == "stale"
    assert controller.modes == [True]
    assert runtime.runtime_fault is None


def test_overrun_latches_balance_and_never_catches_up(monkeypatch):
    clock, controller, runtime, packet = rig(monkeypatch)
    clock.step_ns = 25_000_000
    runtime.tick(packet(0))
    assert runtime.runtime_fault == "control_finished_after_next_deadline"
    assert runtime.runtime_fault_tick == 0
    assert controller.modes == [False]
    assert controller.triggers == ["transport_timeout"]
    assert runtime.next_deadline_ns == 245_000_000
    clock.now, clock.step_ns = runtime.next_deadline_ns, 1_000_000
    runtime.tick(packet(1))
    assert controller.modes == [False, True]
    assert runtime.rows[1]["started_ns"] - runtime.rows[0]["started_ns"] == 45_000_000


def test_whole_period_late_wake_never_runs_sonic(monkeypatch):
    clock, controller, runtime, packet = rig(monkeypatch)
    clock.now += 20_000_000
    runtime.tick(packet(0))
    assert runtime.runtime_fault == "wake_missed_entire_control_period"
    assert controller.modes == [True]


def test_cannot_run_early_or_retry_failed_physics(monkeypatch):
    clock, controller, runtime, packet = rig(monkeypatch)
    clock.now -= 1
    with pytest.raises(ValueError, match="wait until"):
        runtime.tick(packet(0))
    assert controller.completed == 0
    clock.now += 1

    def fail(_):
        raise RuntimeError("injected physical gate")

    controller.step = fail
    with pytest.raises(RuntimeError, match="physical gate"):
        runtime.tick(None)
    assert runtime.session.stopped and len(runtime.rows) == 1
    with pytest.raises(RuntimeError, match="implicitly"):
        runtime.tick(None)


def test_capture_overhead_is_part_of_compute_budget(monkeypatch):
    clock, controller, runtime, packet = rig(monkeypatch)

    def slow_capture():
        clock.now += 21_000_000

    runtime.tick(packet(0), capture=slow_capture)
    assert runtime.rows[0]["execution_ns"] == 23_000_000
    assert controller.fallback_active


@pytest.mark.parametrize("fault", ["timeout", "gap", "payload", "stale"])
def test_external_input_error_keeps_classification(monkeypatch, fault):
    clock, controller, runtime, packet = rig(monkeypatch)
    runtime.tick(None, input_fault=LiveTransportFault(fault, "injected input error"))
    assert runtime.session.fault == fault and controller.modes == [True]


@pytest.mark.parametrize(
    "endpoint",
    [
        "tcp://192.168.123.161:5557",
        "tcp://localhost:5557",
        "tcp://127.0.0.1:0",
        "tcp://127.0.0.1:65536",
        "ipc:///tmp/robot",
        "tcp://127.0.0.1:7/path",
    ],
)
def test_non_loopback_endpoints_rejected_before_socket(endpoint):
    with pytest.raises(ValueError, match="localhost"):
        paced.LoopbackSubInbox(None, endpoint)


def test_wait_does_not_depend_on_packets_and_fifo_is_bounded():
    clock = SimpleNamespace(now=0)
    messages = [b"{}"] * 10

    class Socket:
        def setsockopt(self, *args):
            pass

        def connect(self, endpoint):
            pass

        def recv(self, *, flags):
            assert flags == paced.zmq.NOBLOCK
            if not messages:
                raise paced.zmq.Again()
            return messages.pop(0)

    def sleep(seconds):
        clock.now += int(round(seconds * 1e9))

    inbox = paced.LoopbackSubInbox(
        SimpleNamespace(socket=lambda _: Socket()), "tcp://127.0.0.1:5557", now_ns=lambda: clock.now, sleep=sleep
    )
    inbox.wait_until(20_000_000)
    assert clock.now == 20_000_000
    assert isinstance(inbox.pop(), LiveTransportFault)
    assert inbox.overflow.fault == "gap" and inbox.maximum_depth <= 8
    assert len(inbox.receipts) == 10
    assert np.diff([r["received_ns"] for r in inbox.receipts]).min() >= 0


def test_expensive_model_hash_precedes_stream_receipt(monkeypatch):
    calls = []
    controller = SimpleNamespace(completed=0, data=SimpleNamespace(time=0.0), model="model")
    monkeypatch.setattr(driver, "compiled_model_sha256", lambda model: calls.append(model) or "a" * 64)
    prepared = driver.prepare_stream(controller)
    assert prepared.controller is controller and calls == ["model"]

    class Inbox:
        def startup_pair(self):
            assert calls == ["model"]  # No second serialization after subscription.
            raise LiveTransportFault("timeout", "fixture ends before initialization")

    with pytest.raises(LiveTransportFault):
        driver.run_stream(prepared, Inbox(), source_controls=2, tail_controls=1)
    with pytest.raises(ValueError, match="prepare"):
        driver.run_stream(controller, Inbox(), source_controls=2, tail_controls=1)
    controller.completed = 1
    with pytest.raises(ValueError, match="fresh"):
        driver.prepare_stream(controller)
