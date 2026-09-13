"""Missing input advances the plant; source resumption cannot clear a fault."""

from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.scripts.record_g1_true23_clocked_input_sim import scheduled_packet
from gear_sonic.teleop import clocked_sim_session as clocked
from gear_sonic.utils.g1_true23_frozen_lora_live_teleop import LiveTransportFault


def rig(monkeypatch):
    controller = SimpleNamespace(
        completed=0,
        data=SimpleNamespace(time=0.0),
        fallback_active=False,
        nominal_calls=0,
        balance_calls=0,
        fallback_trigger=None,
    )

    def activate(trigger):
        controller.fallback_active = True
        controller.fallback_trigger = trigger

    def step(_):
        assert controller.fallback_active
        controller.balance_calls += 1
        controller.completed += 1
        controller.data.time += 0.02
        return dict(base_height_m=0.75)

    def nominal(active, packet):
        assert not active.fallback_active
        active.nominal_calls += 1
        active.completed += 1
        active.data.time += 0.02
        return dict(base_height_m=0.75)

    def validate(packet, *, previous, maximum_age_ns, now_ns):
        if not isinstance(packet, dict):
            raise LiveTransportFault("payload", "invalid test payload")
        if previous is not None and packet["index"] != previous["index"] + 1:
            raise LiveTransportFault("gap", "non-contiguous source")
        if not 0 <= now_ns - packet["time"] <= maximum_age_ns:
            raise LiveTransportFault("stale", "test stale input")
        return dict(index=packet["index"])

    controller.activate_fallback = activate
    controller.step = step
    monkeypatch.setattr(clocked, "step_live_packet", nominal)
    monkeypatch.setattr(clocked, "validate_live_packet", validate)
    return controller, clocked.ClockedSimSession(controller, start_ns=200_000_000)


def test_missing_packet_integrates_each_tick_and_returned_input_cannot_rearm(monkeypatch):
    controller, session = rig(monkeypatch)
    for tick in range(30):
        now = 200_000_000 + tick * 20_000_000
        packet = dict(index=tick, time=now) if tick < 2 or tick >= 27 else None
        session.tick(packet, deadline_ns=now)
        assert controller.completed == tick + 1
        assert controller.data.time == pytest.approx((tick + 1) * 0.02)
    assert session.sonic_controls == controller.nominal_calls == 2
    assert session.fault == "timeout" and session.fault_tick == 2
    assert controller.balance_calls == 28
    assert session.packets_ignored_after_latch == 3


@pytest.mark.parametrize(
    "packet,fault",
    [
        (None, "timeout"),
        ([], "payload"),
        (dict(index=0, time=0), "stale"),
        (dict(index=0, time=300_000_000), "stale"),
    ],
)
def test_invalid_input_latches_balance_on_same_tick(monkeypatch, packet, fault):
    controller, session = rig(monkeypatch)
    session.tick(packet, deadline_ns=200_000_000)
    assert session.fault == fault and session.fault_tick == 0
    assert controller.balance_calls == controller.completed == 1
    assert controller.nominal_calls == 0
    assert controller.fallback_trigger == "transport_" + fault


def test_gap_detected_without_skipping_physics(monkeypatch):
    controller, session = rig(monkeypatch)
    session.tick(dict(index=0, time=200_000_000), deadline_ns=200_000_000)
    session.tick(dict(index=2, time=220_000_000), deadline_ns=220_000_000)
    assert controller.completed == 2 and session.fault == "gap"


@pytest.mark.parametrize("deadline", [199_000_000, 220_000_000, 200_000_000.0, True])
def test_skipped_or_duplicate_deadlines_cannot_hide_lost_time(monkeypatch, deadline):
    controller, session = rig(monkeypatch)
    with pytest.raises(ValueError, match="contiguous"):
        session.tick(None, deadline_ns=deadline)
    assert controller.completed == 0


def test_physics_failure_stops_without_retry_or_reset(monkeypatch):
    controller, session = rig(monkeypatch)

    def fail(_):
        controller.completed += 1
        controller.data.time += 0.02
        raise RuntimeError("physical test failure")

    controller.step = fail
    with pytest.raises(RuntimeError, match="physical test failure"):
        session.tick(None, deadline_ns=200_000_000)
    assert controller.completed == 1 and session.stopped
    with pytest.raises(RuntimeError, match="implicitly"):
        session.tick(None, deadline_ns=220_000_000)
    assert controller.completed == 1


def test_noop_physics_cannot_pass_a_clock_tick(monkeypatch):
    controller, session = rig(monkeypatch)
    controller.step = lambda _: {}
    with pytest.raises(RuntimeError, match="exactly 20 ms"):
        session.tick(None, deadline_ns=200_000_000)
    assert session.stopped


def test_schedule_preserves_source_and_pause_is_explicit():
    packets = [
        dict(
            pico_anchor_source_frame_index=i + 9, control_source_frame_index=i + 10, q_ref23_native=list(range(23))
        )
        for i in range(8)
    ]
    source = [dict(p, q_ref23_native=p["q_ref23_native"][:]) for p in packets]
    for scenario in ("nominal", "pause", "gap", "payload", "stale-start", "end-of-stream"):
        result = [
            scheduled_packet(packets, i, scenario=scenario, fault_control=2, pause_controls=3) for i in range(10)
        ]
        assert packets == source
        assert result[8:] == [None, None]
        if scenario == "pause":
            assert result[2:5] == [None] * 3 and result[5] == packets[5]
        if scenario == "gap":
            assert result[2]["control_source_frame_index"] == packets[2]["control_source_frame_index"] + 1


def test_substep_observer_forwards_once_and_does_not_change_control():
    calls = []
    data = SimpleNamespace(qpos=np.zeros(30), qvel=np.zeros(29), ctrl=np.arange(23.0), time=0.0)

    def integrate(model, state):
        calls.append(model)
        state.qpos[7:] += state.ctrl * 0.002
        state.time += 0.002
        return 7

    observer = clocked.SubstepObservation(SimpleNamespace(mj_step=integrate, constant=9))
    assert observer.constant == 9
    assert observer.mj_step("model", data) == 7
    assert calls == ["model"]
    np.testing.assert_array_equal(observer.arrays()["physics_command_torque"], np.arange(23.0)[None])
    np.testing.assert_array_equal(data.ctrl, np.arange(23.0))
    assert observer.arrays()["physics_time"].tolist() == [0.002]
