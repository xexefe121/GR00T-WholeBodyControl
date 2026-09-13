"""The live SIM receiver must not hide a failed controller or mix model pairs."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.scripts import run_g1_true23_frozen_lora_live_teleop as live


def rig(monkeypatch, tmp_path, *, fault=None):
    qpos = np.zeros(30)
    qpos[2:4] = [0.75, 1]
    controller = SimpleNamespace(
        data=SimpleNamespace(qpos=qpos, qvel=np.zeros(29), time=0.0),
        completed=0,
        fallback_active=False,
        fallback_trigger=None,
        fallback_transition=None,
        fallback_tilt_trigger_rad=0.50,
        fallback_policy=SimpleNamespace(query_count=0),
    )
    calls = dict(calibrations=0, closed=False, terminated=False, validations=0)
    socket = SimpleNamespace(
        setsockopt=lambda *args: None,
        connect=lambda *args: None,
        close=lambda: calls.update(closed=True),
    )
    context = SimpleNamespace(socket=lambda *args: socket, term=lambda: calls.update(terminated=True))
    monkeypatch.setattr(live.zmq, "Context", lambda **kwargs: context)
    monkeypatch.setattr(live.zmq, "Poller", lambda: SimpleNamespace(register=lambda *args: None))
    identity = dict(
        diagnostic_pair=dict(validated=True),
        encoder_sha256="a" * 64,
        decoder_sha256="b" * 64,
        legacy_unpaired_diagnostic=False,
    )
    monkeypatch.setattr(live, "build_recording_controller", lambda args: (controller, identity))
    monkeypatch.setattr(live, "sha256_file", lambda path: "c" * 64)
    monkeypatch.setattr(live, "initialize_live_controller", lambda *args: None)
    monkeypatch.setattr(
        live,
        "preserve_calibrated_source_orientation",
        lambda *args: calls.update(calibrations=calls["calibrations"] + 1),
    )
    packets = [dict(index=10 + i) for i in range(4)]

    def receive(*args):
        if fault in ("transport", "transport_and_physics") and len(packets) == 2:
            raise live.LiveTransportFault("timeout", "test timeout")
        return packets.pop(0)

    def validate(packet, **kwargs):
        calls["validations"] += 1
        if fault == "startup_aged" and calls["validations"] == 3:
            raise live.LiveTransportFault("stale", "aged during startup")
        return dict(control_index=packet["index"], age_ns=20_000_000)

    def step(*args):
        controller.completed += 1
        controller.data.time += 0.02
        if controller.completed == 2 and fault in ("physics", "nonfinite", "latched"):
            controller.fallback_active = True
            controller.fallback_trigger = "base_tilt"
            controller.fallback_transition = 1
            if fault != "latched":
                controller.data.qpos[2] = np.nan if fault == "nonfinite" else 0.4
                raise RuntimeError("fallback physical gate failed")
        return dict(base_height_m=0.75, base_tilt_rad=0.1)

    def hold(controller, *, trigger, steps):
        controller.fallback_active = True
        controller.fallback_trigger = trigger
        controller.fallback_transition = controller.completed
        count = 1 if fault == "transport_and_physics" else steps
        controller.completed += count
        controller.data.time += count * 0.02
        controller.fallback_policy.query_count += count
        if fault == "transport_and_physics":
            controller.data.qpos[2] = 0.4
            raise RuntimeError("failed during transport fallback")
        return dict(
            fallback_hold_transitions=steps,
            fallback_stable=True,
            fallback_minimum_base_height_m=0.75,
            fallback_maximum_base_tilt_rad=0.1,
            fallback_maximum_absolute_torque_nm=1.0,
        )

    monkeypatch.setattr(live, "_receive", receive)
    monkeypatch.setattr(live, "validate_live_packet", validate)
    monkeypatch.setattr(live, "step_live_packet", step)
    monkeypatch.setattr(live, "hold_transport_fallback", hold)
    arguments = [
        "--repository-root",
        str(tmp_path),
        "--encoder-report",
        "encoder.json",
        "--decoder-report",
        "decoder.json",
        "--steps",
        "4",
        "--fallback-hold-steps",
        "2",
        "--output",
        str(tmp_path / "report.json"),
        "--trace-output",
        str(tmp_path / "trace.npz"),
    ]
    if fault in ("transport", "transport_and_physics"):
        arguments += ["--expected-transport-fault", "timeout"]
    return arguments, controller, calls


@pytest.mark.parametrize(
    "fault", [None, "physics", "nonfinite", "latched", "transport", "transport_and_physics", "startup_aged"]
)
def test_live_reports_physics_and_transport_outcomes_without_hidden_retries(monkeypatch, tmp_path, fault):
    arguments, controller, calls = rig(monkeypatch, tmp_path, fault=fault)
    result = live.main(arguments)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["passed"] is (fault in (None, "transport"))
    assert result == int(not report["passed"])
    assert calls["closed"] and calls["terminated"]
    assert calls["calibrations"] == 1
    assert report["kind"] == "g1_true23_frozen_lora_live_teleop_v2"
    assert report["diagnostic_pair"]["validated"] is True
    assert report["authorization"]["hardware_authorized"] is False
    assert report["tracking_fidelity_qualified"] is False
    with np.load(tmp_path / "trace.npz", allow_pickle=False) as archive:
        poses, times = archive["qpos"], archive["simulation_time"]
    if fault in ("physics", "nonfinite"):
        assert controller.completed == 2
        assert report["completed_live_transitions"] == 1
        assert report["attempted_live_transitions"] == 2
        assert report["last_attempted_control_source_frame_index"] == 11
        assert len(poses) == 3
        assert report["controller_failure"] == "fallback physical gate failed"
        assert report["terminal_state"]["finite"] is (fault != "nonfinite")
        assert report["terminal_state"]["base_height_m"] == (None if fault == "nonfinite" else 0.4)
    elif fault in ("transport", "transport_and_physics"):
        assert len(poses) == 3  # Hold is explicitly not part of this source-packet trace.
        assert report["observed_transport_fault"] == "timeout"
        assert report["fallback_hold_transitions"] == (1 if fault == "transport_and_physics" else 2)
        if fault == "transport_and_physics":
            assert report["controller_failure"] == "failed during transport fallback"
    elif fault == "startup_aged":
        assert len(poses) == 1
        assert report["completed_live_transitions"] == 0
        assert report["observed_transport_fault"] == "stale"
    else:
        assert len(poses) == 5
        assert calls["validations"] == 8
        assert report["fallback_active"] is (fault == "latched")
    np.testing.assert_allclose(times, np.arange(len(times)) * 0.02)


def test_legacy_cannot_start_implicitly_or_be_mixed_with_paired_reports(monkeypatch, tmp_path):
    arguments, _, _ = rig(monkeypatch, tmp_path)
    index = arguments.index("--encoder-report")
    del arguments[index : index + 2]
    arguments += ["--candidate-summary", "candidate.json"]
    with pytest.raises(SystemExit):
        live.main(arguments)
    arguments += ["--encoder-report", "encoder.json", "--legacy-unpaired-diagnostic"]
    with pytest.raises(SystemExit):
        live.main(arguments)


@pytest.mark.parametrize("extra", [["--fallback-hold-steps", "0"], ["--steps", "100001"]])
def test_unbounded_or_empty_session_rejected(monkeypatch, tmp_path, extra):
    arguments, _, _ = rig(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        live.main([*arguments, *extra])
