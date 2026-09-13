"""Missing receiver and unavailable headset health must not be conflated."""

import json

import pytest

from gear_sonic.scripts import probe_g1_true23_pico_input as probe


def test_missing_receiver_does_not_open_sdk(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "local_receiver_reachable", lambda: False)

    def forbidden(*args):
        raise AssertionError("tracking SDK must not open without the receiver")

    monkeypatch.setattr(probe.tracking, "main", forbidden)
    output = tmp_path / "input.json"
    assert probe.main(["--output", str(output)]) == 1
    result = json.loads(output.read_text())
    assert result["input_status"] == "receiver_unavailable"
    assert result["tracking_probe_attempted"] is False
    assert result["tracking_evidence"] is None
    assert result["hardware_ready"] is False
    assert not (tmp_path / "input.tracking.json").exists()


@pytest.mark.parametrize("health", [None, {}, {"health_available": False}])
def test_zero_defaults_are_unavailable_not_headset_diagnosis(health):
    report = {"passed": False, "latest_health": health}
    assert probe.classify_input(True, report) == "receiver_reachable_tracking_unavailable"


def test_report_boolean_alone_cannot_claim_tracking_ready():
    report = {"passed": True, "latest_health": {"health_available": True}}
    assert probe.classify_input(True, report) == "tracking_not_ready"


def test_full_tracking_health_pass_does_not_authorize_hardware(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "local_receiver_reachable", lambda: True)
    health = dict(
        health_available=True,
        health_valid=True,
        health_supported=True,
        health_connect_state_result=0,
        health_connected_band_count=2,
        health_unique_tracker_count=2,
        health_calibrated=True,
        health_calibration_result=0,
        health_is_tracking=True,
        health_tracking_state_code=0,
        health_body_data_result=0,
        health_body_state_result=0,
        health_body_error_code=0,
        health_body_role_count=24,
    )

    def healthy(arguments):
        path = arguments[arguments.index("--output") + 1]
        probe.tracking._exclusive_json(probe.Path(path), {"passed": True, "latest_health": health})
        return 0

    monkeypatch.setattr(probe.tracking, "main", healthy)
    output = tmp_path / "input.json"
    assert probe.main(["--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["input_status"] == "tracking_ready"
    assert report["receiver_tcp_reachable"] is True
    assert len(report["tracking_evidence"]["sha256"]) == 64
    assert report["hardware_ready"] is False
    assert report["authorization"]["hardware_authorized"] is False


def test_receiver_tcp_check_is_fixed_local_bounded_and_closed(monkeypatch):
    calls = []

    class Connection:
        def __enter__(self):
            calls.append("entered")
            return self

        def __exit__(self, *args):
            calls.append("closed")

    def connect(endpoint, *, timeout):
        calls.append((endpoint, timeout))
        return Connection()

    monkeypatch.setattr(probe.socket, "create_connection", connect)
    assert probe.local_receiver_reachable() is True
    assert calls == [(("127.0.0.1", 60061), 1.0), "entered", "closed"]


@pytest.mark.parametrize("failure", [ConnectionRefusedError, TimeoutError])
def test_receiver_connection_failure_is_not_tracking_failure(monkeypatch, failure):
    def connect(*args, **kwargs):
        raise failure("no local receiver")

    monkeypatch.setattr(probe.socket, "create_connection", connect)
    assert probe.local_receiver_reachable() is False


def test_tracking_probe_failure_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "local_receiver_reachable", lambda: True)

    def fail(*args):
        raise ValueError("XR service SHA256 mismatch")

    monkeypatch.setattr(probe.tracking, "main", fail)
    output = tmp_path / "input.json"
    assert probe.main(["--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["input_status"] == "tracking_probe_failed"
    assert "SHA256 mismatch" in report["tracking_probe_error"]
    assert report["hardware_ready"] is False


def test_existing_output_rejected_before_any_network_check(monkeypatch, tmp_path):
    output = tmp_path / "input.json"
    output.write_text("original")

    def forbidden():
        raise AssertionError("must reject existing output before opening sockets")

    monkeypatch.setattr(probe, "local_receiver_reachable", forbidden)
    with pytest.raises(FileExistsError):
        probe.main(["--output", str(output)])
    assert output.read_text() == "original"


@pytest.mark.parametrize("duration", ["0", "nan", "inf", "31"])
def test_unbounded_probe_duration_rejected(tmp_path, duration):
    with pytest.raises(ValueError):
        probe.main(["--output", str(tmp_path / "input.json"), "--duration-seconds", duration])
