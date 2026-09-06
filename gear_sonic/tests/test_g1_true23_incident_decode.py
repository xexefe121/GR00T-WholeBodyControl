"""Test raw-first recording and immutable offline interpretation, without SDK."""

import gzip
import json

import pytest

from gear_sonic.scripts import decode_g1_true23_incident_no_robot as decoder
from gear_sonic.utils.g1_true23_incident_capture import PacketCapture, sha256


class Packet:
    def __init__(self, value):
        self.value = value

    def serialize(self):
        return bytes([self.value])


def decode(topic, raw):
    value = raw[0]
    return {
        "tick": value * 2 + 2,
        "mode_machine": 4,
        "crc_matches": True,
        "motor_modes": [1] * 35,
        "motor_status": [1 << 30 if value == 1 else 0] + [0] * 34,
    }


def raw_capture(path, *, transport_error=None):
    capture = PacketCapture(path, metadata={"test_only": True})
    for index in range(3):
        capture.receive("rt/lowstate", Packet(index), callback_ns=index * 2_000_000)
    return capture.finish(transport_error=transport_error)


def read_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def test_raw_capture_has_no_decoding_and_offline_decode_retains_transient(tmp_path):
    source, output = tmp_path / "capture", tmp_path / "decode"
    original = raw_capture(source)
    raw = read_records(source / "packets.jsonl.gz")
    assert original["interpretation_deferred"] is True
    assert original["observations"] == {}
    assert raw[0]["interpretation_deferred"] is True
    assert all("decoded" not in packet and packet["observations"] == [] for packet in raw[1:-1])
    report = decoder.decode_recording(source, output, decode=decode)
    interpreted = read_records(output / "decoded.jsonl.gz")
    assert interpreted[2]["observations"][0]["bit30_rising_slots"] == [0]
    assert interpreted[3]["observations"][0]["bit30_falling_slots"] == [0]
    assert report["observations"]["observed_status_change"] == 2
    assert report["source_packets_sha256"] == sha256(source / "packets.jsonl.gz") == original["packets_sha256"]
    assert report["packet_counts"] == {"rt/lowstate": 3}
    assert report["offline_decode_completed"] is True
    assert report["hardware_authorized"] is False
    assert report["deployment_ready"] is False


def test_transport_failure_remains_visible_after_successful_offline_decode(tmp_path):
    source = tmp_path / "interrupted"
    raw_capture(source, transport_error="operator_interrupted_capture_only")
    report = decoder.decode_recording(source, tmp_path / "decoded", decode=decode)
    assert report["offline_decode_completed"] is True
    assert report["source_capture_completed"] is False
    assert report["source_all_observed_callbacks_preserved"] is True


def test_invalid_idl_preserves_raw_payload_and_breaks_edge_baseline(tmp_path):
    source = tmp_path / "invalid-idl"
    raw_capture(source)

    def fail_once(topic, raw):
        if raw == b"\x01":
            raise ValueError("injected invalid IDL")
        return decode(topic, raw)

    report = decoder.decode_recording(source, tmp_path / "decoded", decode=fail_once)
    assert report["errors"] == {"decode_error": 1}
    assert report["observations"]["untrusted_state"] == 1
    assert report["observations"].get("observed_status_change", 0) == 0
    packets = read_records(tmp_path / "decoded/decoded.jsonl.gz")
    assert packets[2]["reserialized_cdr_b64"]
    assert "injected invalid IDL" in packets[2]["decode_error"]


def test_snapshot_error_survives_deferred_decode(tmp_path):
    class Bad:
        def serialize(self):
            raise ValueError("injected serialization failure")

    source = tmp_path / "bad-snapshot"
    capture = PacketCapture(source, metadata={})
    capture.receive("rt/lowstate", Bad())
    capture.finish()
    report = decoder.decode_recording(source, tmp_path / "decoded", decode=decode)
    assert report["errors"] == {"snapshot_error": 1}
    assert report["source_all_observed_callbacks_preserved"] is False


def test_compressed_hash_mismatch_rejected_before_sdk_loading(tmp_path, monkeypatch):
    source = tmp_path / "modified"
    raw_capture(source)
    with (source / "packets.jsonl.gz").open("ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(decoder, "load_readonly_api", lambda: pytest.fail("unexpected SDK load"))
    with pytest.raises(ValueError, match="compressed-packet SHA256"):
        decoder.decode_recording(source, tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("mutation", ["packet_hash", "order", "topic", "footer", "missing_payload", "count"])
def test_hash_matched_container_still_requires_valid_packet_integrity(tmp_path, mutation):
    source = tmp_path / "mutated"
    raw_capture(source)
    path = source / "packets.jsonl.gz"
    data = read_records(path)
    if mutation == "packet_hash":
        data[1]["cdr_sha256"] = "0" * 64
    elif mutation == "order":
        data[2]["capture_index"] = 0
    elif mutation == "topic":
        data[1]["topic"] = "unknown"
    elif mutation == "footer":
        data.pop()
    elif mutation == "missing_payload":
        del data[1]["reserialized_cdr_b64"]
    elif mutation == "count":
        data.pop(1)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in data:
            stream.write(json.dumps(record) + "\n")
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["packets_sha256"] = sha256(path)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        decoder.decode_recording(source, tmp_path / "rejected", decode=decode)
    assert not (tmp_path / "rejected/decode_summary.json").exists()


def test_output_never_overwrites_existing_capture(tmp_path):
    source = tmp_path / "capture"
    original = raw_capture(source)
    with pytest.raises(FileExistsError):
        decoder.decode_recording(source, source, decode=decode)
    assert sha256(source / "packets.jsonl.gz") == original["packets_sha256"]
