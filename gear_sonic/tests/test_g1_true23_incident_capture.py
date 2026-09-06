"""Offline recorder checks; no Unitree SDK, network or robot is used here."""

import base64
import gzip
import hashlib
import json
import threading

import pytest

from gear_sonic.utils.g1_true23_incident_capture import (
    PacketCapture,
    StatusEdges,
    TOPICS,
    TRUE23_HARDWARE_JOINT_IDS,
    json_safe,
    sha256,
)


class Packet:
    def __init__(self, payload=b"snapshot"):
        self.payload = bytearray(payload)

    def serialize(self):
        return self.payload


def state_record(tick=10, index=0, ns=10_000_000, **changes):
    state = {
        "tick": tick,
        "mode_machine": 4,
        "crc_matches": True,
        "motor_modes": [1] * 35,
        "motor_status": [0] * 35,
    }
    state.update(changes)
    return {
        "topic": "rt/lowstate",
        "capture_index": index,
        "callback_monotonic_ns": ns,
        "decoded": state,
    }


def records(output):
    with gzip.open(output / "packets.jsonl.gz", "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def test_status_transition_preserves_raw_bits_and_native23_mapping():
    observer = StatusEdges()
    first = state_record()
    assert observer.observe(first)[0]["kind"] == "initial_observation"
    changed = state_record(tick=12, index=3, ns=12_000_000)
    changed["decoded"]["motor_modes"][0] = 0
    changed["decoded"]["motor_status"][0] = (1 << 30) | 0x80
    edge = observer.observe(changed)[0]
    assert edge["bit30_rising_slots"] == [0]
    assert edge["motor_mode_changes"] == [{"slot": 0, "before": 1, "after": 0}]
    assert edge["raw_motor_status_changes"][0]["after"] == (1 << 30) | 0x80
    assert edge["previous_capture_index"] == 0
    assert edge["tick_delta"] == 2
    assert edge["physical_cause_proven"] is False
    assert edge["upstream_loss_excluded"] is False
    healthy = state_record(tick=14, index=4, ns=14_000_000)
    assert observer.observe(healthy)[0]["bit30_falling_slots"] == [0]
    assert len(TRUE23_HARDWARE_JOINT_IDS) == 23
    assert observer.counts["observed_status_change"] == 2


@pytest.mark.parametrize("slot", sorted(set(range(35)) - set(TRUE23_HARDWARE_JOINT_IDS)))
def test_absent_and_reserved_slots_do_not_become_controlled_motor_edges(slot):
    observer = StatusEdges()
    observer.observe(state_record())
    changed = state_record(tick=12, ns=12_000_000)
    changed["decoded"]["motor_modes"][slot] = 0
    changed["decoded"]["motor_status"][slot] = 1 << 30
    assert observer.observe(changed) == []


@pytest.mark.parametrize(
    "changes",
    [
        {"crc_matches": False},
        {"crc_matches": 1},
        {"motor_modes": []},
        {"motor_status": []},
        {"tick": None},
        {"mode_machine": None},
    ],
)
def test_untrusted_state_resets_edge_baseline(changes):
    observer = StatusEdges()
    observer.observe(state_record())
    bad = state_record(tick=12)
    bad["decoded"].update(changes)
    assert observer.observe(bad)[0]["kind"] == "untrusted_state"
    healthy = state_record(tick=14, ns=14_000_000)
    healthy["decoded"]["motor_status"][0] = 1 << 30
    assert observer.observe(healthy)[0]["kind"] == "initial_observation"
    assert observer.counts["observed_status_change"] == 0


@pytest.mark.parametrize("tick,ns", [(10, 12_000_000), (9, 12_000_000), (12, 0)])
def test_tick_repeat_reset_and_callback_reversal_are_not_fault_edges(tick, ns):
    observer = StatusEdges()
    observer.observe(state_record())
    changed = state_record(tick=tick, ns=ns)
    changed["decoded"]["motor_modes"][0] = 0
    assert observer.observe(changed)[0]["kind"] == "state_discontinuity"
    assert observer.observe(state_record(tick=14, ns=14_000_000))[0]["kind"] == "initial_observation"


def test_uint32_tick_wrap_and_known_collector_gap_are_explicit():
    observer = StatusEdges()
    first = state_record(tick=(1 << 32) - 2)
    first["lowstate_drops_before_callback"] = 3
    observer.observe(first)
    changed = state_record(tick=0, ns=60_000_000, mode_machine=1)
    changed["lowstate_drops_before_callback"] = 5
    edge = observer.observe(changed)[0]
    assert edge["tick_delta"] == 2
    assert edge["callback_gap_ns"] == 50_000_000
    assert edge["gap_exceeds_runtime_40ms_freshness"] is True
    assert edge["collector_lowstate_drops_between_observations"] == 2
    assert edge["mode_machine_before"] == 4
    assert edge["mode_machine_after"] == 1


def test_crc_decode_failure_and_non_state_do_not_infer_edges():
    observer = StatusEdges()
    observer.observe(state_record())
    assert observer.observe({"topic": "rt/api/sport/request"}) == []
    assert observer.observe({"topic": "rt/lowstate", "decode_error": "bad"})[0]["kind"] == "untrusted_state"


def test_capture_owns_snapshot_preserves_raw_bytes_and_remains_non_authorizing(tmp_path):
    output = tmp_path / "evidence"
    entered, release = threading.Event(), threading.Event()

    def decode(topic, payload):
        entered.set()
        assert release.wait(5)
        return state_record()["decoded"]

    capture = PacketCapture(output, metadata={"offline": True}, decode=decode)
    message = Packet()
    capture.receive("rt/lowstate", message, callback_ns=123)
    assert entered.wait(5)
    message.payload[:] = b"mutated!"
    release.set()
    summary = capture.finish()
    evidence = records(output)
    packet = evidence[1]
    assert base64.b64decode(packet["reserialized_cdr_b64"]) == b"snapshot"
    assert packet["cdr_sha256"] == hashlib.sha256(b"snapshot").hexdigest()
    assert packet["callback_monotonic_ns"] == 123
    assert summary["packets_sha256"] == sha256(output / "packets.jsonl.gz")
    assert summary["all_observed_callbacks_preserved"] is True
    assert summary["capture_completed"] is True
    for flag in (
        "robot_commands_published",
        "hardware_authorized",
        "deployment_ready",
        "physical_cause_proven",
        "upstream_DDS_or_network_loss_excluded",
    ):
        assert summary[flag] is False
    assert capture.finish() == summary
    capture.receive("rt/lowstate", Packet())
    assert capture.received["rt/lowstate"] == 1
    assert json.loads((output / "summary.json").read_text()) == summary
    assert evidence[0]["cdr_is_reserialized_message_not_original_wire_bytes"] is True
    assert evidence[-1]["event"] == "capture_end"


def test_queue_overflow_is_counted_without_blocking_or_hiding_loss(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def decode(topic, payload):
        entered.set()
        assert release.wait(5)
        return state_record()["decoded"]

    capture = PacketCapture(tmp_path / "overflow", metadata={}, decode=decode, queue_size=1)
    capture.receive("rt/lowstate", Packet())
    assert entered.wait(5)
    capture.receive("rt/lowstate", Packet())
    capture.receive("rt/lowstate", Packet())
    capture.receive("rt/user_lowcmd", Packet())
    release.set()
    summary = capture.finish()
    assert summary["received_callbacks"] == {"rt/lowstate": 3, "rt/user_lowcmd": 1}
    assert summary["written_packets"] == {"rt/lowstate": 2}
    assert summary["collector_dropped_packets"] == {"rt/lowstate": 1, "rt/user_lowcmd": 1}
    assert summary["maximum_queue_depth"] == 1
    assert summary["all_observed_callbacks_preserved"] is False


@pytest.mark.parametrize("failure", ["oversize", "serialize", "decode"])
def test_bad_packet_evidence_is_kept_and_reported(tmp_path, failure):
    class Broken(Packet):
        def serialize(self):
            if failure == "serialize":
                raise ValueError("injected snapshot failure")
            return b"1234"

    def decode(topic, payload):
        if failure == "decode":
            raise ValueError("injected decode failure")
        return {}

    output = tmp_path / failure
    capture = PacketCapture(
        output, metadata={}, decode=decode, max_packet_bytes=3 if failure == "oversize" else 100
    )
    capture.receive("rt/lowstate", Broken())
    summary = capture.finish()
    evidence = records(output)[1]
    key = "decode_error" if failure == "decode" else "snapshot_error"
    assert key in evidence
    assert summary["errors"] == {key: 1}
    assert summary["all_observed_callbacks_preserved"] is False
    assert evidence["observations"][0]["kind"] == "untrusted_state"
    assert ("reserialized_cdr_b64" in evidence) == (failure == "decode")


def test_nonfinite_numeric_values_are_explicit_json_not_dropped_frames(tmp_path):
    unsafe = {"values": (float("nan"), float("inf"), -float("inf"))}
    assert json_safe(unsafe)["values"] == [
        {"nonfinite_float": "nan"},
        {"nonfinite_float": "inf"},
        {"nonfinite_float": "-inf"},
    ]
    output = tmp_path / "nonfinite"
    capture = PacketCapture(output, metadata=unsafe, decode=lambda topic, payload: unsafe)
    capture.receive("rt/lowcmd", Packet())
    assert capture.finish()["all_observed_callbacks_preserved"] is True
    assert records(output)[1]["decoded"] == json_safe(unsafe)


def test_disk_write_failure_is_not_a_success(tmp_path):
    capture = PacketCapture(tmp_path / "disk", metadata={}, decode=lambda topic, payload: {})

    def fail(record):
        raise OSError("injected disk failure")

    capture._write = fail
    capture.receive("rt/lowcmd", Packet())
    summary = capture.finish()
    assert summary["capture_completed"] is False
    assert summary["all_observed_callbacks_preserved"] is False
    assert "injected disk failure" in summary["worker_error"]


def test_existing_directory_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        PacketCapture(tmp_path, metadata={}, decode=None)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "kwargs",
    [{"queue_size": 0}, {"queue_size": 8193}, {"max_packet_bytes": 0}, {"max_packet_bytes": (1 << 20) + 1}],
)
def test_invalid_bounds_fail_before_creating_evidence(tmp_path, kwargs):
    output = tmp_path / "absent"
    with pytest.raises(ValueError):
        PacketCapture(output, metadata={}, decode=None, **kwargs)
    assert not output.exists()


def test_unknown_topic_rejected_without_fake_packet(tmp_path):
    capture = PacketCapture(tmp_path / "topic", metadata={}, decode=None)
    with pytest.raises(ValueError):
        capture.receive("rt/arm_sdk", Packet())
    assert capture.finish()["received_callbacks"] == {}


def test_multiple_callback_threads_have_unique_serialized_receipt_order(tmp_path):
    output = tmp_path / "concurrent"
    capture = PacketCapture(output, metadata={}, decode=lambda topic, payload: {})
    topics = list(TOPICS)[1:]
    workers = [
        threading.Thread(target=lambda topic=topic: [capture.receive(topic, Packet()) for _ in range(50)])
        for topic in topics
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
        assert not worker.is_alive()
    summary = capture.finish()
    packets = records(output)[1:-1]
    assert summary["all_observed_callbacks_preserved"] is True
    assert [packet["capture_index"] for packet in packets] == list(range(300))
    timestamps = [packet["callback_monotonic_ns"] for packet in packets]
    assert timestamps == sorted(timestamps)
    assert summary["received_callbacks"] == {topic: 50 for topic in topics}
