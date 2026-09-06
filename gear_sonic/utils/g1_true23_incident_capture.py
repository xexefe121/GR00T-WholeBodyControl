"""Bounded, non-actuating packet recorder and observational status-edge detector.

No DDS/robot imports here. The transport supplies reserialized CDR bytes and
an offline decoder. Receipt order does not prove command execution or cause.
"""

from __future__ import annotations

import base64
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time

from gear_sonic.scripts.pico_g1_preflight import TRUE23_HARDWARE_JOINT_IDS


TOPICS = {
    "rt/lowstate": "LowState_",
    "rt/lowcmd": "LowCmd_",
    "rt/user_lowcmd": "LowCmd_",
    "rt/api/sport/request": "Request_",
    "rt/api/sport/response": "Response_",
    "rt/api/motion_switcher/request": "Request_",
    "rt/api/motion_switcher/response": "Response_",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    """Preserve nonfinite values explicitly; do not silently drop bad frames."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    return value


class StatusEdges:
    """Changes between observed CRC-valid true23 states, never fault diagnosis."""

    def __init__(self):
        self.previous = None
        self.counts = Counter()

    def observe(self, record: dict) -> list[dict]:
        if record["topic"] != "rt/lowstate":
            return []
        state = record.get("decoded", {})
        if (
            state.get("crc_matches") is not True
            or len(state.get("motor_modes", [])) != 35
            or len(state.get("motor_status", [])) != 35
            or not isinstance(state.get("tick"), int)
            or not isinstance(state.get("mode_machine"), int)
        ):
            self.previous = None
            self.counts["untrusted_state"] += 1
            return [{"kind": "untrusted_state", "edge_not_inferred": True}]
        current = (
            record["capture_index"],
            record["callback_monotonic_ns"],
            state,
            record.get("lowstate_drops_before_callback", 0),
        )
        if self.previous is None:
            self.previous = current
            self.counts["initial_observation"] += 1
            return [{"kind": "initial_observation", "edge_not_inferred": True}]
        previous_index, previous_ns, old, previous_drops = self.previous
        tick_delta = (state["tick"] - old["tick"]) % (1 << 32)
        gap_ns = record["callback_monotonic_ns"] - previous_ns
        if not 0 < tick_delta < (1 << 31) or gap_ns < 0:
            self.previous = None
            self.counts["state_discontinuity"] += 1
            return [{"kind": "state_discontinuity", "tick_delta": tick_delta, "edge_not_inferred": True}]
        self.previous = current
        modes = []
        statuses = []
        for slot in TRUE23_HARDWARE_JOINT_IDS:
            if old["motor_modes"][slot] != state["motor_modes"][slot]:
                modes.append(
                    {"slot": slot, "before": old["motor_modes"][slot], "after": state["motor_modes"][slot]}
                )
            if old["motor_status"][slot] != state["motor_status"][slot]:
                statuses.append(
                    {"slot": slot, "before": old["motor_status"][slot], "after": state["motor_status"][slot]}
                )
        if not modes and not statuses and old["mode_machine"] == state["mode_machine"]:
            return []
        self.counts["observed_status_change"] += 1
        return [
            {
                "kind": "observed_status_change",
                "previous_capture_index": previous_index,
                "previous_tick": old["tick"],
                "tick": state["tick"],
                "tick_delta": tick_delta,
                "callback_gap_ns": gap_ns,
                "collector_lowstate_drops_between_observations": current[3] - previous_drops,
                "upstream_loss_excluded": False,
                "gap_exceeds_runtime_40ms_freshness": gap_ns > 40_000_000,
                "mode_machine_before": old["mode_machine"],
                "mode_machine_after": state["mode_machine"],
                "motor_mode_changes": modes,
                "raw_motor_status_changes": statuses,
                "bit30_rising_slots": [
                    row["slot"] for row in statuses if not row["before"] & (1 << 30) and row["after"] & (1 << 30)
                ],
                "bit30_falling_slots": [
                    row["slot"] for row in statuses if row["before"] & (1 << 30) and not row["after"] & (1 << 30)
                ],
                "physical_cause_proven": False,
            }
        ]


class PacketCapture:
    def __init__(self, output: Path, *, metadata: dict, decode=None, queue_size=2048, max_packet_bytes=65536):
        if not 1 <= queue_size <= 8192 or not 1 <= max_packet_bytes <= 1 << 20:
            raise ValueError("invalid capture memory bounds")
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.decode = decode
        self.max_packet_bytes = max_packet_bytes
        self.queue = queue.Queue(queue_size)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.accepting = True
        self.next_index = 0
        self.received = Counter()
        self.dropped = Counter()
        self.written = Counter()
        self.errors = Counter()
        self.maximum_queue_depth = 0
        self.failure = None
        self.edges = StatusEdges()
        self.closed_summary = None
        self.capture_path = self.output / "packets.jsonl.gz"
        self.stream = gzip.open(self.capture_path, "xt", encoding="utf-8", compresslevel=1)
        self._write(
            {
                "event": "capture_start",
                "schema_version": 1,
                "metadata": metadata,
                "topics": TOPICS,
                "controlled_motor_slots": TRUE23_HARDWARE_JOINT_IDS,
                "cdr_is_reserialized_message_not_original_wire_bytes": True,
                "queue_capacity": queue_size,
                "max_packet_bytes": max_packet_bytes,
                "interpretation_deferred": decode is None,
                "robot_commands_published": False,
                "physical_cause_proven": False,
            }
        )
        self.worker = threading.Thread(target=self._consume, name="g1-readonly-evidence", daemon=True)
        self.worker.start()

    def _write(self, record):
        self.stream.write(json.dumps(json_safe(record), separators=(",", ":"), allow_nan=False) + "\n")

    def receive(self, topic: str, message, *, callback_ns=None):
        if topic not in TOPICS:
            raise ValueError("unexpected subscription topic")
        # Snapshot in callback, before the SDK or another caller can reuse the
        # message object. Disk I/O and interpretation happen on the consumer.
        with self.lock:
            if not self.accepting:
                return
            index = self.next_index
            self.next_index += 1
            self.received[topic] += 1
            record = {
                "event": "packet",
                "capture_index": index,
                "topic": topic,
                "lowstate_drops_before_callback": self.dropped["rt/lowstate"],
                "callback_monotonic_ns": time.monotonic_ns() if callback_ns is None else callback_ns,
            }
            try:
                payload = bytes(message.serialize())
                if len(payload) > self.max_packet_bytes:
                    raise ValueError("serialized packet exceeds configured byte bound")
            except Exception as error:
                self.errors["snapshot_error"] += 1
                record["snapshot_error"] = f"{type(error).__name__}: {error}"
                payload = None
            try:
                self.queue.put_nowait((record, payload))
                self.maximum_queue_depth = max(self.maximum_queue_depth, self.queue.qsize())
            except queue.Full:
                self.dropped[topic] += 1

    def _consume(self):
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    record, payload = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if payload is not None:
                    record["reserialized_cdr_b64"] = base64.b64encode(payload).decode("ascii")
                    record["cdr_sha256"] = hashlib.sha256(payload).hexdigest()
                    if self.decode is not None:
                        try:
                            record["decoded"] = self.decode(record["topic"], payload)
                        except Exception as error:
                            self.errors["decode_error"] += 1
                            record["decode_error"] = f"{type(error).__name__}: {error}"
                record["observations"] = self.edges.observe(record) if self.decode is not None else []
                self._write(record)
                self.written[record["topic"]] += 1
                if sum(self.written.values()) % 100 == 0:
                    self.stream.flush()
        except Exception as error:
            self.failure = f"{type(error).__name__}: {error}"
            self.stop.set()

    def finish(self, *, transport_error=None, subscriptions_opened=False) -> dict:
        if self.closed_summary is not None:
            return dict(self.closed_summary)
        with self.lock:
            self.accepting = False
        self.stop.set()
        self.worker.join(timeout=10)
        if self.worker.is_alive():
            raise RuntimeError("capture consumer did not stop; evidence is incomplete")
        summary = {
            "kind": "g1_true23_readonly_incident_capture_v1",
            "capture_end_monotonic_ns": time.monotonic_ns(),
            "received_callbacks": dict(self.received),
            "written_packets": dict(self.written),
            "collector_dropped_packets": dict(self.dropped),
            "errors": dict(self.errors),
            "maximum_queue_depth": self.maximum_queue_depth,
            "worker_error": self.failure,
            "transport_error": transport_error,
            "interpretation_deferred": self.decode is None,
            "observations": dict(self.edges.counts),
            "capture_completed": self.failure is None and transport_error is None,
            "all_observed_callbacks_preserved": (
                not self.dropped
                and not self.errors
                and self.failure is None
                and sum(self.received.values()) == sum(self.written.values())
            ),
            "upstream_DDS_or_network_loss_excluded": False,
            "observed_lowstate": self.written["rt/lowstate"] > 0,
            "dds_subscriptions_opened": subscriptions_opened,
            "robot_commands_published": False,
            "physical_cause_proven": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        }
        try:
            self._write({"event": "capture_end", **summary})
            self.stream.close()
        except Exception as error:
            summary["worker_error"] = f"{type(error).__name__}: {error}"
            summary["capture_completed"] = False
            summary["all_observed_callbacks_preserved"] = False
            try:
                self.stream.close()
            except Exception:
                pass
        summary["packets_sha256"] = sha256(self.capture_path)
        with (self.output / "summary.json").open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        self.closed_summary = summary
        return dict(summary)
