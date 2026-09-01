"""Replay validated real PICO reference packets over localhost at exact 50 Hz."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping

import zmq

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import validate_reference_terms

CONTROL_PERIOD_NS = 20_000_000


class WslMonotonicClock:
    """Low-latency persistent bridge to WSL's steady-clock domain."""

    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [
                "wsl.exe",
                "-e",
                "python3",
                "-u",
                "-c",
                (
                    "import sys,time;"
                    "[(print(time.monotonic_ns(),flush=True)) for _ in sys.stdin]"
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None:
            self.close()
            raise RuntimeError("WSL clock bridge pipe creation failed")

    def now_ns(self) -> int:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write("sample\n")
        self.process.stdin.flush()
        output = self.process.stdout.readline()
        if not output:
            raise RuntimeError("WSL clock bridge exited")
        return int(output.strip())

    def close(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()


def estimate_wsl_monotonic_offset_ns(samples: int = 5) -> tuple[int, int]:
    """Estimate WSL steady-clock minus local steady-clock using lowest RTT."""

    if samples < 3:
        raise ValueError("WSL clock calibration needs at least three samples")
    measurements: list[tuple[int, int]] = []
    process = subprocess.Popen(
        [
            "wsl.exe",
            "-e",
            "python3",
            "-u",
            "-c",
            (
                "import sys,time;"
                "[(print(time.monotonic_ns(),flush=True)) for _ in sys.stdin]"
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    try:
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("WSL clock calibration pipe creation failed")
        for _ in range(samples):
            before = time.monotonic_ns()
            process.stdin.write("sample\n")
            process.stdin.flush()
            output = process.stdout.readline()
            after = time.monotonic_ns()
            if not output:
                raise RuntimeError("WSL clock calibration process exited")
            remote = int(output.strip())
            round_trip = after - before
            measurements.append((round_trip, remote - ((before + after) // 2)))
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
    best_round_trip, offset = min(measurements)
    if best_round_trip > 20_000_000:
        raise RuntimeError("WSL monotonic calibration round trip exceeds 20 ms")
    return offset, best_round_trip


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exclusive_json(path: Path, report: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_reference_packets(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "robot_independent_reference_packets",
        "semantic_packets",
    }:
        raise ValueError("saved PICO packet bundle schema mismatch")
    packets = payload["robot_independent_reference_packets"]
    if not isinstance(packets, list) or not packets:
        raise ValueError("saved PICO packet bundle has no reference packets")
    summaries = [validate_reference_terms(packet) for packet in packets]
    for previous, current in zip(summaries, summaries[1:], strict=False):
        if current["control_index"] != previous["control_index"] + 1:
            raise ValueError("saved PICO packet indices are not contiguous")
    return packets


def rebase_reference_packet_time(
    packet: Mapping[str, Any],
    *,
    first_control_index: int,
    first_control_monotonic_ns: int,
) -> dict[str, Any]:
    summary = validate_reference_terms(packet)
    offset = summary["control_index"] - first_control_index
    if offset < 0:
        raise ValueError("packet precedes timed replay origin")
    control_ns = first_control_monotonic_ns + offset * CONTROL_PERIOD_NS
    rebased = copy.deepcopy(dict(packet))
    rebased["pico_anchor_monotonic_ns"] = control_ns - CONTROL_PERIOD_NS
    rebased["control_monotonic_ns"] = control_ns
    validate_reference_terms(rebased)
    return rebased


def publish_timed_bundle(
    *,
    packets: list[dict[str, Any]],
    bind: str,
    subscriber_warmup_s: float,
    fault: str = "none",
    fault_offset: int = 120,
    stale_delay_ms: int = 250,
    timestamp_offset_ns: int = 0,
    timestamp_now_ns: Callable[[], int] | None = None,
) -> dict[str, Any]:
    if not bind.startswith("tcp://127.0.0.1:"):
        raise ValueError("timed PICO replay is restricted to localhost TCP")
    if subscriber_warmup_s < 0.1:
        raise ValueError("subscriber warmup must be at least 0.1 seconds")
    if fault not in {"none", "timeout", "gap", "stale"}:
        raise ValueError("unsupported diagnostic transport fault")
    if fault != "none" and not 2 <= fault_offset < len(packets):
        raise ValueError("fault offset must leave two startup packets and precede stream end")
    if stale_delay_ms <= 100:
        raise ValueError("stale delay must exceed the 100 ms live freshness gate")
    first_summary = validate_reference_terms(packets[0])
    context = zmq.Context(io_threads=1)
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 1000)
    socket.bind(bind)
    schedule_slips_ns: list[int] = []
    published_count = 0
    first_published_control_ns: int | None = None
    last_published_control_ns: int | None = None
    started_ns = time.monotonic_ns()
    try:
        time.sleep(subscriber_warmup_s)
        first_schedule_ns = time.monotonic_ns() + CONTROL_PERIOD_NS
        first_control_ns = (
            timestamp_now_ns() + CONTROL_PERIOD_NS
            if timestamp_now_ns is not None
            else first_schedule_ns + timestamp_offset_ns
        )
        for offset, packet in enumerate(packets):
            if fault == "timeout" and offset == fault_offset:
                break
            if fault == "gap" and offset == fault_offset:
                continue
            deadline_ns = (
                first_control_ns + offset * CONTROL_PERIOD_NS
                if timestamp_now_ns is not None
                else first_schedule_ns + offset * CONTROL_PERIOD_NS
            )
            if fault == "stale" and offset == fault_offset:
                time.sleep(stale_delay_ms / 1000.0)
            while True:
                clock_now_ns = (
                    timestamp_now_ns()
                    if timestamp_now_ns is not None
                    else time.monotonic_ns()
                )
                remaining_ns = deadline_ns - clock_now_ns
                if remaining_ns <= 0:
                    break
                time.sleep(min(remaining_ns / 1_000_000_000, 0.002))
            rebased = rebase_reference_packet_time(
                packet,
                first_control_index=first_summary["control_index"],
                first_control_monotonic_ns=first_control_ns,
            )
            sent_ns = (
                timestamp_now_ns()
                if timestamp_now_ns is not None
                else time.monotonic_ns()
            )
            schedule_slips_ns.append(sent_ns - deadline_ns)
            socket.send_json(rebased)
            published_count += 1
            control_ns = int(rebased["control_monotonic_ns"])
            first_published_control_ns = (
                control_ns if first_published_control_ns is None else first_published_control_ns
            )
            last_published_control_ns = control_ns
        time.sleep(0.1)
    finally:
        socket.close()
        context.term()
    finished_ns = time.monotonic_ns()
    return {
        "schema_version": 1,
        "kind": "g1_true23_saved_pico_timed_zmq_replay",
        "bind": bind,
        "packet_count": published_count,
        "source_packet_count": len(packets),
        "published_packet_count": published_count,
        "first_published_control_monotonic_ns": first_published_control_ns,
        "last_published_control_monotonic_ns": last_published_control_ns,
        "first_control_source_frame_index": first_summary["control_index"],
        "last_control_source_frame_index": validate_reference_terms(packets[-1])["control_index"],
        "control_period_ns": CONTROL_PERIOD_NS,
        "maximum_schedule_slip_ns": max(schedule_slips_ns),
        "mean_schedule_slip_ns": sum(schedule_slips_ns) / len(schedule_slips_ns),
        "wall_duration_ns": finished_ns - started_ns,
        "values_rebased": ["pico_anchor_monotonic_ns", "control_monotonic_ns"],
        "timestamp_clock_offset_ns": timestamp_offset_ns,
        "pose_and_reference_values_unchanged": True,
        "diagnostic_fault": fault,
        "diagnostic_fault_offset": None if fault == "none" else fault_offset,
        "diagnostic_stale_delay_ms": stale_delay_ms if fault == "stale" else None,
        "passed": True,
        "authorization": {
            "localhost_only": True,
            "dds_opened": False,
            "hardware_authorized": False,
            "robot_commands_published": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--bind", default="tcp://127.0.0.1:5557")
    parser.add_argument("--subscriber-warmup-s", type=float, default=2.0)
    parser.add_argument("--fault", choices=("none", "timeout", "gap", "stale"), default="none")
    parser.add_argument("--fault-offset", type=int, default=120)
    parser.add_argument("--stale-delay-ms", type=int, default=250)
    parser.add_argument("--timestamp-clock", choices=("local", "wsl"), default="local")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packets = load_reference_packets(args.packets)
    clock_offset_ns = 0
    calibration_round_trip_ns = None
    clock_bridge = WslMonotonicClock() if args.timestamp_clock == "wsl" else None
    try:
        report = publish_timed_bundle(
            packets=packets,
            bind=args.bind,
            subscriber_warmup_s=args.subscriber_warmup_s,
            fault=args.fault,
            fault_offset=args.fault_offset,
            stale_delay_ms=args.stale_delay_ms,
            timestamp_offset_ns=clock_offset_ns,
            timestamp_now_ns=None if clock_bridge is None else clock_bridge.now_ns,
        )
    finally:
        if clock_bridge is not None:
            clock_bridge.close()
    report["timestamp_clock"] = args.timestamp_clock
    report["timestamp_clock_calibration_round_trip_ns"] = calibration_round_trip_ns
    report["saved_packet_bundle"] = str(args.packets.resolve())
    report["saved_packet_bundle_sha256"] = _sha256_file(args.packets.resolve())
    _exclusive_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
