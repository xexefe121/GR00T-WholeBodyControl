"""Replay validated real PICO reference packets over localhost at exact 50 Hz."""

from __future__ import annotations

import argparse
import copy
import ctypes
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

    def estimate_offset_ns(self, samples: int = 25) -> tuple[int, int]:
        """Map local monotonic time into WSL using the lowest-RTT sample."""

        if samples < 3:
            raise ValueError("WSL clock calibration needs at least three samples")
        measurements: list[tuple[int, int]] = []
        for _ in range(samples):
            before = time.perf_counter_ns()
            remote = self.now_ns()
            after = time.perf_counter_ns()
            round_trip = after - before
            measurements.append((round_trip, remote - ((before + after) // 2)))
        best_round_trip, offset = min(measurements)
        if best_round_trip > 20_000_000:
            raise RuntimeError("WSL monotonic calibration round trip exceeds 20 ms")
        # Keep reference timestamps safely behind receipt time despite midpoint
        # asymmetry. Two milliseconds is tiny versus the 100 ms freshness gate.
        return offset - 2_000_000, best_round_trip

    def close(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()


def estimate_wsl_monotonic_offset_ns(samples: int = 5) -> tuple[int, int]:
    """Estimate WSL steady-clock minus local steady-clock using lowest RTT."""

    bridge = WslMonotonicClock()
    try:
        return bridge.estimate_offset_ns(samples)
    finally:
        bridge.close()


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


def repeat_reference_packets(
    packets: list[dict[str, Any]], repeat_count: int
) -> list[dict[str, Any]]:
    """Repeat immutable reference values with fresh contiguous source indices."""

    if not 1 <= repeat_count <= 100:
        raise ValueError("repeat count must be between 1 and 100")
    if repeat_count == 1:
        return packets
    cycle_frames = len(packets)
    repeated: list[dict[str, Any]] = []
    for cycle in range(repeat_count):
        frame_delta = cycle * cycle_frames
        for packet in packets:
            clone = copy.deepcopy(packet)
            clone["control_source_frame_index"] = (
                int(clone["control_source_frame_index"]) + frame_delta
            )
            clone["pico_anchor_source_frame_index"] = (
                int(clone["pico_anchor_source_frame_index"]) + frame_delta
            )
            validate_reference_terms(clone)
            repeated.append(clone)
    for previous, current in zip(repeated, repeated[1:], strict=False):
        if (
            int(current["control_source_frame_index"])
            != int(previous["control_source_frame_index"]) + 1
        ):
            raise ValueError("repeated PICO packet indices are not contiguous")
    return repeated


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
    stop_file: Path | None = None,
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
    windows_timer_period_active = False
    if os.name == "nt":
        result = ctypes.windll.winmm.timeBeginPeriod(1)  # type: ignore[attr-defined]
        if result != 0:
            socket.close()
            context.term()
            raise RuntimeError("Windows 1 ms timer period request failed")
        windows_timer_period_active = True
    schedule_slips_ns: list[int] = []
    published_count = 0
    first_published_control_ns: int | None = None
    last_published_control_ns: int | None = None
    last_published_source_frame_index: int | None = None
    managed_stop_requested = False
    started_ns = time.perf_counter_ns()
    try:
        warmup_deadline_ns = (
            time.perf_counter_ns() + int(subscriber_warmup_s * 1_000_000_000)
        )
        while time.perf_counter_ns() < warmup_deadline_ns:
            if stop_file is not None and stop_file.exists():
                managed_stop_requested = True
                break
            remaining_s = (warmup_deadline_ns - time.perf_counter_ns()) / 1_000_000_000
            time.sleep(min(max(remaining_s, 0.0), 0.05))
        first_schedule_ns = time.perf_counter_ns() + CONTROL_PERIOD_NS
        first_control_ns = (
            timestamp_now_ns() + CONTROL_PERIOD_NS
            if timestamp_now_ns is not None
            else first_schedule_ns + timestamp_offset_ns
        )
        for offset, packet in enumerate(packets):
            if managed_stop_requested or (
                stop_file is not None and stop_file.exists()
            ):
                managed_stop_requested = True
                break
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
            if timestamp_now_ns is not None:
                local_before_ns = time.perf_counter_ns()
                remote_now_ns = timestamp_now_ns()
                local_after_ns = time.perf_counter_ns()
                local_sample_ns = (local_before_ns + local_after_ns) // 2
                local_deadline_ns = local_sample_ns + deadline_ns - remote_now_ns
            else:
                remote_now_ns = 0
                local_sample_ns = 0
                local_deadline_ns = deadline_ns
            while True:
                remaining_ns = local_deadline_ns - time.perf_counter_ns()
                if remaining_ns <= 0:
                    break
                if remaining_ns > 2_000_000:
                    time.sleep((remaining_ns - 1_000_000) / 1_000_000_000)
            rebased = rebase_reference_packet_time(
                packet,
                first_control_index=first_summary["control_index"],
                first_control_monotonic_ns=first_control_ns,
            )
            local_sent_ns = time.perf_counter_ns()
            sent_ns = (
                remote_now_ns + local_sent_ns - local_sample_ns
                if timestamp_now_ns is not None
                else local_sent_ns
            )
            schedule_slips_ns.append(sent_ns - deadline_ns)
            socket.send_json(rebased)
            published_count += 1
            control_ns = int(rebased["control_monotonic_ns"])
            first_published_control_ns = (
                control_ns if first_published_control_ns is None else first_published_control_ns
            )
            last_published_control_ns = control_ns
            last_published_source_frame_index = int(
                rebased["control_source_frame_index"]
            )
        if published_count > 0:
            time.sleep(0.1)
    finally:
        socket.close()
        context.term()
        if windows_timer_period_active:
            ctypes.windll.winmm.timeEndPeriod(1)  # type: ignore[attr-defined]
    finished_ns = time.perf_counter_ns()
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
        "last_control_source_frame_index": last_published_source_frame_index,
        "source_last_control_source_frame_index": validate_reference_terms(
            packets[-1]
        )["control_index"],
        "control_period_ns": CONTROL_PERIOD_NS,
        "maximum_schedule_slip_ns": (
            max(schedule_slips_ns) if schedule_slips_ns else None
        ),
        "mean_schedule_slip_ns": (
            sum(schedule_slips_ns) / len(schedule_slips_ns)
            if schedule_slips_ns
            else None
        ),
        "wall_duration_ns": finished_ns - started_ns,
        "values_rebased": ["pico_anchor_monotonic_ns", "control_monotonic_ns"],
        "timestamp_clock_offset_ns": timestamp_offset_ns,
        "windows_high_resolution_timer": windows_timer_period_active,
        "managed_stop_requested": managed_stop_requested,
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
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_packets = load_reference_packets(args.packets)
    packets = repeat_reference_packets(source_packets, args.repeat_count)
    clock_offset_ns = 0
    calibration_round_trip_ns = None
    if args.timestamp_clock == "wsl":
        clock_bridge = WslMonotonicClock()
        try:
            clock_offset_ns, calibration_round_trip_ns = (
                clock_bridge.estimate_offset_ns()
            )
            report = publish_timed_bundle(
                packets=packets,
                bind=args.bind,
                subscriber_warmup_s=args.subscriber_warmup_s,
                fault=args.fault,
                fault_offset=args.fault_offset,
                stale_delay_ms=args.stale_delay_ms,
                timestamp_offset_ns=clock_offset_ns,
                timestamp_now_ns=clock_bridge.now_ns,
                stop_file=args.stop_file,
            )
        finally:
            clock_bridge.close()
    else:
        report = publish_timed_bundle(
            packets=packets,
            bind=args.bind,
            subscriber_warmup_s=args.subscriber_warmup_s,
            fault=args.fault,
            fault_offset=args.fault_offset,
            stale_delay_ms=args.stale_delay_ms,
            timestamp_offset_ns=clock_offset_ns,
            stop_file=args.stop_file,
        )
    report["timestamp_clock"] = args.timestamp_clock
    report["timestamp_clock_calibration_round_trip_ns"] = calibration_round_trip_ns
    report["source_bundle_packet_count"] = len(source_packets)
    report["repeat_count"] = args.repeat_count
    report["saved_packet_bundle"] = str(args.packets.resolve())
    report["saved_packet_bundle_sha256"] = _sha256_file(args.packets.resolve())
    _exclusive_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
