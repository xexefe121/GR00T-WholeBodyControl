"""Exercise the native operator-control wire against the safe loopback loop.

This is intentionally a same-host test.  It uses 127.0.0.1, the fixed
domain-232/test-topic binary, and no ``--arm`` or hardware endpoint.  The
shared CLOCK_MONOTONIC timestamp therefore gives a valid send-to-native-action
latency for the manual-abort wire without pretending that two Ethernet hosts
share a clock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import subprocess
import time

import zmq


CONTROL = struct.Struct("<IIQQI")
MAGIC = 0x34434742
HEARTBEAT, ADVANCE, ABORT = range(3)


def send(socket: zmq.Socket, sequence: int, operation: int) -> int:
    sent = time.monotonic_ns()
    socket.send(CONTROL.pack(MAGIC, 2, sequence, sent, operation))
    return sequence + 1


def trial(args: argparse.Namespace, index: int) -> dict:
    base = args.port_base + index * 3
    output = args.output / f"trial_{index:03d}"
    output.mkdir(parents=True)
    command = [
        args.binary, "--source", "replay", "--replay", args.replay,
        "--initial-command", args.initial_command, "--model", args.model,
        "--state-endpoint", f"tcp://127.0.0.1:{base}",
        "--target-endpoint", f"tcp://127.0.0.1:{base + 1}",
        "--operator-endpoint", f"tcp://127.0.0.1:{base + 2}",
        "--bringup-ladder", "--output", str(output), "--ticks", str(args.ticks),
    ]
    with (output / "native.stdout.log").open("w", encoding="utf-8") as stdout, \
            (output / "native.stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        context = zmq.Context.instance()
        socket = context.socket(zmq.PUSH)
        socket.linger = 0
        socket.connect(f"tcp://127.0.0.1:{base + 2}")
        try:
            # Native binds first; this is only the ZMQ handshake allowance.
            time.sleep(0.150)
            sequence = send(socket, 0, HEARTBEAT)
            for _ in range(4):
                time.sleep(0.040)
                sequence = send(socket, sequence, ADVANCE)
            time.sleep(0.040)
            send(socket, sequence, ABORT)
        finally:
            socket.close()
        if process.wait(timeout=args.timeout_seconds) != 0:
            raise RuntimeError(f"native loop failed in trial {index}")
    report = json.loads((output / "loop_report.json").read_text(encoding="utf-8"))
    if report["control_frames_received"] < 6:
        raise RuntimeError(f"trial {index}: native received too few controls: {report['control_frames_received']}")
    if report["abort_reason"] != "manual operator abort":
        raise RuntimeError(f"trial {index}: wrong abort reason: {report['abort_reason']!r}")
    if report["current_stage"] != "aborted":
        raise RuntimeError(f"trial {index}: abort was not latched")
    if report["manual_abort_delivery_latency_ms"]["samples"] != 1:
        raise RuntimeError(f"trial {index}: native did not record manual-abort latency")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--initial-command", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--port-base", type=int, default=5700)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.trials < 50:
        raise ValueError("at least 50 manual aborts are required")
    args.output.mkdir(parents=True, exist_ok=True)
    reports = [trial(args, index) for index in range(args.trials)]
    latencies = [report["manual_abort_delivery_latency_ms"]["summary"] for report in reports]
    combined = {
        "trials": args.trials,
        "all_manual_abort_latched": True,
        "control_frames_received_total": sum(report["control_frames_received"] for report in reports),
        "manual_abort_delivery_latency_ms": {
            "p50": sorted(item["p50"] for item in latencies)[len(latencies) // 2],
            "p95": sorted(item["p95"] for item in latencies)[int((len(latencies) - 1) * .95)],
            "max": max(item["max"] for item in latencies),
        },
        "reports": [str((args.output / f"trial_{index:03d}" / "loop_report.json")) for index in range(args.trials)],
    }
    (args.output / "operator_loopback_summary.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
