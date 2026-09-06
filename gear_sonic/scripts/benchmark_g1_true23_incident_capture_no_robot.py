"""Paced real-IDL recorder throughput test, entirely offline and non-authorizing.

This exercises synthetic callback serialization, native CRC, decode and gzip
writing, not DDS transport, real-time scheduling or robot behavior. It never
initializes a DDS participant or invokes the live collector.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import json
import math
from pathlib import Path
import platform
import time
from unittest.mock import patch

from gear_sonic.scripts.record_g1_true23_incident_readonly import load_readonly_api
from gear_sonic.scripts.decode_g1_true23_incident_no_robot import decode_recording
from gear_sonic.utils.g1_true23_incident_capture import PacketCapture, sha256


def percentile(values, fraction):
    return sorted(values)[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)]


def benchmark(output: Path, *, duration_s=10.0, pairs_per_second=500):
    if not 0.1 <= duration_s <= 30 or not 10 <= pairs_per_second <= 1000:
        raise ValueError("require 0.1..30 s and 10..1000 state/command pairs per second")
    count = math.ceil(duration_s * pairs_per_second)
    if count < 4:
        raise ValueError("need at least four pairs for synthetic status-edge test")
    _, _, _, crc = load_readonly_api()
    from unitree_sdk2py.core import channel
    from unitree_sdk2py.idl.default import (
        unitree_hg_msg_dds__LowCmd_,
        unitree_hg_msg_dds__LowState_,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("offline benchmark attempted DDS or network operation")

    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("record_g1_true23_incident_readonly.py").resolve(),
        Path(__file__).parents[1] / "utils/g1_true23_incident_capture.py",
    ]
    sources = {str(path): sha256(path) for path in paths}
    with ExitStack() as stack:
        for name in ("Domain", "DomainParticipant", "DataWriter", "DataReader", "Topic"):
            stack.enter_context(patch.object(channel, name, forbidden))
        stack.enter_context(patch("socket.socket", forbidden))
        capture = PacketCapture(
            output,
            metadata={"offline_synthetic": True, "source_sha256": sources},
        )
        state = unitree_hg_msg_dds__LowState_()
        command = unitree_hg_msg_dds__LowCmd_()
        state.mode_machine = command.mode_machine = 4
        state.imu_state.quaternion[:] = [1.0, 0.0, 0.0, 0.0]
        callback_ns = []
        lateness_ns = []
        start = time.monotonic_ns()
        transport_error = None
        try:
            for index in range(count):
                deadline = start + round(index * 1e9 / pairs_per_second)
                remaining = (deadline - time.monotonic_ns()) / 1e9
                if remaining > 0:
                    time.sleep(remaining)
                lateness_ns.append(max(0, time.monotonic_ns() - deadline))
                state.tick = index * 2 + 2
                for slot in range(35):
                    q = 0.1 * math.sin(index / 100 + slot)
                    state.motor_state[slot].q = q
                    state.motor_state[slot].dq = 0.5 * math.cos(index / 100 + slot)
                    state.motor_state[slot].tau_est = q * 10
                    state.motor_state[slot].vol = 48.0
                    state.motor_state[slot].mode = 1
                    command.motor_cmd[slot].q = q
                    command.motor_cmd[slot].kp = 20.0
                    command.motor_cmd[slot].kd = 2.0
                # One-sample synthetic raw bit-30 edge; never a robot command.
                state.motor_state[0].motorstate = (1 << 30) if index == count // 2 else 0
                state.crc = crc.Crc(state)
                command.crc = crc.Crc(command)
                for topic, message in (("rt/lowstate", state), ("rt/user_lowcmd", command)):
                    before = time.monotonic_ns()
                    capture.receive(topic, message)
                    callback_ns.append(time.monotonic_ns() - before)
        except BaseException as error:
            transport_error = f"benchmark_producer_error: {type(error).__name__}: {error}"
            raise
        finally:
            producer_seconds = (time.monotonic_ns() - start) / 1e9
            summary = capture.finish(transport_error=transport_error)
        total_seconds = (time.monotonic_ns() - start) / 1e9
        decode_start = time.monotonic()
        interpretation = decode_recording(output, output / "interpretation")
        decode_seconds = time.monotonic() - decode_start
    observed_packets = 0
    rising = falling = 0
    monotonic_indices = True
    previous_index = -1
    with gzip.open(output / "interpretation/decoded.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["event"] == "packet":
                monotonic_indices = monotonic_indices and record["capture_index"] > previous_index
                previous_index = record["capture_index"]
                observed_packets += 1
                for edge in record["observations"]:
                    rising += len(edge.get("bit30_rising_slots", []))
                    falling += len(edge.get("bit30_falling_slots", []))
    report = {
        "kind": "g1_true23_incident_capture_offline_benchmark_v1",
        "source_sha256": sources,
        "source_unchanged": all(sha256(path) == sources[str(path)] for path in paths),
        "host": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "requested_duration_s": duration_s,
        "requested_pairs_per_second": pairs_per_second,
        "generated_pairs": count,
        "producer_seconds": producer_seconds,
        "total_seconds_including_drain": total_seconds,
        "offline_decode_seconds": decode_seconds,
        "effective_callbacks_per_second": 2 * count / producer_seconds,
        "callback_latency_ms_p50": percentile(callback_ns, 0.5) / 1e6,
        "callback_latency_ms_p99": percentile(callback_ns, 0.99) / 1e6,
        "callback_latency_ms_max": max(callback_ns) / 1e6,
        "producer_lateness_ms_p99": percentile(lateness_ns, 0.99) / 1e6,
        "producer_lateness_ms_max": max(lateness_ns) / 1e6,
        "rising_synthetic_bit30_edges": rising,
        "falling_synthetic_bit30_edges": falling,
        "compressed_bytes": (output / "packets.jsonl.gz").stat().st_size,
        "capture": summary,
        "interpretation": interpretation,
        "capture_integrity_pass": summary["all_observed_callbacks_preserved"]
        and monotonic_indices
        and observed_packets == 2 * count
        and rising == falling == 1,
        "dds_transport_tested": False,
        "real_time_scheduling_qualified": False,
        "physical_cause_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    with (output / "benchmark.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--pairs-per-second", type=int, default=500)
    args = parser.parse_args()
    report = benchmark(
        args.output_directory, duration_s=args.duration_seconds, pairs_per_second=args.pairs_per_second
    )
    print(json.dumps(report, indent=2))
    return 0 if report["capture_integrity_pass"] and report["source_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
