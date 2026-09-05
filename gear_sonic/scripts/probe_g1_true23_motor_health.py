"""Read CRC-checked G1 motor state; never publish commands or change modes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import threading
import time

from gear_sonic.scripts.pico_g1_preflight import (
    TRUE23_HARDWARE_JOINT_IDS,
    _load_lowstate_api,
    _validate_lowstate_message,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path, help="Save read-only telemetry; never overwrite existing evidence")
    args = parser.parse_args()
    if not 0.1 <= args.duration_seconds <= 30:
        parser.error("duration-seconds must be between 0.1 and 30")
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    started_utc = datetime.now(timezone.utc).isoformat()
    initialize, subscriber_type, lowstate_type = _load_lowstate_api()
    from unitree_sdk2py.utils.crc import CRC

    crc = CRC()
    lock = threading.Lock()
    latest = None
    first_tick = None
    last_tick = None
    valid_count = 0
    invalid_count = 0
    advancing_count = 0
    errors: list[str] = []
    first_arrival = None
    last_arrival = None
    max_gap_s = 0.0

    def callback(message) -> None:
        nonlocal latest, first_tick, last_tick, valid_count, invalid_count
        nonlocal advancing_count, first_arrival, last_arrival, max_gap_s
        arrival = time.monotonic()
        with lock:
            try:
                sample = _validate_lowstate_message(message, arrival, TRUE23_HARDWARE_JOINT_IDS)
                if crc.Crc(message) != message.crc:
                    raise ValueError("LowState CRC mismatch")
                if last_tick is not None:
                    delta = (sample.tick - last_tick) % (1 << 32)
                    advancing_count += int(0 < delta < (1 << 31))
                if last_arrival is not None:
                    max_gap_s = max(max_gap_s, arrival - last_arrival)
                first_tick = sample.tick if first_tick is None else first_tick
                first_arrival = arrival if first_arrival is None else first_arrival
                last_arrival = arrival
                last_tick = sample.tick
                latest = message
                valid_count += 1
            except Exception as error:
                invalid_count += 1
                if len(errors) < 3:
                    errors.append(str(error))

    route = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        route.connect(("192.168.123.161", 9))
        local_ip = route.getsockname()[0]
    finally:
        route.close()
    membership = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    membership.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    membership.bind(("", 0))
    membership.setsockopt(
        socket.IPPROTO_IP,
        socket.IP_ADD_MEMBERSHIP,
        socket.inet_aton("239.255.0.1") + socket.inet_aton(local_ip),
    )
    subscriber = None
    try:
        initialize(0, args.interface)
        subscriber = subscriber_type("rt/lowstate", lowstate_type)
        subscriber.Init(callback, 10)
        time.sleep(args.duration_seconds)
    finally:
        if subscriber is not None:
            subscriber.Close()
        membership.close()
    with lock:
        result = {
            "kind": "g1_true23_motor_health_readonly_v1",
            "read_only": True,
            "started_utc": started_utc,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "robot_commands_published": False,
            "interface": args.interface,
            "valid_crc_samples": valid_count,
            "invalid_samples": invalid_count,
            "validation_errors": errors,
            "first_tick": first_tick,
            "last_tick": last_tick,
            "advancing_samples": advancing_count,
            "maximum_arrival_gap_seconds": max_gap_s,
            "observed_rate_hz": (
                (valid_count - 1) / (last_arrival - first_arrival)
                if first_arrival is not None and last_arrival > first_arrival
                else None
            ),
            "latest_age_seconds": None if last_arrival is None else time.monotonic() - last_arrival,
        }
        if latest is not None:
            motors = [
                {
                    "compact_index": compact,
                    "motor_slot": slot,
                    "mode": int(latest.motor_state[slot].mode),
                    "motorstate": int(latest.motor_state[slot].motorstate),
                    "q_rad": float(latest.motor_state[slot].q),
                    "dq_rad_s": float(latest.motor_state[slot].dq),
                    "tau_est_nm": float(latest.motor_state[slot].tau_est),
                }
                for compact, slot in enumerate(TRUE23_HARDWARE_JOINT_IDS)
            ]
            result.update(
                {
                    "mode_machine": int(latest.mode_machine),
                    "crc": int(latest.crc),
                    "motors": motors,
                    "nonzero_mode_count": sum(row["mode"] != 0 for row in motors),
                    "motorstate_bit30_slots": [
                        row["motor_slot"] for row in motors if row["motorstate"] & (1 << 30)
                    ],
                    "imu_rpy_rad": [float(value) for value in latest.imu_state.rpy],
                    "imu_quaternion_wxyz": [float(value) for value in latest.imu_state.quaternion],
                    "absent_slot_modes": {
                        str(slot): int(latest.motor_state[slot].mode)
                        for slot in range(29)
                        if slot not in TRUE23_HARDWARE_JOINT_IDS
                    },
                }
            )
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    print(serialized)
    return 0 if valid_count >= 2 and advancing_count > 0 and invalid_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
