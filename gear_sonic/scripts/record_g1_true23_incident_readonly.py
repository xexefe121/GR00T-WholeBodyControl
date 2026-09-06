"""Passively capture G1 state, motor-command and ownership-RPC DDS topics.

No commands, mode queries or changes are sent. This observes traffic published
by other processes; observed command traffic is not proof the robot applied it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import ipaddress
import json
from pathlib import Path
import platform
import socket
import sys
import time

from gear_sonic.utils.g1_true23_incident_capture import PacketCapture, TOPICS, sha256


def load_readonly_api():
    # Importing IDL and subscriber classes does not initialize a participant.
    # Do not instantiate RPC clients: even a query would publish a request.
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_, Response_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    return (
        ChannelFactoryInitialize,
        ChannelSubscriber,
        {
            "LowState_": LowState_,
            "LowCmd_": LowCmd_,
            "Request_": Request_,
            "Response_": Response_,
        },
        CRC(),
    )


def decode_packet(topic, payload, types, crc):
    message = types[TOPICS[topic]].deserialize(payload)
    if topic == "rt/lowstate":
        motors = message.motor_state
        if len(motors) != 35:
            raise ValueError("expected exact 35-slot HG LowState")
        computed_crc = int(crc.Crc(message))
        return {
            "tick": int(message.tick),
            "version_raw": list(message.version),
            "mode_machine": int(message.mode_machine),
            "mode_pr": int(message.mode_pr),
            "crc_received": int(message.crc),
            "crc_computed": computed_crc,
            "crc_matches": int(message.crc) == computed_crc,
            "motor_modes": [int(m.mode) for m in motors],
            "motor_status": [int(m.motorstate) for m in motors],
            "q_rad": [float(m.q) for m in motors],
            "dq_rad_s": [float(m.dq) for m in motors],
            "tau_est_nm": [float(m.tau_est) for m in motors],
            "motor_vol_raw": [float(m.vol) for m in motors],
            "motor_temperature_raw": [list(m.temperature) for m in motors],
            "imu_quaternion_wxyz": list(message.imu_state.quaternion),
            "imu_rpy_rad": list(message.imu_state.rpy),
            "imu_gyroscope": list(message.imu_state.gyroscope),
            "imu_accelerometer": list(message.imu_state.accelerometer),
            "wireless_remote_hex": bytes(message.wireless_remote).hex(),
        }
    if TOPICS[topic] == "LowCmd_":
        return {
            "mode_machine": int(message.mode_machine),
            "mode_pr": int(message.mode_pr),
            "crc_received": int(message.crc),
            "crc_computed": int(crc.Crc(message)),
            "motor_modes": [int(m.mode) for m in message.motor_cmd],
            "q_rad": [float(m.q) for m in message.motor_cmd],
            "kp": [float(m.kp) for m in message.motor_cmd],
            "kd": [float(m.kd) for m in message.motor_cmd],
            "tau_ff": [float(m.tau) for m in message.motor_cmd],
            "robot_received_or_applied_this_command_proven": False,
        }
    result = {
        "request_id": int(message.header.identity.id),
        "api_id": int(message.header.identity.api_id),
        "rpc_match_or_causal_relation_proven": False,
    }
    if TOPICS[topic] == "Request_":
        result["parameter"] = message.parameter
    else:
        result["status_code"] = int(message.header.status.code)
        result["data"] = message.data
    return result


def collect(output: Path, *, interface: str, duration_s: float, api=None, multicast_local_ip=None):
    if not 0.1 <= duration_s <= 300 or not interface:
        raise ValueError("require interface and 0.1..300 s duration")
    if output.exists():
        raise FileExistsError(output)
    if multicast_local_ip is not None:
        multicast_local_ip = str(ipaddress.IPv4Address(multicast_local_ip))
    initialize, subscriber_type, types, crc = load_readonly_api() if api is None else api
    schema_paths = [Path(inspect.getfile(kind)).resolve() for kind in types.values()]
    schema_paths += [Path(inspect.getfile(type(crc))).resolve(), Path(__file__).resolve()]
    schema_paths += [Path(inspect.getfile(PacketCapture)).resolve()]
    schema_paths += [Path(inspect.getfile(initialize)).resolve(), Path(inspect.getfile(subscriber_type)).resolve()]
    schema_paths += [
        Path(module.__file__).resolve()
        for name, module in tuple(sys.modules.items())
        if name.startswith(("unitree_sdk2py.idl.unitree_hg.", "unitree_sdk2py.idl.unitree_api."))
        and getattr(module, "__file__", None)
    ]
    native_crc = getattr(getattr(crc, "crc_lib", None), "_name", None)
    if native_crc is not None and Path(native_crc).is_file():
        schema_paths.append(Path(native_crc).resolve())
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "boot_id": boot_path.read_text().strip() if boot_path.is_file() else None,
        "clock_implementation": time.get_clock_info("monotonic").implementation,
        "python_version": platform.python_version(),
        "start_monotonic_ns": time.monotonic_ns(),
        "interface": interface,
        "duration_seconds": duration_s,
        "multicast_local_ip": multicast_local_ip,
        "schema_source_sha256": {str(path): sha256(path) for path in schema_paths},
        "voltage_temperature_and_version_fields_are_raw_not_firmware_diagnosis": True,
        "received_time_is_local_callback_not_source_clock": True,
    }
    # Full IDL decoding/CRC/expanded JSON cannot keep up with the measured
    # 500 Hz state + 500 Hz command stream on this host. Preserve raw CDR now;
    # decode after every subscriber is closed, in the separate offline tool.
    capture = PacketCapture(output, metadata=metadata)
    subscribers = []
    membership = None
    errors = []
    opened = False
    try:
        if multicast_local_ip is not None:
            membership = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            membership.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            membership.bind(("", 0))
            membership.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_ADD_MEMBERSHIP,
                socket.inet_aton("239.255.0.1") + socket.inet_aton(multicast_local_ip),
            )
        initialize(0, interface)
        for topic, type_name in TOPICS.items():
            subscriber = subscriber_type(topic, types[type_name])
            subscribers.append(subscriber)
            # Avoid SDK's extra, unaccounted user queue. Our bounded queue
            # records its drops; upstream DDS/network losses remain unknown.
            subscriber.Init(lambda message, topic=topic: capture.receive(topic, message), 0)
            opened = True
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline and capture.failure is None:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        errors.append("operator_interrupted_capture_only")
    except Exception as failure:
        errors.append(f"{type(failure).__name__}: {failure}")
    finally:
        for subscriber in reversed(subscribers):
            try:
                subscriber.Close()
            except Exception as failure:
                errors.append(f"subscription_close_error: {failure}")
        if membership is not None:
            try:
                membership.close()
            except Exception as failure:
                errors.append(f"membership_close_error: {failure}")
        for path in sorted(set(schema_paths)):
            try:
                if sha256(path) != metadata["schema_source_sha256"][str(path)]:
                    errors.append(f"capture_source_changed: {path}")
            except OSError as failure:
                errors.append(f"capture_source_check_error: {failure}")
    return capture.finish(transport_error="; ".join(errors) or None, subscriptions_opened=opened)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--duration-seconds", type=float, default=30)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--multicast-local-ip", help="Optional local interface IPv4 for explicit IGMP membership")
    args = parser.parse_args()
    report = collect(
        args.output_directory,
        interface=args.interface,
        duration_s=args.duration_seconds,
        multicast_local_ip=args.multicast_local_ip,
    )
    print(json.dumps(report))
    return (
        0
        if report["capture_completed"]
        and report["all_observed_callbacks_preserved"]
        and report["observed_lowstate"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
