"""Separate local XR receiver availability from PICO tracking health.

Read-only diagnostics. A successful local TCP connection is not a headset,
gRPC-health, tracking-quality, simulator-control, or hardware qualification.
The existing hash-pinned tracking probe remains unchanged.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
from typing import Any, Mapping

from gear_sonic.scripts import probe_g1_true23_pico_tracking_health as tracking

LOCAL_XR_ENDPOINT = ("127.0.0.1", 60061)


def local_receiver_reachable() -> bool:
    """Bounded TCP precheck, restricted to the SDK's local XR endpoint."""
    try:
        with socket.create_connection(LOCAL_XR_ENDPOINT, timeout=1.0):
            return True
    except OSError:
        return False


def classify_input(receiver_reachable: bool, report: Mapping[str, Any] | None) -> str:
    if not receiver_reachable:
        return "receiver_unavailable"
    if report is None:
        return "tracking_probe_failed"
    health = report.get("latest_health")
    if not isinstance(health, Mapping) or health.get("health_available") is not True:
        # Zero-filled SDK defaults cannot prove headset absence or calibration state.
        return "receiver_reachable_tracking_unavailable"
    if report.get("passed") is True and tracking.tracking_health_passes(health):
        return "tracking_ready"
    return "tracking_not_ready"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    args = parser.parse_args(argv)
    if not 0.1 <= args.duration_seconds <= 30.0:
        raise ValueError("probe duration must be between 0.1 and 30 seconds")
    output = args.output.expanduser().resolve()
    tracking_output = output.with_name(output.stem + ".tracking.json")
    if output.exists() or tracking_output.exists():
        raise FileExistsError("refusing to overwrite PICO input evidence")

    reachable = local_receiver_reachable()
    health_report, error = None, None
    if reachable:
        try:
            tracking.main(
                [
                    "--output",
                    str(tracking_output),
                    "--duration-seconds",
                    str(args.duration_seconds),
                ]
            )
            health_report = json.loads(tracking_output.read_text())
        except (OSError, RuntimeError, ValueError, ImportError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    status = classify_input(reachable, health_report)
    report = dict(
        schema_version=1,
        kind="g1_true23_pico_input_probe_v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        local_endpoint=dict(host=LOCAL_XR_ENDPOINT[0], port=LOCAL_XR_ENDPOINT[1]),
        receiver_tcp_reachable=reachable,
        receiver_check_scope="local_tcp_accept_only_not_grpc_or_lan_connectivity",
        input_status=status,
        tracking_probe_attempted=reachable,
        tracking_probe_error=error,
        tracking_evidence=(
            None
            if health_report is None
            else dict(
                path=str(tracking_output),
                sha256=hashlib.sha256(tracking_output.read_bytes()).hexdigest(),
            )
        ),
        passed=status == "tracking_ready",
        hardware_ready=False,
        authorization=dict(
            read_only=True,
            poses_published=False,
            dds_opened=False,
            robot_channel_opened=False,
            hardware_authorized=False,
            robot_commands_published=False,
        ),
    )
    tracking._exclusive_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
