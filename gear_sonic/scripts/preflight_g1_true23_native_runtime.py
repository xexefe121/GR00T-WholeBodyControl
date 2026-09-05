"""Inspect the selected true23 runtime and prepare same-host shadow commands.

Run with the Python interpreter intended for replay, on WSL or the Jetson.
Uses standard-library inspection plus `ldd`/`pgrep`; never executes a control
binary, imports a robot SDK, opens DDS, or launches a publisher. Passing this
preflight means files and dependencies are ready, not that balance is proven.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

# Keep this inventory importable before installing NumPy, XRT, Torch or SOMA.
# These are the selected health probe's reviewed byte identities; a new ARM
# build needs its own reviewed identity, not an automatic architecture bypass.
EXPECTED_SERVICE_SHA256 = "8654b4f3552e36e1223f6589491ebe6c82002a07a09520fae7f257465ce0bbbc"
EXPECTED_XRT_SHA256 = "34eeb4484fb68e860ef4c7a1617022e020084a041b98226aea80f6eb93483de1"
SELECTED_PINS = {
    "encoder": "733353148bef1eb8dd83a96416b7a89f0b5c3530ceb9e0cec9c25fdb04f56ff2",
    "decoder": "44d1fb2701f1e65460f1c2c23f676bce4f1d4a44b3b112798dc5034af37946b8",
    "decoder_report": "02197e5682a9bddc8f11aa6fa9c32ba909b97ec7d1c316c9a0d660cba2d25b7d",
    "packet_bundle": "237910ad5dfc370db9645e52f08ba0ca3b0f409a1383d692e1ce1937c5e3dc9d",
}
ARTIFACT_ROOT = Path("artifacts/g1_true23_frozen_lora")
# The native Jetson may still use Python 3.8, where CompletedProcess is not
# subscriptable when this alias is evaluated at import time.
RunCommand = Callable[[Sequence[str]], Any]


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def elf_architecture(path: Path) -> str:
    """Reject Git LFS pointers and Windows-checkout SONAME text stubs."""
    with path.open("rb") as stream:
        header = stream.read(64)
    if len(header) < 64 or header[:4] != b"\x7fELF" or header[4:6] != b"\x02\x01":
        raise ValueError("expected a materialized 64-bit little-endian ELF")
    machine = int.from_bytes(header[18:20], "little")
    if machine not in {62, 183}:
        raise ValueError(f"unsupported ELF machine {machine}")
    return {62: "x86_64", 183: "aarch64"}[machine]


def inspect_binary(
    path: Path, *, system: str, machine: str, run: RunCommand = _run, executable: bool = True
) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "ready": False}
    try:
        result["architecture"] = elf_architecture(path)
        result["sha256"] = _sha256(path)
        expected = {"AMD64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
        if result["architecture"] != expected:
            raise ValueError(f"binary is {result['architecture']}; host requires {expected}")
        if system != "Linux":
            raise ValueError("ELF runtime checks must run inside WSL or native Linux")
        if executable and not os.access(path, os.X_OK):
            raise ValueError("binary lacks execute permission")
        dependencies = run(["ldd", str(path)])
        result["dependency_output"] = (dependencies.stdout + dependencies.stderr).strip()
        if dependencies.returncode != 0 or "not found" in result["dependency_output"]:
            raise ValueError("dynamic dependencies unresolved; inspect dependency_output")
        result["ready"] = True
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    return result


def inspect_selected_artifacts(paths: dict[str, Path]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for role, expected in SELECTED_PINS.items():
        path = paths[role]
        check: dict[str, Any] = {"path": str(path), "ready": False, "expected_sha256": expected}
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("artifact must be a regular non-symlink file")
            check["sha256"] = _sha256(path)
            if check["sha256"] != expected:
                raise ValueError("selected artifact SHA-256 mismatch")
            check["ready"] = True
        except (OSError, ValueError) as error:
            check["error"] = str(error)
        checks[role] = check
    promotion: dict[str, Any] = {"path": str(paths["promotion"]), "ready": False}
    try:
        value = json.loads(paths["promotion"].read_text(encoding="utf-8"))
        if (
            value.get("kind") != "g1_true23_frozen_lora_dance_shadow_admission_v1"
            or value.get("native_action_dof") != 23
            or value.get("deployment_bytes_authorized_for_shadow") is not True
            or value.get("active_motor_control_authorized") is not False
            or value.get("free_standing_authorized") is not False
        ):
            raise ValueError("expected selected true23 shadow-only admission")
        for role, expected in SELECTED_PINS.items():
            if value.get("source_artifacts", {}).get(f"{role}_sha256") != expected:
                raise ValueError(f"promotion does not bind selected {role}")
        payload = {key: item for key, item in value.items() if key != "promotion_payload_sha256"}
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if hashlib.sha256(encoded).hexdigest() != value.get("promotion_payload_sha256"):
            raise ValueError("promotion payload SHA-256 mismatch")
        promotion["ready"] = True
    except (OSError, ValueError, AttributeError) as error:
        promotion["error"] = str(error)
    checks["promotion"] = promotion
    return checks


def inspect_headset_report(path: Path | None, *, now: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": None if path is None else str(path), "ready": False}
    try:
        if path is None:
            raise ValueError("no recent live headset health report supplied")
        value = json.loads(path.read_text(encoding="utf-8"))
        age = (time.time() if now is None else now) - path.stat().st_mtime
        result["file_age_seconds"] = age
        if not 0 <= age <= 300:
            raise ValueError("headset report is older than 300 seconds or from the future")
        health = value.get("latest_health", {})
        required = {
            "health_available": True,
            "health_valid": True,
            "health_supported": True,
            "health_connect_state_result": 0,
            "health_calibrated": True,
            "health_calibration_result": 0,
            "health_is_tracking": True,
            "health_tracking_state_code": 0,
            "health_body_data_result": 0,
            "health_body_state_result": 0,
            "health_body_error_code": 0,
            "health_body_role_count": 24,
        }
        health_ready = all(
            type(health.get(key)) is type(expected) and health[key] == expected
            for key, expected in required.items()
        ) and all(
            type(health.get(key)) is int and health[key] >= 2
            for key in ("health_connected_band_count", "health_unique_tracker_count")
        )
        if (
            value.get("kind") != "g1_true23_pico_tracking_health_probe_v1"
            or value.get("passed") is not True
            or value.get("xrt_binding_sha256") != EXPECTED_XRT_SHA256
            or value.get("service_binary_sha256") != EXPECTED_SERVICE_SHA256
            or value.get("authorization", {}).get("read_only") is not True
            or value.get("authorization", {}).get("robot_commands_published") is not False
            or not health_ready
        ):
            raise ValueError("headset report provenance or calibrated 24-role tracking failed")
        result["ready"] = True
    except (OSError, ValueError, TypeError, AttributeError) as error:
        result["error"] = str(error)
    return result


def collect(args: argparse.Namespace, *, run: RunCommand = _run) -> dict[str, Any]:
    root = args.repository_root.expanduser().resolve()
    system, machine = platform.system(), platform.machine()
    if re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", args.network) is None:
        raise ValueError("network must name one Linux interface")
    interface = {
        "name": args.network,
        "present": system == "Linux" and (Path("/sys/class/net") / args.network).is_dir(),
    }
    environment = "wsl" if "microsoft" in platform.release().lower() else "native_linux"
    if system != "Linux":
        environment = "windows_run_this_preflight_inside_wsl" if system == "Windows" else system.lower()
    report_path = (
        args.decoder_report
        or root / ARTIFACT_ROOT / "original_sonic_happy_residual_v1/candidate.plus_0p002.decoder.json"
    )
    paths = {
        "encoder": args.encoder.expanduser().resolve(),
        "decoder_report": report_path.expanduser().resolve(),
        "promotion": (
            args.promotion
            or root / ARTIFACT_ROOT / "physical_dance_v1/candidate.plus_0p002.dance_shadow_promotion.v2.json"
        ).resolve(),
        "packet_bundle": (
            args.packet_bundle
            or root / ARTIFACT_ROOT / "physical_dance_v1/original_sonic_happy.true23.causal_packets.json"
        ).resolve(),
    }
    # Only the pinned report's co-located selected decoder is admissible.
    paths["decoder"] = paths["decoder_report"].with_name("candidate.plus_0p002.decoder.onnx")
    artifacts = inspect_selected_artifacts(paths)
    binaries = {
        name: inspect_binary(
            root / "gear_sonic_deploy/target/release" / name, system=system, machine=machine, run=run
        )
        for name in ("g1_true23_live_shadow", "g1_true23_active_gantry")
    }
    python_modules = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "zmq")}
    service_path = args.service_binary.expanduser().resolve()
    xrt_dir = (
        args.xrt_module_dir or root / "external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
    ).resolve()
    bindings = sorted(xrt_dir.glob("xrobotoolkit_sdk*.so"))
    service: dict[str, Any] = {"path": str(service_path), "present": service_path.is_file(), "running": False}
    if service["present"]:
        service["sha256"] = _sha256(service_path)
        service["pinned_identity_matches"] = service["sha256"] == EXPECTED_SERVICE_SHA256
        if system == "Linux":
            try:
                service["running"] = run(["pgrep", "-f", "^" + str(service_path) + "( |$)"]).returncode == 0
            except (OSError, subprocess.SubprocessError) as error:
                service["error"] = str(error)
    binding = {"directory": str(xrt_dir), "candidates": [str(path) for path in bindings], "ready": False}
    if len(bindings) == 1:
        binding.update(inspect_binary(bindings[0], system=system, machine=machine, run=run, executable=False))
        binding["pinned_identity_matches"] = binding.get("sha256") == EXPECTED_XRT_SHA256
        binding["ready"] = binding["ready"] and binding["pinned_identity_matches"]
    health = inspect_headset_report(args.health_report)
    artifacts_ready = all(check["ready"] for check in artifacts.values())
    shadow_ready = (
        artifacts_ready
        and binaries["g1_true23_live_shadow"]["ready"]
        and all(python_modules.values())
        and interface["present"]
    )
    commands = None
    if shadow_ready:
        out = root / ARTIFACT_ROOT / "native_runtime_preflight"
        commands = {
            "working_directory": str(root),
            "environment": {"PYTHONPATH": str(root)},
            "shadow_argv": [
                str(root / "gear_sonic_deploy/target/release/g1_true23_live_shadow"),
                "--mode",
                "shadow",
                "--encoder",
                str(paths["encoder"]),
                "--decoder",
                str(paths["decoder"]),
                "--metadata",
                str(paths["decoder_report"]),
                "--promotion",
                str(paths["promotion"]),
                "--network",
                args.network,
                "--pico-endpoint",
                "tcp://127.0.0.1:5557",
                "--frames",
                "100",
                "--evidence",
                str(out / "new_shadow.jsonl"),
            ],
            "publisher_argv": [
                sys.executable,
                "-m",
                "gear_sonic.scripts.replay_g1_true23_pico_packets_zmq",
                "--packets",
                str(paths["packet_bundle"]),
                "--timestamp-clock",
                "local",
                "--repeat-count",
                "10",
                "--subscriber-warmup-s",
                "2",
                "--output",
                str(out / "new_publisher.json"),
            ],
        }
    return {
        "schema_version": 1,
        "kind": "g1_true23_native_runtime_preflight_v1",
        "repository_root": str(root),
        "runtime": environment,
        "host_architecture": machine,
        "python": sys.executable,
        "python_modules": python_modules,
        "network_interface": interface,
        "artifacts": artifacts,
        "binaries": binaries,
        "pico_service": service,
        "pico_binding": binding,
        "headset_health": health,
        "saved_clip_shadow_runtime_ready": shadow_ready,
        "active_binary_dependencies_ready": binaries["g1_true23_active_gantry"]["ready"],
        "live_headset_prerequisites_present": binding["ready"]
        and service.get("pinned_identity_matches", False)
        and service["running"]
        and health["ready"],
        "same_host_shadow_commands": commands,
        "clock_contract": (
            "publisher and C++ consumer must share the same Linux monotonic clock; "
            "remote host timestamps are not interchangeable"
        ),
        "physical_teleop_ready": False,
        "authorization": {"read_only": True, "dds_opened": False, "robot_commands_published": False},
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--encoder", type=Path, required=True)
    result.add_argument("--decoder-report", type=Path)
    result.add_argument("--promotion", type=Path)
    result.add_argument("--packet-bundle", type=Path)
    result.add_argument("--xrt-module-dir", type=Path)
    result.add_argument(
        "--service-binary", type=Path, default=Path("/opt/apps/roboticsservice/RoboticsServiceProcess")
    )
    result.add_argument("--health-report", type=Path)
    result.add_argument("--network", default="eth0")
    result.add_argument("--output", type=Path, help="Save this read-only inspection without overwriting evidence")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    report = collect(args)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    print(serialized)
    return 0 if report["saved_clip_shadow_runtime_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
