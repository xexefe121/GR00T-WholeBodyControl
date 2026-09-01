"""Run managed real-PICO teleoperation on a gantry-supported true23 G1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Sequence

from gear_sonic.scripts.authorize_g1_true23_causal_gantry import (
    AUTHORIZATION_PHRASE,
)
from gear_sonic.scripts.run_g1_true23_frozen_lora_dance_gantry import (
    _active_command,
    _existing_file,
    _new_output,
    _signal_active_controller,
    _signal_wsl_matching,
    _wsl_path,
)


def _published_reference(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "reference_packet_published":
            return True
        if record.get("event") == "session_failed":
            raise RuntimeError(f"live PICO publisher failed: {record.get('failure')}")
    return False


def _live_publisher_command(
    *,
    distro: str,
    publisher_python: str,
    workspace: str,
    xrt_module_dir: str,
    soma_source_root: str,
    capture_python: str,
    endpoint: str,
    packets: int,
    timeout_seconds: int,
    evidence: str,
    pico_client_apk_sha256: str,
) -> list[str]:
    python_path = f"{workspace}:{xrt_module_dir}"
    return [
        "wsl.exe",
        "-d",
        distro,
        "--",
        "env",
        f"PYTHONPATH={python_path}",
        publisher_python,
        "-m",
        "gear_sonic.scripts.stream_g1_23dof_pico_causal_zmq",
        "--workspace",
        workspace,
        "--xrt-module-dir",
        xrt_module_dir,
        "--soma-source-root",
        soma_source_root,
        "--capture-python",
        capture_python,
        "--bind",
        endpoint,
        "--packets",
        str(packets),
        "--timeout-seconds",
        str(timeout_seconds),
        "--subscriber-warmup-s",
        "2",
        "--frame-timeout-s",
        "2",
        "--pico-client-apk-sha256",
        pico_client_apk_sha256,
        "--evidence",
        evidence,
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--xrt-module-dir", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--active-promotion", type=Path, required=True)
    parser.add_argument("--live-shadow-evidence", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--publisher-evidence", type=Path, required=True)
    parser.add_argument("--pico-client-apk-sha256", required=True)
    parser.add_argument("--gantry-authorize", required=True)
    parser.add_argument("--duration-seconds", type=int, default=10)
    parser.add_argument("--packets", type=int, default=90_000)
    parser.add_argument("--timeout-seconds", type=int, default=1_800)
    parser.add_argument("--startup-timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--publisher-python",
        default="/root/.venvs/g1_true23_soma/bin/python",
    )
    parser.add_argument("--capture-python", default="/usr/bin/python3")
    parser.add_argument("--soma-source-root", default="/root/.cache/g1_true23_soma/source")
    parser.add_argument("--network", default="eth0")
    parser.add_argument("--pico-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--distro", default="Ubuntu-22.04")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.gantry_authorize != AUTHORIZATION_PHRASE:
        raise ValueError("exact explicit gantry authorization phrase is required")
    if not 1 <= args.duration_seconds <= 10:
        raise ValueError("duration-seconds must be between 1 and 10")
    if args.packets < 100 or args.timeout_seconds < 30:
        raise ValueError("live publisher target is too short")
    if args.startup_timeout_seconds < 10:
        raise ValueError("startup timeout must be at least 10 seconds")
    if args.network != "eth0" or args.pico_endpoint != "tcp://127.0.0.1:5557":
        raise ValueError("reviewed live launcher requires eth0 and localhost port 5557")
    apk_sha = args.pico_client_apk_sha256.strip().lower()
    if len(apk_sha) != 64 or any(char not in "0123456789abcdef" for char in apk_sha):
        raise ValueError("pico-client-apk-sha256 must be 64 lowercase hex chars")

    root = args.repository_root.expanduser().resolve(strict=True)
    xrt_dir = args.xrt_module_dir.expanduser().resolve(strict=True)
    if not xrt_dir.is_dir():
        raise ValueError("xrt-module-dir must be a directory")
    binary = _existing_file(
        root / "gear_sonic_deploy/target/release/g1_true23_active_gantry",
        "active controller",
    )
    encoder = _existing_file(args.encoder, "encoder")
    metadata = _existing_file(args.decoder_report, "decoder report")
    decoder_name = json.loads(metadata.read_text(encoding="utf-8")).get(
        "decoder", {}
    ).get("filename")
    if not isinstance(decoder_name, str) or not decoder_name:
        raise ValueError("decoder report does not name decoder bytes")
    decoder = _existing_file(metadata.with_name(decoder_name), "decoder")
    promotion = _existing_file(args.promotion, "promotion")
    active_promotion = _existing_file(args.active_promotion, "active promotion")
    shadow = _existing_file(args.live_shadow_evidence, "live shadow evidence")
    evidence = _new_output(args.evidence, "controller evidence")
    publisher_evidence = _new_output(
        args.publisher_evidence, "publisher evidence"
    )

    converted = {
        "root": _wsl_path(root, args.distro),
        "xrt": _wsl_path(xrt_dir, args.distro),
        "binary": _wsl_path(binary, args.distro),
        "encoder": _wsl_path(encoder, args.distro),
        "decoder": _wsl_path(decoder, args.distro),
        "metadata": _wsl_path(metadata, args.distro),
        "promotion": _wsl_path(promotion, args.distro),
        "active_promotion": _wsl_path(active_promotion, args.distro),
        "shadow": _wsl_path(shadow, args.distro),
        "evidence": _wsl_path(evidence, args.distro),
        "publisher_evidence": _wsl_path(publisher_evidence, args.distro),
    }
    publisher_command = _live_publisher_command(
        distro=args.distro,
        publisher_python=args.publisher_python,
        workspace=converted["root"],
        xrt_module_dir=converted["xrt"],
        soma_source_root=args.soma_source_root,
        capture_python=args.capture_python,
        endpoint=args.pico_endpoint,
        packets=args.packets,
        timeout_seconds=args.timeout_seconds,
        evidence=converted["publisher_evidence"],
        pico_client_apk_sha256=apk_sha,
    )
    controller_command = _active_command(
        distro=args.distro,
        binary=converted["binary"],
        encoder=converted["encoder"],
        decoder=converted["decoder"],
        metadata=converted["metadata"],
        promotion=converted["promotion"],
        active_promotion=converted["active_promotion"],
        live_shadow_evidence=converted["shadow"],
        authorization_id=args.authorization_id,
        network=args.network,
        endpoint=args.pico_endpoint,
        evidence=converted["evidence"],
        duration_seconds=args.duration_seconds,
        gantry_authorize=args.gantry_authorize,
    )

    print(
        "[WAIT] Starting real PICO producer; robot command channel remains closed.",
        flush=True,
    )
    publisher = subprocess.Popen(publisher_command, cwd=root)
    publisher_pattern = (
        f"^{args.publisher_python} -m "
        "gear_sonic.scripts.stream_g1_23dof_pico_causal_zmq "
        f".*--evidence {converted['publisher_evidence']}( |$)"
    )
    controller: subprocess.Popen[bytes] | None = None
    deadline = time.monotonic() + args.startup_timeout_seconds
    try:
        while not _published_reference(publisher_evidence):
            status = publisher.poll()
            if status is not None:
                raise RuntimeError(f"live PICO publisher exited before reference ({status})")
            if time.monotonic() >= deadline:
                raise TimeoutError("live PICO publisher did not produce a fresh reference")
            time.sleep(0.1)
        print("[PASS] Fresh real PICO reference observed.", flush=True)
        print(
            "[OPERATOR] Hold L2; after READY, release A then press A once.",
            flush=True,
        )
        controller = subprocess.Popen(controller_command, cwd=root)
        while controller.poll() is None:
            if publisher.poll() is not None:
                _signal_active_controller(
                    distro=args.distro,
                    binary=converted["binary"],
                    evidence=converted["evidence"],
                )
                controller.wait(timeout=15)
                raise RuntimeError("live PICO publisher stopped during control")
            time.sleep(0.05)
        controller_status = int(controller.returncode)
    except KeyboardInterrupt:
        if controller is not None:
            _signal_active_controller(
                distro=args.distro,
                binary=converted["binary"],
                evidence=converted["evidence"],
            )
            controller_status = controller.wait(timeout=15)
        else:
            controller_status = 130
    finally:
        _signal_wsl_matching(
            distro=args.distro,
            pattern=publisher_pattern,
            signal_name="TERM",
        )
        try:
            publisher_status = publisher.wait(timeout=15)
        except subprocess.TimeoutExpired:
            publisher.terminate()
            publisher_status = publisher.wait(timeout=5)

    if publisher_status != 0:
        raise RuntimeError(f"live PICO publisher cleanup failed ({publisher_status})")
    print(f"[EVIDENCE] controller={evidence}", flush=True)
    print(f"[EVIDENCE] publisher={publisher_evidence}", flush=True)
    return controller_status


if __name__ == "__main__":
    raise SystemExit(main())
