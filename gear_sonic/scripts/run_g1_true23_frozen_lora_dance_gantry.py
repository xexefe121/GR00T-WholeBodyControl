"""Run one managed saved-SONIC dance session on a gantry-supported true23 G1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Sequence

from gear_sonic.scripts.authorize_g1_true23_causal_gantry import (
    AUTHORIZATION_PHRASE,
)
from gear_sonic.scripts.authorize_g1_true23_frozen_lora_dance_gantry import (
    DIRECT_DANCE_COMMAND,
)


def _existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return resolved


def _new_output(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite {label}: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _wsl_path(path: Path, distro: str) -> str:
    del distro
    normalized = str(path.resolve()).replace("\\", "/")
    if len(normalized) < 3 or normalized[1:3] != ":/":
        raise ValueError(f"launcher requires a Windows drive path: {path}")
    drive = normalized[0].lower()
    if not drive.isalpha():
        raise ValueError(f"invalid Windows drive path: {path}")
    return f"/mnt/{drive}/{normalized[3:]}"


def _active_command(
    *,
    distro: str,
    binary: str,
    encoder: str,
    decoder: str,
    metadata: str,
    promotion: str,
    active_promotion: str,
    live_shadow_evidence: str,
    authorization_id: str,
    network: str,
    endpoint: str,
    evidence: str,
    duration_seconds: int,
    gantry_authorize: str,
    direct_dance_command: str,
) -> list[str]:
    return [
        "wsl.exe",
        "-d",
        distro,
        "--",
        "stdbuf",
        "-oL",
        "-eL",
        binary,
        "--encoder",
        encoder,
        "--decoder",
        decoder,
        "--metadata",
        metadata,
        "--promotion",
        promotion,
        "--active-promotion",
        active_promotion,
        "--live-shadow-evidence",
        live_shadow_evidence,
        "--authorization-id",
        authorization_id,
        "--frozen-lora-dance",
        "--network",
        network,
        "--pico-endpoint",
        endpoint,
        "--execute-stage-one",
        "--evidence",
        evidence,
        "--post-arm-duration-seconds",
        str(duration_seconds),
        "--gantry-authorize",
        gantry_authorize,
        "--direct-dance-command",
        direct_dance_command,
    ]


def _signal_active_controller(
    *, distro: str, binary: str, evidence: str
) -> None:
    pattern = f"^{binary} .*--evidence {evidence}( |$)"
    _signal_wsl_matching(distro=distro, pattern=pattern, signal_name="INT")


def _signal_wsl_matching(
    *, distro: str, pattern: str, signal_name: str
) -> None:
    query = subprocess.run(
        ["wsl.exe", "-d", distro, "--", "pgrep", "-f", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    pids = [line for line in query.stdout.splitlines() if line.isdigit()]
    if pids:
        subprocess.run(
            [
                "wsl.exe",
                "-d",
                distro,
                "--",
                "kill",
                f"-{signal_name}",
                *pids,
            ],
            check=False,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--active-promotion", type=Path, required=True)
    parser.add_argument("--live-shadow-evidence", type=Path, required=True)
    parser.add_argument("--packet-bundle", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--publisher-evidence", type=Path, required=True)
    parser.add_argument("--gantry-authorize", required=True)
    parser.add_argument("--direct-dance-command", required=True)
    parser.add_argument("--duration-seconds", type=int, default=5)
    parser.add_argument("--repeat-count", type=int, default=100)
    parser.add_argument("--publisher-warmup-s", type=float, default=5.0)
    parser.add_argument("--network", default="eth0")
    parser.add_argument("--pico-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--distro", default="Ubuntu-22.04")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.gantry_authorize != AUTHORIZATION_PHRASE:
        raise ValueError("exact explicit gantry authorization phrase is required")
    if args.direct_dance_command != DIRECT_DANCE_COMMAND:
        raise ValueError("exact direct dance command DANCE is required")
    if not 1 <= args.duration_seconds <= 5:
        raise ValueError("duration-seconds must be between 1 and 5")
    if not 1 <= args.repeat_count <= 100:
        raise ValueError("repeat-count must be between 1 and 100")
    if args.publisher_warmup_s < 2.0:
        raise ValueError("publisher-warmup-s must be at least 2 seconds")
    if args.network != "eth0" or args.pico_endpoint != "tcp://127.0.0.1:5557":
        raise ValueError("reviewed dance launcher requires eth0 and localhost port 5557")

    root = args.repository_root.expanduser().resolve(strict=True)
    binary = _existing_file(
        root / "gear_sonic_deploy/target/release/g1_true23_active_gantry",
        "active controller",
    )
    encoder = _existing_file(args.encoder, "encoder")
    metadata = _existing_file(args.decoder_report, "decoder report")
    report = json.loads(metadata.read_text(encoding="utf-8"))
    decoder_name = report.get("decoder", {}).get("filename")
    if not isinstance(decoder_name, str) or not decoder_name:
        raise ValueError("decoder report does not name decoder bytes")
    decoder = _existing_file(metadata.with_name(decoder_name), "decoder")
    promotion = _existing_file(args.promotion, "promotion")
    active_promotion = _existing_file(args.active_promotion, "active promotion")
    shadow = _existing_file(args.live_shadow_evidence, "live shadow evidence")
    packets = _existing_file(args.packet_bundle, "saved packet bundle")
    evidence = _new_output(args.evidence, "controller evidence")
    publisher_evidence = _new_output(
        args.publisher_evidence, "publisher evidence"
    )

    converted = {
        "binary": _wsl_path(binary, args.distro),
        "encoder": _wsl_path(encoder, args.distro),
        "decoder": _wsl_path(decoder, args.distro),
        "metadata": _wsl_path(metadata, args.distro),
        "promotion": _wsl_path(promotion, args.distro),
        "active_promotion": _wsl_path(active_promotion, args.distro),
        "shadow": _wsl_path(shadow, args.distro),
        "evidence": _wsl_path(evidence, args.distro),
    }
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
        direct_dance_command=args.direct_dance_command,
    )

    with tempfile.TemporaryDirectory(prefix="g1_true23_dance_") as temporary:
        stop_file = Path(temporary) / "stop-publisher"
        publisher_command = [
            sys.executable,
            "-m",
            "gear_sonic.scripts.replay_g1_true23_pico_packets_zmq",
            "--packets",
            str(packets),
            "--bind",
            args.pico_endpoint,
            "--subscriber-warmup-s",
            str(args.publisher_warmup_s),
            "--timestamp-clock",
            "wsl",
            "--repeat-count",
            str(args.repeat_count),
            "--stop-file",
            str(stop_file),
            "--output",
            str(publisher_evidence),
        ]
        print(
            "[START] Damping-only until fresh policy; DANCE command is bound.",
            flush=True,
        )
        print(
            "[OPERATOR] Physical e-stop ready; app/process STOP remains active.",
            flush=True,
        )
        publisher = subprocess.Popen(publisher_command, cwd=root)
        controller = subprocess.Popen(controller_command, cwd=root)
        controller_status: int | None = None
        try:
            while controller_status is None:
                controller_status = controller.poll()
                publisher_status = publisher.poll()
                if publisher_status is not None and controller_status is None:
                    _signal_active_controller(
                        distro=args.distro,
                        binary=converted["binary"],
                        evidence=converted["evidence"],
                    )
                    controller_status = controller.wait(timeout=15)
                    raise RuntimeError(
                        f"saved-reference publisher exited early ({publisher_status})"
                    )
                time.sleep(0.05)
        except KeyboardInterrupt:
            _signal_active_controller(
                distro=args.distro,
                binary=converted["binary"],
                evidence=converted["evidence"],
            )
            controller_status = controller.wait(timeout=15)
        finally:
            stop_file.touch(exist_ok=True)
            try:
                publisher_status = publisher.wait(timeout=10)
            except subprocess.TimeoutExpired:
                publisher.terminate()
                publisher_status = publisher.wait(timeout=5)

    if publisher_status != 0:
        raise RuntimeError(f"saved-reference publisher failed ({publisher_status})")
    publisher_report = json.loads(publisher_evidence.read_text(encoding="utf-8"))
    if publisher_report.get("passed") is not True:
        raise RuntimeError("saved-reference publisher evidence did not pass")
    if evidence.exists():
        print(f"[EVIDENCE] controller={evidence}", flush=True)
    else:
        print("[NO ACTIVITY EVIDENCE] controller rejected preflight", flush=True)
    print(f"[EVIDENCE] publisher={publisher_evidence}", flush=True)
    return int(controller_status)


if __name__ == "__main__":
    raise SystemExit(main())
