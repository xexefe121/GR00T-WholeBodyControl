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

_EXECUTION_KIND = "g1_true23_stage1_gantry_execution_evidence"
_DIRECT_EVENTS = [
    "session_start",
    "artifact_gate_passed",
    "mutation_gate_open",
    "lowcmd_publisher_created",
    "first_policy_ready_for_arm",
    "pre_arm_hold_prepared",
    "motion_mode_released",
    "pre_arm_hold_gate_open",
    "direct_dance_command_accepted",
    "first_armed_policy_command_written",
    "session_complete",
]


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
    direct_dance_command: str | None = None,
    frozen_lora_policy: bool = True,
) -> list[str]:
    command = [
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
    ]
    if frozen_lora_policy:
        command.append("--frozen-lora-policy")
    if direct_dance_command is not None:
        command.extend(["--direct-dance-command", direct_dance_command])
    return command


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


def validate_direct_dance_execution_evidence(
    path: Path, *, authorization_id: str, duration_seconds: int
) -> dict:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [record.get("event") for record in records] != _DIRECT_EVENTS:
        raise ValueError("controller evidence event sequence is not exact")
    previous_ns = 0
    for record in records:
        if (
            record.get("kind") != _EXECUTION_KIND
            or record.get("authorization_id") != authorization_id
        ):
            raise ValueError("controller evidence identity mismatch")
        monotonic_ns = record.get("monotonic_ns")
        if not isinstance(monotonic_ns, int) or monotonic_ns <= previous_ns:
            raise ValueError("controller evidence clock is not strictly increasing")
        previous_ns = monotonic_ns

    start = records[0]
    publisher = records[3]
    hold_prepared = records[5]
    motion_release = records[6]
    hold_gate = records[7]
    direct = records[8]
    first_policy = records[9]
    terminal = records[10]
    if (
        start.get("operator_contract") != "bounded_direct_dance_command_v1"
        or start.get("post_arm_duration_seconds") != duration_seconds
        or publisher.get("writes_before_event") != 0
        or publisher.get("motion_mode_released") is not False
        or hold_prepared.get("pre_release_lowcmd_writes") != 0
        or hold_prepared.get("kp_fraction") != 0.25
        or hold_prepared.get("feedforward_tau_zero") is not True
        or motion_release.get("pre_release_lowcmd_writes") != 0
        or motion_release.get("first_post_release_command")
        != "sampled_posture_hold"
        or direct.get("command") != DIRECT_DANCE_COMMAND
        or direct.get("policy_ready") is not True
        or first_policy.get("feedforward_tau_zero") is not True
    ):
        raise ValueError("controller startup/command contract mismatch")
    hold_frames = hold_gate.get("pre_arm_hold_frames")
    hold_delay_ns = hold_gate.get("release_to_first_hold_write_ns")
    if (
        not isinstance(hold_frames, int)
        or hold_frames < 25
        or hold_gate.get("required_pre_arm_hold_frames") != 25
        or hold_gate.get("startup_damping_frames") != 0
        or not isinstance(hold_delay_ns, int)
        or not 0 <= hold_delay_ns <= 20_000_000
        or hold_gate.get("maximum_first_hold_write_delay_ns") != 20_000_000
        or hold_gate.get("kp_positive") is not True
        or hold_gate.get("feedforward_tau_zero") is not True
    ):
        raise ValueError("controller pre-arm hold gate failed")
    if (
        terminal.get("passed") is not True
        or terminal.get("policy_prewarmed_before_motion_release") is not True
        or terminal.get("pre_release_lowcmd_writes") != 0
        or terminal.get("pre_arm_hold_gate_open") is not True
        or terminal.get("pre_arm_hold_frames", 0) < 25
        or terminal.get("startup_damping_frames") != 0
        or terminal.get("release_to_first_hold_write_ns") != hold_delay_ns
        or terminal.get("maximum_abs_feedforward_tau_nm") != 0.0
        or terminal.get("final_fault") != "operator_stop"
        or terminal.get("stop_reason")
        != "reviewed_post_arm_duration_complete"
        or terminal.get("required_post_arm_duration_ns")
        != duration_seconds * 1_000_000_000
        or terminal.get("post_arm_elapsed_ns", 0)
        < duration_seconds * 1_000_000_000
        or terminal.get("publisher_write_failed") is not False
    ):
        raise ValueError("controller terminal evidence did not pass safe dance")
    return terminal


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
            "[START] Policy prewarms under Unitree motion mode; first LowCmd "
            "is sampled posture hold. DANCE command is bound.",
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
    if not evidence.exists():
        raise RuntimeError("controller produced no execution evidence")
    if controller_status != 0:
        raise RuntimeError(f"controller failed ({controller_status})")
    validate_direct_dance_execution_evidence(
        evidence,
        authorization_id=args.authorization_id,
        duration_seconds=args.duration_seconds,
    )
    print(f"[EVIDENCE] controller={evidence}", flush=True)
    print(f"[EVIDENCE] publisher={publisher_evidence}", flush=True)
    return int(controller_status)


if __name__ == "__main__":
    raise SystemExit(main())
