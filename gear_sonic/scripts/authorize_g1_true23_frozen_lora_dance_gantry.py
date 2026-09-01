"""Create gantry-only active sidecar from fresh true23 SONIC dance shadow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

from gear_sonic.scripts.authorize_g1_true23_causal_gantry import (
    AUTHORIZATION_PHRASE,
    validate_causal_live_shadow_evidence,
)
from gear_sonic.scripts.promote_g1_true23_frozen_lora_dance import (
    SAFE_TARGET_TRANSFORM_SHA256,
    promotion_body,
)
from gear_sonic.utils.g1_23dof_artifact import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

KIND = "g1_true23_frozen_lora_dance_gantry_active_promotion_v2"
DIRECT_DANCE_COMMAND = "DANCE"
_AUTHORIZATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def active_body(args: argparse.Namespace) -> dict:
    if args.gantry_authorize != AUTHORIZATION_PHRASE:
        raise ValueError("exact explicit gantry authorization phrase is required")
    if not _AUTHORIZATION_ID.fullmatch(args.authorization_id):
        raise ValueError("authorization-id must contain 8-128 safe characters")
    if args.direct_dance_command != DIRECT_DANCE_COMMAND:
        raise ValueError("exact direct dance command DANCE is required")
    promotion = args.promotion.expanduser().resolve(strict=True)
    expected_promotion = promotion_body(args)
    if _object(promotion) != expected_promotion:
        raise ValueError("dance shadow admission differs from re-verified inputs")
    evidence = args.live_shadow_evidence.expanduser().resolve(strict=True)
    age_s = time.time() - evidence.stat().st_mtime
    if not -5.0 <= age_s <= 300.0:
        raise ValueError(f"live-shadow evidence is not fresh (age={age_s:.3f}s)")
    summary = validate_causal_live_shadow_evidence(
        evidence,
        promotion_path=promotion,
        encoder_path=args.encoder,
        decoder_path=args.decoder_report.with_name(
            _object(args.decoder_report)["decoder"]["filename"]
        ),
        metadata_path=args.decoder_report,
        external_safe_target_transform_applied=True,
    )
    if summary["action_frames"] < 100:
        raise ValueError("at least 100 accepted hardware-shadow frames are required")
    report = _object(args.decoder_report)
    decoder = args.decoder_report.with_name(report["decoder"]["filename"]).resolve(strict=True)
    body = {
        "schema_version": 1,
        "kind": KIND,
        "robot_model": "g1_23dof_rev_1_0",
        "required_mode_machine": 4,
        "native_action_dof": 23,
        "deployment_ready": True,
        "active_motor_control_authorized": True,
        "gantry_authorized": True,
        "free_standing_authorized": False,
        "decoder_output_semantics": "raw_native_action",
        "runtime_policy_semantics": "applied_safe_native_action",
        "previous_action_semantics": "applied_safe_native_action",
        "external_safe_target_transform_required": True,
        "safe_target_transform_sha256": SAFE_TARGET_TRANSFORM_SHA256,
        "source_promotion_sha256": sha256_file(promotion),
        "encoder_sha256": sha256_file(args.encoder.resolve(strict=True)),
        "decoder_sha256": sha256_file(decoder),
        "decoder_report_sha256": sha256_file(args.decoder_report.resolve(strict=True)),
        "live_shadow_evidence_sha256": summary["evidence_sha256"],
        "authorization_id": args.authorization_id,
        "stage_one_envelope": {
            "action_fraction": 0.10,
            "maximum_target_rate_rad_per_second": 0.25,
            "maximum_post_arm_duration_seconds": 5,
            "wireless_deadman_required": False,
            "wireless_stop_required": False,
            "direct_dance_command_required": DIRECT_DANCE_COMMAND,
            "physical_estop_required": True,
            "process_signal_stop_required": True,
        },
    }
    body["promotion_payload_sha256"] = sha256_bytes(canonical_json_bytes(body))
    return body


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--happy-dance-report", type=Path, required=True)
    parser.add_argument("--happy-dance-trajectory", type=Path, required=True)
    parser.add_argument("--live-qualification", type=Path, required=True)
    parser.add_argument("--packet-bundle", type=Path, required=True)
    parser.add_argument("--live-shadow-evidence", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--gantry-authorize", required=True)
    parser.add_argument("--direct-dance-command", required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    expected = active_body(args)
    output = args.output.expanduser().resolve()
    if args.verify_only:
        if _object(output) != expected:
            raise ValueError("active sidecar differs from re-verified evidence")
        print(output)
        return 0
    if os.path.lexists(output):
        raise FileExistsError("refusing to overwrite active dance promotion")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical_json_bytes(expected))
        stream.flush()
        os.fsync(stream.fileno())
    print(output)
    print("free-standing authorization: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
