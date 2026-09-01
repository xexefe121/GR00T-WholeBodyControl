"""Create exact-byte admission for read-only true23 SONIC dance shadow.

This is not motor authorization. It binds selected frozen-LoRA decoder,
original SONIC happy-dance replay, live transport qualification, and exact
causal packet bundle so hardware shadow can be collected without LowCmd.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from gear_sonic.utils.g1_23dof_artifact import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)

KIND = "g1_true23_frozen_lora_dance_shadow_admission_v1"
DECODER_SHA256 = "44d1fb2701f1e65460f1c2c23f676bce4f1d4a44b3b112798dc5034af37946b8"
DECODER_REPORT_SHA256 = "02197e5682a9bddc8f11aa6fa9c32ba909b97ec7d1c316c9a0d660cba2d25b7d"
CANDIDATE_SUMMARY_SHA256 = "ed3f5513ed7da8625195b37b674163d64948697d311722ad936be3f4668db801"
HAPPY_REPORT_SHA256 = "bcd674314a48f86ce111ea13a5389ac0acf7a9549ffef6de9f45a059014a1a4f"
HAPPY_TRAJECTORY_SHA256 = "b5d415dacfcd175da08fefeac54cabeb9387e912ec78f49e126a14d65913b697"
LIVE_QUALIFICATION_SHA256 = "ab1b8493d20d5a4e92b2e432f7ab5eeb8c16862115ff3b9c579d7b1781216d35"
PACKET_BUNDLE_SHA256 = "237910ad5dfc370db9645e52f08ba0ca3b0f409a1383d692e1ce1937c5e3dc9d"
ENCODER_SHA256 = "733353148bef1eb8dd83a96416b7a89f0b5c3530ceb9e0cec9c25fdb04f56ff2"
SAFE_TARGET_TRANSFORM_SHA256 = "8313474d1050ca959152afebb2baaefdaad02ec53a1d1312b738192fdf4f449b"


def _object(path: Path, role: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _file(path: Path, role: str, expected_sha256: str | None = None) -> Path:
    lexical = path.expanduser()
    if lexical.is_symlink() or not lexical.is_file():
        raise ValueError(f"{role} must be a regular non-symlink file")
    resolved = lexical.resolve()
    digest = sha256_file(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{role} SHA-256 mismatch")
    return resolved


def promotion_body(args: argparse.Namespace) -> dict[str, Any]:
    encoder = _file(args.encoder, "encoder", ENCODER_SHA256)
    report_path = _file(args.decoder_report, "decoder report", DECODER_REPORT_SHA256)
    summary_path = _file(args.candidate_summary, "candidate summary", CANDIDATE_SUMMARY_SHA256)
    happy_path = _file(args.happy_dance_report, "happy dance report", HAPPY_REPORT_SHA256)
    trajectory = _file(
        args.happy_dance_trajectory, "happy dance trajectory", HAPPY_TRAJECTORY_SHA256
    )
    live_path = _file(
        args.live_qualification, "live qualification", LIVE_QUALIFICATION_SHA256
    )
    packets = _file(args.packet_bundle, "causal packet bundle", PACKET_BUNDLE_SHA256)
    report = _object(report_path, "decoder report")
    summary = _object(summary_path, "candidate summary")
    happy = _object(happy_path, "happy dance report")
    live = _object(live_path, "live qualification")
    decoder_name = report.get("decoder", {}).get("filename")
    if not isinstance(decoder_name, str):
        raise ValueError("decoder report filename missing")
    decoder = _file(report_path.with_name(decoder_name), "decoder", DECODER_SHA256)
    if (
        report.get("kind")
        != "g1_true23_frozen_lora_happy_residual_diagnostic_decoder_onnx"
        or report.get("closed_loop_happy_dance_passed") is not True
        or report.get("diagnostic_only") is not True
        or report.get("hardware_authorized") is not False
        or report.get("active_motor_control_authorized") is not False
        or report.get("decoder", {}).get("input_shape") != [1, 994]
        or report.get("decoder", {}).get("output_shape") != [1, 23]
        or report.get("decoder", {}).get("sha256") != DECODER_SHA256
    ):
        raise ValueError("decoder report contract mismatch")
    if (
        summary.get("kind") != "g1_true23_frozen_lora_parity_candidate_summary_v1"
        or summary.get("original_sonic_happy_dance_parity") is not True
        or summary.get("saved_pico_walk001", {}).get("passed") is not True
        or summary.get("saved_pico_walk001", {}).get("completed_transitions") != 684
        or summary.get("full_suite", {}).get("lost_passing_cases") != []
        or summary.get("candidate", {}).get("decoder_sha256") != DECODER_SHA256
    ):
        raise ValueError("candidate parity summary mismatch")
    if (
        happy.get("kind") != "g1_true23_genuine_sonic_library_motion_mujoco_replay"
        or happy.get("passed") is not True
        or happy.get("completed_transitions") != 535
        or happy.get("physical_dof") != 23
        or happy.get("decoder_output_dof") != 23
        or happy.get("source_29dof_physics_used") is not False
        or happy.get("decoder_sha256") != DECODER_SHA256
        or happy.get("authorization", {}).get("simulator_only") is not True
        or happy.get("authorization", {}).get("hardware_authorized") is not False
    ):
        raise ValueError("happy dance MuJoCo qualification mismatch")
    metrics = happy.get("metrics", {})
    numeric = (
        metrics.get("minimum_base_height_m"),
        metrics.get("maximum_base_tilt_rad"),
        metrics.get("maximum_joint_tracking_rmse_rad"),
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric
    ):
        raise ValueError("happy dance metrics are invalid")
    if (
        live.get("kind") != "g1_true23_frozen_lora_live_qualification_v1"
        or live.get("software_live_teleop_ready") is not True
        or live.get("candidate", {}).get("decoder_sha256") != DECODER_SHA256
        or live.get("candidate", {}).get("physical_dof") != 23
        or live.get("candidate", {}).get("source_29dof_physics_used") is not False
        or live.get("physical_robot_teleop_ready") is not False
    ):
        raise ValueError("live software qualification mismatch")
    packet_document = _object(packets, "causal packet bundle")
    packet_list = packet_document.get("robot_independent_reference_packets")
    if not isinstance(packet_list, list) or len(packet_list) != 535:
        raise ValueError("causal happy-dance packet count mismatch")
    source_motion_hashes = {
        item.get("source_motion_sha256")
        for item in packet_document.get("semantic_packets", [])
        if isinstance(item, Mapping)
    }
    if source_motion_hashes != {happy.get("motion_sha256")}:
        raise ValueError("packet bundle is not bound to qualified happy motion")
    body: dict[str, Any] = {
        "schema_version": 1,
        "kind": KIND,
        "robot_model": "g1_23dof_rev_1_0",
        "required_mode_machine": 4,
        "native_action_dof": 23,
        "deployment_bytes_authorized_for_shadow": True,
        "active_motor_control_authorized": False,
        "gantry_or_rated_support_required": True,
        "free_standing_authorized": False,
        "reference_profile": "true23_causal_step1_history_0p02s_v1",
        "decoder_output_semantics": "raw_native_action",
        "runtime_policy_semantics": "applied_safe_native_action",
        "external_safe_target_transform_required": True,
        "safe_target_transform_sha256": SAFE_TARGET_TRANSFORM_SHA256,
        "source_artifacts": {
            "encoder_sha256": sha256_file(encoder),
            "decoder_sha256": sha256_file(decoder),
            "decoder_report_sha256": sha256_file(report_path),
            "candidate_summary_sha256": sha256_file(summary_path),
            "happy_dance_report_sha256": sha256_file(happy_path),
            "happy_dance_trajectory_sha256": sha256_file(trajectory),
            "live_qualification_sha256": sha256_file(live_path),
            "packet_bundle_sha256": sha256_file(packets),
        },
        "qualification": {
            "happy_dance_passed": True,
            "happy_dance_completed_transitions": 535,
            "saved_pico_walk001_completed_transitions": 684,
            "software_live_transport_fault_drills_passed": True,
            "minimum_base_height_m": metrics["minimum_base_height_m"],
            "maximum_base_tilt_rad": metrics["maximum_base_tilt_rad"],
            "maximum_joint_tracking_rmse_rad": metrics[
                "maximum_joint_tracking_rmse_rad"
            ],
        },
        "stage_one_envelope": {
            "action_fraction": 0.10,
            "maximum_target_rate_rad_per_second": 0.25,
            "maximum_post_arm_duration_seconds": 10,
            "wireless_deadman_required": True,
            "wireless_stop_required": True,
        },
    }
    body["promotion_payload_sha256"] = sha256_bytes(canonical_json_bytes(body))
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--happy-dance-report", type=Path, required=True)
    parser.add_argument("--happy-dance-trajectory", type=Path, required=True)
    parser.add_argument("--live-qualification", type=Path, required=True)
    parser.add_argument("--packet-bundle", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    expected = promotion_body(args)
    output = args.output.expanduser().resolve()
    if args.verify_only:
        if _object(_file(output, "promotion"), "promotion") != expected:
            raise ValueError("promotion differs from re-verified inputs")
        print(output)
        return 0
    if os.path.lexists(output):
        raise FileExistsError("refusing to overwrite dance shadow admission")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical_json_bytes(expected))
        stream.flush()
        os.fsync(stream.fileno())
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
