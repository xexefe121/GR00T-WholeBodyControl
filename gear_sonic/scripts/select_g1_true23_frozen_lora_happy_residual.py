"""Bind one closed-loop-passing residual to a diagnostic decoder report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from gear_sonic.scripts.fit_g1_true23_frozen_lora_happy_residual import (
    _load_base_decoder,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--base-decoder-report", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest_path = args.manifest.expanduser().resolve(strict=True)
    screen_path = args.screen_summary.expanduser().resolve(strict=True)
    base_report_path = args.base_decoder_report.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if os.path.lexists(output):
        raise FileExistsError(f"selected residual report exists: {output}")
    manifest = _object(manifest_path)
    screen = _object(screen_path)
    _base_decoder, base_hash, base_report = _load_base_decoder(base_report_path)
    if (
        manifest.get("kind")
        != "g1_true23_frozen_lora_happy_residual_diagnostic_v1"
        or screen.get("kind")
        != "g1_true23_frozen_lora_happy_residual_closed_loop_screen_v1"
        or screen.get("source_manifest", {}).get("sha256")
        != sha256_file(manifest_path)
        or manifest.get("source", {}).get("base_decoder_report_sha256")
        != sha256_file(base_report_path)
        or manifest.get("source", {}).get("base_decoder_sha256") != base_hash
    ):
        raise ValueError("residual selection provenance mismatch")
    manifest_matches = [
        item
        for item in manifest.get("candidates", [])
        if isinstance(item, dict) and item.get("name") == args.candidate
    ]
    screen_matches = [
        item
        for item in screen.get("candidates", [])
        if isinstance(item, dict) and item.get("name") == args.candidate
    ]
    if len(manifest_matches) != 1 or len(screen_matches) != 1:
        raise ValueError("selected residual candidate is not unique")
    candidate = manifest_matches[0]
    evidence = screen_matches[0]
    if (
        evidence.get("passed") is not True
        or evidence.get("completed_transitions")
        != evidence.get("requested_transitions")
        or evidence.get("decoder_sha256") != candidate.get("decoder_sha256")
    ):
        raise ValueError("selected residual lacks passing closed-loop evidence")
    decoder = manifest_path.with_name(
        str(candidate.get("decoder_filename"))
    ).resolve(strict=True)
    case_report = screen_path.with_name(
        str(evidence.get("report_filename"))
    ).resolve(strict=True)
    if (
        sha256_file(decoder) != candidate.get("decoder_sha256")
        or sha256_file(case_report) != evidence.get("report_sha256")
    ):
        raise ValueError("selected residual artifact identity mismatch")

    value = {
        "schema_version": 1,
        "kind": "g1_true23_frozen_lora_happy_residual_diagnostic_decoder_onnx",
        "source": {
            "base_decoder_report_sha256": sha256_file(base_report_path),
            "base_update_count": base_report["source"]["update_count"],
            "residual_manifest_sha256": sha256_file(manifest_path),
            "closed_loop_screen_sha256": sha256_file(screen_path),
            "closed_loop_case_report_sha256": sha256_file(case_report),
            "candidate": args.candidate,
            "alpha": candidate["alpha"],
        },
        "decoder": {
            "filename": decoder.name,
            "sha256": candidate["decoder_sha256"],
            "input_name": "obs_dict",
            "input_shape": [1, 994],
            "output_name": "action",
            "output_shape": [1, 23],
            "opset": 13,
        },
        "closed_loop_happy_dance_passed": True,
        "diagnostic_only": True,
        "deployment_ready": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "active_motor_control_authorized": False,
        "robot_network_commands": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(output)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
