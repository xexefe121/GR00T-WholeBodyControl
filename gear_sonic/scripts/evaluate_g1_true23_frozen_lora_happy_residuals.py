"""Closed-loop screen tiny happy-dance residuals against their LoRA base."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gear_sonic.scripts.fit_g1_true23_frozen_lora_happy_residual import MOTION
from gear_sonic.utils.g1_23dof_artifact import sha256_file
from gear_sonic.utils.g1_true23_sonic_library_replay import (
    run_library_motion_replay,
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _selection_key(item: Mapping[str, Any]) -> tuple[bool, float, float, float]:
    report = item["report"]
    completed = int(report["completed_transitions"])
    requested = int(report["requested_transitions"])
    error = float(
        report["metrics"]["maximum_relative_tracked_body_position_error_m"]
    )
    return (
        report["passed"] is True,
        completed / requested,
        -abs(float(item["alpha"])),
        -error,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.repository_root.expanduser().resolve(strict=True)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    baseline_path = args.baseline_report.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"residual evaluation exists: {output_dir}")
    manifest = _object(manifest_path)
    if (
        manifest.get("kind")
        != "g1_true23_frozen_lora_happy_residual_diagnostic_v1"
        or manifest.get("diagnostic_only") is not True
        or manifest.get("deployment_ready") is not False
        or manifest.get("hardware_authorized") is not False
        or manifest.get("closed_loop_screening_required") is not True
    ):
        raise ValueError("residual manifest safety contract mismatch")
    motion = (root / MOTION).resolve(strict=True)
    if manifest.get("source", {}).get("motion_sha256") != sha256_file(motion):
        raise ValueError("residual manifest motion identity mismatch")
    baseline = _object(baseline_path)
    if (
        baseline.get("kind")
        != "g1_true23_genuine_sonic_library_motion_mujoco_replay"
        or baseline.get("physical_dof") != 23
        or baseline.get("motion_sha256") != sha256_file(motion)
    ):
        raise ValueError("baseline report contract mismatch")
    raw_candidates = manifest.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("residual manifest has no candidates")

    output_dir.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    names: set[str] = set()
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            raise ValueError("residual candidate must be an object")
        name = candidate.get("name")
        if not isinstance(name, str) or name in names:
            raise ValueError("residual candidate name is invalid or duplicated")
        names.add(name)
        decoder = manifest_path.with_name(
            str(candidate.get("decoder_filename"))
        ).resolve()
        digest = candidate.get("decoder_sha256")
        if not decoder.is_file() or sha256_file(decoder) != digest:
            raise ValueError(f"residual decoder identity mismatch: {name}")
        report, arrays = run_library_motion_replay(
            repository_root=root,
            motion_path=motion,
            decoder_path=decoder,
            expected_decoder_sha256=str(digest),
            controller_mode="sonic",
            gain_profile="released_retained",
        )
        trajectory = output_dir / f"{name}.trajectory.npz"
        with trajectory.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        report["trajectory_npz"] = str(trajectory)
        report_path = output_dir / f"{name}.json"
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        results.append(
            {
                "name": name,
                "alpha": candidate["alpha"],
                "decoder_sha256": digest,
                "report_filename": report_path.name,
                "report_sha256": sha256_file(report_path),
                "report": report,
            }
        )

    selected = max(results, key=_selection_key)
    baseline_done = int(baseline["completed_transitions"])
    selected_done = int(selected["report"]["completed_transitions"])
    summary = {
        "schema_version": 1,
        "kind": "g1_true23_frozen_lora_happy_residual_closed_loop_screen_v1",
        "source_manifest": {
            "filename": manifest_path.name,
            "sha256": sha256_file(manifest_path),
        },
        "baseline": {
            "report_filename": baseline_path.name,
            "report_sha256": sha256_file(baseline_path),
            "completed_transitions": baseline_done,
            "requested_transitions": baseline["requested_transitions"],
            "passed": baseline["passed"],
        },
        "selected": {
            key: selected[key]
            for key in (
                "name",
                "alpha",
                "decoder_sha256",
                "report_filename",
                "report_sha256",
            )
        }
        | {
            "completed_transitions": selected_done,
            "requested_transitions": selected["report"][
                "requested_transitions"
            ],
            "passed": selected["report"]["passed"],
            "transition_delta_from_base": selected_done - baseline_done,
        },
        "candidates": [
            {
                "name": item["name"],
                "alpha": item["alpha"],
                "decoder_sha256": item["decoder_sha256"],
                "report_filename": item["report_filename"],
                "report_sha256": item["report_sha256"],
                "passed": item["report"]["passed"],
                "completed_transitions": item["report"][
                    "completed_transitions"
                ],
                "requested_transitions": item["report"][
                    "requested_transitions"
                ],
                "maximum_relative_tracking_error_m": item["report"][
                    "metrics"
                ]["maximum_relative_tracked_body_position_error_m"],
            }
            for item in results
        ],
        "selection_order": [
            "passed",
            "completion_ratio",
            "smaller_absolute_alpha",
            "lower_maximum_relative_tracking_error",
        ],
        "diagnostic_only": True,
        "deployment_ready": False,
        "hardware_authorized": False,
        "robot_network_commands": False,
    }
    destination = output_dir / "summary.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(destination)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
