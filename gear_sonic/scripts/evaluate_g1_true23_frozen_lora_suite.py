"""Evaluate one diagnostic LoRA decoder on a hash-bound true23 suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from gear_sonic.utils.g1_23dof_artifact import sha256_file
from gear_sonic.utils.g1_true23_sonic_library_replay import (
    run_library_motion_replay,
)

_CATEGORIES = {"in_distribution", "tail", "out_of_distribution"}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _suite_cases(path: Path, repository_root: Path) -> list[dict[str, Any]]:
    value = _load_json(path)
    raw = value.get("cases")
    if value.get("kind") != "g1_true23_frozen_lora_comparison_suite_v1":
        raise ValueError("comparison suite kind mismatch")
    if not isinstance(raw, list) or not raw:
        raise ValueError("comparison suite must contain cases")
    result: list[dict[str, Any]] = []
    labels: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or set(item) != {
            "label",
            "motion",
            "categories",
        }:
            raise ValueError("comparison suite case shape mismatch")
        label = item["label"]
        categories = item["categories"]
        if (
            not isinstance(label, str)
            or re.fullmatch(r"[a-z0-9_]+", label) is None
            or label in labels
            or not isinstance(categories, list)
            or not categories
            or set(categories) - _CATEGORIES
            or len(categories) != len(set(categories))
        ):
            raise ValueError("comparison suite label/category mismatch")
        motion = (repository_root / item["motion"]).resolve()
        if not motion.is_relative_to(repository_root) or not motion.is_file():
            raise ValueError(f"comparison motion unavailable: {motion}")
        labels.add(label)
        result.append(
            {"label": label, "motion": motion, "categories": categories}
        )
    if not all(
        any(name in case["categories"] for case in result)
        for name in _CATEGORIES
    ):
        raise ValueError("comparison suite must cover all categories")
    return result


def _category_metrics(
    reports: list[tuple[dict[str, Any], list[str]]],
    category: str,
) -> dict[str, Any]:
    selected = [report for report, categories in reports if category in categories]
    tracking = [
        float(report["metrics"]["maximum_relative_tracked_body_position_error_m"])
        for report in selected
    ]
    return {
        "case_count": len(selected),
        "passed_count": sum(report["passed"] is True for report in selected),
        "success_rate": sum(report["passed"] is True for report in selected)
        / len(selected),
        "mean_tracking_error": sum(tracking) / len(tracking),
    }


def _decoder_identity(
    report_path: Path,
    value: dict[str, Any],
) -> tuple[Path, str, int]:
    kind = value.get("kind")
    if kind == "g1_true23_frozen_lora_diagnostic_decoder_onnx":
        if (
            value.get("diagnostic_only") is not True
            or value.get("deployment_ready") is not False
            or value.get("hardware_authorized") is not False
            or value.get("active_motor_control_authorized") is not False
        ):
            raise ValueError("LoRA decoder report safety contract mismatch")
        decoder_info = value.get("decoder")
        source = value.get("source")
        if not isinstance(decoder_info, dict) or not isinstance(source, dict):
            raise ValueError("LoRA decoder report identity is missing")
        return (
            report_path.with_name(decoder_info["filename"]).resolve(),
            decoder_info["sha256"],
            int(source["update_count"]),
        )
    if kind == "g1_true23_mjlab_diagnostic_onnx_pair":
        if (
            value.get("diagnostic_only") is not True
            or value.get("deployment_ready") is not False
            or value.get("promotion_eligible") is not False
            or value.get("active_motor_control_authorized") is not False
            or value.get("no_robot_or_network_commands_performed") is not True
        ):
            raise ValueError("baseline decoder report safety contract mismatch")
        artifacts = value.get("artifacts")
        hashes = value.get("hashes")
        source = value.get("source")
        if not all(isinstance(item, dict) for item in (artifacts, hashes, source)):
            raise ValueError("baseline decoder report identity is missing")
        return (
            report_path.with_name(artifacts["decoder_onnx_filename"]).resolve(),
            hashes["decoder_onnx_sha256"],
            int(source["checkpoint_update_count"]),
        )
    if kind == "g1_true23_frozen_lora_happy_residual_diagnostic_decoder_onnx":
        if (
            value.get("closed_loop_happy_dance_passed") is not True
            or value.get("diagnostic_only") is not True
            or value.get("deployment_ready") is not False
            or value.get("promotion_eligible") is not False
            or value.get("hardware_authorized") is not False
            or value.get("active_motor_control_authorized") is not False
            or value.get("robot_network_commands") is not False
        ):
            raise ValueError("residual decoder report safety contract mismatch")
        decoder_info = value.get("decoder")
        source = value.get("source")
        if not isinstance(decoder_info, dict) or not isinstance(source, dict):
            raise ValueError("residual decoder report identity is missing")
        return (
            report_path.with_name(decoder_info["filename"]).resolve(),
            decoder_info["sha256"],
            int(source["base_update_count"]),
        )
    raise ValueError("unsupported diagnostic decoder report kind")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    repository_root = args.repository_root.expanduser().resolve()
    decoder_report_path = args.decoder_report.expanduser().resolve()
    suite_path = args.suite.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output_dir}")
    decoder_report = _load_json(decoder_report_path)
    decoder, expected_decoder_sha256, checkpoint_update_count = (
        _decoder_identity(decoder_report_path, decoder_report)
    )
    if (
        not decoder.is_file()
        or sha256_file(decoder) != expected_decoder_sha256
    ):
        raise ValueError("diagnostic decoder bytes differ from report")
    cases = _suite_cases(suite_path, repository_root)
    output_dir.mkdir(parents=True)
    reports: list[tuple[dict[str, Any], list[str]]] = []
    for case in cases:
        report, arrays = run_library_motion_replay(
            repository_root=repository_root,
            motion_path=case["motion"],
            decoder_path=decoder,
            expected_decoder_sha256=expected_decoder_sha256,
            controller_mode="sonic",
            gain_profile="released_retained",
        )
        report["comparison_categories"] = list(case["categories"])
        trajectory_path = output_dir / f"{case['label']}.trajectory.npz"
        with trajectory_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        report["trajectory_npz"] = str(trajectory_path)
        report_path = output_dir / f"{case['label']}.json"
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        reports.append((report, case["categories"]))

    completed = sum(report["completed_transitions"] for report, _ in reports)
    requested = sum(report["requested_transitions"] for report, _ in reports)
    summary = {
        "kind": "g1_true23_frozen_lora_comparison_result_v1",
        "decoder_report": {
            "filename": decoder_report_path.name,
            "sha256": sha256_file(decoder_report_path),
        },
        "suite": {
            "filename": suite_path.name,
            "sha256": sha256_file(suite_path),
        },
        "checkpoint_update_count": checkpoint_update_count,
        "in_distribution": _category_metrics(reports, "in_distribution"),
        "tail": _category_metrics(reports, "tail"),
        "out_of_distribution": _category_metrics(
            reports, "out_of_distribution"
        ),
        "second_referee": {
            "backend": "cpu_mujoco_reference_vs_training_mujoco_warp",
            "completed_transitions": completed,
            "requested_transitions": requested,
            "survival_rate": completed / requested,
        },
        "cases": [
            {
                "label": case["label"],
                "categories": list(case["categories"]),
                "passed": report["passed"],
                "completed_transitions": report["completed_transitions"],
                "requested_transitions": report["requested_transitions"],
                "failure_type": (
                    None
                    if report["failure"] is None
                    else report["failure"]["type"]
                ),
                "maximum_relative_tracking_error_m": report["metrics"][
                    "maximum_relative_tracked_body_position_error_m"
                ],
            }
            for case, (report, _categories) in zip(cases, reports, strict=True)
        ],
        "diagnostic_only": True,
        "deployment_ready": False,
        "hardware_authorized": False,
        "robot_network_commands": False,
    }
    destination = output_dir / "suite_summary.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(destination)  # noqa: T201
    print("deployment ready: false")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
