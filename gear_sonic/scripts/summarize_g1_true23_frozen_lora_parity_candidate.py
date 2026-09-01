"""Bind parity, preservation, and saved-PICO evidence for one diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from gear_sonic.utils.g1_23dof_artifact import sha256_file


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _cases(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = value.get("cases")
    if not isinstance(raw, list) or not raw:
        raise ValueError("comparison suite cases are missing")
    result = {
        str(item.get("label")): item for item in raw if isinstance(item, dict)
    }
    if len(result) != len(raw):
        raise ValueError("comparison suite labels are invalid or duplicated")
    return result


def build_candidate_summary(
    *,
    decoder_report: Mapping[str, Any],
    decoder_report_sha256: str,
    base_suite: Mapping[str, Any],
    candidate_suite: Mapping[str, Any],
    saved_teleop: Mapping[str, Any],
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        decoder_report.get("kind")
        != "g1_true23_frozen_lora_happy_residual_diagnostic_decoder_onnx"
        or decoder_report.get("diagnostic_only") is not True
        or decoder_report.get("deployment_ready") is not False
        or decoder_report.get("hardware_authorized") is not False
    ):
        raise ValueError("candidate decoder safety contract mismatch")
    decoder_hash = decoder_report.get("decoder", {}).get("sha256")
    if not isinstance(decoder_hash, str):
        raise ValueError("candidate decoder hash is missing")
    for label, suite in (("base", base_suite), ("candidate", candidate_suite)):
        if (
            suite.get("kind")
            != "g1_true23_frozen_lora_comparison_result_v1"
            or suite.get("diagnostic_only") is not True
            or suite.get("deployment_ready") is not False
            or suite.get("hardware_authorized") is not False
        ):
            raise ValueError(f"{label} suite safety contract mismatch")
    if (
        candidate_suite.get("decoder_report", {}).get("sha256")
        != decoder_report_sha256
        or candidate_suite.get("suite", {}).get("sha256")
        != base_suite.get("suite", {}).get("sha256")
    ):
        raise ValueError("candidate suite provenance mismatch")
    base_cases = _cases(base_suite)
    candidate_cases = _cases(candidate_suite)
    if set(base_cases) != set(candidate_cases):
        raise ValueError("base and candidate suite cases differ")
    lost = sorted(
        name
        for name in base_cases
        if base_cases[name].get("passed") is True
        and candidate_cases[name].get("passed") is not True
    )
    if lost:
        raise ValueError(f"candidate loses passing base cases: {lost}")
    gained = sorted(
        name
        for name in base_cases
        if base_cases[name].get("passed") is not True
        and candidate_cases[name].get("passed") is True
    )
    base_survival = float(base_suite["second_referee"]["survival_rate"])
    candidate_survival = float(
        candidate_suite["second_referee"]["survival_rate"]
    )
    if candidate_survival <= base_survival:
        raise ValueError("candidate does not improve aggregate suite survival")
    if (
        saved_teleop.get("kind") != "g1_true23_clean_mujoco_teleop_session"
        or saved_teleop.get("passed") is not True
        or saved_teleop.get("physical_dof") != 23
        or saved_teleop.get("decoder_sha256") != decoder_hash
        or saved_teleop.get("fallback_active") is not False
        or saved_teleop.get("completed_transitions") != 684
    ):
        raise ValueError("saved PICO preservation evidence mismatch")
    if (
        parity.get("kind") != "g1_true23_original_sonic_parity_summary"
        or parity.get("parity", {}).get("achieved") is not True
        or parity.get("provenance", {}).get("chain_validated") is not True
    ):
        raise ValueError("original SONIC parity evidence mismatch")

    return {
        "schema_version": 1,
        "kind": "g1_true23_frozen_lora_parity_candidate_summary_v1",
        "candidate": {
            "decoder_sha256": decoder_hash,
            "base_update_count": decoder_report["source"][
                "base_update_count"
            ],
            "residual_alpha": decoder_report["source"]["alpha"],
        },
        "full_suite": {
            "case_count": len(candidate_cases),
            "base_passed_count": sum(
                item.get("passed") is True for item in base_cases.values()
            ),
            "candidate_passed_count": sum(
                item.get("passed") is True
                for item in candidate_cases.values()
            ),
            "newly_passing_cases": gained,
            "lost_passing_cases": lost,
            "base_survival_rate": base_survival,
            "candidate_survival_rate": candidate_survival,
            "survival_rate_delta": candidate_survival - base_survival,
        },
        "original_sonic_happy_dance_parity": True,
        "saved_pico_walk001": {
            "passed": True,
            "completed_transitions": 684,
            "fallback_active": False,
        },
        "simulator_diagnostic_default_candidate": True,
        "diagnostic_only": True,
        "deployment_ready": False,
        "hardware_authorized": False,
        "robot_network_commands": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-report", type=Path, required=True)
    parser.add_argument("--base-suite", type=Path, required=True)
    parser.add_argument("--candidate-suite", type=Path, required=True)
    parser.add_argument("--saved-teleop", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        name: getattr(args, name).expanduser().resolve(strict=True)
        for name in (
            "decoder_report",
            "base_suite",
            "candidate_suite",
            "saved_teleop",
            "parity",
        )
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"candidate summary exists: {output}")
    value = build_candidate_summary(
        decoder_report=_object(paths["decoder_report"]),
        decoder_report_sha256=sha256_file(paths["decoder_report"]),
        base_suite=_object(paths["base_suite"]),
        candidate_suite=_object(paths["candidate_suite"]),
        saved_teleop=_object(paths["saved_teleop"]),
        parity=_object(paths["parity"]),
    )
    value["evidence"] = {
        name: {"filename": path.name, "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(output)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
