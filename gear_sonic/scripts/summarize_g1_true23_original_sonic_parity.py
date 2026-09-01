"""Bind one released SONIC clip to a native true-23 replay result.

The four inputs form a provenance chain:

released 29-DoF planner clip -> released-policy true23 teacher rollout ->
50 Hz native true23 reference -> candidate native true23 replay.

Only completion ratios are compared across embodiments.  Joint errors from the
29-DoF and 23-DoF models are deliberately not treated as commensurate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _named_record(value: Mapping[str, Any], name: str, *, kind: str) -> Mapping[str, Any]:
    if value.get("kind") != kind or not isinstance(value.get("records"), list):
        raise ValueError(f"unexpected report kind: {value.get('kind')!r}")
    matches = [item for item in value["records"] if isinstance(item, dict) and item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name!r} record")
    return matches[0]


def _positive_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def build_parity_summary(
    *,
    motion_name: str,
    original_policy_report: Mapping[str, Any],
    true23_teacher_report: Mapping[str, Any],
    reference_manifest: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the provenance chain and return a cross-embodiment summary."""

    original = _named_record(
        original_policy_report,
        motion_name,
        kind="g1_released_sonic_policy_mujoco_motion_suite",
    )
    teacher = _named_record(
        true23_teacher_report,
        motion_name,
        kind="g1_released_sonic_feature_teacher_true23_physical_adapter",
    )
    if (
        reference_manifest.get("kind")
        != "g1_true23_physical_rollout_motion_reference_v1"
        or reference_manifest.get("passed") is not True
        or reference_manifest.get("physical_dof") != 23
    ):
        raise ValueError("true23 reference manifest contract mismatch")
    if (
        candidate_report.get("kind")
        != "g1_true23_genuine_sonic_library_motion_mujoco_replay"
        or candidate_report.get("physical_dof") != 23
        or candidate_report.get("source_29dof_physics_used") is not False
    ):
        raise ValueError("candidate true23 replay contract mismatch")

    source_hash = original.get("source_planner_npz_sha256")
    if not isinstance(source_hash, str) or teacher.get("source_sha256") != source_hash:
        raise ValueError("original planner clip hash does not reach true23 teacher")
    teacher_hash = teacher.get("physical_npz_sha256")
    if not isinstance(teacher_hash, str) or reference_manifest.get("source_sha256") != teacher_hash:
        raise ValueError("true23 teacher rollout hash does not reach reference manifest")
    reference_hash = reference_manifest.get("output_sha256")
    if not isinstance(reference_hash, str) or candidate_report.get("motion_sha256") != reference_hash:
        raise ValueError("true23 reference hash does not reach candidate replay")

    original_metrics = original.get("metrics")
    teacher_metrics = teacher.get("metrics")
    if not isinstance(original_metrics, Mapping) or not isinstance(teacher_metrics, Mapping):
        raise ValueError("baseline metrics are missing")
    original_done = _positive_count(original_metrics.get("completed_control_steps"), "original completed")
    original_requested = _positive_count(original_metrics.get("requested_control_steps"), "original requested")
    teacher_done = _positive_count(teacher_metrics.get("completed_control_steps"), "teacher completed")
    teacher_requested = _positive_count(teacher_metrics.get("requested_control_steps"), "teacher requested")
    candidate_done = _positive_count(candidate_report.get("completed_transitions"), "candidate completed")
    candidate_requested = _positive_count(candidate_report.get("requested_transitions"), "candidate requested")
    original_ratio = original_done / original_requested
    teacher_ratio = teacher_done / teacher_requested
    candidate_ratio = candidate_done / candidate_requested
    original_passed = original_metrics.get("passed") is True and original_ratio == 1.0
    teacher_passed = teacher_metrics.get("passed") is True and teacher_ratio == 1.0
    candidate_passed = candidate_report.get("passed") is True and candidate_ratio == 1.0

    return {
        "schema_version": 1,
        "kind": "g1_true23_original_sonic_parity_summary",
        "motion": motion_name,
        "provenance": {
            "original_planner_motion_sha256": source_hash,
            "released_policy_true23_rollout_sha256": teacher_hash,
            "native_true23_reference_sha256": reference_hash,
            "chain_validated": True,
        },
        "released_sonic_29dof": {
            "passed": original_passed,
            "completed": original_done,
            "requested": original_requested,
            "completion_ratio": original_ratio,
        },
        "released_sonic_true23_compatibility_teacher": {
            "passed": teacher_passed,
            "completed": teacher_done,
            "requested": teacher_requested,
            "completion_ratio": teacher_ratio,
        },
        "native_true23_candidate": {
            "passed": candidate_passed,
            "completed": candidate_done,
            "requested": candidate_requested,
            "completion_ratio": candidate_ratio,
            "failure": candidate_report.get("failure"),
            "maximum_joint_tracking_rmse_rad": candidate_report.get("metrics", {}).get(
                "maximum_joint_tracking_rmse_rad"
            ),
            "maximum_relative_tracking_error_m": candidate_report.get("metrics", {}).get(
                "maximum_relative_tracked_body_position_error_m"
            ),
        },
        "parity": {
            "achieved": original_passed and teacher_passed and candidate_passed,
            "completion_ratio_gap_to_original": original_ratio - candidate_ratio,
            "remaining_candidate_transitions": candidate_requested - candidate_done,
            "cross_embodiment_joint_error_comparison_forbidden": True,
        },
        "diagnostic_only": True,
        "deployment_ready": False,
        "hardware_authorized": False,
        "robot_network_commands": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--original-policy-report", type=Path, required=True)
    parser.add_argument("--true23-teacher-report", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite parity report: {output}")
    summary = build_parity_summary(
        motion_name=args.motion,
        original_policy_report=_object(args.original_policy_report.expanduser().resolve(strict=True)),
        true23_teacher_report=_object(args.true23_teacher_report.expanduser().resolve(strict=True)),
        reference_manifest=_object(args.reference_manifest.expanduser().resolve(strict=True)),
        candidate_report=_object(args.candidate_report.expanduser().resolve(strict=True)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(output)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
