"""Rescore immutable simulator reports without rewriting historical evidence.

No simulation, robot SDK, transport, deployment or promotion calls occur here.
This audit accepts legacy native23 library reports and deployment-envelope
summaries. The full clip count must be explicitly supplied for envelope files
because their older schema records only the requested prefix length.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gear_sonic.utils.g1_true23_motion_fidelity import assess_motion_fidelity


def audit(path: Path, available: int) -> dict:
    raw = path.read_bytes()
    source = json.loads(raw)
    kind = source.get("kind")
    if kind == "g1_true23_genuine_sonic_library_motion_mujoco_replay":
        if source["frame_count"] - 11 != available:
            raise ValueError("full clip count differs from library source")
        entries = [source]
    elif kind == "g1_true23_deployment_envelope_diagnostic_v1":
        entries = source["cases"]
    else:
        raise ValueError("unsupported simulator report kind")
    cases = []
    for entry in entries:
        if "metrics" in entry:
            metrics = entry["metrics"]
        else:
            maxima = entry["maximums"]
            metrics = {
                "maximum_pelvis_position_error_m": maxima.get("pelvis_position_error_m"),
                "maximum_pelvis_orientation_error_rad": maxima.get("pelvis_orientation_error_rad"),
                "maximum_relative_tracked_body_position_error_m": maxima.get("relative_body_error_m"),
                "maximum_joint_tracking_rmse_rad": maxima.get("joint_rmse_rad"),
            }
        cases.append({
            "label": entry.get("label", entry.get("motion_path")),
            "legacy_completion_passed": entry.get("library_completion_passed", entry.get("passed")),
            "completed_transitions": entry["completed_transitions"],
            "requested_transitions": entry["requested_transitions"],
            "metrics": metrics,
            "motion_fidelity": assess_motion_fidelity(
                metrics=metrics, completed=entry["completed_transitions"],
                requested=entry["requested_transitions"], available=available,
                failure=entry.get("failure"),
            ),
        })
    return {"source": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(), "cases": cases}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True, type=Path)
    parser.add_argument("--available-transitions", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.available_transitions <= 0:
        parser.error("available-transitions must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = [audit(path.resolve(), args.available_transitions) for path in args.reports]
    cases = [case for source in sources for case in source["cases"]]
    result = {
        "kind": "g1_true23_motion_fidelity_audit_v1", "sources": sources,
        "case_count": len(cases),
        "motion_fidelity_passed_count": sum(case["motion_fidelity"]["passed"] for case in cases),
        "hardware_authorized": False, "robot_commands_published": False,
        "deployment_ready": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases), "fidelity_passed": result["motion_fidelity_passed_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
