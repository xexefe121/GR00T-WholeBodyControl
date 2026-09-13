"""Compare paired compatibility diagnostics without hiding perturbation regressions."""

import argparse
import json
from pathlib import Path

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_generalist_benchmark import FLAGS

CASES = ("nominal", "standing_push_x", "standing_push_y")
MATCH_FIELDS = (
    "compiled_model_sha256",
    "physics_config_sha256",
    "initial_state_and_history_sha256",
    "motion_sha256",
    "available_controls",
    "kp_hardware",
    "kd_hardware",
    "effort_limit_hardware_nm",
)


def indexed_rows(document):
    rows = document.get("rows", document.get("records"))
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("requires exactly three full paired cases")
    indexed = {}
    for row in rows:
        result = row["result"]
        if (
            result["completed_controls"] != result["available_controls"]
            or result["requested_controls"] != result["available_controls"]
            or result["failure"] is not None
        ):
            raise ValueError("full-lifecycle comparison cannot substitute a failed or shortened rollout")
        case = row.get("case", result.get("runtime_adapter", {}).get("case"))
        if case not in CASES or case in indexed:
            raise ValueError("missing, duplicate or unknown case")
        indexed[case] = row
    return indexed


def compare(before, after):
    before, after = indexed_rows(before), indexed_rows(after)
    rows = []
    for case in CASES:
        first, second = before[case], after[case]
        for field in MATCH_FIELDS:
            if first["result"][field] != second["result"][field]:
                raise ValueError(f"unmatched {case}: {field}")
        first_motion = first["lifecycle"]["source_motion_tracking"]
        second_motion = second["lifecycle"]["source_motion_tracking"]
        if first_motion["reference_landmark_thresholds_m"] != second_motion["reference_landmark_thresholds_m"]:
            raise ValueError("comparison cannot change fidelity thresholds")
        first_points = first_motion["landmark_position_p95_m"]
        second_points = second_motion["landmark_position_p95_m"]
        delta = {name: second_points[name] - value for name, value in first_points.items()}
        root_delta = (
            second["result"]["pelvis_world_position_p95_m"] - first["result"]["pelvis_world_position_p95_m"]
        )
        rows.append(
            dict(
                case=case,
                source_controls_before=next(
                    p["completed_controls"] for p in first["lifecycle"]["phases"] if p["name"] == "source_motion"
                ),
                source_controls_after=next(
                    p["completed_controls"] for p in second["lifecycle"]["phases"] if p["name"] == "source_motion"
                ),
                source_landmark_p95_before_m=first_points,
                source_landmark_p95_after_m=second_points,
                source_landmark_p95_delta_m=delta,
                full_lifecycle_root_p95_before_m=first["result"]["pelvis_world_position_p95_m"],
                full_lifecycle_root_p95_after_m=second["result"]["pelvis_world_position_p95_m"],
                full_lifecycle_root_p95_delta_m=root_delta,
                final_standing_joint_max_before_rad=first["lifecycle"]["final_proof_standing_joint_error_max_rad"],
                final_standing_joint_max_after_rad=second["lifecycle"]["final_proof_standing_joint_error_max_rad"],
                any_source_landmark_regressed=any(value > 0 for value in delta.values()),
                root_position_regressed=root_delta > 0,
                after_motion_fidelity_passed=second_motion["provisional_reference_landmark_screen_passed"],
                before_failure=first["result"]["failure"],
                after_failure=second["result"]["failure"],
            )
        )
    return dict(
        cases=rows,
        identical_physics_initialization_reference_and_denominator=True,
        all_cases_improved_without_root_or_landmark_regression=all(
            not r["any_source_landmark_regressed"] and not r["root_position_regressed"] for r in rows
        ),
        all_cases_after_pass_motion_fidelity=all(r["after_motion_fidelity_passed"] for r in rows),
        **FLAGS,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release-before", "release-after", "root-before", "root-after"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: getattr(args, name).resolve(strict=True)
        for name in ("release_before", "release_after", "root_before", "root_after")
    }
    documents = {name: json.loads(path.read_text()) for name, path in paths.items()}
    result = dict(
        kind="native23_sonic_compatibility_matched_comparison_v1",
        inputs={str(path): sha256_file(path) for path in paths.values()},
        zero_training_release=compare(documents["release_before"], documents["release_after"]),
        existing_root100_weights=compare(documents["root_before"], documents["root_after"]),
        candidate_promoted=False,
        conclusion=(
            "Reference geometry fixes standing arm bias; nominal release tracking improves, "
            "but perturbations regress and every full-motion fidelity screen still fails. "
            "Not a deployment solution."
        ),
        **FLAGS,
    )
    # A new, exclusive result; no old failed evidence is rewritten.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for name in ("zero_training_release", "existing_root100_weights"):
        print(
            name,
            [
                (
                    row["case"],
                    round(row["full_lifecycle_root_p95_before_m"], 3),
                    round(row["full_lifecycle_root_p95_after_m"], 3),
                )
                for row in result[name]["cases"]
            ],
        )


if __name__ == "__main__":
    main()
