"""Compare complete, identically configured CPU runs; never promote a policy."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parent
    report_paths = {
        "evaluated_parent_root100": root / "planned_v5_endpoint_priority100_20260907_v1/report.json",
        "legacy_objective_root200": root
        / "root_feedback_campaign_20260907_v1/segment_0100_0200/eval_0200/report.json",
        "measured_posture_objective_root200": directory / "segment_0100_0200/eval_0200/report.json",
    }
    hashes = {name: sha256(path) for name, path in report_paths.items()}
    shared = None
    results = {}
    for name, path in report_paths.items():
        report = json.loads(path.read_text())
        assert report["no_postinitial_robot_pose_rewrites"] and report["no_fallback_controller"]
        assert not report["hardware_authorized"] and not report["deployment_ready"]
        assert [row["case"] for row in report["records"]] == ["nominal", "standing_push_x", "standing_push_y"]
        rows = []
        for row in report["records"]:
            physics, lifecycle = row["result"], row["lifecycle"]
            contract = {
                key: physics[key]
                for key in (
                    "initial_state_and_history_sha256",
                    "compiled_model_sha256",
                    "physics_config_sha256",
                    "motion_sha256",
                    "kp_hardware",
                    "kd_hardware",
                    "effort_limit_hardware_nm",
                    "requested_controls",
                    "available_controls",
                )
            }
            assert shared is None or contract == shared, "comparison changed reference/physics/start/denominator"
            shared = contract
            assert physics["completed_controls"] == physics["requested_controls"] == 1841
            assert physics["state_pose_writes_after_reset"] == physics["history_resets_during_motion"] == 0
            assert lifecycle["source_motion_tracking"]["full_source_motion_completed"]
            assert sha256(Path(row["trace_path"])) == row["trace_sha256"]
            rows.append(
                {
                    "case": row["case"],
                    "completed_controls": physics["completed_controls"],
                    "root_position_p95_m": physics["root_response"]["current_root_position_error_p95_m"],
                    "source_landmark_p95_m": lifecycle["source_motion_tracking"]["landmark_position_p95_m"],
                    "final_standing_joint_error_max_rad": lifecycle["final_proof_standing_joint_error_max_rad"],
                    "final_root_speed_max_m_s": lifecycle["final_proof_root_speed_max_m_s"],
                    "final_root_position_error_max_m": lifecycle["final_proof_root_position_error_max_m"],
                    "source_landmark_screen_passed": lifecycle["source_motion_tracking"][
                        "provisional_reference_landmark_screen_passed"
                    ],
                    "trace_sha256": row["trace_sha256"],
                }
            )
        results[name] = {"input_report_sha256": hashes[name], "cases": rows}
    for name, result in results.items():
        for row, parent in zip(result["cases"], results["evaluated_parent_root100"]["cases"], strict=True):
            row["root_p95_change_vs_root100_m"] = row["root_position_p95_m"] - parent["root_position_p95_m"]
        result["promoted"] = False
    assert all(sha256(path) == hashes[name] for name, path in report_paths.items())
    output = {
        "kind": "g1_true23_posture_objective_matched_full_cpu_comparison_v1",
        "input_reports": {
            name: {"path": str(path), "sha256": hashes[name]} for name, path in report_paths.items()
        },
        "shared_physics_and_reference": shared,
        "all_trace_hashes_verified": True,
        "results": results,
        "conclusion": (
            "Both root200 candidates regress source tracking relative to their evaluated root100 parent. "
            "The posture candidate reduces final speed but does not qualify standing or dance."
        ),
        "test_code_is_not_motion_qualification": True,
        "hardware_authorized": False,
        "deployment_ready": False,
        "simulator_qualified": False,
    }
    destination = directory / "comparison.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(destination)


if __name__ == "__main__":
    main()
