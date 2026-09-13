"""Apply the declared tracking gates to complete, independently scored replays."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


THRESHOLDS = dict(root_p95_m=.20, yaw_p95_deg=15., foot_relative_p95_m=.12,
                  leg_rmse_rad=.15, original_hand_relative_p95_m=.15,
                  original_head_relative_p95_m=.10, range_tolerance_rad=1e-6)


def assess(directory):
    rows=[]
    for clip in ("walk002", "pico", "walk003", "walk008"):
        case=directory/clip
        try:
            report=json.loads((case/"report.json").read_text())
            original=json.loads((case/"original_intent_metrics.json").read_text())
            metrics=report["source_metrics"]
            tests=dict(
                full_timed_source_and_lifecycle=report["full_lifecycle_completed"] and
                    metrics["source_controls"]==metrics["source_requested"],
                root_world_position=metrics["root_p95"]<=THRESHOLDS["root_p95_m"],
                heading=original["yaw_abs_p95_deg"]<=THRESHOLDS["yaw_p95_deg"],
                each_foot_relative_position=max(metrics["relative_landmark_p95"][:2])<=THRESHOLDS["foot_relative_p95_m"],
                twelve_leg_joint_rmse=metrics["leg_rmse"]<=THRESHOLDS["leg_rmse_rad"],
                original_left_hand=original["original_hand_head_relative_p95_m"][0]<=THRESHOLDS["original_hand_relative_p95_m"],
                original_right_hand=original["original_hand_head_relative_p95_m"][1]<=THRESHOLDS["original_hand_relative_p95_m"],
                original_head=original["original_hand_head_relative_p95_m"][2]<=THRESHOLDS["original_head_relative_p95_m"],
                physical_limits=report["range_excess_max"]<=THRESHOLDS["range_tolerance_rad"] and
                    report["velocity_ratio_max"]<=1 and report["effort_ratio_max"]<=1+1e-9)
            rows.append(dict(clip=clip, gates=tests, tracking_passed=all(tests.values()),
                             failed=[key for key,value in tests.items() if not value]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rows.append(dict(clip=clip, tracking_passed=False, error=str(exc)))
    result=dict(time_utc=datetime.now(timezone.utc).isoformat(), thresholds=THRESHOLDS,
                cases=rows, four_clip_tracking_passed=all(row["tracking_passed"] for row in rows),
                online_stream_qualification="separate test required",
                sensor_only_pose_qualification="separate test required",
                hardware_authorized=False, deployment_ready=False)
    (directory/"tracking_gate_assessment.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    assess(p.parse_args().directory)
