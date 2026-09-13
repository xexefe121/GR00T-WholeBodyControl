"""Evaluate each completed residual milestone on full saved-source lifecycles."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", default="2026-09-10T19:15:00+00:00")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = datetime.fromisoformat(args.deadline)
    while datetime.now(timezone.utc) < deadline:
        pending = []
        for path in sorted(args.training.glob("residual_*.pt")):
            try:
                update = int(path.stem.removeprefix("residual_"))
            except ValueError:
                continue
            if update < 200:
                continue
            output = args.output / path.stem
            if not (output / "summary.json").exists():
                pending.append((path, update, output))
        for path, update, output in pending:
            # Torch writes close before the training metric announces a
            # milestone. A brief size-stability check avoids reading partials.
            size = path.stat().st_size
            time.sleep(2)
            if size != path.stat().st_size:
                continue
            output.mkdir(exist_ok=True)
            print(json.dumps(dict(starting_update=update)), flush=True)
            rows = []
            for clip in ("walk002", "pico", "walk003", "walk008"):
                case = output / clip
                report_path = case / "report.json"
                if not report_path.exists():
                    if case.exists():
                        rows.append(dict(clip=clip, error="incomplete prior evaluation preserved"))
                        continue
                    command = [sys.executable, "-m", "gear_sonic.scripts.evaluate_g1_true23_bfmzero",
                               "--clip", clip, "--position-gain", "1", "--yaw-gain", "2",
                               "--residual-checkpoint", str(path), "--output", str(case)]
                    with (output / f"{clip}.log").open("w") as log:
                        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                timeout=600, check=False)
                    if result.returncode:
                        rows.append(dict(clip=clip, error=f"evaluation exited{result.returncode}"))
                        continue
                report = json.loads(report_path.read_text())
                rows.append({key: report.get(key) for key in (
                    "clip", "completed", "requested", "failure", "source_metrics", "range_excess_max",
                    "effort_ratio_max", "velocity_ratio_max", "inference_ms_p95", "residual_updates")})
                with (output / f"{clip}.original_intent.log").open("w") as log:
                    metrics = subprocess.run(
                        [sys.executable, "artifacts/teleop_six_hour_20260910/inspect_bfm_tracking.py", str(case)],
                        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=log,
                        stderr=subprocess.STDOUT, timeout=180, check=False)
                if metrics.returncode == 0:
                    original = json.loads((case / "original_intent_metrics.json").read_text())
                    rows[-1]["original_intent"] = {k: original.get(k) for k in (
                        "original_hand_head_world_p95_m", "original_hand_head_relative_p95_m",
                        "original_hand_head_orientation_p95_rad", "yaw_abs_p95_deg", "own_heading_foot_p95_m",
                        "physical_limits_passed")}
                else:
                    rows[-1]["original_intent_error"] = f"metric process exited{metrics.returncode}"
                print(json.dumps(dict(update=update, case=rows[-1])), flush=True)
            summary = dict(update=update, checkpoint=str(path), cases=rows,
                           full_body_tracking_qualified=False, hardware_authorized=False)
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(dict(completed_update=update, summary=str(output / "summary.json"))), flush=True)
        if (args.training / "outcome.json").exists() and not pending:
            print(json.dumps(dict(training_finished=True, evaluations_finished=True)), flush=True)
            return
        time.sleep(20)
    print(json.dumps(dict(stopped="evaluation_deadline")), flush=True)


if __name__ == "__main__":
    main()
