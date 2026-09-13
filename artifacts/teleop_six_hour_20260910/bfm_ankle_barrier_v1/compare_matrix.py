"""Matched-source-prefix comparison; never compare a truncated baseline as full."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CASES = [
    ("walk002", False, BASE / "bfm_walk002_feedback_v2", HERE / "walk002_k150d2"),
    ("walk008", False, HERE / "walk008_zero", HERE / "walk008_k150d2"),
    ("pico", False, BASE / "bfm_pico_feedback_v2", HERE / "pico_k150d2"),
    ("walk002", True, HERE / "walk002_zero_arms", HERE / "walk002_k150d2_arms"),
    ("walk008", True, BASE / "bfm_walk008_arms_v3", HERE / "walk008_k150d2_arms"),
    ("pico", True, BASE / "bfm_pico_arms_v3", HERE / "pico_k150d2_arms"),
]


def metrics(trace, start, stop):
    return {
        "source_controls": stop - start,
        "leg_rmse": float(np.sqrt(np.mean(trace["joint_error"][start:stop, :12] ** 2))),
        "arm_rmse": float(np.sqrt(np.mean(trace["joint_error"][start:stop, 13:] ** 2))),
        "root_p95": float(np.percentile(np.linalg.norm(trace["root_error"][start:stop], axis=-1), 95)),
        "relative_foot_p95": np.percentile(trace["relative_landmark_error"][start:stop, :2], 95, axis=0).tolist(),
    }


rows = []
for clip, arms, baseline_path, candidate_path in CASES:
    if not (candidate_path / "report.json").exists():
        continue
    baseline = json.loads((baseline_path / "report.json").read_text())
    candidate = json.loads((candidate_path / "report.json").read_text())
    request = json.loads((candidate_path / "request.json").read_text())
    with np.load(baseline_path / "trace.npz") as archive:
        a = {key: archive[key] for key in archive.files}
    with np.load(candidate_path / "trace.npz") as archive:
        b = {key: archive[key] for key in archive.files}
    phase = next(phase for phase in request["source_timeline"]["phases"] if phase["name"] == "source_motion")
    start, stop = phase["control_start"], min(phase["control_stop"], len(a["action"]), len(b["action"]))
    active = np.flatnonzero(np.any(b["physics_barrier_torque"] != 0, axis=1))
    first = int(active[0] // 10) if len(active) else min(len(a["action"]), len(b["action"]))
    pre_exact = all(np.array_equal(a[key][:first], b[key][:first]) for key in ("qpos", "qvel", "action", "target", "state", "history"))
    rows.append({
        "clip": clip,
        "direct_arm_reference": arms,
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "baseline_full_lifecycle": baseline["full_lifecycle_completed"],
        "candidate_full_lifecycle": candidate["full_lifecycle_completed"],
        "baseline_range_excess_rad": baseline["range_excess_max"],
        "candidate_range_excess_rad": candidate["range_excess_max"],
        "candidate_effort_ratio": candidate["effort_ratio_max"],
        "candidate_velocity_ratio": candidate["velocity_ratio_max"],
        "first_barrier_control": first if len(active) else None,
        "pre_intervention_common_arrays_bit_exact": pre_exact,
        "matched_prefix_baseline": metrics(a, start, stop),
        "matched_prefix_candidate": metrics(b, start, stop),
        "full_candidate_source": candidate["source_metrics"],
        "source_geometry_qualified": False,
        "hardware_qualified": False,
        "report_sha256": {str(path): hashlib.sha256((path / "report.json").read_bytes()).hexdigest() for path in (baseline_path, candidate_path)},
    })
result = {"cases": rows, "fixed_parameters": {"margin_rad": .05, "stiffness": 150., "damping": 2.}, "all_requested_candidates_present": len(rows) == len(CASES), "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(HERE / "comparison.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
