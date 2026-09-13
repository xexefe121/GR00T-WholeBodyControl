"""Read-only mean-target clipping/exploration audit of complete CPU source traces.

Uses the independently checked checkpoint's largest Gaussian standard deviation
to upper-bound the probability of an outside mean sampling back into the target
range. This is an evaluation-state diagnosis, not training occupancy or a new
tracking acceptance rule. It never reruns a simulator or changes a policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from gear_sonic.trl.mjlab.native23_projected_target_ppo import reachable_source_bounds
from gear_sonic.utils.g1_23dof_contract import NATIVE_IL23_JOINT_NAMES
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_ACTION_CONVENTION, SOURCE_SCALE_NATIVE_IL23


def measure_source_reachability(raw, std_upper):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 23 or len(raw) == 0 or not np.isfinite(raw).all():
        raise ValueError("reachability requires complete finite [source controls,23] means")
    if not np.isfinite(std_upper) or not 0.02 <= std_upper <= 0.5:
        raise ValueError("checkpoint exploration bound must be finite within .02..0.5")
    low, high = reachable_source_bounds()
    distance = np.maximum(np.maximum(low - raw, raw - high), 0)
    outside = distance > 0
    # Re-entering the finite interval is a subset of crossing the nearest edge.
    probability_upper = ndtr(-distance / std_upper)
    unreachable = outside & (probability_upper < 0.01)
    radians = distance * np.asarray(SOURCE_SCALE_NATIVE_IL23)
    return {
        "source_controls": len(raw),
        "any_mean_projected_fraction": float(np.any(outside, axis=1).mean()),
        "projection_loss_radian_squared": float(np.square(radians).sum(-1).mean()),
        "any_joint_less_than_one_percent_reentry_probability_fraction": float(np.any(unreachable, axis=1).mean()),
        "std_upper_bound_from_independent_checkpoint_check": float(std_upper),
        "probability_bound": "P(sample_reenters_interval)<=Phi(-distance_to_nearest_edge/std_max)",
        "joints": {
            name: {
                "source_raw_target_range": [float(low[j]), float(high[j])],
                "mean_projected_fraction": float(outside[:, j].mean()),
                "less_than_one_percent_reentry_probability_fraction": float(unreachable[:, j].mean()),
                "maximum_outside_distance_in_std_upper_units": float(distance[:, j].max() / std_upper),
                "maximum_target_overreach_rad": float(radians[:, j].max()),
            }
            for j, name in enumerate(NATIVE_IL23_JOINT_NAMES)
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--continuation-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite reachability evidence")
    report_path = args.evaluation_directory.resolve(strict=True) / "report.json"
    check_path = args.continuation_verification.resolve(strict=True)
    bindings = {str(p): sha256_file(p) for p in (report_path, check_path, Path(__file__).resolve())}
    from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure

    repo = Path(__file__).resolve().parents[2]
    for path in collect_local_source_closure(repo, [Path(__file__).resolve()]).files:
        bindings[str(path)] = sha256_file(path)
    report, check = json.loads(report_path.read_text()), json.loads(check_path.read_text())
    if check.get("continuation_update_verification_passed") is not True:
        raise ValueError("reachability needs a passed actual continuation checkpoint check")
    if report.get("release_compatibility", {}).get("action_convention") != SOURCE_ACTION_CONVENTION:
        raise ValueError("reachability requires the original source action units")
    phase = next(p for p in report["timeline"]["phases"] if p["name"] == "source_motion")
    start, stop = phase["control_start"], phase["control_stop"]
    rows = []
    for case in report["records"]:
        source = case["policy_identity"]["source"]
        if (
            source["checkpoint_sha256"] != check["trained_checkpoint_sha256"]
            or source["completed_update_count"] != check["completed_update_count"]
            or source["lineage_sha256"] != check["lineage_sha256"]
        ):
            raise ValueError("evaluation and exploration audit belong to different checkpoints")
        trace = Path(case["trace_path"])
        if sha256_file(trace) != case["trace_sha256"]:
            raise ValueError("reachability trace hash differs")
        bindings[str(trace)] = case["trace_sha256"]
        if case["result"]["completed_controls"] != case["result"]["requested_controls"]:
            raise ValueError("reachability comparison requires complete lifecycle traces")
        with np.load(trace, allow_pickle=False) as data:
            means = data["released_model_raw23"]
            if means.shape != (case["result"]["completed_controls"], 23) or not 0 <= start < stop <= len(means):
                raise ValueError("reachability source span or raw action shape differs")
            measured = measure_source_reachability(means[start:stop], check["std_max_observed"])
        rows.append({"case": case["case"], "policy_source": source, "metrics": measured})
    for path, expected in bindings.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("reachability input changed during measurement")
    result = {
        "kind": "native23_source_action_reachability_saved_trace_audit_v1",
        "inputs": bindings,
        "cases": rows,
        "scope": "complete_source_phase_at_recorded_CPU_evaluation_states",
        "actual_training_occupancy_measured": False,
        "tracking_acceptance_rules_changed": False,
        "simulation_rerun": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": [
                    {
                        "case": row["case"],
                        "projection_loss": row["metrics"]["projection_loss_radian_squared"],
                        "any_mean_projected_fraction": row["metrics"]["any_mean_projected_fraction"],
                    }
                    for row in rows
                ],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
