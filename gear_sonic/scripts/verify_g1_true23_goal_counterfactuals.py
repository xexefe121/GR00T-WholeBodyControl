"""Read-only independent saved-array and pre-control clocking verification."""
import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_mpc_student import sha256


def archive(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def run(args):
    summary = json.loads((args.input / "summary.json").read_text())
    results = []
    for clip in ("pico", "walk002"):
        base = args.input / clip
        report = json.loads((base / "report.json").read_text())
        a = archive(base / "counterfactuals.npz")
        trace = archive(report["baseline_trace_path"])
        phase = report["source_phase"]
        control = a["control"]
        checks = dict(array_hash=sha256(base / "counterfactuals.npz") == report["array_archive_sha256"],
                      baseline_hash=sha256(report["baseline_trace_path"]) == report["baseline_trace_sha256"],
                      sample_count=len(control) == 100 and len(np.unique(control)) == 100,
                      source_endpoints=control[0] == phase["control_start"] and control[-1] == phase["control_stop"] - 1,
                      source_only=bool(np.all((control >= phase["control_start"]) & (control < phase["control_stop"]))),
                      source_frame_alignment=np.array_equal(a["source_frame"], control + 11),
                      precontrol_qpos=np.array_equal(a["pre_qpos"], trace["qpos"][control]),
                      precontrol_qvel=np.array_equal(a["pre_qvel"], trace["qvel"][control]),
                      current_state=np.array_equal(a["actor_state"], trace["state"][control]),
                      previous_history=np.array_equal(a["actor_history"], trace["history"][control]),
                      previous_action=np.array_equal(a["previous_rescaled_action"], trace["action"][control - 1]),
                      original_action_exact=np.array_equal(a["original_rescaled_action"], trace["action"][control]),
                      original_target_exact=np.array_equal(a["original_clipped_target"], trace["target"][control]),
                      finite_arrays=all(np.isfinite(value).all() for value in a.values()))
        checks = {key: bool(value) for key, value in checks.items()}
        errors = []
        pairs = []
        for comparison in report["comparisons"]:
            left, right = comparison["left"], comparison["right"]
            z0, z1 = a[left + "_latent"], a[right + "_latent"]
            cos = np.clip(np.sum(z0 * z1, axis=1) / (np.linalg.norm(z0, axis=1) * np.linalg.norm(z1, axis=1)), -1, 1)
            for name, value in (("p50", np.percentile(cos, 50)), ("p95", np.percentile(cos, 95)),
                                ("max", cos.max()), ("mean", cos.mean())):
                errors.append(abs(float(value) - comparison["latent_cosine"][name]))
            rms = {}
            for kind in ("raw_target", "clipped_target"):
                delta = a[right + "_" + kind] - a[left + "_" + kind]
                for group, span in (("legs", slice(0, 12)), ("waist", slice(12, 13)), ("arms", slice(13, 23))):
                    value = float(np.sqrt(np.mean(np.square(delta[:, span]))))
                    errors.append(abs(value - comparison[kind + "_delta"][group]["rmse_rad"]))
                    rms[kind + "_" + group] = value
            pairs.append(dict(left=left, right=right, latent_cosine_median=float(np.median(cos)), target_rms=rms))
        result = dict(clip=clip, checks=checks, pair_metric_max_abs_error=max(errors), comparisons=pairs,
                      passed=all(checks.values()) and max(errors) < 1e-12)
        results.append(result)
    report = dict(all_passed=all(r["passed"] for r in results), results=results,
                  input_summary_sha256=sha256(args.input / "summary.json"),
                  code_sha256=sha256(Path(__file__)), source_review_precontrol_alignment_correct=True,
                  source_review_pairs="original to A isolates derivative channels; A to B isolates common Z lift and corresponding linear velocity; A to v3 combines pose and its matching derivatives",
                  reference_feature_statistics_are_before_outer_feedback=True,
                  physical_steps=0, closed_loop_quality_established=False, out_of_distribution_established=False)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
