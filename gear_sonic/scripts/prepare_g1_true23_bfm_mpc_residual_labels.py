"""Evaluate frozen BFM on successful physical teacher states and exact history."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import PACKAGE, corrected_goal
from gear_sonic.scripts.evaluate_g1_true23_mjbatch_plan_replay import load_plan, state_difference
from gear_sonic.utils.g1_true23_bfmzero_inference import (BFMHistory, BFMZeroInference,
    load_contract, reference_features, state_and_terms)
from gear_sonic.utils.g1_true23_bfm_mpc_student import ResidualFeatures, BASE_CONTRACT, KIND, zero_checkpoint
from gear_sonic.utils.g1_true23_mpc_student import load_inputs, sha256, OFFSETS


def run(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    model, portable, motion, original, _ = load_inputs(args.bundle)
    contract = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    for key in ("default_q", "kp", "kd"):
        np.testing.assert_array_equal(portable[key], contract[key])
    policy = BFMZeroInference(PACKAGE / "bfmzero_inference_v1/inference.safetensors")
    state, privileged = reference_features(motion, contract)
    request = json.loads((args.teacher / "request.json").read_text())
    teacher = json.loads((args.teacher / "report.json").read_text())
    source_plan = Path(request["source_plan"])
    _, _, plan, _ = load_plan(source_plan, model, args.bundle / "walk002/native_original.npz",
                              args.bundle / "contract.json")
    assert request["feedback_correction_clip_rad"] == .1
    limits = np.asarray(portable["joint_limits"])
    default = contract["default_q"]
    scale = .25 * contract["training_effort"] / contract["kp"]
    features = ResidualFeatures(motion, original, contract, limits)
    rows = {k: [] for k in ("features", "residual_target", "base_target", "expert_target",
                            "expert_preclip_target", "previous_combined_action", "case_id", "source_frame")}
    accepted = [case for case in teacher["case_summaries"] if case["expert_eligible"]]
    provenance = {str(p): sha256(p) for p in (args.teacher / "report.json", args.teacher / "request.json",
                                             source_plan / "trace.npz", Path(__file__),
                                             Path(__file__).resolve().parents[1] / "utils/g1_true23_bfm_mpc_student.py")}
    started = time.perf_counter()
    max_reconstruction = 0.
    for case in accepted:
        path = args.teacher / f"case_{case['case_id']:02d}.npz"
        assert sha256(path) == case["trace_sha256"]
        provenance[str(path)] = sha256(path)
        with np.load(path, allow_pickle=False) as a:
            trajectory = {k: a[k].copy() for k in ("observed_qpos", "observed_qvel", "source_frame", "target")}
        history = BFMHistory()
        action = np.zeros(23, dtype=np.float32)
        for control, frame in enumerate(trajectory["source_frame"]):
            qpos, qvel = trajectory["observed_qpos"][control], trajectory["observed_qvel"][control]
            sensed, terms = state_and_terms(qpos[7:], qvel[6:], qpos[3:7], qvel[3:6], action, default)
            hist = history.before_update(terms)
            goal = corrected_goal(policy, state, privileged, motion, int(frame), qpos, 8, 1., 2.)
            with torch.inference_mode():
                base_action = policy.actor(torch.from_numpy(sensed[None]), torch.from_numpy(action[None]),
                                           torch.from_numpy(hist[None]), goal)[0].numpy() * 5.
            base_target = default + base_action * scale
            error = state_difference(model, plan["planned_state"][control], qpos, qvel)
            preclip = plan["planned_target"][control] + np.clip(plan["feedback_gain"][control] @ error, -.1, .1)
            target = trajectory["target"][control]
            reconstruction = float(np.max(np.abs(np.clip(preclip, limits[:, 0], limits[:, 1]) - target)))
            max_reconstruction = max(max_reconstruction, reconstruction)
            if reconstruction > 1e-12:
                raise ValueError("teacher command reconstruction differs from physical execution")
            row = dict(features=features(qpos, qvel, int(frame), base_target, action),
                       residual_target=preclip - base_target, base_target=base_target,
                       expert_target=target, expert_preclip_target=preclip,
                       previous_combined_action=action.copy(), case_id=case["case_id"], source_frame=int(frame))
            for key, value in row.items():
                rows[key].append(value)
            # Preserve preclip combined command even when the applied target was
            # clipped. Teacher histories never come from the frozen base action.
            action = ((preclip - default) / scale).astype(np.float32)
        print(json.dumps(dict(case=case["case_id"], controls=len(trajectory["target"]),
                              elapsed_s=time.perf_counter() - started)), flush=True)
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    np.savez_compressed(args.output / "labels.npz", **arrays)
    delta = np.abs(arrays["residual_target"])
    report = dict(kind=KIND, accepted_case_ids=[case["case_id"] for case in accepted],
                  rejected_cases=teacher["total_cases"] - len(accepted), actual_physical_samples=len(delta),
                  base_contract=BASE_CONTRACT, goal_offsets=OFFSETS.tolist(), received_goal_buffer_seconds=.74,
                  residual_label="reconstructed expert preclip target minus frozen BFM unclipped target on identical observed state and expert preclip combined history",
                  residual_abs_p50_p90_p95_p99_max_rad=np.percentile(delta, [50, 90, 95, 99, 100]).tolist(),
                  joint_abs_p95_rad=np.percentile(delta, 95, axis=0).tolist(),
                  fraction_labels_exceeding_radius={str(radius): float(np.mean(delta > radius)) for radius in (.15, .25, .4, .6)},
                  expert_native_clip_fraction=float(np.mean(np.abs(arrays["expert_preclip_target"] - arrays["expert_target"]) > 1e-12)),
                  exact_teacher_target_reconstruction_max_abs_rad=max_reconstruction,
                  initial_bfm_action="zero, matching standalone BFM reset", elapsed_s=time.perf_counter() - started,
                  provenance=provenance, labels_sha256=sha256(args.output / "labels.npz"),
                  simulator_qualified=False, hardware_authorized=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "preparer_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "student_snapshot.py").write_bytes((Path(__file__).resolve().parents[1] / "utils/g1_true23_bfm_mpc_student.py").read_bytes())
    zero_checkpoint(args.output / "residual_zero.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
