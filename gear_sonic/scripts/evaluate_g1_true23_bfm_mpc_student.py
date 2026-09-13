"""Native3.2.3 physical closed-loop evaluation of a bounded BFM residual head."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import PACKAGE, corrected_goal
from gear_sonic.utils.g1_true23_bfmzero_inference import (BFMHistory, BFMZeroInference,
    load_contract, reference_features, state_and_terms)
from gear_sonic.utils.g1_true23_bfm_mpc_student import BFMMPCResidualCPU, BASE_CONTRACT
from gear_sonic.utils.g1_true23_mpc_student import load_inputs, physical_failure, sha256, tracking_metrics, OFFSETS


def run(args):
    assert mujoco.__version__ == "3.2.3"; torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    model, portable, motion, original, timeline = load_inputs(args.bundle, args.clip)
    contract = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    limits = np.asarray(portable["joint_limits"])
    student = BFMMPCResidualCPU(args.checkpoint, motion, original, contract, limits)
    policy = BFMZeroInference(PACKAGE / "bfmzero_inference_v1/inference.safetensors")
    state, privileged = reference_features(motion, contract)
    history = BFMHistory(); action = np.zeros(23, np.float32)
    default = contract["default_q"]; scale = .25 * contract["training_effort"] / contract["kp"]
    kp, kd, effort = [np.asarray(portable[k]) for k in ("kp", "kd", "native_effort")]
    data = mujoco.MjData(model)
    data.qpos[:] = np.r_[motion["body_pos_w"][10, 0], motion["body_quat_w"][10, 0], motion["joint_pos"][10]]
    data.qvel[:] = np.r_[motion["body_lin_vel_w"][10, 0],
                        Rotation.from_quat(data.qpos[[4, 5, 6, 3]]).inv().apply(motion["body_ang_vel_w"][10, 0]),
                        motion["joint_vel"][10]]
    if args.initial_case:
        with np.load(args.initial_case, allow_pickle=False) as a:
            data.qpos[:] = a["qpos"][0]; data.qvel[:] = a["qvel"][0]
    mujoco.mj_forward(model, data)
    count = min(args.controls, len(motion["joint_pos"]) - 11)
    request = dict(kind="native23_bfm_mpc_residual_closed_loop", clip=args.clip, mujoco=mujoco.__version__,
                   checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
                   training_step=student.saved["step"], radius_rad=student.radius, base_contract=BASE_CONTRACT,
                   requested_controls=count, initial_case=None if args.initial_case is None else str(args.initial_case.resolve()),
                   goal_offsets=OFFSETS.tolist(), received_goal_frames=38, declared_received_buffer_seconds=.74,
                   no_clip_frame_or_time_inputs=True, native_effort_pd_500hz=True, goal_control_hz=50,
                   combined_action_history="base action plus residual/scale BEFORE final native target clipping",
                   source_time_warped=False, source_frames_removed=0, root_assistance_forces=0,
                   physical_state_rewrites_after_initialization=0, policy_timing_includes_bfm_goal_actor_and_residual=True,
                   hardware_authorized=False, evaluator_sha256=sha256(Path(__file__)),
                   student_sha256=sha256(Path(__file__).resolve().parents[1] / "utils/g1_true23_bfm_mpc_student.py"))
    (args.output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    for name, path in (("evaluator_snapshot.py", Path(__file__)),
                       ("student_snapshot.py", Path(__file__).resolve().parents[1] / "utils/g1_true23_bfm_mpc_student.py")):
        (args.output / name).write_bytes(path.read_bytes())
    trace = {k: [] for k in ("qpos", "qvel", "target", "base_target", "residual", "combined_action", "source_frame",
                             "policy_ms", "joint_error", "root_error", "feet_error", "original_task_error",
                             "original_relative_task_error", "physics_qpos", "physics_qvel", "physics_torque",
                             "range_excess", "velocity_ratio", "effort_ratio")}
    for key, value in (("qpos", data.qpos), ("qvel", data.qvel), ("physics_qpos", data.qpos), ("physics_qvel", data.qvel)):
        trace[key].append(value.copy())
    failure = None; started = time.perf_counter()
    for control in range(count):
        frame = control + 11; tick = time.perf_counter()
        sensed, terms = state_and_terms(data.qpos[7:], data.qvel[6:], data.qpos[3:7], data.qvel[3:6], action, default)
        hist = history.before_update(terms)
        goal = corrected_goal(policy, state, privileged, motion, frame, data.qpos, 8, 1., 2.)
        with torch.inference_mode():
            base_action = policy.actor(torch.from_numpy(sensed[None]), torch.from_numpy(action[None]),
                                       torch.from_numpy(hist[None]), goal)[0].numpy() * 5.
        base_target = default + base_action * scale
        delta = student.predict(data.qpos, data.qvel, frame, base_target, action)
        target = np.clip(base_target + delta, limits[:, 0], limits[:, 1])
        action = (base_action + delta / scale).astype(np.float32)
        policy_ms = (time.perf_counter() - tick) * 1000
        for substep in range(10):
            data.ctrl[:] = np.clip(kp * (target - data.qpos[7:]) - kd * data.qvel[6:], -effort, effort)
            mujoco.mj_step(model, data)
            for key, value in (("physics_qpos", data.qpos), ("physics_qvel", data.qvel), ("physics_torque", data.ctrl)):
                trace[key].append(value.copy())
            failure, physical = physical_failure(model, data, portable)
            for key, value in physical.items(): trace[key].append(value)
            if failure:
                failure.update(control=control, substep=substep + 1, time=float(data.time)); break
        values = dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), target=target, base_target=base_target,
                      residual=delta, combined_action=action.copy(), source_frame=frame, policy_ms=policy_ms,
                      **tracking_metrics(model, data, motion, original, portable, frame))
        for key, value in values.items(): trace[key].append(value)
        if failure: break
    arrays = {k: np.asarray(v) for k, v in trace.items()}
    np.savez_compressed(args.output / "trace.npz", **arrays)
    complete = len(trace["physics_torque"]) // 10
    phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    first = phase["control_start"] if complete > phase["control_start"] else 0
    stop = min(complete, phase["control_stop"]) if first else complete; idx = slice(first, stop)
    metrics = dict(phase="source_motion" if first else "prefix", controls=stop - first,
                   leg_rmse_rad=float(np.sqrt(np.mean(arrays["joint_error"][idx, :12] ** 2))),
                   arm_rmse_rad=float(np.sqrt(np.mean(arrays["joint_error"][idx, 13:] ** 2))),
                   root_p95_m=float(np.percentile(np.linalg.norm(arrays["root_error"][idx], axis=1), 95)),
                   feet_p95_m=np.percentile(arrays["feet_error"][idx], 95, axis=0).tolist(),
                   original_hand_head_p95_m=np.percentile(arrays["original_task_error"][idx], 95, axis=0).tolist())
    result = dict(probe_completed=complete == count and failure is None, completed_full_controls=complete,
                  partial_substeps=len(trace["physics_torque"]) % 10, requested_controls=count,
                  full_source_completed=complete >= phase["control_stop"] and failure is None,
                  failure=failure, metrics=metrics, range_excess_max=max(trace["range_excess"]),
                  velocity_ratio_max=max(trace["velocity_ratio"]), effort_ratio_max=max(trace["effort_ratio"]),
                  policy_ms_p50_p95_max=np.percentile(arrays["policy_ms"], [50, 95, 100]).tolist(),
                  policy_20ms_deadline_misses=int(np.sum(arrays["policy_ms"] > 20)),
                  simulated_seconds=float(data.time), elapsed_s=time.perf_counter() - started,
                  full_body_tracking_qualified=False, hardware_authorized=False,
                  request_sha256=sha256(args.output / "request.json"), trace_sha256=sha256(args.output / "trace.npz"))
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--initial-case", type=Path)
    p.add_argument("--clip", default="walk002"); p.add_argument("--controls", type=int, default=500)
    run(p.parse_args())
