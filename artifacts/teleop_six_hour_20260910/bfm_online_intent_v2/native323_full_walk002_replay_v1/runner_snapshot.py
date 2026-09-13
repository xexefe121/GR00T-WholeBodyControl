"""Independent native-model replay of an offline mjbatch target/feedback plan.

No mjbatch dependency. Loads the original native model directly, and executes
manual 500Hz PD using the selected installed MuJoCo version. Only initialization
writes physical state. Saved nominal states are feedback references, never the
simulated plant. A partial plan cannot qualify a full source or live controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import DATA, PACKAGE, ROOT, load_motion, load_case_motion
from gear_sonic.utils.g1_true23_bfmzero_inference import load_contract
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS, task_points
from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks
from gear_sonic.utils.g1_true23_step1b_mujoco import _quaternion_matrix, prepare_true23_model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_difference(model, nominal, qpos, qvel):
    """MuJoCo tangent: world translation, right-local root rotation, joints, dq."""
    result = np.empty(2 * model.nv)
    mujoco.mj_differentiatePos(model, result[:model.nv], 1., nominal[:model.nq], qpos)
    result[model.nv:] = qvel - nominal[model.nq:]
    return result


def load_plan(plan_dir, model, motion_path, contract_path):
    request = json.loads((plan_dir / "request.json").read_text())
    report = json.loads((plan_dir / "report.json").read_text())
    for field, name in (("trace_sha256", "trace.npz"), ("request_sha256", "request.json"),
                        ("plans_sha256", "plans.json")):
        if report[field] != sha256(plan_dir / name):
            raise ValueError("plan receipt hash mismatch: " + name)
    hashes = request["input_hashes"]
    expected = {"native_original.npz": sha256(motion_path), "contract.json": sha256(contract_path)}
    for name, digest in expected.items():
        matches = [value for key, value in hashes.items() if key.replace("\\", "/").endswith("/" + name)]
        if matches != [digest]:
            raise ValueError("plan input does not match native referee: " + name)
    manifest = request["model_manifest"]
    if manifest["prepared_source_model_sha256"] != sha256(ROOT.parent / "GR00T-WholeBodyControl" / MODEL):
        raise ValueError("plan source model differs")
    if manifest["physics_sha256"] != sha256(ROOT / PHYSICS):
        raise ValueError("plan native physics differs")
    if (request["source_clock_hz"], request["physics_hz"], request["source_reference_lift_m"],
            request["source_frame_removal"]) != (50, 500, 0., 0):
        raise ValueError("plan changed reference time or geometry")
    with np.load(plan_dir / "trace.npz", allow_pickle=False) as archive:
        plan = {key: archive[key].copy() for key in archive.files}
    count = len(plan["target"])
    if count == 0:
        raise ValueError("plan contains no physical controls to replay")
    shapes = {"qpos": (count + 1, 30), "qvel": (count + 1, 29), "target": (count, 23),
              "planned_target": (count, 23), "planned_state": (count, 59),
              "feedback_gain": (count, 23, 58), "source_frame": (count,)}
    for key, shape in shapes.items():
        if plan[key].shape != shape or not np.isfinite(plan[key]).all():
            raise ValueError("invalid plan field: " + key)
    if not np.array_equal(plan["source_frame"], np.arange(count) + 11):
        raise ValueError("plan skipped/repeated source frames")
    # Early prototype traces predate explicit substep counts; their completed
    # controls all contain exactly ten steps. Validate physics array length.
    substeps = plan.get("physics_substeps", np.full(count, 10, dtype=int))
    if substeps.shape != (count,) or np.any(substeps != substeps.astype(int)):
        raise ValueError("invalid physics substep counts")
    if np.any(substeps < 1) or np.any(substeps > 10) or np.any(substeps[:-1] != 10):
        raise ValueError("nonterminal partial physical control")
    if len(plan["physics_qpos"]) != int(substeps.sum()) + 1:
        raise ValueError("physics trace length differs from source clock")
    plan["physics_substeps"] = substeps.astype(int)
    reconstruction_error = 0.
    for control in range(count):
        difference = state_difference(model, plan["planned_state"][control],
                                      plan["qpos"][control], plan["qvel"][control])
        target = np.clip(plan["planned_target"][control] + plan["feedback_gain"][control] @ difference,
                         model.jnt_range[1:, 0], model.jnt_range[1:, 1])
        reconstruction_error = max(reconstruction_error, float(np.max(np.abs(target - plan["target"][control]))))
    if reconstruction_error > 1e-8:
        raise ValueError(f"saved feedback cannot reproduce executed target: {reconstruction_error}")
    return request, report, plan, reconstruction_error


def run(args):
    if args.feedback_clip is not None and (not np.isfinite(args.feedback_clip) or args.feedback_clip <= 0):
        raise ValueError("feedback correction clip must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=False)
    source_request = json.loads((args.plan / "request.json").read_text())
    clip = source_request["clip"]
    base_motion, timeline, base_motion_path = load_motion(clip)
    override = source_request.get("motion_override")
    if bool(override) != (args.motion_override is not None):
        raise ValueError("retargeted plan requires its explicit local --motion-override, original plan requires none")
    motion, _, motion_path = load_case_motion(clip, args.motion_override)
    if override:
        if override["reference_sha256"] != sha256(motion_path) or override["base_native_reference_sha256"] != sha256(base_motion_path):
            raise ValueError("plan retarget/original reference hashes differ")
        portable_receipt = motion_path.parent / "portable_receipt.json"
        if override["portable_receipt_sha256"] != sha256(portable_receipt):
            raise ValueError("portable retarget validation receipt differs")
    _, model, physics = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    contract = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    portable_contract = ROOT / "artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json"
    request, source_report, plan, target_error = load_plan(args.plan, model, base_motion_path, portable_contract)
    portable = json.loads(portable_contract.read_text())
    for key in ("kp", "kd", "default_q"):
        np.testing.assert_array_equal(contract[key], portable[key])
    np.testing.assert_array_equal(physics.effort, portable["native_effort"])
    np.testing.assert_array_equal(model.jnt_range[1:], portable["joint_limits"])
    velocity_limits = np.asarray(portable["native_velocity"])
    initial_qpos = np.r_[motion["body_pos_w"][10, 0], motion["body_quat_w"][10, 0], motion["joint_pos"][10]]
    initial_qvel = np.r_[motion["body_lin_vel_w"][10, 0],
                         _quaternion_matrix(initial_qpos[3:7]).T @ motion["body_ang_vel_w"][10, 0],
                         motion["joint_vel"][10]]
    np.testing.assert_allclose(plan["qpos"][0], initial_qpos, atol=1e-12, rtol=0)
    np.testing.assert_allclose(plan["qvel"][0], initial_qvel, atol=1e-12, rtol=0)
    original_path = DATA / ("pico_freedancing_v1/optical_reference_v2/original29.npz" if clip == "pico" else
                            f"{clip}/original_source_bundle_v1/original_reference.npz")
    with np.load(original_path, allow_pickle=False) as original:
        intent_positions = original["source_task_position_w"].copy()
        intent_roots = original["source_qpos29"][:, :3].copy()
        intent_root_quats = original["source_qpos29"][:, 3:7].copy()
    if override and override["original29_sha256"] != sha256(original_path):
        raise ValueError("retarget original intent source differs")
    source_model_path = ROOT.parent / "GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml"
    source_model = mujoco.MjModel.from_xml_path(str(source_model_path))
    hand_tasks, hand_convention = neutral_wrist_hand_tasks(source_model, model)
    hand_tasks = [next(task for task in hand_tasks if task.name == name)
                  for name in ("left_hand", "right_hand", "head_proxy")]
    geometry_path = ROOT / "gear_sonic/utils/g1_true23_hand_frame_tasks.py"
    provenance = dict(mujoco=mujoco.__version__, source_mujoco=request["mujoco"],
                      mode=args.mode, clip=clip, plan=str(args.plan.resolve()),
                      motion_override=None if args.motion_override is None else str(motion_path),
                      feedback_correction_clip_rad=args.feedback_clip,
                      hashes={str(path): sha256(path) for path in (Path(__file__), args.plan / "trace.npz",
                          args.plan / "request.json", args.plan / "report.json", base_motion_path, motion_path, original_path,
                          ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS, portable_contract,
                          source_model_path, geometry_path)},
                      metric_revision=2, original_intent_convention=hand_convention,
                      declared_future_preview_seconds=request["declared_future_preview_seconds"],
                      conservative_effective_source_preview_seconds=request.get("conservative_effective_source_preview_seconds", request["declared_future_preview_seconds"]),
                      recorded_target_seed=request.get("recorded_target_seed"),
                      feedback_target_reconstruction_error=target_error,
                      full_source_requested_controls=timeline["phases"][2]["requested_controls"],
                      hardware_authorized=False)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    body_ids = [model.body(name).id for name in contract["body_names"]]
    limits = model.jnt_range[1:]
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = initial_qpos, initial_qvel
    mujoco.mj_forward(model, data)
    fields = ("qpos", "qvel", "target", "source_frame", "joint_error", "root_error",
              "yaw_error_deg", "relative_landmark_error", "original_relative_hand_head_error",
              "range_excess", "velocity_ratio", "effort_ratio", "feedback_ms", "physics_qpos",
              "physics_qvel", "physics_torque", "physics_substeps", "raw_feedback_correction",
              "original_root_error", "original_yaw_error_deg")
    trace = {key: [] for key in fields}
    for key, value in (("qpos", data.qpos), ("qvel", data.qvel),
                       ("physics_qpos", data.qpos), ("physics_qvel", data.qvel)):
        trace[key].append(value.copy())
    failure = None
    started = time.perf_counter()
    for control in range(len(plan["target"])):
        tick = time.perf_counter()
        target = plan["target"][control].copy()
        correction = np.zeros(23)
        if args.mode == "feedback":
            error = state_difference(model, plan["planned_state"][control], data.qpos, data.qvel)
            correction = plan["feedback_gain"][control] @ error
            applied_correction = correction if args.feedback_clip is None else np.clip(correction, -args.feedback_clip, args.feedback_clip)
            target = np.clip(plan["planned_target"][control] + applied_correction,
                             limits[:, 0], limits[:, 1])
        feedback_ms = 1000 * (time.perf_counter() - tick)
        peak_range = peak_velocity = peak_effort = 0.
        substeps = 0
        for _ in range(plan["physics_substeps"][control]):
            torque = contract["kp"] * (target - data.qpos[7:]) - contract["kd"] * data.qvel[6:]
            data.ctrl[:] = np.clip(torque, -physics.effort, physics.effort)
            mujoco.mj_step(model, data)
            substeps += 1
            trace["physics_qpos"].append(data.qpos.copy())
            trace["physics_qvel"].append(data.qvel.copy())
            trace["physics_torque"].append(data.ctrl.copy())
            peak_range = max(peak_range, float(np.max(np.maximum(limits[:, 0] - data.qpos[7:], data.qpos[7:] - limits[:, 1]))))
            peak_velocity = max(peak_velocity, float(np.max(np.abs(data.qvel[6:]) / velocity_limits)))
            peak_effort = max(peak_effort, float(np.max(np.abs(data.qfrc_actuator[6:]) / physics.effort)))
            tilt = float(np.arccos(np.clip(1 - 2 * np.sum(data.qpos[4:6] ** 2), -1, 1)))
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.qpos[2] < .25 or tilt > 1.2:
                failure = dict(kind="fall_or_nonfinite", control=control, substep=substeps)
            elif peak_range > .01 or peak_velocity > 1 or peak_effort > 1 + 1e-8:
                failure = dict(kind="physical_limit", control=control, substep=substeps,
                               range_excess=peak_range, velocity_ratio=peak_velocity, effort_ratio=peak_effort)
            if failure:
                break
        frame = int(plan["source_frame"][control])
        mujoco.mj_kinematics(model, data)
        actual = task_points(data.xpos[body_ids], data.xquat[body_ids])
        desired = task_points(motion["body_pos_w"][frame], motion["body_quat_w"][frame])
        root_error = data.qpos[:3] - motion["body_pos_w"][frame, 0]
        actual_intent = np.array([data.xpos[model.body(task.target_body).id] +
                         data.xmat[model.body(task.target_body).id].reshape(3, 3) @ task.target_point
                         for task in hand_tasks])
        relative_intent_error = np.linalg.norm((actual_intent - data.qpos[:3]) -
                                              (intent_positions[frame] - intent_roots[frame]), axis=-1)
        actrot, refrot = _quaternion_matrix(data.qpos[3:7]), _quaternion_matrix(motion["body_quat_w"][frame, 0])
        dyaw = np.arctan2(actrot[1, 0], actrot[0, 0]) - np.arctan2(refrot[1, 0], refrot[0, 0])
        original_rot = _quaternion_matrix(intent_root_quats[frame])
        original_dyaw = np.arctan2(actrot[1, 0], actrot[0, 0]) - np.arctan2(original_rot[1, 0], original_rot[0, 0])
        values = dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), target=target, source_frame=frame,
                      joint_error=data.qpos[7:] - motion["joint_pos"][frame], root_error=root_error,
                      yaw_error_deg=np.degrees(abs(np.arctan2(np.sin(dyaw), np.cos(dyaw)))),
                      original_root_error=data.qpos[:3] - intent_roots[frame],
                      original_yaw_error_deg=np.degrees(abs(np.arctan2(np.sin(original_dyaw), np.cos(original_dyaw)))),
                      relative_landmark_error=np.linalg.norm(actual - desired - root_error, axis=-1),
                      original_relative_hand_head_error=relative_intent_error, range_excess=peak_range,
                      velocity_ratio=peak_velocity, effort_ratio=peak_effort, feedback_ms=feedback_ms,
                      raw_feedback_correction=correction,
                      physics_substeps=substeps)
        for key, value in values.items():
            trace[key].append(value)
        if failure:
            break
    arrays = {key: np.asarray(value) for key, value in trace.items()}
    np.savez_compressed(args.output / "trace.npz", **arrays)
    count = len(arrays["target"])
    phase = next(phase for phase in timeline["phases"] if phase["name"] == "source_motion")
    first, last = phase["control_start"], min(count, phase["control_stop"])
    selection = slice(first, last) if last > first else slice(0, count)
    metrics = dict(phase="source_motion" if last > first else "prefix",
                   controls=last - first if last > first else count,
                   includes_partial_final_control=bool(arrays["physics_substeps"][selection][-1] != 10),
                   leg_rmse=float(np.sqrt(np.mean(arrays["joint_error"][selection, :12] ** 2))),
                   arm_rmse=float(np.sqrt(np.mean(arrays["joint_error"][selection, 13:] ** 2))),
                   root_p95=float(np.percentile(np.linalg.norm(arrays["root_error"][selection], axis=-1), 95)),
                   yaw_p95_deg=float(np.percentile(arrays["yaw_error_deg"][selection], 95)),
                   original_root_p95=float(np.percentile(np.linalg.norm(arrays["original_root_error"][selection], axis=-1), 95)),
                   original_yaw_p95_deg=float(np.percentile(arrays["original_yaw_error_deg"][selection], 95)),
                   relative_landmark_p95_m=np.percentile(arrays["relative_landmark_error"][selection], 95, axis=0).tolist(),
                   original_relative_hand_head_p95_m=np.percentile(arrays["original_relative_hand_head_error"][selection], 95, axis=0).tolist())
    nphys = len(arrays["physics_qpos"])
    full_slots = arrays["physics_substeps"] == 10
    result = dict(clip=clip, mode=args.mode, mujoco=mujoco.__version__, source_mujoco=request["mujoco"],
                  metric_revision=2,
                  motion_override=None if args.motion_override is None else str(motion_path),
                  feedback_correction_clip_rad=args.feedback_clip,
                  feedback_clip_active_controls=0 if args.feedback_clip is None else int(np.sum(np.max(np.abs(arrays["raw_feedback_correction"]), axis=1) > args.feedback_clip)),
                  completed_controls=count, planned_controls=len(plan["target"]),
                  completed_full_controls=int(full_slots.sum()),
                  partial_final_control_substeps=None if full_slots[-1] else int(arrays["physics_substeps"][-1]),
                  plan_replayed=count == len(plan["target"]) and failure is None,
                  source_plan_failed=source_report["failure"] is not None,
                  full_source_completed=count >= phase["control_stop"] and bool(np.all(full_slots[:phase["control_stop"]])) and failure is None,
                  full_lifecycle_completed=count == len(motion["joint_pos"]) - 11 and bool(np.all(full_slots)) and failure is None,
                  failure=failure, metrics=metrics,
                  range_excess_max=float(arrays["range_excess"].max()),
                  velocity_ratio_max=float(arrays["velocity_ratio"].max()),
                  effort_ratio_max=float(arrays["effort_ratio"].max()),
                  source_physics_qpos_max_abs_difference=float(np.max(np.abs(arrays["physics_qpos"] - plan["physics_qpos"][:nphys]))),
                  source_physics_qvel_max_abs_difference=float(np.max(np.abs(arrays["physics_qvel"] - plan["physics_qvel"][:nphys]))),
                  feedback_only_ms_p50_p95_max=np.percentile(arrays["feedback_ms"], [50, 95, 100]).tolist(),
                  feedback_timing_excludes_offline_planning=True,
                  planning_ms_p50_p95_max=source_report["planning_ms_p50_p95_max"],
                  declared_future_preview_seconds=request["declared_future_preview_seconds"],
                  conservative_effective_source_preview_seconds=request.get("conservative_effective_source_preview_seconds", request["declared_future_preview_seconds"]),
                  recorded_target_seed=request.get("recorded_target_seed"),
                  physical_state_rewrites_after_initialization=0, root_assistance_forces=0,
                  elapsed_wall_seconds=time.perf_counter() - started, simulated_seconds=float(data.time),
                  physical_steps=nphys - 1, hardware_authorized=False, received_stream_controller=False,
                  full_body_tracking_qualified=False, timing_qualified=False,
                  trace_sha256=sha256(args.output / "trace.npz"),
                  provenance_sha256=sha256(args.output / "provenance.json"))
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mode", choices=("executed-target", "feedback"), default="feedback")
    parser.add_argument("--feedback-clip", type=float)
    parser.add_argument("--motion-override", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
