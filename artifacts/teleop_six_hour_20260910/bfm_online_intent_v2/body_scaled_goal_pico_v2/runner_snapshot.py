"""Standalone CPU MuJoCo referee for the public native23 BFM-Zero policy.

Simulation only. Uses the exact native model and effort caps, BFM's published
gain/action contract, and source-only latent goals. No controller fallback,
root-force assistance, pose rewrites after initialization, or robot transport.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from gear_sonic.utils.g1_true23_bfmzero_inference import (
    BFMHistory, BFMZeroInference, load_contract, reference_features, state_and_terms,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS, task_points
from gear_sonic.utils.g1_true23_step1b_mujoco import _quaternion_matrix, prepare_true23_model

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "artifacts/g1_true23_six_hour_replan_20260910_v1"
DATA = Path("C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")


def platform_path(value):
    if value.startswith("/mnt/c/"):
        return Path("C:/" + value[7:])
    return Path(value)


def load_motion(name):
    report_path = DATA / ("normal_core_pico_v1/report.json" if name == "pico" else
                          f"released_core_comparison_v1/normal/{name}/report.json")
    report = json.loads(report_path.read_text())
    timeline = report["timeline"]
    path = platform_path(timeline["timeline_path"])
    with np.load(path, allow_pickle=False) as z:
        motion = {k: z[k].copy() for k in z.files}
    return motion, timeline, path


def load_case_motion(name, override=None):
    motion,timeline,path=load_motion(name)
    if override is None:
        return motion,timeline,path
    path=Path(override).resolve()
    with np.load(path,allow_pickle=False) as archive:
        candidate={key:archive[key].copy() for key in archive.files}
    if set(candidate)!=set(motion):
        raise ValueError("declared retarget motion fields differ")
    for key in motion:
        if candidate[key].shape!=motion[key].shape or not np.isfinite(candidate[key]).all():
            raise ValueError("retarget must preserve all finite timed source samples: "+key)
    if not np.array_equal(candidate["fps"],motion["fps"]):
        raise ValueError("retarget must preserve original playback rate")
    if np.max(np.abs(np.linalg.norm(candidate["body_quat_w"],axis=-1)-1.))>1e-5:
        raise ValueError("retarget contains invalid body rotations")
    receipt=json.loads((path.parent/"report.json").read_text())
    if receipt.get("clip")!=name or receipt.get("full_original_timeline") is not True:
        raise ValueError("retarget requires a complete matching source receipt")
    if receipt.get("reference_sha256")!=hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("retarget bytes differ from its declared receipt")
    dt=1./float(candidate["fps"][0])
    _,native,_=prepare_true23_model(ROOT.parent/"GR00T-WholeBodyControl"/MODEL,ROOT/PHYSICS)
    limits=native.jnt_range[1:]
    joint=candidate["joint_pos"]
    if np.max(np.maximum(limits[:,0]-joint,joint-limits[:,1]))>1e-8:
        raise ValueError("retarget joint pose exceeds native position limits")
    speed=np.asarray(json.loads((ROOT/PHYSICS).read_text())["physics"]["velocity_limit_hardware_radps"])
    ratios=np.abs(np.diff(joint,axis=0))/dt/speed
    if np.max(ratios)>1.+1e-8:
        frame,axis=np.unravel_index(np.argmax(ratios),ratios.shape)
        raise ValueError(f"retarget interval {frame}->{frame+1} joint {axis} exceeds native speed by ratio {ratios[frame,axis]:.6f}")
    for key,position in (("joint_vel",joint),("body_lin_vel_w",candidate["body_pos_w"])):
        expected=np.gradient(position,dt,axis=0)
        if not np.allclose(candidate[key],expected,atol=1e-6,rtol=1e-5):
            raise ValueError("retarget derivative differs from timed poses: "+key)
    from scipy.spatial.transform import Rotation
    count=len(joint)
    earlier=np.maximum(np.arange(count)-1,0)
    later=np.minimum(np.arange(count)+1,count-1)
    for body in range(24):
        rotations=Rotation.from_quat(candidate["body_quat_w"][:,body,[1,2,3,0]])
        expected=(rotations[later]*rotations[earlier].inv()).as_rotvec()/((later-earlier)*dt)[:,None]
        if not np.allclose(candidate["body_ang_vel_w"][:,body],expected,atol=1e-6,rtol=1e-5):
            raise ValueError("retarget angular velocity differs from timed body rotations")
    data=mujoco.MjData(native)
    for frame in range(count):
        data.qpos[:]=np.r_[candidate["body_pos_w"][frame,0],candidate["body_quat_w"][frame,0],joint[frame]]
        mujoco.mj_kinematics(native,data)
        if np.max(np.abs(data.xpos[1:]-candidate["body_pos_w"][frame]))>1e-7:
            raise ValueError(f"retarget body positions inconsistent with native FK at frame {frame}")
        dot=np.sum(data.xquat[1:]*candidate["body_quat_w"][frame],axis=-1)
        if np.max(np.abs(np.abs(dot)-1.))>1e-7:
            raise ValueError(f"retarget body orientations inconsistent with native FK at frame {frame}")
    return candidate,timeline,path


def corrected_goal(policy, state, privileged, motion, frame, qpos, horizon, position_gain, yaw_gain,
                   goal_gyro_convention="published-world-unscaled"):
    """Task-space outer feedback through BFM's existing desired-velocity inputs.

    This consumes simulator ground-truth pose, so it is separately labelled
    simulation-only until a physical pose estimator is validated. No forces or
    simulator state changes are applied. The source pose remains unchanged.
    """
    stop = min(frame + horizon, len(state))
    states = state[frame:stop].copy()
    priv = privileged[frame:stop].copy()
    if not position_gain and not yaw_gain:
        with torch.inference_mode():
            z = policy.backward(torch.from_numpy(states), torch.from_numpy(priv)).mean(0, keepdim=True)
            return 16 * F.normalize(z, dim=-1)
    ref_rot = _quaternion_matrix(motion["body_quat_w"][frame, 0])
    act_rot = _quaternion_matrix(qpos[3:7])
    ref_yaw = np.arctan2(ref_rot[1, 0], ref_rot[0, 0])
    act_yaw = np.arctan2(act_rot[1, 0], act_rot[0, 0])
    yaw_error = np.arctan2(np.sin(ref_yaw - act_yaw), np.cos(ref_yaw - act_yaw))
    omega = np.array([0., 0., np.clip(yaw_gain * yaw_error, -.8, .8)])
    delta_world = position_gain * (motion["body_pos_w"][frame, 0] - qpos[:3])
    delta_world[2] = 0.  # Height remains the unmodified explicit pose goal.
    magnitude = np.linalg.norm(delta_world)
    delta_world *= min(1., .6 / max(magnitude, 1e-8))
    actual_heading = np.array([[np.cos(act_yaw), -np.sin(act_yaw), 0.],
                               [np.sin(act_yaw), np.cos(act_yaw), 0.], [0., 0., 1.]])
    for offset, idx in enumerate(range(frame, stop)):
        rot = _quaternion_matrix(motion["body_quat_w"][idx, 0])
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        heading = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                            [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
        local_positions = np.vstack((np.zeros(3), priv[offset, 1:73].reshape(24, 3)))
        # Goal articulation stays in its canonical local frame. Translate the
        # desired world root velocity into the ACTUAL heading so an entry yaw
        # error cannot rotate an XY correction away from its requested path.
        original_root_velocity = motion["body_lin_vel_w"][idx, 0]
        local_delta = (actual_heading.T @ (original_root_velocity + delta_world)
                       - heading.T @ original_root_velocity)
        priv[offset, 223:298] += (local_delta + np.cross(omega, local_positions)).reshape(-1)
        priv[offset, 298:373] += np.tile(omega, 25)
        states[offset, -3:] += .25 * (rot.T @ omega) if goal_gyro_convention == "actor-body-scaled" else omega
    with torch.inference_mode():
        z = policy.backward(torch.from_numpy(states), torch.from_numpy(priv)).mean(0, keepdim=True)
        return 16 * F.normalize(z, dim=-1)


def run(args):
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    contract = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    policy = BFMZeroInference(PACKAGE / "bfmzero_inference_v1/inference.safetensors")
    _, model, physics = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    assert (model.nq, model.nv, model.nu) == (30, 29, 23)
    body_ids = [model.body(n).id for n in contract["body_names"]]
    limits = model.jnt_range[1:].copy()
    ankle_barrier = None
    if getattr(args, "ankle_barrier", False):
        from gear_sonic.utils.g1_true23_bfmzero_ankle_barrier import AnkleRollRepulsion
        ankle_barrier = AnkleRollRepulsion([model.joint(i).name for i in range(1, 24)], limits)
    velocity_limits = np.asarray(json.loads((ROOT / PHYSICS).read_text())["physics"]["velocity_limit_hardware_radps"])
    motion_override=getattr(args,"motion_override",None)
    motion, timeline, path = load_case_motion(args.clip,motion_override)
    goal_gyro_convention = getattr(args, "goal_gyro_convention", "published-world-unscaled")
    arm_ik_enabled = getattr(args, "arm_intent_ik", False)
    arm_ik_velocity_feedforward = getattr(args, "arm_ik_velocity_feedforward", 0.)
    if not np.isfinite(arm_ik_velocity_feedforward) or not 0 <= arm_ik_velocity_feedforward <= 1:
        raise ValueError("arm IK velocity feedforward must be within [0,1]")
    waist_intent_yaw = getattr(args, "waist_intent_yaw", False)
    arm_ik = None
    if arm_ik_enabled:
        if args.arm_reference or motion_override or getattr(args, "residual_checkpoint", None):
            raise ValueError("measured-body arm IK is a separate native-original baseline ablation")
        from gear_sonic.utils.g1_true23_intent_arm_ik import Native23ArmIK
        arm_ik = Native23ArmIK(model, orientation_weight=.03, posture_weight=.02,
                              max_nfev=getattr(args, "arm_ik_evaluations", 6),
                              joint_margin=getattr(args, "arm_ik_joint_margin", 0.))
        original_ik_path = DATA / ("pico_freedancing_v1/optical_reference_v2/original29.npz" if args.clip == "pico" else
                                   f"{args.clip}/original_source_bundle_v1/original_reference.npz")
        with np.load(original_ik_path, allow_pickle=False) as archive:
            ik_intent = archive["source_task_position_w"].copy()
            ik_rotations = archive["source_task_quaternion_wxyz"].copy()
            ik_original_root = archive["source_qpos29"][:, :3].copy()
        if ik_intent.shape != (len(motion["joint_pos"]), 3, 3):
            raise ValueError("original IK intent timeline differs")
    elif waist_intent_yaw:
        raise ValueError("waist yaw ablation requires measured-body arm IK")
    residual = None
    residual_path = getattr(args, "residual_checkpoint", None)
    if residual_path:
        from gear_sonic.utils.g1_true23_bfm_residual_replay import BFMResidualCPU, residual_features_numpy
        if (args.position_gain, args.yaw_gain, args.goal_horizon) != (1., 2., 8) or args.arm_reference or getattr(args, "leg_error_gain", 0.) or goal_gyro_convention != "published-world-unscaled":
            raise ValueError("trained residual requires exact feedback v2 base without additional controllers")
        residual = BFMResidualCPU(residual_path)
        original_path = DATA / ("pico_freedancing_v1/optical_reference_v2/original29.npz" if args.clip == "pico" else
                                f"{args.clip}/original_source_bundle_v1/original_reference.npz")
        with np.load(original_path, allow_pickle=False) as z:
            original_vr21 = z["virtual_vr21"].copy()
        if original_vr21.shape != (len(motion["joint_pos"]), 21):
            raise ValueError("residual original29 intent and motion timeline differ")
    source_root = Path(__file__).resolve().parents[2]
    source_files = [Path(__file__), source_root / "gear_sonic/utils/g1_true23_bfmzero_inference.py", path]
    if ankle_barrier:
        source_files.append(source_root / "gear_sonic/utils/g1_true23_bfmzero_ankle_barrier.py")
    if arm_ik:
        source_files.extend((original_ik_path, source_root / "gear_sonic/utils/g1_true23_intent_arm_ik.py"))
    if residual_path:
        source_files.extend((Path(residual_path), source_root / "gear_sonic/utils/g1_true23_bfm_residual_replay.py"))
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    state, privileged = reference_features(motion, contract)
    if goal_gyro_convention == "actor-body-scaled":
        # Explicit ablation: target proprioception matches the actor sensor
        # convention instead of the published target converter's world gyro.
        for frame in range(len(state)):
            state[frame, -3:] = .25 * _quaternion_matrix(motion["body_quat_w"][frame, 0]).T @ motion["body_ang_vel_w"][frame, 0]
    latent = policy.backward(torch.from_numpy(state), torch.from_numpy(privileged))
    # Published current+7 future goal mean, or explicit current-only ablation.
    # Last standing samples are terminal-hold, not a cropped source recording.
    count = len(state)
    latent = torch.cat((latent, latent[-1:].repeat(args.goal_horizon - 1, 1)))
    goals = torch.stack([latent[i:i + count] for i in range(args.goal_horizon)]).mean(0)
    goals = 16 * F.normalize(goals, dim=-1)
    data = mujoco.MjData(model)
    data.qpos[:] = np.r_[motion["body_pos_w"][10, 0], motion["body_quat_w"][10, 0], motion["joint_pos"][10]]
    data.qvel[:] = np.r_[motion["body_lin_vel_w"][10, 0],
                         _quaternion_matrix(data.qpos[3:7]).T @ motion["body_ang_vel_w"][10, 0],
                         motion["joint_vel"][10]]
    mujoco.mj_forward(model, data)
    history = BFMHistory()
    action = np.zeros(23, dtype=np.float32)
    available = count - 11
    requested = min(available, args.max_controls) if args.max_controls else available
    traces = {k: [] for k in ("qpos", "qvel", "action", "target", "state", "history",
                             "joint_error", "root_error", "landmark_error", "relative_landmark_error",
                             "range_excess", "velocity_ratio", "effort_ratio", "inference_ms")}
    traces["qpos"].append(data.qpos.copy())
    traces["qvel"].append(data.qvel.copy())
    record_physics = bool(ankle_barrier or getattr(args, "record_physics", False))
    if record_physics:
        traces["physics_qpos"] = [data.qpos.copy()]
        traces["physics_qvel"] = [data.qvel.copy()]
        traces["physics_torque"] = []
        traces["physics_barrier_torque"] = []
    previous_ik_pose = data.qpos.copy()
    if arm_ik:
        for key in ("ik_ms", "ik_position_error", "ik_nfev", "ik_success", "ik_target", "ik_hand_goal"):
            traces[key] = []
    failure = None
    started = time.perf_counter()
    for control in range(requested):
        frame = control + 11
        sensed, terms = state_and_terms(data.qpos[7:], data.qvel[6:], data.qpos[3:7],
                                        data.qvel[3:6], action, contract["default_q"])
        previous_combined_action = action.copy()
        hist = history.before_update(terms)
        tick = time.perf_counter()
        goal = goals[frame:frame + 1]
        if args.position_gain or args.yaw_gain:
            goal = corrected_goal(policy, state, privileged, motion, frame, data.qpos,
                                  args.goal_horizon, args.position_gain, args.yaw_gain, goal_gyro_convention)
        raw = policy.actor(torch.from_numpy(sensed[None]), torch.from_numpy(action[None]),
                           torch.from_numpy(hist[None]), goal)[0].numpy()
        action = raw * 5.0
        target = contract["default_q"] + action * .25 * contract["training_effort"] / contract["kp"]
        if residual:
            features = residual_features_numpy(data.qpos, data.qvel, previous_combined_action, target,
                                               motion, original_vr21, frame, data.xpos[np.asarray(body_ids)[[6, 12]]], (6, 12))
            delta = residual.predict(features)
            target += delta
            action = (action + delta / (.25 * contract["training_effort"] / contract["kp"])).astype(np.float32)
        leg_error_gain = getattr(args, "leg_error_gain", 0.)
        if leg_error_gain:
            # Bounded diagnostic residual in the same .15rad envelope as the
            # new learned head. This is feedback, not direct pose playback.
            error = motion["joint_pos"][frame, :12] - data.qpos[7:19]
            target[:12] += .15 * np.tanh(leg_error_gain * error / .15)
        if args.arm_reference:
            target[13:] = (motion["joint_pos"][frame, 13:] +
                           contract["kd"][13:] / contract["kp"][13:] * motion["joint_vel"][frame, 13:])
        if arm_ik:
            ik_tick = time.perf_counter()
            ik_seed = data.qpos.copy()
            ik_seed[20:] = previous_ik_pose[20:]
            if waist_intent_yaw:
                reference_rotation = _quaternion_matrix(motion["body_quat_w"][frame, 0])
                actual_rotation = _quaternion_matrix(data.qpos[3:7])
                yaw_error = np.arctan2(reference_rotation[1, 0], reference_rotation[0, 0]) - np.arctan2(actual_rotation[1, 0], actual_rotation[0, 0])
                yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
                waist_goal = np.clip(motion["joint_pos"][frame, 12] + yaw_error, limits[12, 0], limits[12, 1])
                target[12] = np.clip(waist_goal, previous_ik_pose[19] - .12, previous_ik_pose[19] + .12)
                # Arm geometry uses measured waist; it does not assume that the
                # new waist target has already been reached by the physical joint.
            posture = ik_seed.copy()
            posture[20:] = motion["joint_pos"][frame, 13:]
            hand_goal = data.qpos[:3] + ik_intent[frame, :2] - ik_original_root[frame]
            ik_result = arm_ik.solve(ik_seed, hand_goal, ik_rotations[frame, :2], posture_qpos=posture,
                                     previous_qpos=previous_ik_pose, max_step_rad=.12)
            ik_velocity = (ik_result.qpos[20:] - previous_ik_pose[20:]) / .02
            target[13:] = ik_result.qpos[20:] + arm_ik_velocity_feedforward * contract["kd"][13:] / contract["kp"][13:] * ik_velocity
            target[13:] = np.clip(target[13:], arm_ik.lower.ravel(), arm_ik.upper.ravel())
            previous_ik_pose = ik_result.qpos.copy()
            previous_ik_pose[19] = target[12]
            for key, value in dict(ik_ms=(time.perf_counter() - ik_tick) * 1000,
                    ik_position_error=ik_result.position_errors_m, ik_nfev=ik_result.nfev,
                    ik_success=ik_result.success, ik_target=ik_result.qpos[20:], ik_hand_goal=hand_goal).items():
                traces[key].append(value)
        clipped = np.clip(target, limits[:, 0], limits[:, 1])
        if args.arm_reference or leg_error_gain or arm_ik:
            action = ((target - contract["default_q"]) * contract["kp"] /
                      (.25 * contract["training_effort"])).astype(np.float32)
        inference_ms = (time.perf_counter() - tick) * 1000
        # No gain/effort substitution: use BFM gains with native effort caps.
        peak_effort = 0.0
        peak_range = 0.0
        peak_velocity = 0.0
        for _ in range(physics.decimation):
            torque = contract["kp"] * (clipped - data.qpos[7:]) - contract["kd"] * data.qvel[6:]
            barrier_torque = np.zeros(23) if ankle_barrier is None else ankle_barrier.torque(data.qpos[7:], data.qvel[6:])
            torque += barrier_torque
            data.ctrl[:] = np.clip(torque, -physics.effort, physics.effort)
            mujoco.mj_step(model, data)
            if record_physics:
                traces["physics_qpos"].append(data.qpos.copy())
                traces["physics_qvel"].append(data.qvel.copy())
                traces["physics_torque"].append(data.ctrl.copy())
                traces["physics_barrier_torque"].append(barrier_torque)
            peak_effort = max(peak_effort, float(np.max(np.abs(data.qfrc_actuator[6:]) / physics.effort)))
            peak_range = max(peak_range, float(np.max(np.maximum(limits[:, 0] - data.qpos[7:], data.qpos[7:] - limits[:, 1]))))
            peak_velocity = max(peak_velocity, float(np.max(np.abs(data.qvel[6:]) / velocity_limits)))
        mujoco.mj_kinematics(model, data)
        actual = task_points(data.xpos[body_ids], data.xquat[body_ids])
        desired = task_points(motion["body_pos_w"][frame], motion["body_quat_w"][frame])
        root_error = data.qpos[:3] - motion["body_pos_w"][frame, 0]
        values = dict(qpos=data.qpos.copy(), qvel=data.qvel.copy(), action=action.copy(), target=clipped,
                      state=sensed, history=hist, joint_error=data.qpos[7:] - motion["joint_pos"][frame],
                      root_error=root_error, landmark_error=np.linalg.norm(actual - desired, axis=-1),
                      relative_landmark_error=np.linalg.norm(actual - desired - root_error, axis=-1),
                      range_excess=peak_range, velocity_ratio=peak_velocity, effort_ratio=peak_effort,
                      inference_ms=inference_ms)
        for key, value in values.items():
            traces[key].append(value)
        tilt = float(np.arccos(np.clip(_quaternion_matrix(data.qpos[3:7])[2, 2], -1, 1)))
        if not np.isfinite(data.qpos).all() or data.qpos[2] < .25 or tilt > 1.2:
            failure = dict(kind="fall", control=control, height=float(data.qpos[2]), tilt=tilt)
            break
        if peak_range > .01 or peak_velocity > 1.0:
            failure = dict(kind="physical_limit", control=control, range_excess=peak_range, velocity_ratio=peak_velocity)
            break
        if (control + 1) % 500 == 0:
            print(json.dumps(dict(clip=args.clip, controls=control + 1, height=float(data.qpos[2]),
                                  root_error=float(np.linalg.norm(root_error)))), flush=True)
    arrays = {k: np.asarray(v) for k, v in traces.items()}
    np.savez_compressed(args.output / "trace.npz", **arrays)
    completed = len(arrays["action"])
    phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    start, stop = phase["control_start"], min(completed, phase["control_stop"])
    metrics = None
    if stop > start:
        error = arrays["joint_error"][start:stop]
        metrics = dict(source_controls=stop - start, source_requested=phase["requested_controls"],
                       leg_rmse=float(np.sqrt(np.mean(error[:, :12] ** 2))),
                       arm_rmse=float(np.sqrt(np.mean(error[:, 13:] ** 2))),
                       root_p95=float(np.percentile(np.linalg.norm(arrays["root_error"][start:stop], axis=-1), 95)),
                       landmark_p95=np.percentile(arrays["landmark_error"][start:stop], 95, axis=0).tolist(),
                       relative_landmark_p95=np.percentile(arrays["relative_landmark_error"][start:stop], 95, axis=0).tolist())
    result = dict(clip=args.clip, completed=completed, requested=requested, available=available, failure=failure,
                  full_lifecycle_completed=completed == available and failure is None, source_metrics=metrics,
                  range_excess_max=float(arrays["range_excess"].max()), effort_ratio_max=float(arrays["effort_ratio"].max()),
                  velocity_ratio_max=float(arrays["velocity_ratio"].max()),
                  inference_ms_p95=float(np.percentile(arrays["inference_ms"], 95)),
                  elapsed_seconds=time.perf_counter() - started, goal_horizon=args.goal_horizon,
                  goal_gyro_convention=goal_gyro_convention,
                  goal_buffer_ms=(args.goal_horizon - 1) * 20, reference_path=str(path),
                  motion_override=None if motion_override is None else str(path),
                  reference_geometry_retargeted=motion_override is not None,
                  ground_truth_pose_feedback=bool(args.position_gain or args.yaw_gain),
                  root_xy_feedback_used=bool(args.position_gain or residual),
                  heading_feedback_sensor_equivalent="pelvis_IMU_orientation_with_initial_calibration",
                  position_gain=args.position_gain, yaw_gain=args.yaw_gain,
                  arm_reference=args.arm_reference,
                  ankle_barrier=None if ankle_barrier is None else ankle_barrier.report(),
                  full_physics_trace_recorded=record_physics,
                  arm_intent_ik=arm_ik_enabled, waist_intent_yaw=waist_intent_yaw,
                  arm_ik_goal=None if not arm_ik else "original29 hand positions relative to original root, world axes, translated to measured current root; original hand rotations",
                  arm_ik_evaluation_budget=None if not arm_ik else arm_ik.max_nfev,
                  arm_ik_target_step_rad=None if not arm_ik else .12,
                  arm_ik_step_bound_applies_to=None if not arm_ik else "IK pose; velocity feedforward may increase final PD target step",
                  arm_ik_joint_margin=None if not arm_ik else arm_ik.joint_margin,
                  arm_ik_final_pd_target_inside_joint_margin=bool(arm_ik),
                  arm_ik_velocity_feedforward=None if not arm_ik else arm_ik_velocity_feedforward,
                  arm_ik_ms_p95=None if not arm_ik else float(np.percentile(arrays["ik_ms"], 95)),
                  arm_ik_position_residual_p95=None if not arm_ik else np.percentile(arrays["ik_position_error"], 95, axis=0).tolist(),
                  arm_ik_budget_exhausted_count=None if not arm_ik else np.sum(~arrays["ik_success"], axis=0).tolist(),
                  leg_error_gain=getattr(args, "leg_error_gain", 0.),
                  residual_checkpoint=None if residual_path is None else str(residual_path),
                  residual_updates=None if residual is None else residual.completed_updates,
                  model_path=str(ROOT.parent / "GR00T-WholeBodyControl" / MODEL),
                  action_contract="target=default_q+5*tanh_actor*.25*training_effort/training_kp; native effort clipped",
                  hardware_authorized=False, deployment_ready=False)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", choices=("pico", "walk002", "walk003", "walk008"), default="walk002")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal-horizon", type=int, choices=(1, 8), default=8)
    parser.add_argument("--max-controls", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--position-gain", type=float, default=0.)
    parser.add_argument("--yaw-gain", type=float, default=0.)
    parser.add_argument("--arm-reference", action="store_true")
    parser.add_argument("--leg-error-gain", type=float, default=0.)
    parser.add_argument("--residual-checkpoint", type=Path)
    parser.add_argument("--motion-override",type=Path)
    parser.add_argument("--arm-intent-ik", action="store_true")
    parser.add_argument("--waist-intent-yaw", action="store_true")
    parser.add_argument("--arm-ik-evaluations", type=int, default=6)
    parser.add_argument("--arm-ik-velocity-feedforward", type=float, default=0.)
    parser.add_argument("--arm-ik-joint-margin", type=float, default=0.)
    parser.add_argument("--goal-gyro-convention", choices=("published-world-unscaled", "actor-body-scaled"), default="published-world-unscaled")
    parser.add_argument("--ankle-barrier", action="store_true")
    parser.add_argument("--record-physics", action="store_true")
    run(parser.parse_args())
