"""Offline source-policy control: all 29 joints, saved original dance and physics.

The earlier original29 baseline used the normal release's G1 encoder, whereas
native23 training starts from the low-latency teleop encoder. This diagnostic
tests the latter without deleting joints. It is neither native23 qualification
nor full C++/firmware equivalence. Saved future frames are explicitly consumed.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import platform
import sys

import mujoco
import numpy as np
import torch

from gear_sonic.scripts import simulate_g1_sonic_library_motions as stock
from gear_sonic.utils.g1_29dof_low_latency_teacher import ExactLowLatencyTeacher
from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters, file_sha256
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


def reference_features(model, poses, velocity_hardware):
    """Source29 FK and step-1 future lower-body values, left six then right six.

    Original TrackingCommand selects hardware indices range(12), mapping those
    into IL29 before indexing the motion library. Sorting IL indices instead
    silently interleaves legs and does not reproduce the released encoder.
    """
    if (
        poses.ndim != 2
        or poses.shape[1] != 36
        or len(poses) < 10
        or velocity_hardware.shape != (len(poses), 29)
        or not np.isfinite(poses).all()
        or not np.isfinite(velocity_hardware).all()
        or not np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("source reference must retain finite complete hardware29 poses and velocities")
    bodies = (
        ("left_wrist_yaw_link", (0.18, -0.025, 0)),
        ("right_wrist_yaw_link", (0.18, 0.025, 0)),
        ("torso_link", (0, 0, 0.35)),
    )
    ids = [model.body(name).id for name, _ in bodies]
    features = np.zeros((len(poses), 267), dtype=np.float32)
    q = poses[:, 7:19]
    dq = velocity_hardware[:, :12]
    data = mujoco.MjData(model)
    for frame, pose in enumerate(poses):
        indices = np.minimum(frame + np.arange(10), len(poses) - 1)
        features[frame, :240] = np.concatenate((q[indices].reshape(-1), dq[indices].reshape(-1)))
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        inverse = stock._quat_conjugate(pose[3:7])
        positions, orientations = [], []
        for body, (_, offset) in zip(ids, bodies, strict=True):
            point = data.xpos[body] + stock._quat_rotate(data.xquat[body], np.asarray(offset))
            positions.extend(stock._quat_rotate(inverse, point - pose[:3]))
            orientations.extend(stock._quat_multiply(inverse, data.xquat[body]))
        features[frame, 240:261] = [*positions, *orientations]
    return features


def record_case(model, parameters, teacher, poses, features, initial_pose):
    data = mujoco.MjData(model)
    data.qpos[:] = initial_pose
    data.qvel[:] = 0
    history = deque((stock._zero_history_frame() for _ in range(9)), maxlen=10)
    previous = np.zeros(29)
    arrays = {
        name: []
        for name in (
            "pre_qpos",
            "qpos",
            "qvel",
            "encoder267",
            "history930",
            "raw29",
            "target29",
            "physics_torque29",
        )
    }
    failure = None
    for frame in range(len(poses)):
        mujoco.mj_forward(model, data)
        state = stock._current_history_frame(data, previous)
        state["joint_position"] = (data.qpos[7:] - parameters.default_angles)[stock.ISAAC_TO_MUJOCO_INDEX]
        history.append(state)
        proprio = stock._policy_observation(history)
        semantic = features[frame].copy()
        relative = stock._quat_multiply(stock._quat_conjugate(data.qpos[3:7]), poses[frame, 3:7])
        semantic[261:] = stock._rotation_6d(relative)
        action = teacher.infer(semantic, proprio)
        if not np.isfinite(action).all() or np.max(np.abs(action)) >= 10:
            failure = "invalid_or_original_baseline_action_bound"
            break
        target = parameters.target(action)
        arrays["pre_qpos"].append(data.qpos.copy())
        for _ in range(10):
            torque = parameters.kps * (target - data.qpos[7:]) - parameters.kds * data.qvel[6:]
            data.ctrl[:] = np.clip(torque, -parameters.effort, parameters.effort)
            arrays["physics_torque29"].append(data.ctrl.copy())
            mujoco.mj_step(model, data)
        for key, value in (
            ("qpos", data.qpos),
            ("qvel", data.qvel),
            ("encoder267", semantic),
            ("history930", proprio),
            ("raw29", action),
            ("target29", target),
        ):
            arrays[key].append(value.copy())
        previous = action.copy()
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.qpos[2] < 0.08:
            failure = "invalid_or_low_state"
            break
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    count = len(arrays["qpos"])
    if not count:
        raise ValueError(f"source baseline completed zero controls: {failure}")
    return arrays, {
        "requested_controls": len(poses),
        "completed_controls": count,
        "failure": failure,
        "pre_control_root_error_p95_m": float(
            np.percentile(np.linalg.norm(arrays["pre_qpos"][:, :3] - poses[:count, :3], axis=1), 95)
        ),
        "post_control_joint_rmse_rad_with_20ms_offset": float(
            np.sqrt(np.mean((arrays["qpos"][:, 7:] - poses[:count, 7:]) ** 2))
        ),
        "minimum_root_height_m": float(arrays["qpos"][:, 2].min()),
        "horizontal_displacement_m": float(np.linalg.norm(arrays["qpos"][-1, :2] - arrays["pre_qpos"][0, :2])),
        "source_horizontal_displacement_m": float(np.linalg.norm(poses[-1, :2] - poses[0, :2])),
        "maximum_joint_velocity_rad_s": float(np.abs(arrays["qvel"][:, 6:]).max()),
        "state_resets_after_initialization": 0,
        "native23_qualification": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-directory", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    root, output = args.baseline_directory.resolve(strict=True), args.output_directory.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite baseline evidence")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"baseline input hash mismatch: {path}")
        inputs[str(path)] = digest
        return path

    prior = json.loads(bind(root / "report.json").read_text())
    case = json.loads(bind(root / "happy_dance.cpp_parameters_and_float32_targets.report.json").read_text())
    model_path = bind(root / "original29.mjb", prior["compiled_model_sha256"])
    model = mujoco.MjModel.from_binary_path(str(model_path))
    model_hash = compiled_model_sha256(model)
    if model_hash != prior["compiled_model_sha256"] or (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("original29 compiled model identity or ABI mismatch")
    if model.opt.timestep != 0.002:
        raise ValueError("baseline requires unchanged 2-ms physics")
    parameter_path = root / "cpp_capture/parameters.json"
    parameters = CppParameters(json.loads(bind(parameter_path, prior["inputs"][str(parameter_path)]).read_text()))
    with np.load(
        bind(root / "happy_dance.cpp_parameters_and_float32_targets.npz", case["trace_sha256"]), allow_pickle=False
    ) as archive:
        poses = archive["planned_qpos50"].copy()
        velocity = archive["planned_joint_velocity_hardware29"].copy()
        initial = archive["pre_qpos"][0].copy()
    features = reference_features(model, poses, velocity)
    torch.set_num_threads(1)
    teacher = ExactLowLatencyTeacher(args.source_checkpoint, device="cpu")
    bind(args.source_checkpoint, teacher.checkpoint_sha256)
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    bind(__file__)
    bind(Path(__file__).resolve().parents[1] / "envs/manager_env/mdp/commands.py")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    initializations = [("original_baseline_standing_joints", initial)]
    if not np.array_equal(initial, poses[0]):
        initializations.append(("source_reference_joints_zero_velocity", poses[0]))
    for name, start in initializations:
        arrays, metrics = record_case(model, parameters, teacher, poses, features, start)
        trace = output / (name + ".npz")
        with trace.open("xb") as stream:
            np.savez_compressed(stream, **arrays, planned_qpos=poses)
        records.append(dict(case=name, metrics=metrics, trace_path=str(trace), trace_sha256=file_sha256(trace)))
        print(json.dumps(records[-1]), flush=True)
    for path, digest in inputs.items():
        if file_sha256(path) != digest:
            raise ValueError("baseline input changed during execution")
    if compiled_model_sha256(model) != model_hash:
        raise ValueError("diagnostic mutated source physics")
    report = {
        "kind": "sonic_low_latency29_untrimmed_source_control_v2",
        "inputs": inputs,
        "compiled_model_sha256": model_hash,
        "teacher": teacher.descriptor(),
        "records": records,
        "original_normal_release_metrics": case["details"]["stock_metrics"],
        "future_frames_consumed": 9,
        "encoder_lower_body_order": "hardware_left_six_then_right_six_not_sorted_il29",
        "reference_start_identical_to_baseline": bool(np.array_equal(initial, poses[0])),
        "future_tail_handling": "clamp_last_frame_like_original_baseline",
        "history_startup": "nine_zero_frames_then_actual_measured_frames_like_original_baseline",
        "original_baseline_gains_effort_and_physics_preserved": True,
        "runtime": {"python": platform.python_version(), "mujoco": mujoco.__version__, "torch": torch.__version__},
        "native23_motion_qualification": False,
        "full_cpp_deployment_equivalence": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
