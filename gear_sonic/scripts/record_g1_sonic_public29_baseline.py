"""SIM-only all29 low-latency SONIC control on pinned public TWIST2 motions.

Uses the saved original deployment scene and captured C++ command parameters,
not native23 physics. This is a source-side reference baseline, not a controlled
morphology-only ablation or complete C++/hardware reproduction. No DDS is opened.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

from gear_sonic.scripts import simulate_g1_sonic_library_motions as stock
from gear_sonic.scripts.audit_g1_sonic_low_latency29_baseline import reference_features
from gear_sonic.scripts.prepare_g1_true23_twist2_replay import load_pinned_recording, native23_poses
from gear_sonic.teleop.buffered_source_horizon import ReceivedSourceHorizon
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES, SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_29dof_low_latency_teacher import ExactLowLatencyTeacher
from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters, file_sha256
from gear_sonic.utils.g1_true23_generalist_lifecycle import (
    PHASE_CONTROLS,
    _blend_poses,
    _planned_endpoint_standing,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

POINTS = (
    ("left_ankle_roll_link", (0, 0, 0)),
    ("right_ankle_roll_link", (0, 0, 0)),
    ("left_wrist_yaw_link", (0.18, -0.025, 0)),
    ("right_wrist_yaw_link", (0.18, 0.025, 0)),
    ("torso_link", (0, 0, 0.35)),
)
FLAGS = dict(hardware_authorized=False, deployment_ready=False, native23_qualified=False)


def source29_poses(recording):
    selected, times, queries = native23_poses(recording)
    q = np.column_stack([np.interp(queries, times, recording["dof_pos"][:, j]) for j in range(29)])
    result = np.column_stack((selected[:, :7], q))
    np.testing.assert_array_equal(result[:, 7:][:, SOURCE_MJ29_KEEP_INDICES], selected[:, 7:])
    return result


def lifecycle29(source, standing):
    source, standing = np.asarray(source), np.asarray(standing)
    if source.ndim != 2 or source.shape[1] != 36 or len(source) < 12 or standing.shape != (36,):
        raise ValueError("requires complete source29 poses and one standing pose")
    if not np.isfinite(source).all() or not np.isfinite(standing).all():
        raise ValueError("nonfinite lifecycle poses")
    if not np.allclose(np.linalg.norm(source[:, 3:7], axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("source quaternion is not normalized")
    if abs(np.linalg.norm(standing[3:7]) - 1) > 1e-6:
        raise ValueError("standing quaternion is not normalized")
    returned = _planned_endpoint_standing(standing, source[-1])
    pieces, phases, cursor = [np.tile(standing, (11, 1))], [], 0
    for name, count in PHASE_CONTROLS:
        count = len(source) if count is None else count
        if name == "source_motion":
            poses = source.copy()
        elif name == "acquisition_ramp":
            poses = _blend_poses(standing, source[0], count)
        elif name == "return_ramp":
            poses = _blend_poses(source[-1], returned, count)
        else:
            poses = np.tile(returned if name != "initial_standing" else standing, (count, 1))
        pieces.append(poses)
        phases.append(dict(name=name, control_start=cursor, control_stop=cursor + count))
        cursor += count
    return np.concatenate(pieces), phases


def points_at(model, data):
    return np.asarray(
        [
            data.xpos[model.body(name).id]
            + stock._quat_rotate(data.xquat[model.body(name).id], np.asarray(offset))
            for name, offset in POINTS
        ]
    )


def run_case(model, source_model, parameters, teacher, poses, phases):
    if (model.nq, model.nv, model.nu) != (36, 35, 29) or model.opt.timestep != 0.002:
        raise ValueError("requires original29 model at 500 Hz")
    if (source_model.nq, source_model.nv, source_model.nu) != (36, 35, 29):
        raise ValueError("reference geometry must contain all29 joints")
    if poses.ndim != 2 or poses.shape[1] != 36 or len(poses) < 22 or not np.isfinite(poses).all():
        raise ValueError("requires a finite original29 lifecycle")
    if not np.array_equal(poses[-10:], np.repeat(poses[-1:], 10, axis=0)):
        raise ValueError("only an explicit constant standing tail may extend")
    before = compiled_model_sha256(model)
    requested = len(poses) - 11
    extended = np.concatenate((poses, np.repeat(poses[-1:], 10, axis=0)))
    # Only VR terms are consumed from this helper. Lower-body differences below
    # are formed by the shared buffer AFTER float32 serialization, as in native23.
    vr = reference_features(source_model, extended, np.zeros((len(extended), 29)))[:, 240:261]
    source = dict(
        joint_pos=extended[:, 7:][:, SOURCE_MJ29_KEEP_INDICES].astype(np.float32),
        root_position_w=extended[:, :3].astype(np.float32),
        root_quaternion_wxyz=extended[:, 3:7].astype(np.float32),
        virtual_vr21=vr,
    )
    buffer = ReceivedSourceHorizon()

    def emit(index):
        return buffer.push(
            source_timestamp_s=index * 0.02,
            arrival_timestamp_s=index * 0.02,
            joint_names=HARDWARE_23_JOINT_NAMES,
            joint_position23=source["joint_pos"][index],
            root_position_w=source["root_position_w"][index],
            root_quaternion_wxyz=source["root_quaternion_wxyz"][index],
            virtual_source_vr21=vr[index],
        )

    for index in range(19):
        emit(index)
    data, probe, reference = mujoco.MjData(model), mujoco.MjData(model), mujoco.MjData(source_model)
    data.qpos[:] = poses[10]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    previous = np.zeros(29, dtype=np.float32)

    def measured():
        value = stock._current_history_frame(data, previous)
        value["joint_position"] = (data.qpos[7:] - parameters.default_angles)[stock.ISAAC_TO_MUJOCO_INDEX]
        return value

    history = deque((measured() for _ in range(10)), maxlen=10)
    arrays = {
        name: []
        for name in (
            "encoder267",
            "history930",
            "raw29",
            "target29",
            "physics_pre_qpos",
            "physics_post_qpos",
            "physics_pre_qvel",
            "physics_post_qvel",
            "requested_torque29",
            "applied_torque29",
            "engine_force29",
            "contact_count",
            "landmark_error_m",
            "relative_landmark_error_m",
            "root_error_m",
            "joint_error29",
            "source_timestamps_s",
        )
    }
    arrays.update(qpos=[data.qpos.copy()], qvel=[data.qvel.copy()])
    failure = None
    try:
        for control in range(requested):
            sample = emit(19 + control)
            semantic = sample.encoder267(data.qpos[3:7].astype(np.float32))
            history.append(measured())
            proprio = stock._policy_observation(history)
            action = teacher.infer(semantic, proprio)
            if action.shape != (29,) or not np.isfinite(action).all() or np.max(np.abs(action)) >= 10:
                raise RuntimeError("invalid or raw-bound original29 action")
            target = parameters.target(action)
            for key, value in (
                ("encoder267", semantic),
                ("history930", proprio),
                ("raw29", action),
                ("target29", target),
            ):
                arrays[key].append(value.copy())
            arrays["source_timestamps_s"].append(
                [
                    sample.emission_source_timestamp_s,
                    sample.encoder_anchor_timestamp_s,
                    sample.root_setpoint_timestamp_s,
                ]
            )
            for _ in range(10):
                arrays["physics_pre_qpos"].append(data.qpos.copy())
                arrays["physics_pre_qvel"].append(data.qvel.copy())
                torque = parameters.kps * (target - data.qpos[7:]) - parameters.kds * data.qvel[6:]
                applied = np.clip(torque, -parameters.effort, parameters.effort)
                data.ctrl[:] = applied
                start_time = float(data.time)
                mujoco.mj_step(model, data)
                if abs(float(data.time) - start_time - 0.002) > 1e-10:
                    raise RuntimeError("original29 integration time discontinuity")
                for key, value in (
                    ("physics_post_qpos", data.qpos),
                    ("physics_post_qvel", data.qvel),
                    ("requested_torque29", torque),
                    ("applied_torque29", applied),
                    ("engine_force29", data.qfrc_actuator[6:]),
                ):
                    arrays[key].append(value.copy())
                arrays["contact_count"].append(int(data.ncon))
            arrays["qpos"].append(data.qpos.copy())
            arrays["qvel"].append(data.qvel.copy())
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("nonfinite original29 integrated state")
            probe.qpos[:], reference.qpos[:] = data.qpos, poses[11 + control]
            mujoco.mj_fwdPosition(model, probe)
            mujoco.mj_fwdPosition(source_model, reference)
            actual, desired = points_at(model, probe), points_at(source_model, reference)
            arrays["landmark_error_m"].append(np.linalg.norm(actual - desired, axis=1))
            arrays["relative_landmark_error_m"].append(
                np.linalg.norm((actual - data.qpos[:3]) - (desired - poses[11 + control, :3]), axis=1)
            )
            arrays["root_error_m"].append(float(np.linalg.norm(data.qpos[:3] - poses[11 + control, :3])))
            arrays["joint_error29"].append(data.qpos[7:] - poses[11 + control, 7:])
            previous = action.copy()
            gravity = stock._quat_rotate(stock._quat_conjugate(data.qpos[3:7]), np.array([0, 0, -1]))
            if data.qpos[2] < 0.12 or np.arccos(np.clip(-gravity[2], -1, 1)) > 2.2:
                raise RuntimeError("absolute height/tilt diagnostic stop")
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    completed = len(arrays["qpos"]) - 1
    phase = next(row for row in phases if row["name"] == "source_motion")
    stop = min(completed, phase["control_stop"], len(arrays["landmark_error_m"]))
    count = max(0, stop - phase["control_start"])
    section = slice(phase["control_start"], stop)
    errors = arrays["landmark_error_m"][section]
    metrics = dict(
        requested_controls=requested,
        completed_controls=completed,
        failure=failure,
        requested_source_controls=phase["control_stop"] - phase["control_start"],
        completed_source_controls=count,
        source_landmark_p95_m=np.percentile(errors, 95, axis=0).tolist() if count else None,
        source_root_position_p95_m=float(np.percentile(arrays["root_error_m"][section], 95)) if count else None,
        source_leg_joint_rmse_rad=float(np.sqrt(np.mean(arrays["joint_error29"][section, :12] ** 2)))
        if count
        else None,
        source_landmark_screen_passed=bool(
            count == phase["control_stop"] - phase["control_start"]
            and count
            and np.all(np.percentile(errors, 95, axis=0) <= [0.05, 0.05, 0.1, 0.1, 0.1])
        ),
        no_postinitial_state_rewrites=True,
        no_fallback_controller=True,
        **FLAGS,
    )
    if compiled_model_sha256(model) != before:
        raise ValueError("original29 simulation mutated its physical model")
    return arrays, source, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline-directory",
        "source-checkpoint",
        "source-model",
        "native-sim-config",
        "output-directory",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    if len({path.name for path in args.sources}) != len(args.sources):
        raise ValueError("duplicate public source names")
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError("baseline refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("baseline input hash mismatch: " + str(path))
        inputs[str(path)] = digest
        return path

    baseline = args.baseline_directory.resolve(strict=True)
    prior = json.loads(bind(baseline / "report.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(bind(baseline / "original29.mjb", prior["compiled_model_sha256"])))
    parameter_path = baseline / "cpp_capture/parameters.json"
    parameters = CppParameters(json.loads(bind(parameter_path, prior["inputs"][str(parameter_path)]).read_text()))
    source_model = mujoco.MjModel.from_xml_path(str(bind(args.source_model)))
    initial = json.loads(bind(args.native_sim_config).read_text())["initial_state"]
    np.testing.assert_array_equal(
        parameters.default_angles[list(SOURCE_MJ29_KEEP_INDICES)], initial["joint_position_hardware_rad"]
    )
    standing = np.r_[initial["base_position_m"], initial["base_quaternion_wxyz"], parameters.default_angles]
    torch.set_num_threads(1)
    teacher = ExactLowLatencyTeacher(args.source_checkpoint, device="cpu")
    bind(args.source_checkpoint, teacher.checkpoint_sha256)
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    bind(__file__)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for path in args.sources:
        recording = load_pinned_recording(bind(path))
        poses, phases = lifecycle29(source29_poses(recording), standing)
        arrays, received, metrics = run_case(model, source_model, parameters, teacher, poses, phases)
        target = output / (path.stem + ".npz")
        with target.open("xb") as stream:
            np.savez_compressed(
                stream, **arrays, reference_qpos29=poses, **{"received_" + k: v for k, v in received.items()}
            )
        record = dict(
            source=str(path.resolve()),
            recorded_source_frames=len(recording["root_pos"]),
            recorded_source_fps=float(recording["fps"]),
            resampled_source_frames=len(poses) - 761,
            source_speed_factor=1.0,
            unsampled_original_final_fraction_s=(
                (len(recording["root_pos"]) - 1) / float(recording["fps"]) - (len(poses) - 762) * 0.02
            ),
            trace=str(target),
            trace_sha256=file_sha256(target),
            phases=phases,
            metrics=metrics,
        )
        records.append(record)
        with (output / (path.stem + ".json")).open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
        print(json.dumps(dict(clip=path.name, **metrics)), flush=True)
    for path, digest in inputs.items():
        if file_sha256(path) != digest:
            raise ValueError("baseline input changed during execution")
    report = dict(
        kind="sonic_public_untrimmed29_buffered_baseline_v1",
        inputs=inputs,
        records=records,
        physical_model_sha256=compiled_model_sha256(model),
        source_geometry_sha256=compiled_model_sha256(source_model),
        teacher=teacher.descriptor(),
        all29_joints_actuated=True,
        reference_all29_joints_retained=True,
        history_start="ten_repeated_measured_standing_frames_zero_previous_action",
        source_phase="encoder_anchor_9_plus_t_emission_19_plus_t_scoring_11_plus_t",
        added_terminal_standing_source_samples=10,
        root_feedback9_controller_added=False,
        source_only_baseline_not_controlled_morphology_ablation=True,
        cpp_command_parameters_used=True,
        complete_cpp_firmware_equivalence=False,
        physical_model_and_gain_differences_from_native23_are_confounders=True,
        no_hardware_connection=True,
        **FLAGS,
    )
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
