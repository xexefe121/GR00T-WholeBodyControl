"""Independent source29 replay/input/network audit; never connects to a robot."""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.nn import functional as F

from gear_sonic.scripts.record_g1_sonic_public29_full_pose import BASELINE_REPORT_SHA256, CLIPS, teacher_digest
from gear_sonic.utils.g1_29dof_low_latency_g1_teacher import LowLatencyG1Teacher
from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters, file_sha256
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

POINTS = (
    ("left_ankle_roll_link", (0, 0, 0)),
    ("right_ankle_roll_link", (0, 0, 0)),
    ("left_wrist_yaw_link", (0.18, -0.025, 0)),
    ("right_wrist_yaw_link", (0.18, 0.025, 0)),
    ("torso_link", (0, 0, 0.35)),
)


def rotation(q):
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def expected_history(trace, parameter_payload, count):
    order = np.asarray(parameter_payload["mujoco_to_isaaclab"])
    default = np.asarray(parameter_payload["default_angles"])
    result = []
    for control in range(count):
        indices = np.maximum(0, np.arange(control - 9, control + 1))
        q, v = trace["qpos"][indices], trace["qvel"][indices]
        previous = np.stack([trace["raw29"][i - 1] if i else np.zeros(29) for i in indices])
        result.append(
            np.concatenate(
                (
                    v[:, 3:6].ravel(),
                    (q[:, 7:] - default)[:, order].ravel(),
                    v[:, 6:][:, order].ravel(),
                    previous.ravel(),
                    np.stack([-rotation(quat)[2] for quat in q[:, 3:7]]).ravel(),
                )
            ).astype(np.float32)
        )
    return np.asarray(result).reshape(-1, 930)


def input_audit(trace, payload):
    count = len(trace["attempt_encoder640"])
    n = len(trace["qpos"]) - 1
    assert n <= count <= n + 1
    ref = trace["reference_qpos29"]
    extended = np.concatenate((ref, np.repeat(ref[-1:], 10, axis=0))).astype(np.float32)
    order = np.asarray(payload["mujoco_to_isaaclab"])
    np.testing.assert_array_equal(trace["attempt_anchor_sample_index"], np.arange(count) + 9)
    np.testing.assert_array_equal(trace["attempt_measured_wxyz"], trace["qpos"][:count, 3:7].astype(np.float32))
    np.testing.assert_array_equal(trace["source_timestamps_s"], (np.arange(n)[:, None] + [19, 9, 10]) * 0.02)
    np.testing.assert_allclose(
        expected_history(trace, payload, count), trace["attempt_history930"], atol=1e-7, rtol=0
    )
    orientation_max = 0.0
    for control in range(count):
        window = extended[9 + control : 20 + control]
        q = window[:, 7:][:, order]
        dq = (q[1:] - q[:-1]) / np.float32(0.02)
        joint_input = np.concatenate((q[:-1].ravel(), dq.ravel())).reshape(10, 58)
        actual = trace["attempt_encoder640"][control].reshape(10, 64)
        np.testing.assert_array_equal(actual[:, :58], joint_input)
        measured = Rotation.from_quat(trace["attempt_measured_wxyz"][control, [1, 2, 3, 0]])
        desired = Rotation.from_quat(window[:-1, [4, 5, 6, 3]])
        orientations = (measured.inv() * desired).as_matrix()[:, :, :2].reshape(10, 6).astype(np.float32)
        np.testing.assert_allclose(actual[:, 58:], orientations, atol=2e-7, rtol=0)
        orientation_max = max(orientation_max, float(np.max(np.abs(actual[:, 58:] - orientations))))
    for key in ("encoder640", "token64", "raw29", "history930"):
        np.testing.assert_array_equal(trace[key][:n], trace["attempt_" + key][:n])
    assert trace["attempt_output_present"][:n].all()
    return dict(
        attempts=count,
        executed=n,
        joint_velocity_layout_bit_exact=True,
        maximum_orientation_input_error=orientation_max,
        history_verified=True,
        received_window_verified=True,
        unreceived_future_consumed=False,
    )


def network_audit(teacher, trace):
    def mlp(layers, value):
        for i, (weight, bias) in enumerate(layers):
            value = F.linear(value, weight, bias)
            if i != len(layers) - 1:
                value = F.silu(value)
        return value

    checked = 0
    with torch.inference_mode():
        for i, present in enumerate(trace["attempt_output_present"]):
            if not present:
                continue
            latent = mlp(teacher.encoder, torch.from_numpy(trace["attempt_encoder640"][i : i + 1]))
            # Independent expression of source FSQ forward, not runtime.infer/exact_fsq32.
            half = 31 * 1.001 / 2
            token = torch.round(torch.tanh(latent + math.atanh(0.5 / half)) * half - 0.5) / 16
            raw = mlp(
                teacher.decoder, torch.cat((token, torch.from_numpy(trace["attempt_history930"][i : i + 1])), -1)
            )
            np.testing.assert_array_equal(token.numpy()[0], trace["attempt_token64"][i])
            np.testing.assert_array_equal(raw.numpy()[0], trace["attempt_raw29"][i])
            checked += 1
    return dict(singleton_outputs_checked=checked, token_and_action_bit_exact=True)


def physics_audit(model, parameters, trace):
    n = len(trace["qpos"]) - 1
    steps = n * 10
    assert (model.nq, model.nv, model.nu) == (36, 35, 29)
    assert model.opt.timestep == 0.002
    for kind in ("qpos", "qvel"):
        assert len(trace["physics_pre_" + kind]) == len(trace["physics_post_" + kind]) == steps
        np.testing.assert_array_equal(trace["physics_pre_" + kind][1:], trace["physics_post_" + kind][:-1])
        np.testing.assert_array_equal(trace["physics_pre_" + kind][::10], trace[kind][:-1])
        np.testing.assert_array_equal(trace["physics_post_" + kind][9::10], trace[kind][1:])
    # Independently redo target arithmetic, preserving captured C++ float32 stores.
    from gear_sonic.scripts.simulate_g1_sonic_library_motions import MUJOCO_TO_ISAAC_INDEX

    targets = (
        (
            parameters.default_angles
            + trace["raw29"].astype(float)[:, MUJOCO_TO_ISAAC_INDEX] * parameters.action_scale
        )
        .astype(np.float32)
        .astype(float)
    )
    np.testing.assert_array_equal(targets, trace["target29"])
    torque = (
        parameters.kps * (np.repeat(targets, 10, axis=0) - trace["physics_pre_qpos"][:, 7:])
        - parameters.kds * trace["physics_pre_qvel"][:, 6:]
    )
    np.testing.assert_array_equal(torque, trace["requested_torque29"])
    np.testing.assert_array_equal(
        np.clip(torque, -parameters.effort, parameters.effort), trace["applied_torque29"]
    )
    data = mujoco.MjData(model)
    data.qpos[:] = trace["qpos"][0]
    data.qvel[:] = trace["qvel"][0]
    mujoco.mj_forward(model, data)
    post_q, post_v, forces = (
        np.empty_like(trace[key]) for key in ("physics_post_qpos", "physics_post_qvel", "engine_force29")
    )
    contacts = np.empty(steps, dtype=int)
    for step in range(steps):
        target = targets[step // 10]
        requested = parameters.kps * (target - data.qpos[7:]) - parameters.kds * data.qvel[6:]
        data.ctrl[:] = np.clip(requested, -parameters.effort, parameters.effort)
        mujoco.mj_step(model, data)
        post_q[step], post_v[step], forces[step], contacts[step] = (
            data.qpos,
            data.qvel,
            data.qfrc_actuator[6:],
            data.ncon,
        )
    np.testing.assert_array_equal(post_q, trace["physics_post_qpos"])
    np.testing.assert_array_equal(post_v, trace["physics_post_qvel"])
    np.testing.assert_array_equal(forces, trace["engine_force29"])
    np.testing.assert_array_equal(contacts, trace["contact_count"])
    assert abs(data.time - steps * 0.002) < 1e-8
    limits = model.jnt_range[1:]
    limited = model.jnt_limited[1:].astype(bool)
    excess = np.maximum(np.maximum(limits[:, 0] - post_q[:, 7:], post_q[:, 7:] - limits[:, 1]), 0)
    excess[:, ~limited] = 0
    return dict(
        physical_substeps_reintegrated=steps,
        bit_exact_qpos_qvel_engine_force_contacts=True,
        postinitial_state_overwrites=0,
        maximum_actual_hard_range_excess_rad=float(excess.max()),
        actual_hard_range_violating_substeps=int(np.count_nonzero(np.any(excess > 0, axis=1))),
        maximum_engine_effort_excess_nm=float(np.maximum(np.abs(forces) - parameters.effort, 0).max()),
        maximum_target_jump_rad=float(np.abs(np.diff(targets, axis=0)).max()) if n > 1 else 0.0,
        target_rate_hardware_qualified=False,
    )


def fk(model, poses):
    data = mujoco.MjData(model)
    points = []
    for pose in poses:
        # Separate kinematic observer, never the integrated physical data.
        data.qpos[:] = pose
        mujoco.mj_fwdPosition(model, data)
        points.append(
            [
                data.xpos[model.body(name).id] + data.xmat[model.body(name).id].reshape(3, 3) @ offset
                for name, offset in POINTS
            ]
        )
    return np.asarray(points)


def tracking_audit(model, source_model, trace, record):
    n = len(trace["qpos"]) - 1
    desired = trace["reference_qpos29"][11 : 11 + n]
    q = trace["qpos"][1:]
    root_error = np.linalg.norm(q[:, :3] - desired[:, :3], axis=1)
    joint_error = q[:, 7:] - desired[:, 7:]
    landmark_error = np.linalg.norm(fk(model, q) - fk(source_model, desired), axis=-1)
    np.testing.assert_allclose(root_error, trace["root_error_m"], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(joint_error, trace["joint_error29"])
    np.testing.assert_allclose(landmark_error, trace["landmark_error_m"], atol=1e-12, rtol=0)
    phase = next(row for row in record["phases"] if row["name"] == "source_motion")
    stop = min(n, phase["control_stop"])
    count = max(0, stop - phase["control_start"])
    section = slice(phase["control_start"], stop)
    result = dict(
        source_controls_scored=count,
        full_source=count == phase["control_stop"] - phase["control_start"],
        root_p95_m=None,
        leg_rmse_rad=None,
        world_landmark_p95_m=None,
        tracking_screen_passed=False,
    )
    if count:
        result.update(
            root_p95_m=float(np.percentile(root_error[section], 95)),
            leg_rmse_rad=float(np.sqrt(np.mean(joint_error[section, :12] ** 2))),
            world_landmark_p95_m=np.percentile(landmark_error[section], 95, axis=0).tolist(),
        )
        result["tracking_screen_passed"] = bool(
            result["full_source"]
            and np.all(np.asarray(result["world_landmark_p95_m"]) <= [0.05, 0.05, 0.1, 0.1, 0.1])
        )
        np.testing.assert_allclose(
            result["root_p95_m"], record["metrics"]["source_root_position_p95_m"], atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            result["leg_rmse_rad"], record["metrics"]["source_leg_joint_rmse_rad"], atol=1e-12, rtol=0
        )
    assert result["tracking_screen_passed"] == record["metrics"]["source_landmark_screen_passed"]
    return result


def continuation_gate(cases):
    complete = all(row["complete_lifecycle"] and row["tracking"]["full_source"] for row in cases)
    if not complete:
        return dict(
            passed=False,
            all_lifecycles_complete=False,
            reason="incomplete lifecycle; partial/full comparisons prohibited",
        )
    root_ratio = float(
        np.mean([row["tracking"]["root_p95_m"] for row in cases])
        / np.mean([row["baseline_tracking"]["root_p95_m"] for row in cases])
    )
    leg_ratios = [row["tracking"]["leg_rmse_rad"] / row["baseline_tracking"]["leg_rmse_rad"] for row in cases]
    return dict(
        passed=bool(root_ratio <= 0.9 and max(leg_ratios) <= 1.05),
        all_lifecycles_complete=True,
        mean_root_p95_ratio=root_ratio,
        each_leg_rmse_ratio=leg_ratios,
        mean_root_p95_ratio_max=0.9,
        each_leg_rmse_ratio_max=1.05,
        authorizes_native23_or_hardware=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("audit refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("audit input changed: " + str(path))
        inputs[str(path)] = digest
        return path

    def archive(path, expected):
        with np.load(bind(path, expected), allow_pickle=False) as value:
            return {key: value[key].copy() for key in value.files}

    bind(__file__)
    report = json.loads(bind(args.report).read_text())
    for path, digest in report["inputs"].items():
        bind(path, digest)
    for flag in ("hardware_authorized", "deployment_ready", "native23_qualified", "native23_physics_equivalent"):
        assert report[flag] is False
    assert report["training_updates"] == 0
    assert tuple(Path(row["source"]).stem for row in report["records"]) == CLIPS
    prior = json.loads(bind(report["original29_baseline_report"], BASELINE_REPORT_SHA256).read_text())
    parameters_payload = json.loads(bind(report["cpp_parameters"]).read_text())
    parameters = CppParameters(parameters_payload)
    model = mujoco.MjModel.from_binary_path(str(bind(report["original_model"])))
    source_model = mujoco.MjModel.from_xml_path(str(bind(report["source_model"])))
    assert compiled_model_sha256(model) == report["physical_model_sha256"] == prior["physical_model_sha256"]
    assert (
        compiled_model_sha256(source_model) == report["source_geometry_sha256"] == prior["source_geometry_sha256"]
    )
    torch.set_num_threads(1)
    print("Loading immutable source weights for independent singleton arithmetic", flush=True)
    teacher = LowLatencyG1Teacher(report["teacher"]["checkpoint_path"])
    assert teacher_digest(teacher) == report["teacher_tensor_sha256"]
    cases = []
    for row, old in zip(report["records"], prior["records"], strict=True):
        trace, previous = archive(row["trace"], row["trace_sha256"]), archive(old["trace"], old["trace_sha256"])
        assert row["phases"] == old["phases"]
        for key in ("reference_qpos29", *[key for key in previous if key.startswith("received_")]):
            np.testing.assert_array_equal(trace[key], previous[key])
        for key in ("qpos", "qvel"):
            np.testing.assert_array_equal(trace[key][0], previous[key][0])
        n = len(trace["qpos"]) - 1
        assert n == row["metrics"]["completed_controls"]
        assert len(row["attempts"]) == len(trace["attempt_encoder640"])
        for i, attempt in enumerate(row["attempts"]):
            assert attempt["index"] == i
            assert attempt["applied"] == (i < n)
            assert attempt["output_present"] == bool(trace["attempt_output_present"][i])
        result = dict(
            clip=Path(row["source"]).stem,
            complete_lifecycle=n == row["metrics"]["requested_controls"] and row["metrics"]["failure"] is None,
            inputs=input_audit(trace, parameters_payload),
            network=network_audit(teacher, trace),
            physics=physics_audit(model, parameters, trace),
            tracking=tracking_audit(model, source_model, trace, row),
            baseline_tracking=tracking_audit(model, source_model, previous, old),
        )
        cases.append(result)
        print(json.dumps(result), flush=True)
    assert teacher_digest(teacher) == report["teacher_tensor_sha256"]
    assert compiled_model_sha256(model) == report["physical_model_sha256"]
    assert compiled_model_sha256(source_model) == report["source_geometry_sha256"]
    for path, digest in inputs.items():
        if file_sha256(path) != digest:
            raise ValueError("audit input changed during execution: " + path)
    result = dict(
        kind="source29_full_pose_encoder_independent_audit_v1",
        inputs=inputs,
        cases=cases,
        continuation_gate=continuation_gate(cases),
        mujoco_version=mujoco.__version__,
        hardware_authorized=False,
        deployment_ready=False,
        native23_qualified=False,
        training_updates=0,
        no_hardware_connection=True,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["continuation_gate"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
