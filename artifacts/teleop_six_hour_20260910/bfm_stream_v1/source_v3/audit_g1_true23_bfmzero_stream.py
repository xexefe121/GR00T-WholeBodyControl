"""Independent replay of saved stream torques and sensor/history evidence.

No policy controller is called during physics replay. All saved torques must
reproduce every 2 ms state; source admission is audited from arrival records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import (
    MODEL,
    PACKAGE,
    PHYSICS,
    ROOT,
    corrected_goal,
    load_motion,
)
from gear_sonic.utils.g1_true23_bfmzero_inference import BFMZeroInference, load_contract, reference_features
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory, *, actor_samples=24):
    request = json.loads((directory / "request.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    assert report["request_sha256"] == sha256(directory / "request.json")
    assert report["trace_sha256"] == sha256(directory / "trace.npz")
    for path, digest in request["inputs"].items():
        if Path(path).suffix != ".py":
            assert sha256(path) == digest, path
    with np.load(directory / "trace.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    _, model, physics = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    contract = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    physical = json.loads((ROOT / PHYSICS).read_text())
    velocity_limits = np.asarray(physical["physics"]["velocity_limit_hardware_radps"])
    n = len(data["action"])
    assert len(data["physics_torque"]) == n * 10
    assert data["physics_qpos"].shape == (n * 10 + 1, 30)
    assert data["physics_qvel"].shape == (n * 10 + 1, 29)
    np.testing.assert_array_equal(data["qpos"], data["physics_qpos"][::10])
    np.testing.assert_array_equal(data["qvel"], data["physics_qvel"][::10])
    probe = mujoco.MjData(model)
    probe.qpos[:] = data["physics_qpos"][0]
    probe.qvel[:] = data["physics_qvel"][0]
    mujoco.mj_forward(model, probe)
    replay_qpos_error = replay_qvel_error = requested_torque_error = 0.0
    for index, torque in enumerate(data["physics_torque"]):
        expected = (
            contract["kp"] * (data["target"][index // 10] - probe.qpos[7:]) - contract["kd"] * probe.qvel[6:]
        )
        requested_torque_error = max(
            requested_torque_error, float(np.max(np.abs(expected - data["physics_requested_torque"][index])))
        )
        np.testing.assert_array_equal(
            torque, np.clip(data["physics_requested_torque"][index], -physics.effort, physics.effort)
        )
        probe.ctrl[:] = torque
        mujoco.mj_step(model, probe)
        replay_qpos_error = max(
            replay_qpos_error, float(np.max(np.abs(probe.qpos - data["physics_qpos"][index + 1])))
        )
        replay_qvel_error = max(
            replay_qvel_error, float(np.max(np.abs(probe.qvel - data["physics_qvel"][index + 1])))
        )
    assert replay_qpos_error == replay_qvel_error == requested_torque_error == 0.0
    limits = model.jnt_range[1:]
    actual_q = data["physics_qpos"][1:, 7:]
    range_excess = float(max(0, np.max(limits[:, 0] - actual_q), np.max(actual_q - limits[:, 1])))
    velocity_ratio = float(np.max(np.abs(data["physics_qvel"][1:, 6:]) / velocity_limits))
    effort_ratio = float(np.max(np.abs(data["physics_torque"]) / physics.effort))
    assert abs(range_excess - report["range_excess_max_rad"]) < 1e-12
    assert abs(velocity_ratio - report["velocity_ratio_max"]) < 1e-12
    assert abs(effort_ratio - report["effort_ratio_max"]) < 1e-12

    # Independent SciPy rotation and explicit public feature slices. Histories
    # are indexed from preceding sensor records, never copied from BFMHistory.
    qpos, qvel = data["qpos"][:-1], data["qvel"][:-1]
    gravity = Rotation.from_quat(qpos[:, [4, 5, 6, 3]]).inv().apply(np.tile([0, 0, -1.0], (n, 1)))
    expected_state = np.column_stack(
        (qpos[:, 7:] - contract["default_q"], qvel[:, 6:], gravity, qvel[:, 3:6] * 0.25)
    ).astype(np.float32)
    state_error = float(np.max(np.abs(expected_state - data["state"])))
    np.testing.assert_allclose(expected_state, data["state"], atol=1e-7, rtol=0)
    previous_action = np.concatenate((np.zeros((1, 23), np.float32), data["action"][:-1]))
    expected_history = np.zeros((n, 300), np.float32)
    for index in range(n):
        for offset in range(1, min(index, 4) + 1):
            row = index - offset
            slot = offset - 1
            for start, values in (
                (0, previous_action[row]),
                (92, expected_state[row, 49:52]),
                (104, expected_state[row, :23]),
                (196, expected_state[row, 23:46]),
                (288, expected_state[row, 46:49]),
            ):
                width = len(values)
                expected_history[index, start + slot * width : start + (slot + 1) * width] = values
    history_error = float(np.max(np.abs(expected_history - data["history"])))
    np.testing.assert_allclose(expected_history, data["history"], atol=1e-7, rtol=0)
    expected_target = contract["default_q"] + data["action"] * 0.25 * contract["training_effort"] / contract["kp"]
    np.testing.assert_array_equal(expected_target, data["unclipped_target"])
    np.testing.assert_array_equal(np.clip(expected_target, limits[:, 0], limits[:, 1]), data["target"])

    accepted = data["arrivals"][data["arrivals"][:, 4] == 1]
    consumed = set()
    control_times = np.arange(n) * 0.02
    if "control_receiver_time" in data:
        control_times = data["control_receiver_time"]
        clock_witness = "explicit_actual_receiver_clock"
    elif request["paced"]:
        # Earlier traces clamped negative start lateness to zero. Recover the
        # recorded clock only on source controls from receipt + logged age;
        # disclose that these are not an independent receiver-clock witness.
        clock_witness = "legacy_clock_reconstructed_from_receipt_and_source_age"
        for index in np.flatnonzero(data["source_sequence"] >= 0):
            original = accepted[
                (accepted[:, 1] == data["source_epoch"][index])
                & (accepted[:, 2] == data["source_sequence"][index])
            ]
            assert len(original) == 1
            control_times[index] = original[0, 0] + data["source_buffer_age"][index]
    else:
        clock_witness = "declared_unpaced_fixed_control_clock"
    for index in np.flatnonzero(data["source_sequence"] >= 0):
        epoch, sequence = int(data["source_epoch"][index]), int(data["source_sequence"][index])
        now = control_times[index]
        eligible = accepted[(accepted[:, 1] == epoch) & (accepted[:, 0] <= now + 1e-7)]
        available = {int(row[2]): row for row in eligible}
        assert (epoch, sequence) not in consumed
        consumed.add((epoch, sequence))
        window_size = int(data["goal_window_samples"][index])
        assert 1 <= window_size <= 8
        for used in range(sequence, sequence + window_size):
            assert used in available, (index, epoch, used, "goal used source before receipt")
        assert abs(data["source_timestamp"][index] - sequence * 0.02) < 1e-12
        assert abs(data["source_latest_timestamp_used"][index] - (sequence + window_size - 1) * 0.02) < 1e-12
        assert abs(data["source_buffer_age"][index] - (now - available[sequence][0])) < 1e-7
    for epoch in report["epochs"]:
        assert epoch["consumed"] == sum(item[0] == epoch["epoch"] for item in consumed)
        if epoch["fault"]:
            assert not epoch["full_source_consumed"]
            # After a fault, this epoch can never control the robot again.
            indices = np.flatnonzero((data["source_epoch"] == epoch["epoch"]) & (data["source_sequence"] >= 0))
            assert np.all(control_times[indices] < epoch["fault"]["time"] + 1e-7)
    if report["any_fault_latched"]:
        assert not report["full_uninterrupted_source_consumed"]

    torch.set_num_threads(1)
    policy = BFMZeroInference(PACKAGE / "bfmzero_inference_v1/inference.safetensors")
    original_motion, _, _ = load_motion(request["clip"])
    source_controls = np.flatnonzero(data["source_sequence"] >= 0)
    selected_goals = (
        source_controls[
            np.unique(
                np.linspace(0, len(source_controls) - 1, min(actor_samples, len(source_controls))).astype(int)
            )
        ]
        if len(source_controls)
        else []
    )
    source_goal_error = 0.0
    for index in selected_goals:
        sequence, epoch_id = int(data["source_sequence"][index]), int(data["source_epoch"][index])
        window_size = int(data["goal_window_samples"][index])
        source = {
            name: original_motion[name][sequence + 11 : sequence + 11 + window_size].copy()
            for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        }
        epoch = next(row for row in report["epochs"] if row["epoch"] == epoch_id)
        calibration = epoch["calibration"]
        if calibration is not None:
            quaternion = np.asarray(calibration["rotation_wxyz"])
            rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
            original_z = source["body_pos_w"][..., 2].copy()
            positions = source["body_pos_w"] - np.asarray(calibration["source_origin"])
            source["body_pos_w"] = rotation.apply(positions.reshape(-1, 3)).reshape(positions.shape) + np.asarray(
                calibration["measured_origin"]
            )
            source["body_pos_w"][..., 2] = original_z
            quaternions = source["body_quat_w"].reshape(-1, 4)
            source["body_quat_w"] = (
                (rotation * Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]))
                .as_quat()[:, [3, 0, 1, 2]]
                .reshape(window_size, 24, 4)
            )
            for name in ("body_lin_vel_w", "body_ang_vel_w"):
                source[name] = rotation.apply(source[name].reshape(-1, 3)).reshape(window_size, 24, 3)
        np.testing.assert_allclose(
            np.r_[source["body_pos_w"][0, 0], source["body_quat_w"][0, 0], source["joint_pos"][0]],
            data["reference_qpos"][index],
            atol=1e-12,
            rtol=0,
        )
        # Encode individually, matching arrival-time float32 quantization, then
        # run the separately maintained referee correction on this received window.
        encoded = [
            reference_features({name: value[offset : offset + 1] for name, value in source.items()}, contract)
            for offset in range(window_size)
        ]
        states = np.concatenate([row[0] for row in encoded])
        privileged = np.concatenate([row[1] for row in encoded])
        goal = corrected_goal(
            policy,
            states,
            privileged,
            source,
            0,
            data["qpos"][index],
            window_size,
            request["position_gain"],
            request["yaw_gain"],
        )[0].numpy()
        source_goal_error = max(source_goal_error, float(np.max(np.abs(goal - data["goal"][index]))))
    assert source_goal_error <= 3e-5
    selected = np.unique(np.linspace(0, n - 1, min(n, actor_samples)).astype(int))
    actor_error = 0.0
    for index in selected:
        action = (
            policy.actor(
                torch.from_numpy(data["state"][index : index + 1]),
                torch.from_numpy(previous_action[index : index + 1]),
                torch.from_numpy(data["history"][index : index + 1]),
                torch.from_numpy(data["goal"][index : index + 1]),
            )[0].numpy()
            * 5.0
        )
        actor_error = max(actor_error, float(np.max(np.abs(action - data["action"][index]))))
    assert actor_error == 0.0
    return {
        "kind": "independent_bfm_stream_torque_and_observation_audit",
        "passed": True,
        "controls": n,
        "physics_steps": n * 10,
        "torque_replay_qpos_max_error": replay_qpos_error,
        "torque_replay_qvel_max_error": replay_qvel_error,
        "requested_torque_max_error": requested_torque_error,
        "scipy_sensor_state_max_error": state_error,
        "independent_history_max_error": history_error,
        "actor_reinference_samples": selected.tolist(),
        "actor_action_max_error": actor_error,
        "source_goal_reconstruction_samples": [int(index) for index in selected_goals],
        "source_goal_max_error": source_goal_error,
        "source_goal_tolerance": 3e-5,
        "received_only_goal_controls_audited": len(consumed),
        "receiver_clock_witness": clock_witness,
        "range_excess_max_rad": range_excess,
        "velocity_ratio_max": velocity_ratio,
        "effort_ratio_max": effort_ratio,
        "source_tracking_qualified": False,
        "hardware_qualified": False,
        "artifact_hashes": {
            name: sha256(directory / name) for name in ("request.json", "report.json", "trace.npz")
        },
        "audit_source_sha256": sha256(__file__),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor-samples", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("audit evidence refuses overwrite")
    result = audit(args.directory, actor_samples=args.actor_samples)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)
