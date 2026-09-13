"""Recompute the declared ankle law and replay every saved native motor torque."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import MODEL, PHYSICS, PACKAGE, ROOT
from gear_sonic.utils.g1_true23_bfmzero_inference import load_contract
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory):
    report = json.loads((directory / "report.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    assert sha256(directory / "request.json") == report["request_sha256"]
    assert sha256(directory / "trace.npz") == report["trace_sha256"]
    for name, digest in request["inputs"].items():
        if Path(name).suffix != ".py":
            assert sha256(name) == digest, name
    with np.load(directory / "trace.npz", allow_pickle=False) as archive:
        trace = {name: archive[name].copy() for name in archive.files}
    _, model, physics = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    contract = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    law = request["barrier"]
    k, d, margin = law["stiffness_Nm_per_rad"], law["damping_Nm_per_rad_per_second"], law["margin_rad"]
    indices = [model.joint(name).id - 1 for name in ("left_ankle_roll_joint", "right_ankle_roll_joint")]
    assert indices == law["joint_indices"]
    assert len(trace["physics_torque"]) == len(trace["action"]) * 10
    np.testing.assert_array_equal(trace["qpos"], trace["physics_qpos"][::10])
    np.testing.assert_array_equal(trace["qvel"], trace["physics_qvel"][::10])
    state = mujoco.MjData(model)
    state.qpos[:] = trace["physics_qpos"][0]
    state.qvel[:] = trace["physics_qvel"][0]
    mujoco.mj_forward(model, state)
    torque_error = qpos_error = qvel_error = 0.0
    for step, actual in enumerate(trace["physics_torque"]):
        pd = contract["kp"] * (trace["target"][step // 10] - state.qpos[7:]) - contract["kd"] * state.qvel[6:]
        np.testing.assert_array_equal(pd, trace["physics_base_torque"][step])
        extra = np.zeros(23)
        for joint in indices:
            position, speed = state.qpos[joint + 7], state.qvel[joint + 6]
            low, high = model.jnt_range[joint + 1]
            if position < low + margin:
                extra[joint] = k * (low + margin - position)
                if speed < 0:
                    extra[joint] -= d * speed
            if position > high - margin:
                extra[joint] = -k * (position - high + margin)
                if speed > 0:
                    extra[joint] -= d * speed
        torque_error = max(torque_error, float(np.abs(extra - trace["physics_barrier_torque"][step]).max()))
        np.testing.assert_array_equal(
            pd + trace["physics_barrier_torque"][step], trace["physics_requested_torque"][step]
        )
        np.testing.assert_array_equal(
            actual, np.clip(trace["physics_requested_torque"][step], -physics.effort, physics.effort)
        )
        state.ctrl[:] = actual
        mujoco.mj_step(model, state)
        qpos_error = max(qpos_error, float(np.abs(state.qpos - trace["physics_qpos"][step + 1]).max()))
        qvel_error = max(qvel_error, float(np.abs(state.qvel - trace["physics_qvel"][step + 1]).max()))
    assert torque_error < 1e-12 and qpos_error == qvel_error == 0
    q = trace["physics_qpos"][1:, 7:]
    range_excess = float(max(0, (model.jnt_range[1:, 0] - q).max(), (q - model.jnt_range[1:, 1]).max()))
    assert range_excess == report["range_excess_max"]
    return {
        "passed": True,
        "physical_steps_replayed": len(trace["physics_torque"]),
        "declared_ankle_law_max_torque_error": torque_error,
        "native_torque_replay_qpos_max_error": qpos_error,
        "native_torque_replay_qvel_max_error": qvel_error,
        "raw_joint_range_excess_rad": range_excess,
        "evidence_is_physical_qualification": False,
        "artifact_hashes": {
            name: sha256(directory / name) for name in ("request.json", "report.json", "trace.npz")
        },
        "audit_source_sha256": sha256(__file__),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("audit evidence refuses overwrite")
    result = audit(args.directory)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
