"""Independent CPU reconstruction of actual compact learner input samples."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from gear_sonic.utils.g1_23dof_contract import ISAACLAB_TO_MUJOCO_DOF, NATIVE_IL23_TO_CANONICAL_IL29
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads((args.run / "request.json").read_text())
    outcome = json.loads((args.run / "outcome.json").read_text())
    if not outcome.get("completed"):
        raise ValueError("input audit requires completed bounded run")
    for path, expected in request["source_hashes"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("training source changed " + path)
    bank = json.loads(Path(request["bank_report"]).read_text())
    spec = json.loads(Path(bank["files"]["spec"]).read_text())
    with np.load(spec["files"]["native_motion"]["path"], allow_pickle=False) as z:
        motion = {k: z[k].copy() for k in z.files}
    with np.load(spec["files"]["original_reference"]["path"], allow_pickle=False) as z:
        original = {k: z[k].copy() for k in z.files}
    with np.load(args.run / "sampled_inputs_rewards.npz", allow_pickle=False) as z:
        samples = {k: z[k].copy() for k in z.files}
    model = mujoco.MjModel.from_xml_path(spec["files"]["native_model"]["path"])
    data = mujoco.MjData(model)
    feet = [model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    keep = np.asarray(NATIVE_IL23_TO_CANONICAL_IL29)[list(ISAACLAB_TO_MUJOCO_DOF)]
    maximum = dict(goal90=0.0, packed165=0.0, measured_joint=0.0, measured_velocity=0.0)
    count = 0
    for t in range(len(samples["update"])):
        for e in range(samples["policy"].shape[1]):
            q0 = int(samples["reference_q0"][t, e])
            q1 = q0 + 1
            hist = samples["policy"][t, e]
            position = samples["root_position_w"][t, e]
            quaternion = samples["root_quaternion_wxyz"][t, e]
            velocity = samples["root_velocity_w"][t, e]
            origin = samples["env_origins"][t, e]
            q = samples["qpos_hw"][t, e]
            data.qpos[:] = np.r_[position, quaternion, q]
            mujoco.mj_kinematics(model, data)
            rot = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
            wanted = motion["body_pos_w"][q1, 0].astype(np.float32) + origin
            wanted_v = (
                motion["body_pos_w"][q1, 0].astype(np.float32) - motion["body_pos_w"][q0, 0].astype(np.float32)
            ) / np.float32(0.02)
            wanted_rot = Rotation.from_quat(motion["body_quat_w"][q1, 0][[1, 2, 3, 0]]).as_matrix()
            yaw = np.arctan2(rot[1, 0], rot[0, 0])
            c, s = np.cos(yaw), np.sin(yaw)
            yaw_rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            feedback = np.r_[yaw_rot @ (wanted - position), yaw_rot @ wanted_v, yaw_rot @ velocity]
            reference_feet = motion["body_pos_w"][q1, np.asarray(feet) - 1] + origin
            foot_error = (reference_feet - data.xpos[feet]) @ rot
            goal = np.r_[
                motion["joint_pos"][q1],
                (motion["joint_pos"][q1].astype(np.float32) - motion["joint_pos"][q0].astype(np.float32))
                / np.float32(0.02),
                feedback,
                (rot.T @ wanted_rot)[:, :2].reshape(-1),
                original["virtual_vr21"][q1],
                foot_error.reshape(-1),
                wanted[2] - origin[2],
                position[2] - origin[2],
            ].astype(np.float32)
            actual = samples["native_goal"][t, e]
            maximum["goal90"] = max(maximum["goal90"], float(np.max(np.abs(goal - actual))))
            latest = [
                hist[27:30],
                *[hist[start : start + 290].reshape(10, 29)[-1, keep] for start in (30, 320, 610)],
                hist[927:930],
            ]
            packed = np.concatenate((*latest, actual))
            maximum["packed165"] = max(
                maximum["packed165"], float(np.max(np.abs(packed - samples["native_controller"][t, e])))
            )
            q_actual = latest[1] + np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, np.float32)
            maximum["measured_joint"] = max(maximum["measured_joint"], float(np.max(np.abs(q_actual - q))))
            maximum["measured_velocity"] = max(
                maximum["measured_velocity"], float(np.max(np.abs(latest[2] - samples["dqpos_hw"][t, e])))
            )
            count += 1
    if (
        maximum["goal90"] > 1e-5
        or maximum["packed165"] != 0
        or maximum["measured_joint"] > 5e-7
        or maximum["measured_velocity"] != 0
    ):
        raise ValueError("compact training input mismatch " + str(maximum))
    zero = torch.load(args.run / "compact_0000.pt", map_location="cpu", weights_only=True)
    final = torch.load(args.run / f"compact_{outcome['updates']:04d}.pt", map_location="cpu", weights_only=True)
    changed = [
        name
        for name, value in final["actor_state"].items()
        if name.startswith("mlp.") and not torch.equal(value, zero["actor_state"][name])
    ]
    steps = sorted(set(int(item["step"]) for item in final["optimizer_state"]["state"].values()))
    if not changed or steps != [outcome["updates"] * 4 * 4]:
        raise ValueError("compact weights or optimizer update count wrong")
    result = dict(
        passed=True,
        sampled_actual_input_states=count,
        max_abs_errors=maximum,
        changed_mlp_tensors=changed,
        optimizer_steps_per_parameter=steps,
        sampled_input_sha256=sha256_file(args.run / "sampled_inputs_rewards.npz"),
        actor_checkpoint_sha256=sha256_file(args.run / f"compact_{outcome['updates']:04d}.pt"),
        deployment_ready=False,
        hardware_authorized=False,
    )
    with (args.run / "input_audit.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
