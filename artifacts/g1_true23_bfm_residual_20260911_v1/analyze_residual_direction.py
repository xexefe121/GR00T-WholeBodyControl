"""Read-only residual direction on fixed, real zero-residual replay states.

This counterfactual input audit does not replace closed-loop simulation.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from gear_sonic.utils.g1_true23_bfmzero_inference import load_contract
from gear_sonic.utils.g1_true23_bfm_residual_replay import residual_features_numpy, functional_residual


ROOT = Path(__file__).resolve().parents[2]


def unix_path(value):
    value = str(value).replace("\\", "/")
    return Path("/mnt/c/" + value[3:] if value.startswith("C:/") else value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = ROOT / "artifacts/teleop_six_hour_20260910/bfm_residual_zero_walk002_v2"
    report = json.loads((folder / "report.json").read_text())
    motion = dict(np.load(unix_path(report["reference_path"]), allow_pickle=False))
    trace = dict(np.load(folder / "trace.npz", allow_pickle=False))
    original = dict(np.load("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/walk002/original_source_bundle_v1/original_reference.npz", allow_pickle=False))
    config = load_contract(ROOT / "artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inspect_v1/config.yaml")
    model = mujoco.MjModel.from_xml_path(str(ROOT.parent / "GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"))
    data = mujoco.MjData(model)
    feet = [model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    scale = .25 * config["training_effort"] / config["kp"]
    features, errors = [], []
    for control in np.linspace(0, len(trace["action"]) - 1, 256).astype(int):
        data.qpos[:] = trace["qpos"][control]
        mujoco.mj_kinematics(model, data)
        previous = trace["action"][control-1] if control else np.zeros(23)
        target = config["default_q"] + trace["action"][control] * scale
        features.append(residual_features_numpy(data.qpos, trace["qvel"][control], previous, target, motion,
                        original["virtual_vr21"], control+11, data.xpos[feet], (6, 12)))
        frame = min(control+11, len(motion["joint_pos"])-1)
        errors.append(motion["joint_pos"][frame] - data.qpos[7:])
    features = torch.from_numpy(np.array(features))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    with torch.inference_mode():
        raw = functional_residual(checkpoint["actor_state"], features)
        delta = .15 * raw.tanh().numpy()
    errors = np.array(errors)
    result = dict(scope="counterfactual residual on 256 frozen CPU baseline walk002 states; not closed-loop qualification",
                  checkpoint=str(args.checkpoint), update=checkpoint["completed_updates"],
                  residual_rms_rad=float(np.sqrt(np.mean(delta**2))),
                  residual_p95_abs_rad=float(np.quantile(np.abs(delta), .95)),
                  residual_max_abs_rad=float(np.max(np.abs(delta))), joints=[])
    for j in range(23):
        result["joints"].append(dict(name=model.joint(j+1).name, mean_rad=float(delta[:, j].mean()),
             rms_rad=float(np.sqrt(np.mean(delta[:, j]**2))),
             p95_abs_rad=float(np.quantile(np.abs(delta[:, j]), .95)),
             correction_error_dot=float(np.mean(delta[:, j] * errors[:, j])),
             mean_reference_error_rad=float(errors[:, j].mean())))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
