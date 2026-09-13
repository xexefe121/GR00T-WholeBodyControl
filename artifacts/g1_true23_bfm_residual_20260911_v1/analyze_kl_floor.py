"""Read-only early-checkpoint KL decomposition on actual zero-residual replay states.

This does not claim to recover unlogged minibatch KL from the running learner.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from gear_sonic.utils.g1_true23_bfmzero_inference import load_contract
from gear_sonic.utils.g1_true23_bfm_residual_replay import residual_features_numpy, functional_residual


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent


def unix_path(value):
    value = str(value).replace("\\", "/")
    return Path("/mnt/c/" + value[3:] if value.startswith("C:/") else value)


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
features = []
for control in np.linspace(0, len(trace["action"]) - 1, 256).astype(int):
    data.qpos[:] = trace["qpos"][control]
    mujoco.mj_kinematics(model, data)
    previous = trace["action"][control-1] if control else np.zeros(23)
    target = config["default_q"] + trace["action"][control] * scale
    features.append(residual_features_numpy(data.qpos, trace["qvel"][control], previous, target, motion,
                    original["virtual_vr21"], control+11, data.xpos[feet], (6, 12)))
features = torch.from_numpy(np.array(features))
two = torch.load(BASE / "smoke64_v2/residual_00002_portable.pt", map_location="cpu", weights_only=True)
four = torch.load(BASE / "smoke256_v2/residual_00004.pt", map_location="cpu", weights_only=True)
norm_only = dict(four["actor_state"])
norm_only.update({k: v for k, v in two["actor_state"].items() if k.startswith("obs_normalizer.")})
weight_only = dict(two["actor_state"])
weight_only.update({k: v for k, v in four["actor_state"].items() if k.startswith("obs_normalizer.")})
with torch.inference_mode():
    final = functional_residual(four["actor_state"], features)
    std = .01 + .29 * four["actor_state"]["distribution.raw_std"].sigmoid()
    result = {"scope": "checkpoint2-to4 decomposition on256 actual CPU replay states; not current minibatch KL", "samples": 256,
              "normalization_counts": [int(two["actor_state"]["obs_normalizer.count"]), int(four["actor_state"]["obs_normalizer.count"])]}
    for name, state in (("normalizer_change_only", norm_only), ("actor_weights_change_only", weight_only)):
        delta = final - functional_residual(state, features)
        kl = (delta.square() / (2 * std.square())).sum(-1)
        result[name] = dict(mean_kl=float(kl.mean()), p95_kl=float(torch.quantile(kl, .95)),
                           mean_action_shift_rms_u=float(delta.square().mean().sqrt()))
print(json.dumps(result, indent=2))
with (BASE / "early_kl_decomposition.json").open("x") as stream:
    json.dump(result, stream, indent=2)
