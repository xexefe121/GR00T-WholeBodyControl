"""Numerical CPU/GPU parity and frozen BFM residual feature checks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import corrected_goal
from gear_sonic.utils.g1_true23_bfmzero_inference import BFMHistory, load_contract, state_and_terms
from gear_sonic.utils.g1_true23_bfm_residual import FrozenBFMController, bfm_state_and_terms
from gear_sonic.trl.mjlab.g1_true23_bfm_residual_actor import BFMResidualActor


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "artifacts/g1_true23_six_hour_replan_20260910_v1"


def run(args):
    torch.set_num_threads(1)
    bank = json.loads((ROOT / "artifacts/g1_true23_pico_training_20260910_v1/bank_v1/report.json").read_text())
    spec = json.loads(Path(bank["files"]["spec"]).read_text())
    with np.load(spec["files"]["native_motion"]["path"], allow_pickle=False) as z:
        motion = {k: v[:800].copy() for k, v in z.items()}
    contract = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    weights = PACKAGE / "bfmzero_inference_v1/inference.safetensors"
    controller = FrozenBFMController(weights, contract, motion, count=4, device="cpu")
    frames = np.array([12, 350, 450, 600])
    qpos = np.c_[motion["body_pos_w"][frames, 0], motion["body_quat_w"][frames, 0], motion["joint_pos"][frames]].astype(np.float32)
    qpos[:, :2] += np.array([.08, -.06], np.float32)
    # Exercise the measured-heading transform and clipped yaw correction, not
    # merely same-heading goals where those terms would vanish.
    for i, angle in enumerate((.12, -.25, .65, -1.0)):
        w, x, y, z = qpos[i, 3:7].copy()
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        qpos[i, 3:7] = [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]
    qvel = np.c_[motion["body_lin_vel_w"][frames, 0], np.zeros((4, 3)), motion["joint_vel"][frames]].astype(np.float32)
    qvel[:, 3:6] = np.array([.1, -.2, .03])
    last = np.random.default_rng(32).normal(0, .2, (4, 23)).astype(np.float32)
    controller.last_action.copy_(torch.from_numpy(last))
    state, terms = bfm_state_and_terms(torch.from_numpy(qpos), torch.from_numpy(qvel), torch.from_numpy(last), controller.default_q)
    numpy_histories = [BFMHistory() for _ in range(4)]
    maxima = dict(state=0., history=0., goal=0., actor=0., zero_target=0.)
    for step in range(6):
        tensor_history = controller.history.before_update(terms)
        for i in range(4):
            wanted_state, wanted_terms = state_and_terms(qpos[i, 7:], qvel[i, 6:], qpos[i, 3:7], qvel[i, 3:6], last[i], contract["default_q"])
            wanted_history = numpy_histories[i].before_update(wanted_terms)
            maxima["state"] = max(maxima["state"], float(np.max(np.abs(state[i].numpy() - wanted_state))))
            maxima["history"] = max(maxima["history"], float(np.max(np.abs(tensor_history[i].numpy() - wanted_history))))
    controller.history.reset(slice(None))
    base, sensed, hist, goal = controller.observe(torch.from_numpy(qpos), torch.from_numpy(qvel), torch.from_numpy(frames), torch.full((4,), 799))
    for i in range(4):
        wanted_goal = corrected_goal(controller.policy, controller.reference_state.numpy(), controller.reference_privileged.numpy(), motion,
                                     int(frames[i]), qpos[i], 8, 1., 2.)
        maxima["goal"] = max(maxima["goal"], float((goal[i] - wanted_goal[0]).abs().max()))
        wanted = 5 * controller.policy.actor(sensed[i:i+1], torch.from_numpy(last[i:i+1]), hist[i:i+1], wanted_goal)
        maxima["actor"] = max(maxima["actor"], float((base[i] - wanted[0]).abs().max()))
    target, delta, combined = controller.targets(torch.zeros(4, 23), torch.full((23,), -100.), torch.full((23,), 100.))
    maxima["zero_target"] = float((target - (controller.default_q + base * controller.action_scale)).abs().max())
    assert maxima["state"] < 2e-6 and maxima["history"] < 2e-6 and maxima["goal"] < 2e-5 and maxima["actor"] < 2e-4
    assert maxima["zero_target"] == 0 and torch.count_nonzero(delta) == 0
    assert torch.equal(combined, base)
    obs = TensorDict({"bfm_residual": torch.randn(4, 188)}, batch_size=[4])
    actor = BFMResidualActor(obs, {"actor": ["bfm_residual"]}, "actor", 23)
    assert torch.count_nonzero(actor(obs)) == 0
    actions = actor(obs, stochastic_output=True)
    assert torch.isfinite(actor.get_output_log_prob(actions)).all()
    before = actor(obs).clone()
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    optimizer.zero_grad()
    (actor(obs) - .1).square().mean().backward()
    optimizer.step()
    assert float((actor(obs) - before).detach().abs().max()) > .001
    if args.cuda:
        gpu = FrozenBFMController(weights, contract, motion, count=4, device="cuda:0")
        gpu.last_action.copy_(torch.from_numpy(last).cuda())
        gpu_base, _, _, gpu_goal = gpu.observe(torch.from_numpy(qpos).cuda(), torch.from_numpy(qvel).cuda(), torch.from_numpy(frames).cuda(), torch.full((4,), 799, device="cuda:0"))
        maxima["cuda_goal"] = float((gpu_goal.cpu() - goal).abs().max())
        maxima["cuda_actor"] = float((gpu_base.cpu() - base).abs().max())
        assert maxima["cuda_goal"] < 2e-4 and maxima["cuda_actor"] < 2e-3
    result = dict(passed=True, maxima=maxima, actor_parameters=sum(p.numel() for p in actor.parameters()),
                  learned_residual_changed=True, zero_residual_target_exact=True, hardware_authorized=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())
