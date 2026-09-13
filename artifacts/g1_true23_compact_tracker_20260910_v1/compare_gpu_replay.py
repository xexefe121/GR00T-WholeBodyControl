"""Deterministic standing-start GPU diagnostic using recovered compact1000.

Only reset sampling is replaced, to select one exact lifecycle per environment
without reset perturbations. Actor, reference timing, PD and physics unchanged.
Stops each trace at its first training termination; never claims a full pass.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gear_sonic.scripts import train_g1_true23_normal_lora as recipe
from gear_sonic.scripts.train_g1_true23_compact_tracker_v4 import observations
from gear_sonic.scripts.train_g1_true23_existing_pico import validate_bank
from gear_sonic.envs.mjlab import sonic_true23_compact_tracker as compact_env
from gear_sonic.envs.mjlab.sonic_true23_mixed_reference_starts import MixedReferenceRootFeedbackCommand
from gear_sonic.utils.g1_true23_compact_replay import CompactPolicy


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def main():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-aligned", action="store_true")
    options = parser.parse_args()
    os.environ["MUJOCO_GL"] = "egl"
    torch.set_num_threads(1)
    output = BASE / ("gpu_replay1000_clean_aligned_v1" if options.clean_aligned else "gpu_replay1000_v1")
    output.mkdir(exist_ok=False)
    bank = validate_bank(ROOT / "artifacts/g1_true23_pico_training_20260910_v1/bank_v1/report.json")
    args = SimpleNamespace(spec=Path(bank["files"]["spec"]), spans=Path(bank["files"]["spans"]),
                           num_envs=3, seed=20260910,
                           sim_config=ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json")
    spec = recipe.intent.validate_spec(json.loads(args.spec.read_text()))
    rows = json.loads(args.spans.read_text())["spans"]

    def deterministic_sample(command, env_ids):
        choice = env_ids.remainder(len(rows))
        command._lifecycle_choice[env_ids] = choice
        command.time_steps[env_ids] = command._lifecycle_starts[choice]
        command._lifecycle_last_anchor[env_ids] = command._lifecycle_ends[choice]
        command._env_clip_stop[env_ids] = command._lifecycle_last_anchor[env_ids] + 1
        command._episode_initial_anchor[env_ids] = command.time_steps[env_ids]
        command._episode_began_standing[env_ids] = True

    MixedReferenceRootFeedbackCommand._uniform_sampling = deterministic_sample
    cfg = compact_env.configure(recipe.environment_configuration(args, spec, rows), spec)
    cfg.commands["motion"].pose_range = {}
    cfg.commands["motion"].velocity_range = {}
    if options.clean_aligned:
        cfg.observations["policy"].enable_corruption = False
    policy = CompactPolicy(BASE / "train2000_v2/compact_1000.pt")
    actor = policy.actor.to("cuda:0")
    actor.eval()
    traces = {row["name"]: [] for row in rows}
    active = np.ones(3, dtype=bool)
    summary = {"reset_sampling": "exact_lifecycle_standing_no_perturbation", "cases": {}, "actor_and_dynamics_unchanged": True,
               "observation_corruption": not options.clean_aligned,
               "prime_reference_aligned_to_cpu": options.clean_aligned,
               "hardware_authorized": False, "deployment_ready": False}
    with recipe.ieee_training_precision():
        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=9.999)
        try:
            recipe.prime_sonic_true23_training_environment(wrapped)
            command = env.command_manager.get_term("motion")
            if options.clean_aligned:
                command.time_steps[:] = command._lifecycle_starts[command._lifecycle_choice]
                command._refresh_targets_from_causal_anchor()
            obs = observations(wrapped.get_observations())
            print(json.dumps({"initial_reference": command.time_steps.cpu().tolist(), "origins": env.scene.env_origins.cpu().tolist()}), flush=True)
            with torch.inference_mode():
                for step in range(1000):
                    actions = actor(obs)
                    qpos = env.sim.data.qpos.cpu().numpy().copy()
                    qpos[:, :3] -= env.scene.env_origins.cpu().numpy()
                    qvel = env.sim.data.qvel.cpu().numpy().copy()
                    features = obs["native_controller"].cpu().numpy().copy()
                    for i, row in enumerate(rows):
                        if active[i]:
                            traces[row["name"]].append({"qpos": qpos[i], "qvel": qvel[i], "features": features[i], "actions": actions[i].cpu().numpy().copy()})
                    obs, rewards, dones, extras = wrapped.step(actions)
                    obs = observations(obs)
                    for i, row in enumerate(rows):
                        if active[i] and bool(dones[i]):
                            active[i] = False
                            summary["cases"][row["name"]] = {"first_training_done_control": step + 1, "last_pre_step_root": qpos[i, :7].tolist(),
                                "termination_terms": {k: bool(env.termination_manager.get_term(k)[i]) for k in env.termination_manager.active_terms}}
                            print(json.dumps({row["name"]: summary["cases"][row["name"]]}), flush=True)
                    if not active.any():
                        break
            for name, samples in traces.items():
                arrays = {k: np.asarray([s[k] for s in samples]) for k in samples[0]}
                np.savez_compressed(output / f"{name}.npz", **arrays)
                with np.load(BASE / "eval1000_recovered_v1" / name / "attempts.npz", allow_pickle=False) as z:
                    cpu = dict(z)
                n = min(len(arrays["qpos"]), len(cpu["measured_qpos"]))
                summary["cases"].setdefault(name, {})["comparisons"] = {
                    str(i): {"qpos_max_abs": float(np.max(np.abs(arrays["qpos"][i] - cpu["measured_qpos"][i]))),
                             "qvel_max_abs": float(np.max(np.abs(arrays["qvel"][i] - cpu["measured_qvel"][i]))),
                             "features_max_abs": float(np.max(np.abs(arrays["features"][i] - cpu["native_controller165"][i]))),
                             "actions_max_abs": float(np.max(np.abs(arrays["actions"][i] - cpu["released_raw23"][i])))}
                    for i in (0, 1, 10, 50, 100, 200, 300, 349, 400, n - 1) if 0 <= i < n}
        finally:
            env.close()
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
