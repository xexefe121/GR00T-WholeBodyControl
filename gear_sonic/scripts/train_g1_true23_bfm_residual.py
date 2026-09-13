"""Bounded simulation-only residual PPO over a frozen native23 BFM policy."""

import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from gear_sonic.scripts import train_g1_true23_normal_lora as recipe
from gear_sonic.scripts.train_g1_true23_existing_pico import validate_bank
from gear_sonic.scripts.train_g1_true23_compact_tracker_v4 import audit_reward
from gear_sonic.envs.mjlab import sonic_true23_compact_tracker as compact_env
from gear_sonic.envs.mjlab.g1_true23_bfm_residual_env import BFMResidualDriver, configure_bfm_action
from gear_sonic.utils.g1_true23_bfmzero_inference import load_contract, WEIGHTS_SHA, CONFIG_SHA
from gear_sonic.utils.g1_true23_bfm_residual import KIND, residual_contract
from gear_sonic.utils.g1_true23_foot_precision_reward import FootPrecisionStep
from gear_sonic.utils.g1_true23_world_quality import WorldQualityStep
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "artifacts/g1_true23_six_hour_replan_20260910_v1"
FLAGS = dict(deployment_ready=False, hardware_authorized=False, simulator_qualified=False)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def configuration():
    from mjlab.rl import RslRlModelCfg
    cfg = recipe.true23_mjlab_ppo_runner_cfg()
    cfg.actor = RslRlModelCfg(class_name="gear_sonic.trl.mjlab.g1_true23_bfm_residual_actor:BFMResidualActor",
                            hidden_dims=(256, 256), activation="elu", obs_normalization=True, distribution_cfg=None)
    cfg.critic = RslRlModelCfg(hidden_dims=(256, 256, 128), activation="elu", obs_normalization=True)
    cfg.obs_groups = {"actor": ("bfm_residual",), "critic": ("critic", "native_goal", "bfm_residual")}
    cfg.algorithm.learning_rate = 3e-4
    cfg.algorithm.schedule = "adaptive"
    cfg.algorithm.desired_kl = .01
    cfg.algorithm.entropy_coef = .001
    cfg.algorithm.use_clipped_value_loss = False
    cfg.algorithm.max_grad_norm = .5
    cfg.algorithm.num_learning_epochs = 4
    cfg.algorithm.num_mini_batches = 4
    cfg.num_steps_per_env = 24
    result = asdict(cfg)
    for name in ("actor", "critic"):
        result[name].pop("cnn_cfg", None)
    result["multi_gpu"] = None
    return result


def verify_input_snapshot(env, driver, obs, actor, motion, bfm, spec):
    """Independent CPU FK/features and functional actor vs actual GPU input."""
    import mujoco
    from gear_sonic.utils.g1_true23_bfm_residual_replay import residual_features_numpy, functional_residual
    model = mujoco.MjModel.from_xml_path(spec["files"]["native_model"]["path"])
    data = mujoco.MjData(model)
    names = ("left_ankle_roll_link", "right_ankle_roll_link")
    feet = [model.body(name).id for name in names]
    motion_feet = [bfm["body_names"].index(name) for name in names]
    with np.load(spec["files"]["original_reference"]["path"], allow_pickle=False) as z:
        vr = z["virtual_vr21"]
    qpos = env.sim.data.qpos.cpu().numpy().copy()
    qpos[:, :3] -= env.scene.env_origins.cpu().numpy()
    qvel = env.sim.data.qvel.cpu().numpy()
    previous = driver.controller.last_action.cpu().numpy()
    base_target = (driver.controller.default_q + driver.controller.base_action * driver.controller.action_scale).cpu().numpy()
    frames = env.command_manager.get_term("motion").time_steps.cpu().numpy() + 1
    actual = obs["bfm_residual"].cpu().numpy()
    maximum = 0.
    for i in range(min(8, env.num_envs)):
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        expected = residual_features_numpy(qpos[i], qvel[i], previous[i], base_target[i], motion, vr,
                                           int(frames[i]), data.xpos[feet], motion_feet)
        maximum = max(maximum, float(np.max(np.abs(expected - actual[i]))))
    with torch.inference_mode():
        cpu_state = {key: value.detach().cpu() for key, value in actor.state_dict().items()}
        expected_action = functional_residual(cpu_state, torch.from_numpy(actual))
        actor_error = float((actor(obs).cpu() - expected_action).abs().max())
    if maximum > 2e-5 or actor_error > 2e-5:
        raise ValueError(f"residual CPU/GPU input mismatch: features={maximum},actor={actor_error}")
    return dict(states=min(8, env.num_envs), feature_max_abs_error=maximum, actor_max_abs_error=actor_error)


def run(args):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from rsl_rl.algorithms.ppo import PPO

    if args.output.exists():
        raise FileExistsError("choose a new residual output folder")
    args.output.mkdir(parents=True)
    args.seed = 20260911
    torch.set_num_threads(1)
    os.environ["MUJOCO_GL"] = "egl"
    bank_path = ROOT / "artifacts/g1_true23_pico_training_20260910_v1/bank_v1/report.json"
    bank = validate_bank(bank_path)
    args.spec, args.spans = Path(bank["files"]["spec"]), Path(bank["files"]["spans"])
    spec = recipe.intent.validate_spec(json.loads(args.spec.read_text()))
    rows = json.loads(args.spans.read_text())["spans"]
    bfm = load_contract(PACKAGE / "bfmzero_inspect_v1/config.yaml")
    weights = PACKAGE / "bfmzero_inference_v1/inference.safetensors"
    with np.load(spec["files"]["native_motion"]["path"], allow_pickle=False) as archive:
        motion = dict(archive)
    cfg = compact_env.configure(recipe.environment_configuration(args, spec, rows), spec)
    cfg = configure_bfm_action(cfg, bfm)
    algorithm_cfg = configuration()
    manifest = dict(kind=KIND, contract=residual_contract(), bank_sha256=sha256_file(bank_path),
                    weights_sha256=WEIGHTS_SHA, config_sha256=CONFIG_SHA, num_envs=args.num_envs,
                    planned_updates=args.updates, wall_hours=args.wall_hours, steps=24, seed=args.seed,
                    actor_config=algorithm_cfg["actor"], agent=algorithm_cfg,
                    resumed_from=None if not args.resume else dict(path=str(args.resume), sha256=sha256_file(args.resume)),
                    reset_schedule="25pct standing;75pct source reference reset;zero BFM history on reset",
                    bfm_pd_gains=cfg.actions["joint_pos"].profile.contract(),
                    source_files={str(p): sha256_file(p) for p in (
                        Path(__file__), ROOT / "gear_sonic/utils/g1_true23_bfm_residual.py",
                        ROOT / "gear_sonic/envs/mjlab/g1_true23_bfm_residual_env.py",
                        ROOT / "gear_sonic/trl/mjlab/g1_true23_bfm_residual_actor.py")}, **FLAGS)
    # Persist only plain JSON metadata; NumPy scalar objects otherwise break
    # weights_only checkpoint loading across NumPy/OS versions.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    write_json(args.output / "request.json", manifest)
    env, alg = None, None
    completed = 0
    started = time.perf_counter()
    outcome = dict(completed=False, **FLAGS)
    try:
        with recipe.ieee_training_precision() as (precision, guard):
            write_json(args.output / "precision.json", precision)
            env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
            wrapped = RslRlVecEnvWrapper(env, clip_actions=1000.)
            write_json(args.output / "prime.json", recipe.prime_sonic_true23_training_environment(wrapped))
            write_json(args.output / "physics_parity.json", recipe.verify_nominal_scene(
                env.sim.mj_model, spec["files"]["native_model"]["path"], args.sim_config))
            step = FootPrecisionStep(WorldQualityStep(recipe.bounded.install_progress_step(env, spec)))
            env.step = step
            driver = BFMResidualDriver(env, weights, bfm, motion)
            obs = driver.augment(wrapped.get_observations())
            alg = PPO.construct_algorithm(obs, wrapped, copy.deepcopy(algorithm_cfg), "cuda:0")
            recipe.guard_algorithm(alg, guard)
            alg.train_mode()
            actor = alg.actor
            actor.update_normalization(obs)
            alg.critic.update_normalization(obs)
            start_updates = 0
            if args.resume:
                saved = torch.load(args.resume, map_location="cuda:0", weights_only=True)
                if saved["kind"] != KIND or saved["manifest"]["contract"] != manifest["contract"]:
                    raise ValueError("incompatible BFM residual checkpoint")
                actor.load_state_dict(saved["actor_state"])
                alg.critic.load_state_dict(saved["critic_state"])
                alg.optimizer.load_state_dict(saved["optimizer_state"])
                alg.learning_rate = saved["learning_rate"]
                start_updates = saved["completed_updates"]
            else:
                with torch.inference_mode():
                    if torch.count_nonzero(actor(obs)):
                        raise ValueError("residual initialization is not exact zero")
            input_checks = [verify_input_snapshot(env, driver, obs, actor, motion, bfm, spec)]

            def save(label, count):
                value = dict(kind=KIND, manifest=manifest, completed_updates=count,
                             actor_state=actor.state_dict(), critic_state=alg.critic.state_dict(),
                             optimizer_state=alg.optimizer.state_dict(), learning_rate=alg.learning_rate,
                             elapsed_s=time.perf_counter() - started, exact_environment_resume_supported=False, **FLAGS)
                with (args.output / f"residual_{label}.pt").open("xb") as stream:
                    torch.save(value, stream)

            save(f"{start_updates:05d}", start_updates)
            print(json.dumps(dict(ready=True, actor_parameters=sum(p.numel() for p in actor.parameters()),
                                  num_envs=args.num_envs, start_updates=start_updates)), flush=True)
            with (args.output / "metrics.jsonl").open("x") as stream:
                for update in range(1, args.updates + 1):
                    tick = time.perf_counter()
                    rewards_all, root_all, feet_all, deltas, terminations = [], [], [], [], {}
                    done_count = 0
                    with torch.inference_mode():
                        for index in range(24):
                            actions = alg.act(obs)
                            values = alg.transition.values.detach().squeeze(-1).clone()
                            obs, reward, dones, extras = wrapped.step(actions)
                            obs = driver.augment(obs, dones)
                            if update == 1 and index == 0:
                                input_checks.append(verify_input_snapshot(env, driver, obs, actor, motion, bfm, spec))
                                write_json(args.output / "input_parity.json", dict(passed=True, checks=input_checks))
                            stored_index = alg.storage.step
                            alg.process_env_step(obs, reward, dones, extras)
                            row = step.rows[-1]
                            audit_reward(row, reward, alg.storage.rewards[stored_index].squeeze(-1), values, alg.gamma)
                            rewards_all.append(float(reward.mean().cpu()))
                            parts = row["world_cost_parts_before"]
                            root_all.append(float(torch.sqrt(parts[:, 0] / 10).mean()))
                            feet_all.append(float(torch.sqrt(parts[:, 1]).mean() * .05))
                            deltas.append(float(env.action_manager.get_term("joint_pos").residual_delta.square().mean().sqrt()))
                            done_count += int(dones.sum())
                            for term in env.termination_manager.active_terms:
                                terminations[term] = terminations.get(term, 0) + int(env.termination_manager.get_term(term).sum())
                            step.rows.clear()
                        alg.compute_returns(obs)
                    losses = alg.update()
                    completed = start_updates + update
                    if any(not torch.isfinite(p).all() for p in actor.parameters()):
                        raise ValueError("nonfinite residual actor weights")
                    metric = dict(update=completed, segment_update=update,
                                  elapsed_s=time.perf_counter() - started, iteration_s=time.perf_counter() - tick,
                                  mean_reward=float(np.mean(rewards_all)), root_error_mean_m=float(np.mean(root_all)),
                                  feet_rms_mean_m=float(np.mean(feet_all)), residual_rms_rad=float(np.mean(deltas)),
                                  done_fraction=done_count/(args.num_envs*24), termination_counts=terminations,
                                  learning_rate=alg.learning_rate, std_mean=float(actor.distribution.bounded_std.detach().mean()),
                                  losses={k: float(v) for k, v in losses.items()}, **FLAGS)
                    stream.write(json.dumps(metric, allow_nan=False) + "\n")
                    stream.flush()
                    if update <= 3 or update % 25 == 0:
                        print(json.dumps(metric), flush=True)
                    if completed % 200 == 0 or update == args.updates:
                        save(f"{completed:05d}", completed)
                        print(json.dumps(dict(checkpoint=completed, output=str(args.output))), flush=True)
                    if time.perf_counter() - started > args.wall_hours * 3600:
                        if completed % 200 and update != args.updates:
                            save(f"{completed:05d}", completed)
                        outcome["stop_reason"] = "requested_wall_budget"
                        break
            outcome.update(completed=True, updates=completed)
    except BaseException as error:
        outcome.update(error=repr(error), traceback=traceback.format_exc(), updates=completed)
        if alg is not None and "save" in locals():
            save("interrupted", completed)
        raise
    finally:
        outcome["elapsed_s"] = time.perf_counter() - started
        write_json(args.output / "outcome.json", outcome)
        if env is not None:
            env.close()
        print(json.dumps({k: v for k, v in outcome.items() if k != "traceback"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--updates", type=int, default=12000)
    parser.add_argument("--wall-hours", type=float, default=3.)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--sim-config", type=Path, default=ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
