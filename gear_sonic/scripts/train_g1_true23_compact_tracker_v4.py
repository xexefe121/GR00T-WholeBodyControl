"""One bounded native23 tracker run using existing PICO references; SIM only."""

import argparse
import copy
from dataclasses import asdict
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from gear_sonic.scripts import train_g1_true23_normal_lora as recipe
from gear_sonic.scripts.train_g1_true23_existing_pico import validate_bank
from gear_sonic.envs.mjlab import sonic_true23_compact_tracker as compact_env
from gear_sonic.trl.mjlab import native23_compact_actor_v2 as native23_compact_actor
from gear_sonic.utils.g1_true23_compact_features import contract, pack_observation
from gear_sonic.utils.g1_true23_foot_precision_reward import FootPrecisionStep, foot_precision_contract
from gear_sonic.utils.g1_true23_world_quality import WorldQualityStep, quality_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure

ROOT = Path(__file__).resolve().parents[2]
FLAGS = dict(deployment_ready=False, hardware_authorized=False, simulator_qualified=False)
KIND = "native23_compact_reference_tracker_training_v1"


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def observations(obs):
    # No second observation-manager call and no history advance.
    obs["native_controller"] = pack_observation(obs["policy"], obs["native_goal"])
    return obs


def agent_config():
    from mjlab.rl import RslRlModelCfg

    config = recipe.true23_mjlab_ppo_runner_cfg()
    config.actor = RslRlModelCfg(
        class_name="gear_sonic.trl.mjlab.native23_compact_actor_v2:CompactNative23ActorV2",
        hidden_dims=(256, 256, 256),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
    )
    config.critic = RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True)
    config.obs_groups = {"actor": ("native_controller",), "critic": ("critic", "native_goal")}
    config.algorithm.learning_rate = 3e-4
    config.algorithm.schedule = "adaptive"
    config.algorithm.desired_kl = 0.01
    config.algorithm.entropy_coef = 0.002
    config.algorithm.use_clipped_value_loss = False
    config.algorithm.max_grad_norm = 0.5
    config.algorithm.num_learning_epochs = 4
    config.algorithm.num_mini_batches = 4
    config.num_steps_per_env = 24
    result = asdict(config)
    for model in ("actor", "critic"):
        if result[model].pop("cnn_cfg", None) is not None:
            raise ValueError("compact recipe forbids CNN configuration")
    return result


def audit_reward(row, rewards, stored, value, gamma):
    # RewardManager and CPU torch.sum use different float32 reduction orders.
    # Bound their rounding error by gamma_n * sum(abs(components)), not an
    # arbitrary fixed epsilon near the -100 terminal reward.
    components = row["weighted_base_components"].double()
    base = components.sum(-1)
    base_error = (base - row["base_reward"].double()).abs()
    eps = torch.finfo(torch.float32).eps
    n = 2 * components.shape[-1] + 2
    base_bound = (n * eps / (1 - n * eps)) * components.abs().sum(-1).clamp_min(1.0)
    expected = (
        row["base_reward"] + row["shaping_reward"] + row["world_quality_bonus"] + row["foot_precision_bonus"]
    )
    returned_error = (expected - rewards.cpu()).abs()
    bootstrap = gamma * value.cpu() * row["timeouts"]
    stored_expected = rewards.cpu() + bootstrap
    stored_error = (stored.cpu() - stored_expected).abs()
    stored_bound = 4 * eps * (stored.cpu().abs() + rewards.cpu().abs() + bootstrap.abs()).clamp_min(1.0)
    errors = dict(
        base=float(base_error.max()), returned=float(returned_error.max()), stored=float(stored_error.max())
    )
    if (
        not torch.isfinite(base_error).all()
        or not torch.isfinite(returned_error).all()
        or not torch.isfinite(stored_error).all()
        or torch.any(base_error > base_bound)
        or torch.any(returned_error != 0)
        or torch.any(stored_error > stored_bound)
    ):
        raise ValueError("compact learner reward or timeout bootstrap mismatch: " + str(errors))
    done = row["terminated"] | row["timeouts"]
    if row["world_quality_bonus"][done].count_nonzero() or row["foot_precision_bonus"][done].count_nonzero():
        raise ValueError("compact reward paid quality bonus on a reset state")
    return errors


def run(args):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from rsl_rl.algorithms.ppo import PPO

    smoke = args.mode == "smoke"
    updates, envs, steps = (2, 8, 24) if smoke else (2000, 256, 24)
    if args.output.exists():
        raise FileExistsError("compact run refuses overwrite or implicit resume")
    bank = validate_bank(args.bank_report)
    args.spec, args.spans = Path(bank["files"]["spec"]), Path(bank["files"]["spans"])
    spec = recipe.intent.validate_spec(json.loads(args.spec.read_text()))
    rows = json.loads(args.spans.read_text())["spans"]
    args.num_envs, args.seed, args.rollout_steps, args.updates = envs, 20260910, steps, updates
    config = agent_config()
    config["multi_gpu"] = None
    env_cfg = compact_env.configure(recipe.environment_configuration(args, spec, rows), spec)
    closure = collect_local_source_closure(ROOT, [Path(__file__)])
    sources = {str(path.resolve()): sha256_file(path) for path in closure.as_source_files(ROOT).values()}
    sources[str(Path(inspect.getfile(PPO)))] = sha256_file(Path(inspect.getfile(PPO)))
    sources[str(Path(native23_compact_actor.__file__).resolve())] = sha256_file(
        Path(native23_compact_actor.__file__)
    )
    sources[str(args.experiment.resolve())] = sha256_file(args.experiment)
    source_digest = recipe.build_file_manifest(
        {f"source/{i:04d}/{Path(k).name}": Path(k) for i, k in enumerate(sorted(sources))},
        kind="source_files",
    )
    if not smoke:
        prior = json.loads(args.smoke_report.read_text())
        if (
            not prior.get("completed")
            or prior.get("updates") != 2
            or prior.get("bank_sha256") != sha256_file(args.bank_report)
            or prior.get("source_hashes") != sources
        ):
            raise ValueError("main requires completed smoke of identical code and bank")
    if shutil.disk_usage(args.output.parent).free < 700_000_000:
        raise RuntimeError("compact run needs at least700MB free; no old evidence removed")
    args.output.mkdir(parents=True)
    manifest = dict(
        kind=KIND,
        contract=contract(),
        bank_sha256=sha256_file(args.bank_report),
        bank_report=str(args.bank_report),
        source_hashes=sources,
        source_manifest=source_digest,
        agent=copy.deepcopy(config),
        physics_sha256=sha256_file(args.sim_config),
        spec_sha256=sha256_file(args.spec),
        seed=args.seed,
        num_envs=envs,
        rollout_steps=steps,
        planned_updates=updates,
        milestones=[0, 2] if smoke else [0, 200, 500, 1000, 2000],
        goal_reference_timing=recipe.NORMAL_TIMING,
        objective=dict(
            bounded=recipe.normal_training_contract()["bounded_reward"],
            world_quality=quality_contract(),
            foot_precision=foot_precision_contract(),
        ),
        no_frozen_sonic_weight_used=True,
        no_teacher_labels_used=True,
        start_schedule="mixed_reference_reset_v1",
        **FLAGS,
    )
    write_json(args.output / "request.json", manifest)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    os.environ["MUJOCO_GL"], os.environ["MUJOCO_EGL_DEVICE_ID"] = "egl", "0"
    started = time.perf_counter()
    env, alg = None, None
    completed = 0
    audit_max = dict(base=0.0, returned=0.0, stored=0.0)
    sampled = []
    outcome = dict(completed=False, source_hashes=sources, bank_sha256=manifest["bank_sha256"], **FLAGS)
    try:
        with recipe.ieee_training_precision() as (precision, guard):
            write_json(args.output / "precision.json", precision)
            print("compact: constructing unchanged native23 physics", flush=True)
            env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
            wrapped = RslRlVecEnvWrapper(env, clip_actions=9.999)
            write_json(args.output / "prime.json", recipe.prime_sonic_true23_training_environment(wrapped))
            write_json(
                args.output / "physics_parity.json",
                recipe.verify_nominal_scene(
                    env.sim.mj_model, spec["files"]["native_model"]["path"], args.sim_config
                ),
            )
            write_json(
                args.output / "action_parity.json",
                recipe.verify_executed_release_environment(env, spec["files"]["source_model"]["path"]),
            )
            write_json(args.output / "source_parity.json", recipe.verify_normal_source_environment(env, spec))
            recipe.world_failure.verify_runtime(env)
            progress = recipe.bounded.install_progress_step(env, spec)
            reward_step = FootPrecisionStep(WorldQualityStep(progress))
            env.step = reward_step
            obs = observations(wrapped.get_observations())
            alg = PPO.construct_algorithm(obs, wrapped, copy.deepcopy(config), "cuda:0")
            recipe.guard_algorithm(alg, guard)
            alg.train_mode()
            actor = alg.get_policy()
            actor.update_normalization(obs)
            alg.critic.update_normalization(obs)
            actor_count = sum(p.numel() for p in actor.parameters())
            print(
                json.dumps(dict(actor_parameters=actor_count, num_envs=envs, planned_updates=updates)), flush=True
            )

            def save(count):
                if shutil.disk_usage(args.output).free < 300_000_000:
                    raise RuntimeError("compact training storage guard reached")
                value = dict(
                    kind=KIND,
                    manifest=manifest,
                    completed_updates=count,
                    actor_state=actor.state_dict(),
                    critic_state=alg.critic.state_dict(),
                    optimizer_state=alg.optimizer.state_dict(),
                    learning_rate=alg.learning_rate,
                    elapsed_s=time.perf_counter() - started,
                    **FLAGS,
                )
                checkpoint = args.output / f"compact_{count:04d}.pt"
                with checkpoint.open("xb") as stream:
                    torch.save(value, stream)
                write_json(
                    args.output / f"checkpoint_{count:04d}.json",
                    dict(path=str(checkpoint), sha256=sha256_file(checkpoint), updates=count, **FLAGS),
                )

            save(0)
            with (args.output / "metrics.jsonl").open("x") as metrics_stream:
                for update in range(1, updates + 1):
                    tick = time.perf_counter()
                    returns, dones_count, roots, feet_cost = [], 0, [], []
                    with torch.inference_mode():
                        for step in range(steps):
                            if step == 0 and (update <= 2 or update % 200 == 0):
                                command = env.command_manager.get_term("motion")
                                sampled.append(
                                    dict(
                                        update=update,
                                        **{
                                            k: obs[k][:8].cpu().numpy().copy()
                                            for k in ("policy", "native_goal", "native_controller")
                                        },
                                        qpos_hw=command.robot_joint_pos[:8].cpu().numpy().copy(),
                                        root_position_w=command.robot_anchor_pos_w[:8].cpu().numpy().copy(),
                                        root_quaternion_wxyz=command.robot_anchor_quat_w[:8].cpu().numpy().copy(),
                                        root_velocity_w=command.robot_anchor_lin_vel_w[:8].cpu().numpy().copy(),
                                        env_origins=env.scene.env_origins[:8].cpu().numpy().copy(),
                                        dqpos_hw=command.robot_joint_vel[:8].cpu().numpy().copy(),
                                        reference_q0=command.time_steps[:8].cpu().numpy().copy(),
                                    )
                                )
                            actions = alg.act(obs)
                            if not torch.isfinite(actions).all() or actions.abs().max() >= 10:
                                raise ValueError("compact stochastic request exceeds unchanged raw action guard")
                            values = alg.transition.values.detach().squeeze(-1).clone()
                            obs, rewards, dones, extras = wrapped.step(actions)
                            obs = observations(obs)
                            index = alg.storage.step
                            alg.process_env_step(obs, rewards, dones, extras)
                            row = reward_step.rows[-1]
                            stored = alg.storage.rewards[index].squeeze(-1)
                            for name, value in audit_reward(row, rewards, stored, values, alg.gamma).items():
                                audit_max[name] = max(audit_max[name], value)
                            if step == 0 and (update <= 2 or update % 200 == 0):
                                sampled[-1].update(
                                    actions=actions[:8].cpu().numpy().copy(),
                                    **{"reward_" + k: v[:8].numpy().copy() for k, v in row.items()},
                                )
                            returns.append(float(rewards.mean().cpu()))
                            dones_count += int(dones.sum().cpu())
                            parts = row["world_cost_parts_before"]
                            roots.append(float(torch.sqrt(parts[:, 0] / 10).mean()))
                            feet_cost.append(float(torch.sqrt(parts[:, 1]).mean() * 0.05))
                            # This run saves selected raw rows and all aggregate metrics, not unbounded tensors.
                            reward_step.rows.clear()
                        alg.compute_returns(obs)
                    losses = alg.update()
                    completed = update
                    if any(not torch.isfinite(p).all() for p in actor.parameters()):
                        raise ValueError("compact parameters became nonfinite")
                    metric = dict(
                        update=update,
                        transitions=update * envs * steps,
                        elapsed_s=time.perf_counter() - started,
                        iteration_s=time.perf_counter() - tick,
                        mean_reward=float(np.mean(returns)),
                        done_fraction=dones_count / (envs * steps),
                        root_error_mean_m=float(np.mean(roots)),
                        feet_rms_mean_m=float(np.mean(feet_cost)),
                        learning_rate=alg.learning_rate,
                        action_std_mean=float(actor.distribution.bounded_std.detach().mean().cpu()),
                        losses={k: float(v) for k, v in losses.items()},
                        **FLAGS,
                    )
                    metrics_stream.write(json.dumps(metric, allow_nan=False) + "\n")
                    metrics_stream.flush()
                    if update <= 2 or update % 25 == 0:
                        print(json.dumps(metric), flush=True)
                    if update in manifest["milestones"]:
                        save(update)
                    if time.perf_counter() - started > 4 * 3600:
                        raise RuntimeError("compact four-hour wall cap reached")
            outcome.update(completed=True, actor_parameters=actor_count)
    except BaseException as error:
        outcome.update(error=repr(error), traceback=traceback.format_exc())
        if alg is not None:
            partial = dict(
                kind=KIND,
                manifest=manifest,
                completed_updates=completed,
                actor_state=alg.actor.state_dict(),
                critic_state=alg.critic.state_dict(),
                optimizer_state=alg.optimizer.state_dict(),
                learning_rate=alg.learning_rate,
                partial_rollout_controls=alg.storage.step,
                exact_environment_resume_supported=False,
                **FLAGS,
            )
            with (args.output / "interrupted_snapshot.pt").open("xb") as stream:
                torch.save(partial, stream)
        raise
    finally:
        outcome.update(
            updates=completed,
            transitions=completed * envs * steps,
            elapsed_s=time.perf_counter() - started,
            reward_audit_max_abs=audit_max,
            sampled_controls=len(sampled),
        )
        if sampled:
            common = set.intersection(*(set(row) for row in sampled))
            with (args.output / "sampled_inputs_rewards.npz").open("xb") as stream:
                np.savez_compressed(stream, **{k: np.asarray([row[k] for row in sampled]) for k in common})
        write_json(args.output / "outcome.json", outcome)
        if env is not None:
            env.close()
        print(
            json.dumps({k: v for k, v in outcome.items() if k not in ("source_hashes", "traceback")}), flush=True
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "main"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument(
        "--bank-report",
        type=Path,
        default=ROOT / "artifacts/g1_true23_pico_training_20260910_v1/bank_v1/report.json",
    )
    parser.add_argument(
        "--experiment", type=Path, default=ROOT / "artifacts/g1_true23_compact_tracker_20260910_v1/EXPERIMENT.md"
    )
    parser.add_argument(
        "--sim-config", type=Path, default=ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
    )
    args = parser.parse_args()
    if args.mode == "main" and args.smoke_report is None:
        parser.error("main requires --smoke-report")
    run(args)


if __name__ == "__main__":
    main()
