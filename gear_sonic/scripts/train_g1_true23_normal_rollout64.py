"""Bounded normal-core rollout64 experiment; original rollout16 source is frozen.

Same actor, reward, limits and204800-transition100-update budget.32physical
worlds x64controls replaces128 x16. This changes rollout geometry and sampling,
not only horizon; full-motion evaluation is mandatory. Capture actual final
bootstrap values and PPO returns without an additional critic forward pass.
"""

import argparse
from dataclasses import asdict, dataclass
import inspect
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from gear_sonic.envs.mjlab import (
    sonic_true23_bounded_progress as bounded,
    sonic_true23_original_intent as intent,
    sonic_true23_world_tracking_termination as world_failure,
)
from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment
from gear_sonic.envs.mjlab.sonic_true23_buffered_source import configure_buffered_source_environment
from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import make_causal_multimotion_v14_env_cfg
from gear_sonic.envs.mjlab.sonic_true23_native_model_actuation import apply_native_model_actuation_profile
from gear_sonic.envs.mjlab.sonic_true23_nominal_scene import configure_nominal_scene, verify_nominal_scene
from gear_sonic.envs.mjlab.sonic_true23_normal_source import (
    configure_normal_source_environment,
    verify_normal_source_environment,
)
from gear_sonic.envs.mjlab.sonic_true23_release_compatible import (
    configure_release_compatible_environment,
    verify_executed_release_environment,
)
from gear_sonic.envs.mjlab.sonic_true23_root_feedback import (
    configure_root_feedback_environment,
    install_root_feedback_command,
)
from gear_sonic.scripts.train_g1_23dof_mjlab import (
    UNITREE_ROOT,
    _manifest_files,
    _resolved_training_config,
    _source_files,
)
from gear_sonic.scripts.train_g1_23dof_mjlab_teleop_v13 import _install_corpus_command
from gear_sonic.trl.mjlab.config import True23SonicActorCfg, true23_mjlab_ppo_runner_cfg
from gear_sonic.trl.mjlab.native23_normal_lora_runner import (
    Native23NormalLoraRunner,
    normal_training_contract,
    validate_normal_checkpoint,
)
from gear_sonic.utils.g1_23dof_contract import REFERENCE_PROFILE_NORMAL
from gear_sonic.utils.g1_23dof_mjlab_training import build_file_manifest
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_normal_reference import NORMAL_TIMING, normal_reference_contract
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_ACTION_CONVENTION
from gear_sonic.utils.g1_true23_training_precision import guard_algorithm, ieee_training_precision, write_runtime

ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_SHA256 = "f11da3df8ca218bd50eccb9e3752dfa101e3b65dfa593bb20fe2abb5fbd2e30e"


@dataclass
class NormalActorCfg(True23SonicActorCfg):
    class_name: str = "gear_sonic.trl.mjlab.native23_normal_lora_actor:True23NormalLoraActorModel"
    source_checkpoint_path: str = "sonic_release/last.pt"
    root_feedback_obs_group: str = "root_feedback"
    reference_timing: str = NORMAL_TIMING
    std_min: float = 0.02
    std_max: float = 0.5


def agent_configuration(args):
    cfg = true23_mjlab_ppo_runner_cfg()
    cfg.actor = NormalActorCfg(
        warm_start_path=str(args.warm_start),
        source_checkpoint_path=str(args.source_checkpoint),
        distribution_cfg=dict(class_name="GaussianDistribution", init_std=0.1, std_type="scalar"),
    )
    cfg.obs_groups = {
        "actor": ("tokenizer", "policy", "root_feedback"),
        "critic": ("critic", "root_feedback", "original_intent_value_reference"),
    }
    cfg.experiment_name = "sonic_native23_normal_core_lora_simulation"
    cfg.run_name = "original_core_920ms_fresh"
    cfg.seed = args.seed
    cfg.max_iterations = cfg.save_interval = args.updates
    cfg.num_steps_per_env = args.rollout_steps
    cfg.algorithm.learning_rate = 5e-7
    cfg.algorithm.schedule = "fixed"
    cfg.algorithm.desired_kl = None
    cfg.algorithm.entropy_coef = 0.002
    cfg.algorithm.max_grad_norm = 0.5
    cfg.algorithm.use_clipped_value_loss = False
    cfg.algorithm.num_learning_epochs = 2
    cfg.algorithm.num_mini_batches = 4
    return cfg


def environment_configuration(args, spec, rows):
    _install_corpus_command(args.spans)
    install_root_feedback_command(rows, start_schedule="mixed_reference_reset_v1")
    cfg = make_causal_multimotion_v14_env_cfg(
        motion_file=spec["files"]["native_motion"]["path"], num_envs=args.num_envs, play=False
    )
    cfg = apply_native_model_actuation_profile(cfg, NativeModelActuationProfile.from_sim_config(args.sim_config))
    cfg = configure_root_feedback_environment(
        cfg,
        rows,
        reset_position_range_m=0.1,
        reset_velocity_range_m_s=0.1,
        objective_profile="root_and_upper_feet_world_v4",
    )
    cfg = configure_release_compatible_environment(
        cfg, spec["files"]["source_model"]["path"], SOURCE_ACTION_CONVENTION
    )
    cfg = configure_nominal_scene(cfg, spec["files"]["native_model"]["path"], args.sim_config)
    cfg = configure_buffered_source_environment(cfg)
    cfg = intent.configure_original_intent_environment(cfg, spec)
    cfg = configure_normal_source_environment(cfg)
    cfg = bounded.configure_environment(cfg)
    cfg = world_failure.configure_environment(cfg)
    cfg.seed = args.seed
    return cfg


def material_manifests(args, spec):
    from rsl_rl.algorithms.ppo import PPO

    closure = collect_local_source_closure(ROOT, [Path(__file__)])
    sources = {**_source_files(), **closure.as_source_files(ROOT)}
    sources["external/rsl_rl/algorithms/ppo.py"] = Path(inspect.getfile(PPO))
    sources["normal/preflight.json"] = args.preflight
    sources["normal/execution_diagnostic.json"] = args.execution_diagnostic
    sources["normal/rollout_diagnostic.json"] = args.rollout_diagnostic
    assets = _manifest_files(UNITREE_ROOT / "src/assets/robots/unitree_g1", logical_prefix="unitree_g1")
    # The nominal scene uses original MJCF meshes, not just the construction assets.
    import mujoco

    for key in ("native_model", "source_model"):
        path = Path(spec["files"][key]["path"])
        assets[f"original/{key}.xml"] = path
        model = mujoco.MjSpec.from_file(str(path))
        for i, mesh in enumerate(model.meshes):
            assets[f"original/{key}/mesh_{i:03d}"] = path.parent / model.meshdir / mesh.file
    assets["physics/sim_config.json"] = args.sim_config
    dataset = {
        f"reference/{key}": Path(value["path"]) for key, value in spec["files"].items() if "model" not in key
    }
    dataset.update({"reference/spec.json": args.spec, "reference/spans.json": args.spans})
    return (
        build_file_manifest(sources, kind="source_files"),
        build_file_manifest(assets, kind="robot_assets"),
        build_file_manifest(dataset, kind="motion_dataset"),
    )


def copied_groups(runner):
    return {g["name"]: [p.detach().cpu().clone() for p in g["params"]] for g in runner.alg.optimizer.param_groups}


def install_learning_capture(runner):
    gradient_rows, input_rows = [], []
    original_step = runner.alg.optimizer.step
    original_act = runner.alg.act

    def step(*args, **kwargs):
        norms = {}
        for group in runner.alg.optimizer.param_groups:
            values = [p.grad.detach() for p in group["params"] if p.grad is not None]
            if not values or not all(torch.isfinite(v).all() for v in values):
                raise ValueError("normal training group missing finite gradients")
            norms[group["name"]] = float(torch.sqrt(sum(v.double().square().sum() for v in values)).cpu())
        gradient_rows.append(norms)
        return original_step(*args, **kwargs)

    def act(obs):
        command = runner.env.unwrapped.command_manager.get_term("motion")
        row = {name: obs[name].detach().cpu().clone() for name in ("tokenizer", "policy", "root_feedback")}
        row.update(
            reference_q0=command.time_steps.detach().cpu().clone(),
            measured_root_position_w=command.robot_anchor_pos_w.detach().cpu().clone(),
            measured_root_quaternion_wxyz=command.robot_anchor_quat_w.detach().cpu().clone(),
        )
        actions = original_act(obs)
        row["sampled_raw_action23"] = actions.detach().cpu().clone()
        input_rows.append(row)
        return actions

    runner.alg.optimizer.step = step
    runner.alg.act = act
    return gradient_rows, input_rows


def install_returns_capture(runner):
    """Observe the real PPO bootstrap forward and stored returns, without recomputation."""
    rows = []
    original = runner.alg.compute_returns

    def compute(obs):
        seen = []
        hook = runner.alg.critic.register_forward_hook(
            lambda module, args, output: seen.append(output.detach().cpu().numpy().copy())
        )
        try:
            result = original(obs)
        finally:
            hook.remove()
        if len(seen) != 1:
            raise ValueError("PPO return capture expected exactly one actual critic bootstrap")
        storage = runner.alg.storage
        rows.append(
            dict(
                last_values=seen[0],
                values=storage.values.detach().cpu().numpy().copy(),
                returns=storage.returns.detach().cpu().numpy().copy(),
                normalized_advantages=storage.advantages.detach().cpu().numpy().copy(),
                rewards=storage.rewards.detach().cpu().numpy().copy(),
                dones=storage.dones.detach().cpu().numpy().copy(),
            )
        )
        return result

    runner.alg.compute_returns = compute
    return rows


def run(args):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    if (args.updates, args.num_envs, args.rollout_steps) not in ((2, 4, 64), (100, 32, 64)):
        raise ValueError("rollout64 run limited to2x4x64 smoke or100x32x64 full experiment")
    diagnostic = json.loads(args.execution_diagnostic.read_text())
    rollout_diagnostic = json.loads(args.rollout_diagnostic.read_text())
    if (
        diagnostic.get("kind") != "native23_normal_cpu_gpu_saved_observation_audit_v1"
        or not diagnostic.get("completed")
        or rollout_diagnostic.get("kind") != "normal16_actual_rollout_and_failure_return_audit_v1"
        or not rollout_diagnostic.get("completed")
        or rollout_diagnostic.get("reward_shape") != [1600, 128]
    ):
        raise ValueError("rollout64 requires the completed original numerical and rollout diagnostics")
    if args.num_envs * args.rollout_steps % 4:
        raise ValueError("rollout must divide into four PPO minibatches")
    if sha256_file(args.preflight) != PREFLIGHT_SHA256:
        raise ValueError("normal source-weight/gradient preflight differs")
    if args.run_dir.exists():
        raise FileExistsError("normal training refuses an existing run directory")
    spec = intent.validate_spec(json.loads(args.spec.read_text()))
    spans = json.loads(args.spans.read_text())
    rows = spans["spans"]
    motion_path = Path(spec["files"]["native_motion"]["path"])
    agent = agent_configuration(args)
    env_cfg = environment_configuration(args, spec, rows)
    sources, assets, dataset = material_manifests(args, spec)
    args.run_dir.mkdir(parents=True)
    write_runtime(
        args.run_dir / "request.json",
        {
            "arguments": {
                key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
            },
            "preflight_sha256": PREFLIGHT_SHA256,
            "reference": normal_reference_contract(),
            "pico_used_for_training": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        },
    )
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    env = runner = None
    gradients, inputs, returns_capture = [], [], []
    outcome = dict(
        kind="native23_normal_rollout64_actual_simulator_training_v1",
        completed=False,
        simulator_updates=0,
        full_motion_tracking_verified=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    try:
        with ieee_training_precision() as (precision, guard):
            resolved = _resolved_training_config(
                agent_cfg=agent,
                mode="smoke" if args.updates == 2 else "bounded_fresh_training",
                motion_path=motion_path,
                num_envs=args.num_envs,
                seed=args.seed,
                planned_updates=args.updates,
                reference_profile=NORMAL_TIMING,
            )
            resolved.update(
                schema="g1_true23_normal_core_received_920ms_rollout64_training_v1",
                rollout_geometry_comparison=dict(
                    previous_num_envs=128,
                    previous_rollout_steps=16,
                    requested_num_envs=args.num_envs,
                    requested_rollout_steps=args.rollout_steps,
                    full_run_transition_budget=204800,
                    same_parameter_initialization_claimed_before_comparison=False,
                    one_variable_ablation_claimed=False,
                ),
                task="Unitree-G1-Native23-OriginalSONIC-Received920ms",
                normal_adaptation=normal_training_contract(),
                original_intent_spec=spec,
                reference=normal_reference_contract(),
                precision=precision,
                source_weight_profile=REFERENCE_PROFILE_NORMAL,
                randomize_initial_episode_lengths=False,
                push_disturbances_enabled=False,
                domain_randomization=[],
                actual_event_names=list(env_cfg.events),
                start_schedule="mixed_reference_reset_v1",
                reset_xy_position_range_m=0.1,
                reset_xy_velocity_range_m_s=0.1,
                pico_used_for_training=False,
                optimizer_or_critic_reused=False,
                low_latency_checkpoint_relabelled=False,
                deployment_ready=False,
                hardware_authorized=False,
            )
            write_runtime(args.run_dir / "resolved_training.json", resolved)
            print("normal: constructing nominal native23 environment", flush=True)
            env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            prime = prime_sonic_true23_training_environment(wrapped)
            error = float(intent.ee_height_error(env, spec).max().cpu())
            if error > 0.25:
                raise ValueError("normal original-intent prime contains terminal hand/foot target")
            prime.update(
                legacy_ee_error_uses_native_body_origins=True, original_hand_foot_height_error_max_m=error
            )
            write_runtime(args.run_dir / "environment_prime.json", prime)
            parity = verify_nominal_scene(env.sim.mj_model, spec["files"]["native_model"]["path"], args.sim_config)
            write_runtime(args.run_dir / "nominal_scene_parity.json", parity)
            previous_probe = verify_executed_release_environment(env, spec["files"]["source_model"]["path"])
            action_probe = {
                key: previous_probe[key]
                for key in (
                    "native_joint_count",
                    "probe_count",
                    "target_max_abs_error_rad",
                    "normalized_history_max_abs_error",
                    "action_convention",
                    "physical_state_mutated",
                    "physics_steps",
                )
            }
            write_runtime(args.run_dir / "action_history_parity.json", action_probe)
            write_runtime(args.run_dir / "normal_source_parity.json", verify_normal_source_environment(env, spec))
            world_failure.verify_runtime(env)
            print("normal: reference/physics parity passed; constructing frozen core and adapters", flush=True)
            runner = Native23NormalLoraRunner(
                wrapped,
                asdict(agent),
                str(args.run_dir),
                "cuda:0",
                warm_start_checkpoint_path=args.warm_start,
                resolved_config=resolved,
                source_manifest=sources,
                asset_manifest=assets,
                dataset_manifest=dataset,
                checkpoint_dir=args.run_dir / "checkpoints",
            )
            guard_algorithm(runner.alg, guard)
            write_runtime(args.run_dir / "lineage.json", runner.training_lineage)
            actor = runner.alg.get_policy()
            fixed_obs = wrapped.get_observations().clone()
            with torch.no_grad():
                mean_before = actor(fixed_obs).detach().cpu().clone()
                semantic = fixed_obs["tokenizer"][..., 1:]
                base = actor.core.decoder(
                    torch.cat(
                        (actor.core.encode(semantic), actor.core.codec.encode_proprioception(fixed_obs["policy"])),
                        -1,
                    )
                )
                if not torch.equal(mean_before, base.cpu()):
                    raise ValueError("actual simulator normal zero-adapter mean differs from original base")
            groups_before = copied_groups(runner)
            gradients, inputs = install_learning_capture(runner)
            returns_capture = install_returns_capture(runner)
            print(
                f"normal: training {args.updates} updates x {args.num_envs} envs x {args.rollout_steps} controls",
                flush=True,
            )
            runner.learn(args.updates, init_at_random_ep_len=False)
            with torch.no_grad():
                mean_after = actor(fixed_obs).detach().cpu().clone()
            changes = {
                name: max(
                    float((a - b).abs().max()) for a, b in zip(values, copied_groups(runner)[name], strict=True)
                )
                for name, values in groups_before.items()
            }
            if not all(np.isfinite(value) and value > 0 for value in changes.values()):
                raise ValueError("actual simulator update failed to change each trainable group")
            actor.core.assert_frozen_encoder_unchanged()
            final_path = runner._numbered_checkpoint_path(args.updates)
            checkpoint = torch.load(final_path, map_location="cpu", weights_only=True)
            validate_normal_checkpoint(checkpoint, actor=actor, lineage=runner.training_lineage)
            np.savez_compressed(
                args.run_dir / "fixed_observation_mean_change.npz",
                before=mean_before.numpy(),
                after=mean_after.numpy(),
                **{key: value.cpu().numpy() for key, value in fixed_obs.items()},
            )
            outcome.update(
                completed=True,
                simulator_updates=runner.completed_update_count,
                transitions=len(inputs) * args.num_envs,
                controls_per_env=len(inputs),
                optimizer_steps=len(gradients),
                maximum_parameter_change_by_group=changes,
                fixed_observation_mean_max_abs_change=float((mean_after - mean_before).abs().max()),
                initial_actual_observation_mean_bit_exact=True,
                frozen_base_unchanged=True,
                checkpoint=str(final_path),
                checkpoint_sha256=sha256_file(final_path),
            )
            print(json.dumps(outcome), flush=True)
    except BaseException as exc:
        outcome.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        if runner is not None:
            outcome["simulator_updates"] = runner.completed_update_count
            capture = runner._reward_step.capture()
            if capture:
                np.savez_compressed(args.run_dir / "reward_capture.npz", **capture)
            failures = env._world_root_failure_capture
            if failures:
                np.savez_compressed(
                    args.run_dir / "world_failure_capture.npz",
                    **{
                        key: np.stack(
                            [
                                row[key].numpy() if isinstance(row[key], torch.Tensor) else row[key]
                                for row in failures
                            ]
                        )
                        for key in failures[0]
                    },
                )
        if inputs:
            np.savez_compressed(
                args.run_dir / "actual_inputs.npz",
                **{key: torch.stack([row[key] for row in inputs]).numpy() for key in inputs[0]},
            )
        if returns_capture:
            np.savez_compressed(
                args.run_dir / "returns_capture.npz",
                **{key: np.stack([row[key] for row in returns_capture]) for key in returns_capture[0]},
            )
        write_runtime(args.run_dir / "gradient_capture.json", gradients)
        write_runtime(args.run_dir / "outcome.json", outcome)
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "spec",
        "spans",
        "sim-config",
        "warm-start",
        "source-checkpoint",
        "preflight",
        "run-dir",
        "execution-diagnostic",
        "rollout-diagnostic",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve(strict=key != "run_dir"))
    run(args)


if __name__ == "__main__":
    main()
