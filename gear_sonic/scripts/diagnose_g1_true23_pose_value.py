"""Actual native23 SIM rollouts plus disposable critic-only value-clip probes.

Reuses the checked pose-LoRA environment constructor in a dedicated process.
Checkpoint actor/critic/Adam state is loaded only for diagnostic evaluation;
this is NOT a training resume, and no new actor checkpoint may be written.
"""

import argparse
import copy
import inspect
import json
from pathlib import Path

import torch

from gear_sonic.scripts import train_g1_true23_pose_lora as pose
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256
from gear_sonic.trl.mjlab.native23_pose_lora_runner import Native23PoseLoraRunner, validate_pose_lora_checkpoint
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_pose_lora_checkpoint import validate_semantics
from gear_sonic.utils.g1_true23_training_precision import write_runtime
from gear_sonic.utils.g1_true23_value_diagnostic import (
    calibration,
    cpu_state,
    critic_adam_state,
    fit_probe,
    stats,
    verify_gae,
)


def file_row(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def capture_block(runner, obs):
    alg, env = runner.alg, runner.env.unwrapped
    command = env.command_manager.get_term("motion")
    reward = env.reward_manager
    scale = float(env.step_dt) if reward._scale_by_dt else 1.0
    saved = {
        name: []
        for name in (
            "raw_rewards",
            "timeouts",
            "reward_components",
            "source_anchors",
            "measured_joint_position",
            "measured_root_position_w",
            "measured_root_quaternion_wxyz",
        )
    }
    critic_before = cpu_state(alg.critic)
    for _ in range(runner.cfg["num_steps_per_env"]):
        with torch.inference_mode():
            saved["source_anchors"].append(command.time_steps.detach().cpu().clone())
            saved["measured_joint_position"].append(command.robot_joint_pos.detach().cpu().clone())
            saved["measured_root_position_w"].append(command.robot_anchor_pos_w.detach().cpu().clone())
            saved["measured_root_quaternion_wxyz"].append(command.robot_anchor_quat_w.detach().cpu().clone())
            actions = alg.act(obs)
            obs, rewards, dones, extras = runner.env.step(actions.to(runner.env.device))
            obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
            if any(not value.isfinite().all() for value in obs.values()) or not rewards.isfinite().all():
                raise ValueError("nonfinite diagnostic rollout")
            saved["raw_rewards"].append(rewards.detach().cpu().clone().reshape(-1, 1))
            saved["timeouts"].append(
                extras.get("time_outs", torch.zeros_like(dones)).detach().cpu().clone().reshape(-1, 1)
            )
            components = reward._step_reward.detach().cpu().clone() * scale
            torch.testing.assert_close(components.sum(-1), rewards.cpu(), atol=1e-4, rtol=3e-6)
            saved["reward_components"].append(components)
            # Exact training behavior: update input normalization on next obs,
            # then stock timeout bootstrapping and transition storage.
            alg.process_env_step(obs, rewards, dones, extras)
            extras["log"] = {}
    with torch.inference_mode():
        last_values = alg.critic(obs).detach().cpu().clone()
        alg.compute_returns(obs)
    storage = alg.storage
    block = {name: torch.stack(values) for name, values in saved.items()}
    block.update(
        {
            name: getattr(storage, name).detach().cpu().clone()
            for name in (
                "rewards",
                "values",
                "returns",
                "advantages",
                "dones",
                "actions",
                "actions_log_prob",
            )
        }
    )
    block.update(
        observations={key: value.detach().cpu().clone() for key, value in storage.observations.items()},
        last_observations={key: value.detach().cpu().clone() for key, value in obs.items()},
        last_values=last_values,
        critic_before=critic_before,
        critic_after=cpu_state(alg.critic),
        reward_component_names=list(reward._term_names),
        normalization_mode="stock_process_env_step_updates_input_statistics_no_weight_updates",
        physical_samples="pre_control_manager_state_same_as_training_not_substep_replay",
    )
    storage.clear()
    return block, obs


def run_diagnostic(runner, options):
    output = options.diagnostic_output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = options.diagnostic_checkpoint.resolve(strict=True)
    checkpoint_row = file_row(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    semantics = validate_semantics(checkpoint)
    actor, critic = runner.alg.actor, runner.alg.critic
    validate_pose_lora_checkpoint(checkpoint, actor=actor, lineage=checkpoint["lineage"])
    existing = semantics["resolved"]
    fresh = runner.training_lineage["materials"]["resolved_config"]["payload"]
    for key in (
        "agent",
        "action_count",
        "policy_dim",
        "tokenizer_dim",
        "semantic_profile",
        "seed",
        "num_envs",
        "domain_randomization",
        "recovery",
        "push_disturbances_enabled",
        "history_length",
        "initial_simulation_prime_steps",
        "randomize_initial_episode_lengths",
    ):
        if fresh.get(key) != existing.get(key):
            raise ValueError(f"diagnostic environment factory differs from checkpoint: {key}")
    for key in (
        "objective_contract",
        "objective_profile_contract",
        "feature_contract",
        "reference_timing",
        "release_compatibility",
        "training_physics",
        "start_schedule",
        "reset_position_range_m",
        "reset_velocity_range_m_s",
    ):
        if fresh["native23_root_feedback"].get(key) != existing["native23_root_feedback"].get(key):
            raise ValueError(f"diagnostic task differs from checkpoint: {key}")
    old_files = semantics["spec"]["files"]
    new_files = fresh["native23_root_feedback"]["original_intent_spec"]["files"]
    if {k: v["sha256"] for k, v in old_files.items()} != {k: v["sha256"] for k, v in new_files.items()}:
        raise ValueError("diagnostic source and physical geometry differ from checkpoint")
    actor.load_training_artifact(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    runner.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    actor_hash = _state_sha256(cpu_state(actor))
    critic_parameters_hash = _state_sha256(dict(critic.named_parameters()))
    optimizer_snapshot = critic_adam_state(runner.alg.optimizer)
    initial_anchors = runner.env.unwrapped.command_manager.get_term("motion").time_steps.cpu().tolist()
    root = Path(__file__).resolve().parents[2]
    closure = collect_local_source_closure(root, [Path(__file__)])
    source_rows = [file_row(value) for value in closure.as_source_files(root).values()]
    from rsl_rl.algorithms import PPO
    from rsl_rl.storage import RolloutStorage

    external_rows = [
        file_row(inspect.getfile(value))
        for value in (
            PPO,
            RolloutStorage,
            type(critic),
            type(critic.obs_normalizer),
            type(runner.env.unwrapped.reward_manager),
        )
    ]
    contract = {
        "kind": "native23_pose_value_learning_diagnostic_v1",
        "checkpoint": checkpoint_row,
        "checkpoint_updates": checkpoint["trainer_state"]["completed_update_count"],
        "checkpoint_lineage_sha256": checkpoint["lineage_sha256"],
        "factory_lineage_sha256": runner.lineage_sha256,
        "seed": fresh.get("seed"),
        "initial_anchors": initial_anchors,
        "blocks": options.diagnostic_blocks,
        "environments": runner.env.num_envs,
        "steps_per_block": runner.cfg["num_steps_per_env"],
        "normalization_mode": "stock_process_env_step_updates_input_statistics_no_weight_updates",
        "probe": "fixed_rollout_returns_two_epochs_four_identical_minibatches_clipped_vs_unclipped_critic_only",
        "sources": source_rows,
        "external_sources": external_rows,
        "training_resume_claimed": False,
        "actor_updates": 0,
        "deployable_snapshot_written": False,
        "deployment_ready": False,
        "hardware_authorized": False,
        "interpretation": (
            "Local optimizer constraint; not generalization, control improvement or full-motion qualification."
        ),
    }
    if runner.env.num_envs != 32 or runner.cfg["num_steps_per_env"] != 64:
        raise ValueError("diagnostic requires the actual32x64 regression batch geometry")
    alg = runner.alg
    if alg.num_learning_epochs != 2 or alg.num_mini_batches != 4 or not alg.use_clipped_value_loss:
        raise ValueError("diagnostic requires the actual clipped2x4 PPO configuration")
    contract["algorithm"] = {
        name: getattr(alg, name)
        for name in (
            "gamma",
            "lam",
            "clip_param",
            "max_grad_norm",
            "value_loss_coef",
            "num_learning_epochs",
            "num_mini_batches",
            "use_clipped_value_loss",
            "normalize_advantage_per_mini_batch",
        )
    }
    write_runtime(output / "contract.json", contract)
    print(
        json.dumps(
            {
                "diagnostic_preflight_complete": True,
                "checkpoint_updates": contract["checkpoint_updates"],
                "planned_transitions": options.diagnostic_blocks * runner.env.num_envs * 64,
            }
        ),
        flush=True,
    )

    # No actor/critic optimizer call may accidentally turn this into training.
    def forbidden(*args, **kwargs):
        raise RuntimeError("diagnostic forbids actual PPO updates and checkpoint saves")

    alg.update = forbidden
    alg.optimizer.step = forbidden
    alg.train_mode()
    obs = runner.env.get_observations().to(runner.device)
    reports = []
    for index in range(options.diagnostic_blocks):
        block, obs = capture_block(runner, obs)
        block["critic_optimizer_state"] = copy.deepcopy(optimizer_snapshot)
        report = {
            "block": index,
            "gae": verify_gae(block, alg.gamma, alg.lam),
            "stored_value_calibration": calibration(block["values"], block["returns"]),
            "raw_rewards": stats(block["raw_rewards"]),
            "done_count": int(block["dones"].sum()),
            "timeout_count": int(block["timeouts"].sum()),
            "reward_components": {
                name: stats(block["reward_components"][..., j])
                for j, name in enumerate(block["reward_component_names"])
            },
        }
        observations = {key: value.flatten(0, 1) for key, value in block["observations"].items()}
        values, returns = block["values"].flatten(0, 1), block["returns"].flatten(0, 1)
        shuffle = torch.randperm(values.numel(), generator=torch.Generator().manual_seed(20260909 + index))
        block["probe_shuffle"] = shuffle
        for clip in (True, False):
            report["clipped_probe" if clip else "unclipped_probe"] = fit_probe(
                critic,
                observations,
                values,
                returns,
                optimizer_snapshot,
                use_value_clip=clip,
                clip_param=alg.clip_param,
                max_grad_norm=alg.max_grad_norm,
                value_loss_coef=alg.value_loss_coef,
                indices=shuffle,
                num_mini_batches=4,
                num_epochs=2,
            )
        report["normalizer_only_replay_delta"] = {
            "mse_before_probe": report["clipped_probe"]["before"]["mse"],
            "stored_rollout_mse": report["stored_value_calibration"]["mse"],
        }
        artifact_path = output / f"rollout_{index}.pt"
        with artifact_path.open("xb") as stream:
            torch.save(block, stream)
        report["capture"] = file_row(artifact_path)
        write_runtime(output / f"block_{index}.json", report)
        reports.append(report)
        print(
            json.dumps(
                {
                    "diagnostic_block_complete": index,
                    "checkpoint_updates": contract["checkpoint_updates"],
                    "stored_value_rmse": report["stored_value_calibration"]["error"]["rms"],
                    "clipped_fit_mse": report["clipped_probe"]["after"]["mse"],
                    "unclipped_fit_mse": report["unclipped_probe"]["after"]["mse"],
                }
            ),
            flush=True,
        )
    if _state_sha256(cpu_state(actor)) != actor_hash:
        raise ValueError("diagnostic mutated actor")
    if _state_sha256(dict(critic.named_parameters())) != critic_parameters_hash:
        raise ValueError("diagnostic mutated simulator critic parameters")
    if runner.completed_update_count != 0 or runner.current_learning_iteration != 0:
        raise ValueError("diagnostic changed training counters")
    for row in [checkpoint_row, *source_rows, *external_rows]:
        if file_row(row["path"]) != row:
            raise ValueError("diagnostic input changed during execution")
    write_runtime(
        output / "summary.json",
        {
            **contract,
            "blocks_completed": len(reports),
            "transitions": options.diagnostic_blocks * runner.env.num_envs * runner.cfg["num_steps_per_env"],
            "actor_state_sha256": actor_hash,
            "actor_unchanged": True,
            "critic_parameters_unchanged": True,
            "blocks_reports": [file_row(output / f"block_{i}.json") for i in range(len(reports))],
            "completed": True,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--diagnostic-checkpoint", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument("--diagnostic-blocks", type=int, choices=(1, 2), default=2)
    options, remaining = parser.parse_known_args(argv)
    if options.diagnostic_checkpoint.is_symlink() or options.diagnostic_output.exists():
        raise ValueError("diagnostic needs a real checkpoint and a new output directory")

    class DiagnosticRunner(Native23PoseLoraRunner):
        def learn(self, num_learning_iterations, init_at_random_ep_len=False):
            del num_learning_iterations, init_at_random_ep_len
            return run_diagnostic(self, options)

        def save(self, *args, **kwargs):
            raise RuntimeError("value diagnostic cannot write policy checkpoints")

    previous = pose.Native23PoseLoraRunner
    pose.Native23PoseLoraRunner = DiagnosticRunner
    try:
        return pose.main(remaining)
    finally:
        pose.Native23PoseLoraRunner = previous


if __name__ == "__main__":
    raise SystemExit(main())
