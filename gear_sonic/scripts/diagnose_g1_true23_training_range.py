"""Fixed-policy stochastic training rollouts; no optimization or robot I/O.

Reuses the world-quality environment constructor and strict checkpoint reader.
The optional observer reads every physics step without adding a termination.
Control-only mode provides a fresh-process observational identity comparison.
"""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23_passive_range_capture import PassiveRangeCapture, snapshot
from gear_sonic.scripts import train_g1_true23_world_quality as quality
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256
from gear_sonic.trl.mjlab.native23_world_quality_runner import Native23WorldQualityRunner, validate_checkpoint
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_training_precision import write_runtime
from gear_sonic.utils.g1_true23_value_diagnostic import cpu_state
from gear_sonic.utils.g1_true23_world_quality_checkpoint import validate_semantics


def file_row(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha256_file(path), size_bytes=path.stat().st_size)


def validate_factory(fresh, existing):
    # Paths of newly constructed curriculum files necessarily differ. All other
    # resolved top-level settings and all task/physics/actor contracts must match.
    exclusions = {"motion_filename", "native23_root_feedback", "native23_generalist"}
    if set(fresh) != set(existing):
        raise ValueError("diagnostic factory resolved keys differ")
    for key in fresh.keys() - exclusions:
        if fresh[key] != existing[key]:
            raise ValueError(f"diagnostic factory differs from checkpoint: {key}")
    for section, omitted in (
        ("native23_root_feedback", {"curriculum", "original_intent_spec"}),
        ("native23_generalist", {"curriculum", "current_curriculum", "training_inputs"}),
    ):
        if set(fresh[section]) != set(existing[section]):
            raise ValueError(f"diagnostic factory keys differ: {section}")
        for key in fresh[section].keys() - omitted:
            if fresh[section][key] != existing[section][key]:
                raise ValueError(f"diagnostic task differs from checkpoint: {section}/{key}")
    old_files = existing["native23_root_feedback"]["original_intent_spec"]["files"]
    new_files = fresh["native23_root_feedback"]["original_intent_spec"]["files"]
    if {k: v["sha256"] for k, v in old_files.items()} != {k: v["sha256"] for k, v in new_files.items()}:
        raise ValueError("diagnostic source and physical geometry differ from checkpoint")


def run_diagnostic(runner, options):
    runner.require_runtime()
    output = options.diagnostic_output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_row = file_row(options.diagnostic_checkpoint)
    checkpoint = torch.load(checkpoint_row["path"], map_location="cpu", weights_only=True)
    semantics = validate_semantics(checkpoint)
    actor, critic, alg = runner.alg.actor, runner.alg.critic, runner.alg
    validate_checkpoint(checkpoint, actor=actor, lineage=checkpoint["lineage"])
    fresh = runner.training_lineage["materials"]["resolved_config"]["payload"]
    validate_factory(fresh, semantics["resolved"])
    actor.load_training_artifact(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    actor_hash = _state_sha256(cpu_state(actor))
    critic_hash = _state_sha256(dict(critic.named_parameters()))
    if runner.env.num_envs != 32 or runner.cfg["num_steps_per_env"] != 64:
        raise ValueError("diagnostic requires actual32x64 training batch geometry")

    def forbidden(*args, **kwargs):
        raise RuntimeError("range diagnostic forbids optimizer updates and checkpoint saves")

    alg.update = forbidden
    alg.optimizer.step = forbidden
    env = runner.env.unwrapped
    command = env.command_manager.get_term("motion")
    manager = env.termination_manager
    root = Path(__file__).resolve().parents[2]
    closure = collect_local_source_closure(root, [Path(__file__)])
    source_rows = [file_row(path) for path in closure.as_source_files(root).values()]
    external_rows = [
        file_row(inspect.getfile(value)) for value in (type(env), type(env.sim), type(manager), type(alg))
    ]
    contract = dict(
        kind="native23_passive_actual_training_range_diagnostic_v1",
        checkpoint=checkpoint_row,
        checkpoint_updates=checkpoint["trainer_state"]["completed_update_count"],
        checkpoint_lineage_sha256=checkpoint["lineage_sha256"],
        factory_lineage_sha256=runner.lineage_sha256,
        controls_per_env=options.diagnostic_controls,
        num_envs=env.num_envs,
        capture_mode=options.capture_mode,
        seed=fresh["seed"],
        initial_anchors=command.time_steps.cpu().tolist(),
        termination_names=list(manager.active_terms),
        termination_timeouts=[manager.get_term_cfg(n).time_out for n in manager.active_terms],
        measured_range_termination_added=False,
        existing_failure_reward_and_reset_rules_unchanged=True,
        physics_steps_per_control=10,
        normalization_mode="stock_process_env_step_updates_input_statistics_no_weight_updates",
        action_mode="actual_stochastic_training_distribution_not_deterministic_replay",
        optimizer_state_loaded=False,
        actor_updates=0,
        critic_updates=0,
        training_resume_claimed=False,
        sources=source_rows,
        external_sources=external_rows,
        deployment_ready=False,
        hardware_authorized=False,
    )
    write_runtime(output / "contract.json", contract)
    print(
        json.dumps(dict(preflight_passed=True, planned_transitions=options.diagnostic_controls * 32)), flush=True
    )
    observer = PassiveRangeCapture(env) if options.capture_mode == "passive" else None
    # The original reward wrapper identity remains intact even while observing.
    runner.require_runtime()
    alg.train_mode()
    obs = runner.env.get_observations().to(runner.device)
    rows, observations = [], []
    for index in range(options.diagnostic_controls):
        with torch.inference_mode():
            row = dict(
                source_anchor=snapshot(command.time_steps),
                episode_length_before=snapshot(env.episode_length_buf),
                pre_qpos=snapshot(env.sim.data.qpos),
                pre_qvel=snapshot(env.sim.data.qvel),
            )
            observations.append({k: snapshot(v) for k, v in obs.items()})
            actions = alg.act(obs)
            row["actions"] = snapshot(actions)
            if observer is None:
                obs, rewards, dones, extras = runner.env.step(actions.to(runner.env.device))
            else:
                obs, rewards, dones, extras = observer.run(runner.env.step, actions.to(runner.env.device))
            row.update(
                reward=snapshot(rewards),
                done=snapshot(dones).bool(),
                terminated=snapshot(env.reset_terminated),
                timeouts=snapshot(env.reset_time_outs),
                term_flags=torch.stack([snapshot(manager.get_term(n)) for n in manager.active_terms], -1),
                post_qpos=snapshot(env.sim.data.qpos),
                post_qvel=snapshot(env.sim.data.qvel),
                next_source_anchor=snapshot(command.time_steps),
                episode_length_after=snapshot(env.episode_length_buf),
            )
            rows.append(row)
            obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
            if any(not value.isfinite().all() for value in obs.values()) or not rewards.isfinite().all():
                raise ValueError("nonfinite actual training diagnostic transition")
            alg.process_env_step(obs, rewards, dones, extras)
            extras["log"] = {}
            if (index + 1) % runner.cfg["num_steps_per_env"] == 0:
                alg.storage.clear()
        if (index + 1) % 32 == 0 or index + 1 == options.diagnostic_controls:
            print(json.dumps(dict(controls_complete=index + 1, transitions=(index + 1) * 32)), flush=True)

    arrays = {key: torch.stack([row[key] for row in rows]).numpy() for key in rows[0]}
    for key in observations[0]:
        arrays["obs_" + key] = torch.stack([row[key] for row in observations]).numpy()
    if observer is not None:
        arrays.update({"passive_" + key: value for key, value in observer.capture().items()})
    world_rows = env._world_root_failure_capture
    if len(world_rows) != len(rows):
        raise ValueError("actual world failure count differs from controls")
    for key in ("desired_position_w", "measured_position_w", "error_m", "failure", "reference_q0"):
        arrays["world_" + key] = torch.stack([row[key] for row in world_rows]).numpy()
    arrays["world_common_step_counter"] = np.asarray([row["common_step_counter"] for row in world_rows])
    arrays.update({"reward_capture_" + key: value for key, value in runner._reward_step.capture().items()})
    arrays_path = output / "rollout.npz"
    with arrays_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    if _state_sha256(cpu_state(actor)) != actor_hash:
        raise ValueError("diagnostic mutated actor")
    if _state_sha256(dict(critic.named_parameters())) != critic_hash:
        raise ValueError("diagnostic mutated critic parameters")
    if runner.completed_update_count != 0 or runner.current_learning_iteration != 0:
        raise ValueError("diagnostic performed training updates")
    runner.require_runtime()
    for row in [checkpoint_row, *source_rows, *external_rows]:
        if file_row(row["path"]) != row:
            raise ValueError("diagnostic input changed during execution")
    write_runtime(
        output / "summary.json",
        dict(
            **contract,
            capture=file_row(arrays_path),
            actor_state_sha256=actor_hash,
            actor_unchanged=True,
            critic_parameters_unchanged=True,
            transitions=len(rows) * env.num_envs,
            actual_physics_steps=len(rows) * env.num_envs * 10,
            actual_physics_steps_captured=len(rows) * env.num_envs * 10 if observer else 0,
            independent_audit_passed=False,
            completed=True,
        ),
    )
    print(json.dumps(dict(completed=True, transitions=len(rows) * 32, optimizer_updates=0)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--diagnostic-checkpoint", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument("--diagnostic-controls", type=int, choices=(8, 256), default=256)
    parser.add_argument("--capture-mode", choices=("passive", "control-only"), default="passive")
    options, remaining = parser.parse_known_args(argv)
    if options.diagnostic_checkpoint.is_symlink() or options.diagnostic_output.exists():
        raise ValueError("diagnostic needs a real checkpoint and a new output directory")

    class DiagnosticRunner(Native23WorldQualityRunner):
        def learn(self, num_learning_iterations, init_at_random_ep_len=False):
            del num_learning_iterations, init_at_random_ep_len
            return run_diagnostic(self, options)

        def save(self, *args, **kwargs):
            raise RuntimeError("range diagnostic cannot write policy checkpoints")

    previous = quality.Native23WorldQualityRunner
    quality.Native23WorldQualityRunner = DiagnosticRunner
    try:
        return quality.main(remaining)
    finally:
        quality.Native23WorldQualityRunner = previous


if __name__ == "__main__":
    raise SystemExit(main())
