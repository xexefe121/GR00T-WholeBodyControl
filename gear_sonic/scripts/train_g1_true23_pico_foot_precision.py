"""PICO foot-precision fine-tuning from verified learner500; simulation only.

Preserves learned actor, critic and Adam state. Fresh simulation/RNG state is
explicit; the only objective change is an independent dense foot-placement term.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import inspect
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from gear_sonic.scripts import train_g1_true23_normal_lora as recipe
from gear_sonic.trl.mjlab.native23_pico_foot_precision_runner import (
    Native23PicoFootPrecisionRunner,
    PARENT_SHA256,
    equal_state,
    training_contract,
    validate_checkpoint,
)
from gear_sonic.utils.g1_23dof_mjlab_training import build_file_manifest
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure

RECORDINGS = ("existing_pico_derived_walk002", "existing_pico_derived_walk003", "existing_pico_derived_pico")
FLAGS = dict(deployment_ready=False, hardware_authorized=False, simulator_qualified=False)


def input_capture_selected(control, total):
    return control < 16 or control % 64 == 0 or control >= total - 16


def validate_bank(path):
    bank = json.loads(path.read_text())
    if (
        bank.get("training_recording_ids") != list(RECORDINGS)
        or bank.get("evaluation_only_recording_ids") != ["existing_pico_derived_walk008"]
        or bank.get("pico_used_for_training") is not True
        or bank.get("source_frames") != 7266
        or bank.get("lifecycle_frames") != 9549
        or bank.get("saved_native_lifecycles_bit_exact") is not True
        or bank.get("all_original29_tasks_bit_exact") is not True
        or bank.get("deployment_ready") is not False
    ):
        raise ValueError("existing-PICO bank ownership or preservation contract differs")
    for label in ("inputs", "outputs"):
        for item, expected in bank[label].items():
            if sha256_file(Path(item)) != expected:
                raise ValueError("existing-PICO bank source changed: " + item)
    return bank


def telemetry(runner, total):
    gradients, inputs = [], []
    state = dict(controls=0, transitions=0)
    original_step, original_act = runner.alg.optimizer.step, runner.alg.act

    def optimizer_step(*args, **kwargs):
        norms = {}
        for group in runner.alg.optimizer.param_groups:
            values = [p.grad.detach() for p in group["params"] if p.grad is not None]
            if not values or not all(torch.isfinite(value).all() for value in values):
                raise ValueError("PICO trainable group lacks finite gradients")
            norms[group["name"]] = float(torch.sqrt(sum(value.double().square().sum() for value in values)).cpu())
        gradients.append(norms)
        return original_step(*args, **kwargs)

    def act(obs):
        control = state["controls"]
        row = None
        if input_capture_selected(control, total):
            command = runner.env.unwrapped.command_manager.get_term("motion")
            row = {name: obs[name].detach().cpu().clone() for name in ("tokenizer", "policy", "root_feedback")}
            row.update(
                control_index=np.int64(control),
                reference_q0=command.time_steps.detach().cpu().clone(),
                measured_root_position_w=command.robot_anchor_pos_w.detach().cpu().clone(),
                measured_root_quaternion_wxyz=command.robot_anchor_quat_w.detach().cpu().clone(),
            )
        action = original_act(obs)
        state["controls"] += 1
        state["transitions"] += action.shape[0]
        if row is not None:
            row["sampled_raw_action23"] = action.detach().cpu().clone()
            inputs.append(row)
        return action

    runner.alg.optimizer.step, runner.alg.act = optimizer_step, act
    return gradients, inputs, state


def run(args):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from rsl_rl.algorithms.ppo import PPO

    if (args.updates, args.num_envs, args.rollout_steps) not in ((2, 4, 8), (500, 128, 16)):
        raise ValueError("PICO run must be declared2x4x8 smoke or500x128x16 fixed trial")
    if args.run_dir.exists():
        raise FileExistsError("foot precision trainer refuses overwrite")
    if sha256_file(args.preflight) != recipe.PREFLIGHT_SHA256:
        raise ValueError("normal-core weight/gradient preflight changed")
    bank = validate_bank(args.bank_report)
    if sha256_file(args.parent_checkpoint) != PARENT_SHA256:
        raise ValueError("foot precision requires evaluated PICO500 parent")
    spec_path, spans_path = Path(bank["files"]["spec"]), Path(bank["files"]["spans"])
    args.spec, args.spans = spec_path, spans_path
    spec = recipe.intent.validate_spec(json.loads(spec_path.read_text()))
    rows = json.loads(spans_path.read_text())["spans"]
    if [row["name"] for row in rows] != ["walk002", "walk003", "pico"]:
        raise ValueError("training spans changed recording ownership")
    if args.updates == 500:
        if args.smoke_report is None:
            raise ValueError("fixed PICO training requires completed new-bank smoke")
        smoke = json.loads(args.smoke_report.read_text())
        if (
            not smoke.get("completed")
            or smoke.get("simulator_updates") != 502
            or smoke.get("additional_updates") != 2
            or smoke.get("parent_checkpoint_sha256") != PARENT_SHA256
            or smoke.get("bank_sha256") != sha256_file(args.bank_report)
        ):
            raise ValueError("new-bank smoke missing or belongs to another bank")
    agent = recipe.agent_configuration(args)
    agent.save_interval = args.updates
    env_cfg = recipe.environment_configuration(args, spec, rows)
    _, assets, dataset = recipe.material_manifests(args, spec)
    closure = collect_local_source_closure(recipe.ROOT, [Path(__file__)])
    source_files = {
        **recipe._source_files(),
        **closure.as_source_files(recipe.ROOT),
        "external/rsl_rl/algorithms/ppo.py": Path(inspect.getfile(PPO)),
        "pico/bank_provenance.json": args.bank_report,
        "pico/parent_checkpoint.pt": args.parent_checkpoint,
        "pico/experiment.md": args.experiment,
        "normal/preflight.json": args.preflight,
    }
    if args.smoke_report is not None:
        source_files["pico/completed_smoke.json"] = args.smoke_report
    sources = build_file_manifest(source_files, kind="source_files")
    args.run_dir.mkdir(parents=True)
    recipe.write_runtime(
        args.run_dir / "request.json",
        dict(
            arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            bank_sha256=sha256_file(args.bank_report),
            pico_used_for_training=True,
            training_recording_ids=RECORDINGS,
            validation_recording_ids=["existing_pico_derived_walk008"],
            untouched_test_generalization_claimed=False,
            input_telemetry="first16,last16,every64_control;all_envs_at_selected_controls",
            milestones=[500, 502] if args.updates == 2 else [500, 600, 1000],
            parent_checkpoint_sha256=PARENT_SHA256,
            learner_state_preserved=True,
            uninterrupted_simulation_resume=False,
            **FLAGS,
        ),
    )
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    os.environ["MUJOCO_GL"], os.environ["MUJOCO_EGL_DEVICE_ID"] = "egl", "0"
    env = runner = None
    gradients, inputs, counts = [], [], dict(controls=0, transitions=0)
    started = time.perf_counter()
    outcome = dict(
        kind="native23_pico_foot_precision_continuation_v1",
        parent_checkpoint_sha256=PARENT_SHA256,
        completed=False,
        simulator_updates=0,
        bank_sha256=sha256_file(args.bank_report),
        pico_used_for_training=True,
        **FLAGS,
    )
    try:
        with recipe.ieee_training_precision() as (precision, guard):
            resolved = recipe._resolved_training_config(
                agent_cfg=agent,
                mode="foot_precision_smoke" if args.updates == 2 else "bounded_pico_foot_precision",
                motion_path=Path(spec["files"]["native_motion"]["path"]),
                num_envs=args.num_envs,
                seed=args.seed,
                planned_updates=args.updates,
                reference_profile=recipe.NORMAL_TIMING,
            )
            resolved.update(
                schema="g1_true23_pico_foot_precision_training_v1",
                task="Unitree-G1-Native23-PICO-FootPrecision-Received920ms",
                foot_precision_adaptation=training_contract(),
                original_intent_spec=spec,
                reference=recipe.normal_reference_contract(),
                precision=precision,
                source_weight_profile=recipe.REFERENCE_PROFILE_NORMAL,
                randomize_initial_episode_lengths=False,
                push_disturbances_enabled=False,
                domain_randomization=[],
                actual_event_names=list(env_cfg.events),
                start_schedule="mixed_reference_reset_v1",
                reset_xy_position_range_m=0.1,
                reset_xy_velocity_range_m_s=0.1,
                pico_used_for_training=True,
                training_recording_ids=RECORDINGS,
                evaluation_only_recording_ids=bank["evaluation_only_recording_ids"],
                original_bank_report_sha256=sha256_file(args.bank_report),
                optimizer_or_critic_reused=True,
                parent_checkpoint_sha256=PARENT_SHA256,
                parent_completed_updates=500,
                planned_additional_updates=args.updates,
                planned_total_updates=500 + args.updates,
                simulator_and_rng_state_reinitialized=True,
                low_latency_checkpoint_relabelled=False,
                **FLAGS,
            )
            recipe.write_runtime(args.run_dir / "resolved_training.json", resolved)
            print("PICO foot precision: constructing unchanged nominal native23 physics", flush=True)
            env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            prime = recipe.prime_sonic_true23_training_environment(wrapped)
            error = float(recipe.intent.ee_height_error(env, spec).max().cpu())
            if error > 0.25:
                raise ValueError("PICO prime has terminal original hand/foot target")
            prime["original_hand_foot_height_error_max_m"] = error
            checks = {
                "environment_prime": prime,
                "nominal_scene_parity": recipe.verify_nominal_scene(
                    env.sim.mj_model, spec["files"]["native_model"]["path"], args.sim_config
                ),
                "action_history_parity": recipe.verify_executed_release_environment(
                    env, spec["files"]["source_model"]["path"]
                ),
                "normal_source_parity": recipe.verify_normal_source_environment(env, spec),
                "world_failure_contract": recipe.world_failure.verify_runtime(env),
            }
            for name, value in checks.items():
                recipe.write_runtime(args.run_dir / (name + ".json"), value)
            runner = Native23PicoFootPrecisionRunner(
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
            recipe.guard_algorithm(runner.alg, guard)
            recipe.write_runtime(args.run_dir / "lineage.json", runner.training_lineage)
            imported = runner.import_parent(args.parent_checkpoint)
            recipe.write_runtime(args.run_dir / "parent_import.json", imported)
            runner._save_numbered_checkpoint()
            actor = runner.alg.get_policy()
            fixed = wrapped.get_observations().clone()
            with torch.no_grad():
                mean_before = actor(fixed).cpu().clone()
                parent = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=True)
                if not equal_state(actor.export_training_artifact(), parent["actor"]):
                    raise ValueError("initial foot precision actor differs from evaluated parent")
                del parent
            before = recipe.copied_groups(runner)
            gradients, inputs, counts = telemetry(runner, args.updates * args.rollout_steps)
            original_log = runner.logger.log
            with (args.run_dir / "training.log").open("x") as log_stream:

                def log(**values):
                    with redirect_stdout(log_stream):
                        original_log(**values)
                    log_stream.flush()
                    completed = runner.completed_update_count
                    if completed == 600:
                        runner._save_numbered_checkpoint()
                    if completed == 501 or completed % 25 == 0 or completed == 500 + args.updates:
                        progress = dict(
                            completed_updates=completed,
                            planned_updates=500 + args.updates,
                            additional_updates=completed - 500,
                            transitions=counts["transitions"],
                            elapsed_s=time.perf_counter() - started,
                            sampled_input_controls=len(inputs),
                            pico_used_for_training=True,
                            **FLAGS,
                        )
                        recipe.write_runtime(args.run_dir / f"progress_{completed:04d}.json", progress)
                        print(json.dumps(progress), flush=True)

                runner.logger.log = log
                print(f"PICO foot precision: training {args.updates} continuous updates", flush=True)
                runner.learn(args.updates, init_at_random_ep_len=False)
            with torch.no_grad():
                mean_after = actor(fixed).cpu().clone()
            after = recipe.copied_groups(runner)
            changes = {
                name: max(float((a - b).abs().max()) for a, b in zip(values, after[name], strict=True))
                for name, values in before.items()
            }
            if not all(np.isfinite(value) and value > 0 for value in changes.values()):
                raise ValueError("PICO update did not change all trainable groups")
            actor.core.assert_frozen_encoder_unchanged()
            final = runner._numbered_checkpoint_path(500 + args.updates)
            validate_checkpoint(
                torch.load(final, map_location="cpu", weights_only=True),
                actor=actor,
                lineage=runner.training_lineage,
            )
            with (args.run_dir / "fixed_observation_mean_change.npz").open("xb") as stream:
                np.savez_compressed(
                    stream,
                    before=mean_before.numpy(),
                    after=mean_after.numpy(),
                    **{key: value.cpu().numpy() for key, value in fixed.items()},
                )
            outcome.update(
                completed=True,
                checkpoint=str(final),
                checkpoint_sha256=sha256_file(final),
                maximum_parameter_change_by_group=changes,
                initial_actor_matches_evaluated_parent_bit_exact=True,
                exact_actor_critic_and_adam_import=True,
                frozen_base_unchanged=True,
                fixed_observation_mean_max_abs_change=float((mean_after - mean_before).abs().max()),
            )
    except BaseException as exc:
        outcome.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        if runner is not None:
            outcome["simulator_updates"] = runner.completed_update_count
            outcome["additional_updates"] = max(0, runner.completed_update_count - 500)
            reward = runner._reward_step.capture()
            if reward:
                with (args.run_dir / "reward_capture.npz").open("xb") as stream:
                    np.savez_compressed(stream, **reward)
            failure = env._world_root_failure_capture
            if failure:
                with (args.run_dir / "world_failure_capture.npz").open("xb") as stream:
                    np.savez_compressed(
                        stream,
                        **{
                            key: np.stack(
                                [
                                    row[key].numpy() if isinstance(row[key], torch.Tensor) else row[key]
                                    for row in failure
                                ]
                            )
                            for key in failure[0]
                        },
                    )
        if inputs:
            with (args.run_dir / "sampled_actual_inputs.npz").open("xb") as stream:
                np.savez_compressed(
                    stream,
                    **{
                        key: np.stack(
                            [
                                row[key].numpy() if isinstance(row[key], torch.Tensor) else row[key]
                                for row in inputs
                            ]
                        )
                        for key in inputs[0]
                    },
                )
        outcome.update(
            controls_per_env=counts["controls"],
            transitions=counts["transitions"],
            sampled_input_controls=len(inputs),
            optimizer_steps=len(gradients),
            elapsed_s=time.perf_counter() - started,
        )
        recipe.write_runtime(args.run_dir / "gradient_capture.json", gradients)
        recipe.write_runtime(args.run_dir / "outcome.json", outcome)
        print(json.dumps(outcome), flush=True)
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "bank-report",
        "parent-checkpoint",
        "sim-config",
        "warm-start",
        "source-checkpoint",
        "preflight",
        "experiment",
        "run-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve(strict=key != "run_dir"))
    run(args)


if __name__ == "__main__":
    main()
