"""Simulation-only native23 SONIC training with observable fixed-world root error.

Regression is a bounded experiment on existing local clips, not audited corpus
training. It cannot establish held-out generalization or authorize hardware.
Every invocation stops after at most 100 updates for independent CPU evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gear_sonic.envs.mjlab.sonic_true23_root_feedback import (
    configure_root_feedback_environment,
    install_root_feedback_command,
    root_objective_contract,
)
from gear_sonic.scripts import train_g1_true23_generalist as generalist
from gear_sonic.scripts.train_g1_true23_generalist_curriculum import parent_actor_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_curriculum import STAGES, derive_curriculum, write_curriculum_bundle
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract


def initialize_base_actor(runner, parent):
    """Verified old decoder transfer; new conditioner remains exactly zero."""
    if parent is None:
        return
    import torch
    from gear_sonic.scripts.export_g1_true23_generalist import validate_export_semantics
    from gear_sonic.trl.mjlab.native23_generalist_runner import validate_generalist_checkpoint

    path = Path(parent["checkpoint_path"])
    if sha256_file(path) != parent["checkpoint_sha256"]:
        raise ValueError("parent checkpoint no longer matches its bound hash")
    if runner.completed_update_count != 0 or runner.alg.optimizer.state:
        raise ValueError("parent transfer requires fresh optimizer and counters")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_export_semantics(checkpoint)
    actor = runner.alg.get_policy()
    proxy = SimpleNamespace(validate_training_artifact=actor.validate_base_training_artifact)
    validate_generalist_checkpoint(checkpoint, actor=proxy, lineage=checkpoint["lineage"])
    if checkpoint["actor"]["state_sha256"] != parent["actor_state_sha256"]:
        raise ValueError("parent actor state differs from transfer contract")
    if sha256_file(path) != parent["checkpoint_sha256"]:
        raise ValueError("parent checkpoint changed during transfer validation")
    actor.load_base_training_artifact(checkpoint["actor"])
    runner._assert_boundary()
    del checkpoint
    gc.collect()


def feedback_training_contract(args, curriculum):
    from gear_sonic.trl.mjlab.native23_root_feedback_actor import ROOT_FEEDBACK_ARCHITECTURE
    from gear_sonic.trl.mjlab.native23_root_feedback_runner import OPTIMIZER_PROFILES

    return {
        "feature_contract": root_feedback_contract(),
        "architecture": ROOT_FEEDBACK_ARCHITECTURE,
        "optimizer_profile": args.optimizer_profile,
        "optimizer_learning_rate_multipliers": dict(OPTIMIZER_PROFILES[args.optimizer_profile]),
        "curriculum": curriculum,
        "experiment_class": experiment_class(args.mode),
        "continuation": getattr(args, "continuation_contract", None),
        "reset_position_range_m": args.reset_position_range_m,
        "reset_velocity_range_m_s": args.reset_velocity_range_m_s,
        "root_world_tracking_error_weight": -10.0,
        "reward_task_target_frame": "q10_fixed_world",
        "objective_contract": root_objective_contract(),
        "root_linear_velocity_source": "simulator_ground_truth",
        "root_world_position_source": "simulator_ground_truth",
        "physical_estimator_qualified": False,
        "trainable": "all_decoder_affine_layers_plus_root_conditioner_bounded_exploration_and_critic",
        "critic_observes_root_feedback": True,
        "checkpoint_pattern": "root_feedback_model_N.pt",
        "held_out_policy_generalization_verified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def experiment_class(mode):
    return {"regression": "local_regression_only", "campaign": "evaluated_local_training_campaign"}.get(mode, mode)


def install_hooks(args, inputs, curriculum, precision, guard):
    from gear_sonic.envs.mjlab import sonic_true23_causal_history as task
    from gear_sonic.trl.mjlab import causal_history_runner, config
    from gear_sonic.trl.mjlab.native23_root_feedback_runner import Native23RootFeedbackRunner
    from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
    from gear_sonic.utils.g1_true23_training_precision import guard_algorithm, write_runtime

    bound_inputs = {**inputs, "derived_curriculum": curriculum}
    base = generalist.install_hooks(args, bound_inputs, precision, guard)
    rows = curriculum["derived_spans"]["spans"]
    install_root_feedback_command(rows)
    original_builder = task.make_causal_history_recovery_env_cfg

    def build_env(**kwargs):
        return configure_root_feedback_environment(
            original_builder(**kwargs),
            rows,
            reset_position_range_m=args.reset_position_range_m,
            reset_velocity_range_m_s=args.reset_velocity_range_m_s,
        )

    task.make_causal_history_recovery_env_cfg = build_env
    original_cfg = config.true23_mjlab_ppo_runner_cfg
    inherited_actor_type = type(original_cfg().actor)

    @dataclass
    class RootActorCfg(inherited_actor_type):
        class_name: str = "gear_sonic.trl.mjlab.native23_root_feedback_actor:True23RootFeedbackActorModel"
        root_feedback_obs_group: str = "root_feedback"

    def agent_cfg():
        cfg = original_cfg()
        cfg.actor = RootActorCfg(
            **{
                field.name: getattr(cfg.actor, field.name)
                for field in fields(cfg.actor)
                if field.name != "class_name"
            }
        )
        cfg.obs_groups = {"actor": ("tokenizer", "policy", "root_feedback"), "critic": ("critic", "root_feedback")}
        cfg.experiment_name = "sonic_native23_root_feedback_simulation"
        cfg.run_name = args.curriculum_stage
        cfg.num_steps_per_env = args.rollout_steps
        cfg.algorithm.num_learning_epochs = args.ppo_epochs
        cfg.algorithm.num_mini_batches = args.ppo_minibatches
        return cfg

    config.true23_mjlab_ppo_runner_cfg = agent_cfg
    feedback = feedback_training_contract(args, curriculum)

    class RootFeedbackRunner(Native23RootFeedbackRunner):
        def __init__(self, *values, **kwargs):
            guard()
            super().__init__(*values, **kwargs)
            if args.mode == "campaign":
                self.initialize_evaluated_continuation(args.continuation_contract)
            elif args.resume is None:
                initialize_base_actor(self, curriculum["parent_actor_initialization"])
            guard_algorithm(self.alg, guard)
            write_runtime(
                self.checkpoint_dir.parent / "root_feedback_runtime.json",
                {
                    "actor": self.alg.get_policy().artifact_contract(),
                    "training_inputs": bound_inputs,
                    "root_feedback": feedback,
                    "precision": precision,
                    "optimizer_groups": [
                        {
                            "name": group["name"],
                            "learning_rate": group["lr"],
                            "elements": sum(p.numel() for p in group["params"]),
                        }
                        for group in self.alg.optimizer.param_groups
                    ],
                    "deployment_ready": False,
                    "hardware_authorized": False,
                },
            )

        def learn(self, num_learning_iterations, init_at_random_ep_len=False):
            if args.mode == "campaign":
                last_evaluated = args.continuation_contract["completed_update_count"]
                if self.completed_update_count + num_learning_iterations > last_evaluated + 100:
                    raise ValueError("fresh CPU evaluation required before another campaign block")
                self._save_numbered_checkpoint()
            try:
                return super().learn(num_learning_iterations, init_at_random_ep_len=False)
            finally:
                receipt = self.env.unwrapped.command_manager.get_term("motion").curriculum_receipt()
                receipt.update(
                    stage=curriculum["stage"],
                    experiment_class=feedback["experiment_class"],
                    deployment_ready=False,
                    physical_estimator_qualified=False,
                )
                write_runtime(
                    self.checkpoint_dir.parent / f"curriculum_runtime_{self.completed_update_count}.json", receipt
                )

    causal_history_runner.CausalHistoryMjlabOnPolicyRunner = RootFeedbackRunner
    original_resolved = base._resolved_training_config

    def resolved(*values, **kwargs):
        result = original_resolved(*values, **kwargs)
        result["native23_generalist"]["current_curriculum"] = curriculum["stage"]
        result["native23_generalist"]["curriculum"] = curriculum
        result["native23_root_feedback"] = feedback
        return result

    base._resolved_training_config = resolved
    original_sources = base._source_files

    def source_files():
        result = original_sources()
        roots = [Path(__file__)] + [
            generalist.ROOT / path
            for path in (
                "gear_sonic/trl/mjlab/native23_root_feedback_actor.py",
                "gear_sonic/trl/mjlab/native23_root_feedback_runner.py",
                "gear_sonic/scripts/export_g1_true23_root_feedback.py",
            )
        ]
        result.update(collect_local_source_closure(generalist.ROOT, roots).as_source_files(generalist.ROOT))
        result["native23_root_feedback/curriculum.spans.json"] = args.spans
        return result

    base._source_files = source_files
    return base


def make_parser():
    from gear_sonic.scripts import train_g1_23dof_mjlab_causal_history as base

    parser = base._parser()
    subparsers = parser._subparsers._group_actions[0]
    regression = subparsers.add_parser("regression", help="bounded local-clip experiment; no corpus qualification")
    base._common(regression)
    regression.set_defaults(num_envs=16, iterations=100)
    campaign = subparsers.add_parser("campaign", help="evaluated local continuation; not corpus qualification")
    base._common(campaign)
    campaign.set_defaults(num_envs=16, iterations=2000)
    for subparser in subparsers.choices.values():
        subparser.add_argument("--source-checkpoint", type=Path, required=True)
        subparser.add_argument("--spans", type=Path, required=True)
        subparser.add_argument("--sim-config", type=Path, default=generalist.SIM_CONFIG)
        subparser.add_argument("--corpus-manifest", type=Path)
        subparser.add_argument("--effort-penalty-weight", type=float, default=0.1)
        subparser.add_argument("--curriculum-stage", choices=STAGES, required=True)
        subparser.add_argument("--curriculum-directory", type=Path, required=True)
        subparser.add_argument("--initialize-actor-from", type=Path)
        subparser.add_argument("--continue-from", type=Path)
        subparser.add_argument("--continuation-evaluation", type=Path)
        subparser.add_argument("--reset-position-range-m", type=float, default=0.1)
        subparser.add_argument("--reset-velocity-range-m-s", type=float, default=0.1)
        subparser.add_argument("--ppo-epochs", type=int, default=2)
        subparser.add_argument("--ppo-minibatches", type=int, default=4)
        subparser.add_argument("--rollout-steps", type=int, default=32)
        subparser.add_argument(
            "--optimizer-profile", choices=("feedback_priority", "legacy_uniform"), default="feedback_priority"
        )
        subparser.add_argument(
            "--return-target", choices=("planned_endpoint", "configured_origin"), default="planned_endpoint"
        )
        subparser.add_argument(
            "--native-model",
            type=Path,
            default=generalist.ROOT.parent
            / "GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        )
        subparser.set_defaults(learning_rate=5e-7, save_interval=100, session_updates=100)
    return parser


def validate_bounds(args):
    if args.mode == "smoke" and (args.iterations > 2 or args.num_envs > 4):
        raise ValueError("root-feedback smoke is bounded to 2 updates and 4 environments")
    if args.mode == "regression" and (args.iterations > 100 or args.num_envs > 32):
        raise ValueError("local regression is bounded to 100 updates and 32 environments")
    if args.mode == "campaign":
        if args.iterations > 5000 or args.num_envs > 32:
            raise ValueError("local campaign is bounded to 5000 total updates and 32 environments")
        if args.continue_from is None or args.continuation_evaluation is None:
            raise ValueError("campaign requires an evaluated root-feedback parent")
        if args.resume is not None or args.initialize_actor_from is not None:
            raise ValueError("campaign uses evaluated full-state continuation, not resume or actor transfer")
    elif args.continue_from is not None or args.continuation_evaluation is not None:
        raise ValueError("evaluated continuation is available only in explicit campaign mode")
    if args.session_updates > 100 or args.save_interval > 100:
        raise ValueError("independent CPU evaluation required at most every 100 updates")
    if not 1 <= args.ppo_epochs <= 5 or not 1 <= args.ppo_minibatches <= 8:
        raise ValueError("PPO epochs/minibatches exceed bounded training configuration")
    if not 8 <= args.rollout_steps <= 64:
        raise ValueError("rollout steps must be between 8 and 64")
    if not 0 <= args.reset_position_range_m <= 0.15 or not 0 <= args.reset_velocity_range_m_s <= 0.15:
        raise ValueError("root reset perturbation exceeds bounded nominal acquisition range")
    if args.resume is not None and args.initialize_actor_from is not None:
        raise ValueError("resume and base actor initialization are mutually exclusive")
    if args.curriculum_directory.resolve() == args.run_dir.resolve():
        raise ValueError("curriculum inputs and training outputs need separate directories")


def main(argv=None):
    from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision
    from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        validate_bounds(args)
    except ValueError as error:
        parser.error(str(error))
    for name in (
        "source_checkpoint",
        "warm_start",
        "motion_file",
        "motion_metadata",
        "spans",
        "sim_config",
        "native_model",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    inputs = generalist.validate_training_inputs(
        args.motion_file,
        args.spans,
        args.corpus_manifest,
        smoke_only=args.mode in {"smoke", "preflight", "regression", "campaign"},
    )
    inputs["experiment_class"] = experiment_class(args.mode)
    spans = json.loads(args.spans.read_text())
    if spans.get("source_motion_sha256", inputs["motion_sha256"]) != inputs["motion_sha256"]:
        raise ValueError("source span payload binding differs from selected motion")
    with np.load(args.motion_file, allow_pickle=False) as archive:
        motion = {key: archive[key].copy() for key in generalist.MOTION_KEYS}
    if (
        sha256_file(args.motion_file) != inputs["motion_sha256"]
        or sha256_file(args.spans) != inputs["spans_sha256"]
    ):
        raise ValueError("original curriculum inputs changed after validation")
    _, model, _ = prepare_true23_model(args.native_model, args.sim_config)
    derived, generated_spans, curriculum = derive_curriculum(
        motion,
        spans,
        inputs,
        stage=args.curriculum_stage,
        model=model,
        simulation_config=args.sim_config,
        return_target=args.return_target,
    )
    curriculum["native_reference_model_sha256"] = sha256_file(args.native_model)
    curriculum["native_reference_sim_config_sha256"] = sha256_file(args.sim_config)
    if args.mode == "campaign":
        from gear_sonic.utils.g1_true23_root_feedback_campaign import campaign_parent_contract

        if len(spans["spans"]) != 1:
            raise ValueError("local campaign evaluation currently supports one complete reference")
        args.continuation_contract = campaign_parent_contract(
            args.continue_from, args.continuation_evaluation, args=args, curriculum=curriculum
        )
        parent = args.continuation_contract["original_base_parent"]
    elif args.resume is not None:
        previous = json.loads((args.curriculum_directory / "curriculum.json").read_text())["curriculum"]
        previous_parent = previous["parent_actor_initialization"]
        parent = parent_actor_contract(
            None if previous_parent is None else Path(previous_parent["checkpoint_path"]), inputs
        )
        if parent != previous_parent:
            raise ValueError("resume parent actor binding changed")
    else:
        parent = parent_actor_contract(args.initialize_actor_from, inputs)
    curriculum["parent_actor_initialization"] = parent
    if args.resume is not None:
        directory = args.curriculum_directory.resolve(strict=True)
        metadata = json.loads((directory / "curriculum.json").read_text())
        generated = directory / "curriculum.npz"
        if metadata["curriculum"] != curriculum or metadata["output"]["sha256"] != sha256_file(generated):
            raise ValueError("resume curriculum payload or contract changed")
        args.motion_file, args.motion_metadata, args.spans = (
            generated,
            directory / "curriculum.json",
            directory / "curriculum.spans.json",
        )
        if json.loads(args.spans.read_text()) != generated_spans:
            raise ValueError("resume curriculum span mismatch")
    else:
        args.motion_file, args.motion_metadata, args.spans = write_curriculum_bundle(
            args.curriculum_directory.resolve(), derived, generated_spans, curriculum
        )
    del model, motion, derived
    gc.collect()
    with ieee_training_precision() as (precision, guard):
        base = install_hooks(args, inputs, curriculum, precision, guard)
        if args.mode == "preflight":
            report = base.preflight(args)
            report["native23_root_feedback"] = feedback_training_contract(args, curriculum)
            print(json.dumps(report, indent=2, sort_keys=True))
            if args.json_output:
                with args.json_output.open("x") as stream:
                    json.dump(report, stream, indent=2, sort_keys=True)
            return 0 if report["ready"] else 2
        print(base.run_training(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
