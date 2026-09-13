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
from gear_sonic.utils.g1_true23_buffered_reference import BUFFERED_TIMING, CAUSAL_TIMING, REFERENCE_TIMINGS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_curriculum import STAGES, derive_curriculum, write_curriculum_bundle
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract
from gear_sonic.utils.g1_true23_start_schedule import START_SCHEDULES, start_schedule_contract


def selected_root_objectives(timing):
    result = root_objective_contract()
    if timing == BUFFERED_TIMING:
        result = {
            **result,
            "kind": "g1_native23_buffered_source_root_objectives_v2",
            "reference_positions_orientations_and_joint_positions": "received_horizon_q1_age_180ms_fixed_world",
            "linear_and_joint_velocity_reference": "received_q1_minus_received_q0_over_0p02s",
            "angular_velocity_reference": "world_shortest_arc_received_q1_times_inverse_received_q0_over_0p02s",
            "tokenizer_and_original_command_properties": (
                "received_horizon_q0_age_200ms_current_measured_orientation"
            ),
            "reward_and_termination_phase": "post_physics_before_reference_advance_against_held_received_q1",
            "reference_timing": timing,
        }
    return result


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
    from gear_sonic.trl.mjlab.native23_projected_target_ppo import projection_objective_contract
    from gear_sonic.trl.mjlab.native23_root_feedback_actor import ROOT_FEEDBACK_ARCHITECTURE
    from gear_sonic.trl.mjlab.native23_root_feedback_runner import OPTIMIZER_PROFILES
    from gear_sonic.utils.g1_true23_root_feedback_objectives import objective_profile_contract

    timing = getattr(args, "reference_timing", CAUSAL_TIMING)
    objective = objective_profile_contract(args.objective_profile)
    return {
        **(
            {"ppo_auxiliary_objective": projection_objective_contract(args.ppo_auxiliary_objective)}
            if getattr(args, "ppo_auxiliary_objective", "none") != "none"
            else {}
        ),
        **(
            {
                "simulation_batching": {
                    "profile": "parallel128_v1",
                    "num_envs": args.num_envs,
                    "rollout_steps": args.rollout_steps,
                    "transitions_per_update": args.num_envs * args.rollout_steps,
                    "ppo_epochs": args.ppo_epochs,
                    "ppo_minibatches": args.ppo_minibatches,
                    "maximum_updates_between_cpu_evaluations": 100,
                    "robot_physics_or_acceptance_limits_changed": False,
                }
            }
            if getattr(args, "campaign_batching", "standard32") == "parallel128_v1"
            else {}
        ),
        **({"reference_timing": timing} if timing != CAUSAL_TIMING else {}),
        **(
            {"training_physics": args.training_physics_contract}
            if getattr(args, "training_physics_contract", None) is not None
            else {}
        ),
        **(
            {"release_compatibility": args.release_compatibility_contract}
            if getattr(args, "release_compatibility_contract", None) is not None
            else {}
        ),
        "objective_profile": args.objective_profile,
        "start_schedule": start_schedule_contract(args.start_schedule),
        "objective_profile_contract": objective,
        "feature_contract": root_feedback_contract(timing),
        "architecture": ROOT_FEEDBACK_ARCHITECTURE,
        "optimizer_profile": args.optimizer_profile,
        "optimizer_learning_rate_multipliers": dict(OPTIMIZER_PROFILES[args.optimizer_profile]),
        "curriculum": curriculum,
        "experiment_class": experiment_class(args.mode),
        "continuation": getattr(args, "continuation_contract", None),
        "reset_position_range_m": args.reset_position_range_m,
        "reset_velocity_range_m_s": args.reset_velocity_range_m_s,
        "root_world_tracking_error_weight": objective.get("root_world_tracking_error_weight", -10.0),
        "reward_task_target_frame": "received_horizon_q1_fixed_world"
        if timing == BUFFERED_TIMING
        else "q10_fixed_world",
        "objective_contract": selected_root_objectives(timing),
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
    install_root_feedback_command(rows, start_schedule=args.start_schedule)
    original_builder = task.make_causal_history_recovery_env_cfg

    def build_env(**kwargs):
        cfg = configure_root_feedback_environment(
            original_builder(**kwargs),
            rows,
            reset_position_range_m=args.reset_position_range_m,
            reset_velocity_range_m_s=args.reset_velocity_range_m_s,
            objective_profile=args.objective_profile,
        )
        if args.start_schedule == "staggered_standing_start_v1":
            from gear_sonic.envs.mjlab.sonic_true23_staggered_starts import configure_staggered_start_timeout

            cfg = configure_staggered_start_timeout(cfg, rows)
        if args.release_source_geometry is not None:
            from gear_sonic.envs.mjlab.sonic_true23_release_compatible import (
                configure_release_compatible_environment,
            )

            cfg = configure_release_compatible_environment(
                cfg, args.release_source_geometry, args.release_action_convention
            )
        if args.training_physics_profile == "pinned_cpu_referee_scene_v1":
            from gear_sonic.envs.mjlab.sonic_true23_nominal_scene import configure_nominal_scene

            cfg = configure_nominal_scene(cfg, args.native_model, args.sim_config)
        if args.reference_timing == BUFFERED_TIMING:
            from gear_sonic.envs.mjlab.sonic_true23_buffered_source import configure_buffered_source_environment

            cfg = configure_buffered_source_environment(cfg)
        return cfg

    task.make_causal_history_recovery_env_cfg = build_env
    original_cfg = config.true23_mjlab_ppo_runner_cfg
    inherited_actor_type = type(original_cfg().actor)

    @dataclass
    class RootActorCfg(inherited_actor_type):
        class_name: str = "gear_sonic.trl.mjlab.native23_root_feedback_actor:True23RootFeedbackActorModel"
        root_feedback_obs_group: str = "root_feedback"
        release_compatibility: dict | None = None

    def agent_cfg():
        cfg = original_cfg()
        cfg.actor = RootActorCfg(
            **{
                field.name: getattr(cfg.actor, field.name)
                for field in fields(cfg.actor)
                if field.name != "class_name"
            }
        )
        cfg.actor.release_compatibility = getattr(args, "release_compatibility_contract", None)
        cfg.obs_groups = {"actor": ("tokenizer", "policy", "root_feedback"), "critic": ("critic", "root_feedback")}
        cfg.experiment_name = "sonic_native23_root_feedback_simulation"
        cfg.run_name = args.curriculum_stage
        cfg.num_steps_per_env = args.rollout_steps
        cfg.algorithm.num_learning_epochs = args.ppo_epochs
        cfg.algorithm.num_mini_batches = args.ppo_minibatches
        if args.ppo_auxiliary_objective != "none":
            cfg.algorithm.class_name = "gear_sonic.trl.mjlab.native23_projected_target_ppo:ProjectedTargetPPO"
        return cfg

    config.true23_mjlab_ppo_runner_cfg = agent_cfg
    feedback = feedback_training_contract(args, curriculum)

    class RootFeedbackRunner(Native23RootFeedbackRunner):
        @property
        def semantic_profile(self):
            from gear_sonic.utils.g1_true23_buffered_reference import reference_profile_contract

            return reference_profile_contract(args.reference_timing)["profile"]

        def __init__(self, *values, **kwargs):
            guard()
            super().__init__(*values, **kwargs)
            if args.mode == "campaign":
                self.initialize_evaluated_continuation(args.continuation_contract)
            elif args.resume is None:
                initialize_base_actor(self, curriculum["parent_actor_initialization"])
            guard_algorithm(self.alg, guard)
            if args.ppo_auxiliary_objective != "none":
                from gear_sonic.trl.mjlab.native23_projected_target_ppo import WEIGHT, ProjectedTargetPPO

                if type(self.alg) is not ProjectedTargetPPO or self.alg.mean_projection_weight != WEIGHT:
                    raise ValueError("executed PPO auxiliary objective differs from bound configuration")
                write_runtime(
                    self.checkpoint_dir.parent / "ppo_auxiliary_initial.json",
                    self.alg.projection_runtime_receipt(),
                )
            if args.training_physics_profile == "pinned_cpu_referee_scene_v1":
                from gear_sonic.envs.mjlab.sonic_true23_nominal_scene import verify_nominal_scene

                parity = verify_nominal_scene(self.env.unwrapped.sim.mj_model, args.native_model, args.sim_config)
                write_runtime(self.checkpoint_dir.parent / "nominal_scene_parity.json", parity)
            if args.release_source_geometry is not None:
                from gear_sonic.envs.mjlab.sonic_true23_release_compatible import (
                    verify_executed_release_environment,
                )

                parity = verify_executed_release_environment(self.env.unwrapped, args.release_source_geometry)
                write_runtime(self.checkpoint_dir.parent / "release_compatibility_parity.json", parity)
            if args.reference_timing == BUFFERED_TIMING:
                from gear_sonic.envs.mjlab.sonic_true23_buffered_source import verify_buffered_source_environment

                parity = verify_buffered_source_environment(self.env.unwrapped)
                write_runtime(self.checkpoint_dir.parent / "buffered_source_parity.json", parity)
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
                if args.ppo_auxiliary_objective != "none":
                    write_runtime(
                        self.checkpoint_dir.parent / f"ppo_auxiliary_runtime_{self.completed_update_count}.json",
                        self.alg.projection_runtime_receipt(),
                    )
                receipt = self.env.unwrapped.command_manager.get_term("motion").curriculum_receipt()
                receipt.update(
                    stage=curriculum["stage"],
                    experiment_class=feedback["experiment_class"],
                    deployment_ready=False,
                    physical_estimator_qualified=False,
                )
                if args.reference_timing == BUFFERED_TIMING:
                    receipt.update(
                        reference_timing=args.reference_timing,
                        root_feedback=root_feedback_contract(args.reference_timing),
                        reward_task_target_frame=feedback["reward_task_target_frame"],
                        objective_contract=feedback["objective_contract"],
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
        if args.release_source_geometry is not None:
            result["native23_root_feedback/source29_reference_geometry.xml"] = args.release_source_geometry
        transition = (getattr(args, "continuation_contract", None) or {}).get("reference_transition")
        if transition is not None:
            for index, path in enumerate(sorted(transition["inputs"])):
                result[f"native23_root_feedback/reference_transition/{index:03d}-{Path(path).name}"] = Path(path)
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
        subparser.add_argument("--reference-timing", choices=REFERENCE_TIMINGS, default=CAUSAL_TIMING)
        subparser.add_argument(
            "--release-source-geometry",
            type=Path,
            help="Opt-in versioned source29 reference and bounded linear action training",
        )
        subparser.add_argument("--continue-from", type=Path)
        subparser.add_argument(
            "--training-physics-profile",
            choices=("legacy_training_asset", "pinned_cpu_referee_scene_v1"),
            default="legacy_training_asset",
            help="Opt-in training asset/contact/solver match; CPU evaluation and motor limits unchanged",
        )
        subparser.add_argument(
            "--release-action-convention",
            choices=("released_bounded_linear", "released29_scale_bounded_linear_v2"),
            default="released_bounded_linear",
        )
        subparser.add_argument("--continuation-evaluation", type=Path)
        subparser.add_argument("--allow-reference-transition", action="store_true")
        subparser.add_argument("--previous-reference-report", type=Path)
        subparser.add_argument("--reference-original-time-audit", type=Path)
        subparser.add_argument(
            "--objective-profile",
            choices=(
                "legacy_root_tracking",
                "root_and_posture_v1",
                "root_and_upper_posture_v2",
                "root_and_upper_world_priority_v3",
                "root_and_upper_feet_world_v4",
                "root_and_upper_feet_sole_world_v5",
                "root_and_upper_feet_swing_load_v6",
            ),
            default="legacy_root_tracking",
        )
        subparser.add_argument("--allow-objective-transition", action="store_true")
        subparser.add_argument(
            "--reference-bank",
            action="store_true",
            help="Use an audited local bank and an index of separate per-clip CPU comparisons",
        )
        subparser.add_argument("--allow-reference-bank-transition", action="store_true")
        subparser.add_argument("--lifecycle-reference-repairs", type=Path)
        subparser.add_argument("--reference-bank-repair-plan", type=Path)
        subparser.add_argument(
            "--source-start-registration",
            choices=("none", "once_only_source_start_se2_v1"),
            default="none",
            help="Explicit fixed first-source SE(2) calibration; original audited bank stays unchanged",
        )
        subparser.add_argument("--allow-source-start-registration-transition", action="store_true")
        subparser.add_argument(
            "--source-reference-conditioning",
            choices=("none", "bounded_contact_conditioned_source_v1"),
            default="none",
            help="Explicit changed-reference SIM experiment; no inherited raw fidelity or force feasibility",
        )
        subparser.add_argument("--allow-contact-conditioning-transition", action="store_true")
        subparser.add_argument(
            "--contact-step-references",
            type=Path,
            help="Explicit generated contact-step reference index; source bank stays unchanged",
        )
        subparser.add_argument("--allow-contact-step-transition", action="store_true")
        subparser.add_argument(
            "--ppo-auxiliary-objective", choices=("none", "source_target_projection_l2_v1"), default="none"
        )
        subparser.add_argument("--allow-ppo-objective-transition", action="store_true")
        subparser.add_argument("--start-schedule", choices=START_SCHEDULES, default="synchronous")
        subparser.add_argument("--allow-start-schedule-transition", action="store_true")
        subparser.add_argument("--reset-position-range-m", type=float, default=0.1)
        subparser.add_argument("--reset-velocity-range-m-s", type=float, default=0.1)
        subparser.add_argument("--ppo-epochs", type=int, default=2)
        subparser.add_argument("--ppo-minibatches", type=int, default=4)
        subparser.add_argument("--rollout-steps", type=int, default=32)
        subparser.add_argument(
            "--campaign-batching",
            choices=("standard32", "parallel128_v1"),
            default="standard32",
            help="Opt-in local simulation resource scale; never changes physical or qualification limits.",
        )
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
    if getattr(args, "objective_profile", "legacy_root_tracking") == "root_and_upper_feet_swing_load_v6" and (
        args.training_physics_profile != "pinned_cpu_referee_scene_v1"
    ):
        raise ValueError("swing-load objective requires pinned nominal physics and its fixed flat floor")
    bank_mode = getattr(args, "reference_bank", False)
    registration = getattr(args, "source_start_registration", "none")
    registration_transition = getattr(args, "allow_source_start_registration_transition", False)
    conditioning = getattr(args, "source_reference_conditioning", "none")
    contact_transition = getattr(args, "allow_contact_conditioning_transition", False)
    steps = getattr(args, "contact_step_references", None)
    step_transition = getattr(args, "allow_contact_step_transition", False)
    if steps is not None and (
        not bank_mode
        or args.mode != "campaign"
        or args.curriculum_stage != "lifecycle"
        or conditioning != "bounded_contact_conditioned_source_v1"
        or registration_transition
        or getattr(args, "allow_reference_bank_transition", False)
        or getattr(args, "reference_bank_repair_plan", None) is not None
    ):
        raise ValueError("contact-step references require an explicit conditioned full-lifecycle campaign")
    if step_transition and steps is None:
        raise ValueError("contact-step transition requires its separately bound reference index")
    if conditioning not in {"none", "bounded_contact_conditioned_source_v1"} or (
        conditioning != "none"
        and (
            not bank_mode
            or registration != "once_only_source_start_se2_v1"
            or getattr(args, "lifecycle_reference_repairs", None) is None
        )
    ):
        raise ValueError("contact conditioning requires an explicitly registered evaluated bank")
    if contact_transition and (
        conditioning == "none"
        or not bank_mode
        or registration_transition
        or getattr(args, "allow_reference_bank_transition", False)
        or getattr(args, "reference_bank_repair_plan", None) is not None
    ):
        raise ValueError("contact conditioning requires its own explicit changed-reference transition")
    if registration not in {"none", "once_only_source_start_se2_v1"}:
        raise ValueError("unsupported source-start registration profile")
    if registration != "none" and (not bank_mode or getattr(args, "lifecycle_reference_repairs", None) is None):
        raise ValueError("registered training requires an evaluated bank and every registered ramp repair")
    if registration_transition and (
        registration == "none"
        or not bank_mode
        or getattr(args, "allow_reference_bank_transition", False)
        or getattr(args, "reference_bank_repair_plan", None) is not None
    ):
        raise ValueError("source-start registration requires its own explicit unchanged-source bank transition")
    if getattr(args, "lifecycle_reference_repairs", None) is not None and not bank_mode:
        raise ValueError("generated ramp reference index requires explicit evaluated --reference-bank")
    if getattr(args, "reference_bank_repair_plan", None) is not None and (
        not bank_mode or getattr(args, "allow_reference_bank_transition", False)
    ):
        raise ValueError("reference repair plan requires evaluated bank mode and cannot also request expansion")
    if getattr(args, "allow_reference_bank_transition", False) and not bank_mode:
        raise ValueError("bank expansion requires explicit --reference-bank")
    if bank_mode and (
        args.mode != "campaign"
        or args.curriculum_stage != "lifecycle"
        or getattr(args, "reference_timing", CAUSAL_TIMING) != BUFFERED_TIMING
        or args.training_physics_profile != "pinned_cpu_referee_scene_v1"
        or getattr(args, "allow_reference_transition", False)
    ):
        raise ValueError(
            "reference bank requires evaluated buffered nominal lifecycle without another reference change"
        )
    reference_change = getattr(args, "allow_reference_transition", False)
    reference_proofs = (
        getattr(args, "previous_reference_report", None),
        getattr(args, "reference_original_time_audit", None),
    )
    if reference_change:
        if (
            args.mode != "campaign"
            or args.curriculum_stage != "lifecycle"
            or getattr(args, "reference_timing", CAUSAL_TIMING) != BUFFERED_TIMING
            or args.training_physics_profile != "pinned_cpu_referee_scene_v1"
            or any(path is None for path in reference_proofs)
        ):
            raise ValueError(
                "reference transition requires evaluated buffered nominal lifecycle and both proof files"
            )
    elif any(path is not None for path in reference_proofs):
        raise ValueError("reference proof files require explicit --allow-reference-transition")
    if getattr(args, "allow_ppo_objective_transition", False) and args.mode != "campaign":
        raise ValueError("PPO objective transition requires an evaluated campaign")
    if getattr(args, "ppo_auxiliary_objective", "none") != "none" and (
        args.reference_timing != BUFFERED_TIMING
        or args.training_physics_profile != "pinned_cpu_referee_scene_v1"
        or args.release_action_convention != "released29_scale_bounded_linear_v2"
    ):
        raise ValueError("source projection objective requires buffered source-scaled nominal-physics training")
    batching = getattr(args, "campaign_batching", "standard32")
    if batching == "parallel128_v1" and (
        args.mode != "campaign"
        or args.num_envs < 4
        or args.num_envs % 4
        or args.reference_timing != BUFFERED_TIMING
        or args.training_physics_profile != "pinned_cpu_referee_scene_v1"
        or args.rollout_steps != 64
    ):
        raise ValueError(
            "parallel128 batching requires evaluated buffered nominal-physics campaign and 64-step rollouts"
        )
    if getattr(args, "reference_timing", CAUSAL_TIMING) == BUFFERED_TIMING and (
        args.release_source_geometry is None
        or args.release_action_convention != "released29_scale_bounded_linear_v2"
        or args.training_physics_profile != "pinned_cpu_referee_scene_v1"
        or args.curriculum_stage != "lifecycle"
        or args.resume is not None
        or args.initialize_actor_from is not None
    ):
        raise ValueError(
            "buffered source currently requires fresh lifecycle initialization, "
            "nominal physics and source-scaled actions"
        )
    if args.release_action_convention != "released_bounded_linear" and args.release_source_geometry is None:
        raise ValueError("source-scaled action requires explicit release source geometry")
    if args.start_schedule in ("mixed_reference_reset_v1", "phase_balanced_reference_reset_v1") and (
        args.curriculum_stage != "lifecycle" or args.num_envs < 4 or args.num_envs % 4
    ):
        raise ValueError("mixed reference resets require lifecycle curriculum and a multiple of four environments")
    if args.allow_start_schedule_transition and args.mode != "campaign":
        raise ValueError("explicit start schedule transitions apply only to evaluated campaign continuations")
    if args.release_source_geometry is not None and args.initialize_actor_from is not None:
        raise ValueError("new release semantics require fresh initialization, not legacy actor transfer")
    if args.allow_objective_transition and args.mode != "campaign":
        raise ValueError("explicit objective transitions apply only to evaluated campaign continuations")
    if args.mode == "smoke" and (args.iterations > 2 or args.num_envs > 4):
        raise ValueError("root-feedback smoke is bounded to 2 updates and 4 environments")
    if args.mode == "regression" and (args.iterations > 100 or args.num_envs > 32):
        raise ValueError("local regression is bounded to 100 updates and 32 environments")
    if args.mode == "campaign":
        maximum_envs = 128 if batching == "parallel128_v1" else 32
        if args.iterations > 5000 or args.num_envs > maximum_envs:
            raise ValueError(f"local campaign is bounded to 5000 total updates and {maximum_envs} environments")
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


def validate_campaign_reference_count(args, count):
    if type(count) is not int:
        raise ValueError("campaign reference count must be integral")
    if getattr(args, "reference_bank", False):
        if not 2 <= count <= 16:
            raise ValueError("explicit local bank campaign requires 2..16 complete references")
    elif count != 1:
        raise ValueError("local campaign requires one complete reference or explicit audited --reference-bank")


def main(argv=None):
    from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model
    from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision

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
    if args.release_source_geometry is not None:
        from gear_sonic.utils.g1_true23_release_compatibility import release_compatibility_contract

        args.release_source_geometry = args.release_source_geometry.expanduser().resolve(strict=True)
        args.release_compatibility_contract = release_compatibility_contract(
            sha256_file(args.release_source_geometry), args.release_action_convention, args.reference_timing
        )
    if args.training_physics_profile == "pinned_cpu_referee_scene_v1":
        from gear_sonic.envs.mjlab.sonic_true23_nominal_scene import nominal_scene_contract

        args.training_physics_contract = nominal_scene_contract(args.native_model, args.sim_config)
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
    lifecycle_repairs = None
    if args.lifecycle_reference_repairs is not None:
        from gear_sonic.scripts.g1_true23_reference_bank_campaign import load_bank_lifecycle_repairs

        lifecycle_repairs, repair_inputs = load_bank_lifecycle_repairs(
            args.lifecycle_reference_repairs,
            args.motion_metadata,
            source_start_registration=args.source_start_registration,
            source_reference_conditioning=args.source_reference_conditioning,
        )
        inputs["generated_lifecycle_repair_inputs"] = repair_inputs
    contact_steps = None
    if args.contact_step_references is not None:
        from gear_sonic.utils.g1_true23_contact_step_bank_reference import load_step_reference_index

        contact_steps, step_inputs = load_step_reference_index(args.contact_step_references, args.motion_metadata)
        inputs["generated_contact_step_inputs"] = step_inputs
    derived, generated_spans, curriculum = derive_curriculum(
        motion,
        spans,
        inputs,
        stage=args.curriculum_stage,
        model=model,
        simulation_config=args.sim_config,
        return_target=args.return_target,
        lifecycle_repairs=lifecycle_repairs,
        source_start_registration=args.source_start_registration,
        source_reference_conditioning=args.source_reference_conditioning,
        contact_step_references=contact_steps,
    )
    curriculum["native_reference_model_sha256"] = sha256_file(args.native_model)
    curriculum["native_reference_sim_config_sha256"] = sha256_file(args.sim_config)
    if args.start_schedule in ("mixed_reference_reset_v1", "phase_balanced_reference_reset_v1"):
        curriculum["reset_start"] = (
            "mixed_configured_standing_and_sampled_source_q10"
            if args.start_schedule == "mixed_reference_reset_v1"
            else "quarter_standing_entry_source_and_return_q10"
        )
        curriculum["every_training_episode_requests_full_lifecycle"] = False
        curriculum["sampled_training_suffixes_qualify_full_lifecycle"] = False
    if args.mode == "campaign":
        from gear_sonic.utils.g1_true23_root_feedback_campaign import campaign_parent_contract

        validate_campaign_reference_count(args, len(spans["spans"]))
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
