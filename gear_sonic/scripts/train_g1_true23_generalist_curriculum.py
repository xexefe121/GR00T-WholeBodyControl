"""Bounded nominal acquisition/lifecycle training; no deployment authorization.

Each invocation selects one stage. To continue acquisition into lifecycle,
explicitly transfer its actor with --initialize-actor-from. This intentionally
starts a fresh critic/optimizer/lineage, unlike exact same-stage --resume.
Generated references are stored separately from the training output directory.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from gear_sonic.envs.mjlab.sonic_true23_generalist_curriculum import (
    configure_curriculum_environment,
    install_curriculum_command,
)
from gear_sonic.scripts import train_g1_true23_generalist as generalist
from gear_sonic.utils.g1_true23_generalist_curriculum import STAGES, derive_curriculum, write_curriculum_bundle
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def parent_actor_contract(path, training_inputs=None):
    if path is None:
        return None
    import torch
    from gear_sonic.scripts.export_g1_true23_generalist import validate_export_semantics

    path = path.expanduser().resolve(strict=True)
    before = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    semantics = validate_export_semantics(checkpoint)
    if checkpoint["trainer_state"]["completed_update_count"] < 1:
        raise ValueError("parent actor transfer requires at least one completed training update")
    if training_inputs is not None and not training_inputs["smoke_only"]:
        previous = semantics["training_configuration"]["training_inputs"].get("corpus_audit")
        current = training_inputs.get("corpus_audit")
        if (
            previous is None
            or current is None
            or any(previous[key] != current[key] for key in ("manifest_sha256", "split_sha256"))
        ):
            raise ValueError("production parent transfer requires identical audited corpus ownership and split")
    if sha256_file(path) != before:
        raise ValueError("parent actor checkpoint changed during validation")
    contract = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": before,
        "actor_state_sha256": checkpoint["actor"]["state_sha256"],
        "lineage_sha256": checkpoint["lineage_sha256"],
        "completed_update_count": checkpoint["trainer_state"]["completed_update_count"],
        "actor_and_bounded_noise_transferred": True,
        "critic_optimizer_counters_or_rng_resumed": False,
    }
    del checkpoint
    gc.collect()
    return contract


def initialize_actor_from_parent(runner, parent):
    """Explicit weight transfer, never optimizer/counter resume or relabeling."""
    if parent is None:
        return
    import torch
    from gear_sonic.trl.mjlab.native23_generalist_runner import validate_generalist_checkpoint
    from gear_sonic.scripts.export_g1_true23_generalist import validate_export_semantics

    path = Path(parent["checkpoint_path"])
    if sha256_file(path) != parent["checkpoint_sha256"]:
        raise ValueError("parent checkpoint no longer matches bound hash")
    if runner.completed_update_count != 0 or runner.alg.optimizer.state:
        raise ValueError("parent transfer requires fresh optimizer and counters")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_export_semantics(checkpoint)
    actor = runner.alg.get_policy()
    validate_generalist_checkpoint(checkpoint, actor=actor, lineage=checkpoint["lineage"])
    if checkpoint["actor"]["state_sha256"] != parent["actor_state_sha256"]:
        raise ValueError("parent actor differs from bound transfer contract")
    actor.load_training_artifact(checkpoint["actor"])
    runner._assert_boundary()
    del checkpoint
    gc.collect()


def install_curriculum_hooks(args, original_inputs, contract, precision, guard):
    from gear_sonic.envs.mjlab import sonic_true23_causal_history as task
    from gear_sonic.trl.mjlab import causal_history_runner
    from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
    from gear_sonic.utils.g1_true23_training_precision import write_runtime

    inputs = {**original_inputs, "derived_curriculum": contract}
    base = generalist.install_hooks(args, inputs, precision, guard)
    rows = contract["derived_spans"]["spans"]
    install_curriculum_command(rows)
    original_builder = task.make_causal_history_recovery_env_cfg

    def build_env(**kwargs):
        return configure_curriculum_environment(original_builder(**kwargs), rows)

    task.make_causal_history_recovery_env_cfg = build_env
    previous_runner = causal_history_runner.CausalHistoryMjlabOnPolicyRunner

    class CurriculumRunner(previous_runner):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, **kwargs)
            if args.resume is None:
                initialize_actor_from_parent(self, contract["parent_actor_initialization"])

        def learn(self, num_learning_iterations, init_at_random_ep_len=False):
            # A random global time limit can cut a full source before its end.
            # It is not a reference-start curriculum and is disabled explicitly.
            try:
                return super().learn(num_learning_iterations, init_at_random_ep_len=False)
            finally:
                env = self.env.unwrapped
                receipt = env.command_manager.get_term("motion").curriculum_receipt()
                receipt.update(
                    stage=contract["stage"],
                    original_inputs_smoke_only=original_inputs["smoke_only"],
                    full_lifecycle_training_completed=False,
                )
                write_runtime(
                    self.checkpoint_dir.parent / f"curriculum_runtime_{self.completed_update_count}.json", receipt
                )

    causal_history_runner.CausalHistoryMjlabOnPolicyRunner = CurriculumRunner
    original_resolved = base._resolved_training_config

    def resolved(*values, **kwargs):
        result = original_resolved(*values, **kwargs)
        result["native23_generalist"]["current_curriculum"] = contract["stage"]
        result["native23_generalist"]["curriculum"] = contract
        return result

    base._resolved_training_config = resolved
    original_files = base._source_files

    def source_files():
        result = original_files()
        result.update(
            collect_local_source_closure(generalist.ROOT, [Path(__file__)]).as_source_files(generalist.ROOT)
        )
        result["native23_generalist/curriculum.spans.json"] = args.spans
        return result

    base._source_files = source_files
    return base


def main(argv=None):
    from gear_sonic.scripts import train_g1_23dof_mjlab_causal_history as base
    from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision
    from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

    parser = base._parser()
    for subparser in parser._subparsers._group_actions[0].choices.values():
        subparser.add_argument("--source-checkpoint", type=Path, required=True)
        subparser.add_argument("--spans", type=Path, required=True)
        subparser.add_argument("--sim-config", type=Path, default=generalist.SIM_CONFIG)
        subparser.add_argument("--corpus-manifest", type=Path)
        subparser.add_argument("--effort-penalty-weight", type=float, default=0.1)
        subparser.add_argument("--curriculum-stage", choices=STAGES, required=True)
        subparser.add_argument("--curriculum-directory", type=Path, required=True)
        subparser.add_argument("--initialize-actor-from", type=Path)
        subparser.add_argument(
            "--native-model",
            type=Path,
            default=generalist.ROOT.parent
            / "GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        )
        subparser.set_defaults(learning_rate=5e-6, save_interval=100, session_updates=100)
    args = parser.parse_args(argv)
    if args.mode == "smoke" and (args.iterations > 2 or args.num_envs > 4):
        parser.error("curriculum smoke is bounded to 2 updates and 4 environments")
    if args.session_updates > 100 or args.save_interval > 100:
        parser.error("curriculum requires checkpoints/evaluation at most every 100 updates")
    if args.resume is not None and args.initialize_actor_from is not None:
        parser.error("exact resume and parent actor initialization are mutually exclusive")
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
    if args.curriculum_directory.resolve() == args.run_dir.resolve():
        parser.error("curriculum inputs and training output require separate directories")
    original_inputs = generalist.validate_training_inputs(
        args.motion_file, args.spans, args.corpus_manifest, smoke_only=args.mode in {"smoke", "preflight"}
    )
    spans = json.loads(args.spans.read_text())
    with np.load(args.motion_file, allow_pickle=False) as archive:
        motion = {key: archive[key].copy() for key in generalist.MOTION_KEYS}
    if (
        sha256_file(args.motion_file) != original_inputs["motion_sha256"]
        or sha256_file(args.spans) != original_inputs["spans_sha256"]
    ):
        raise ValueError("original curriculum inputs changed after validation")
    _, model, _ = prepare_true23_model(args.native_model, args.sim_config)
    derived, generated_spans, contract = derive_curriculum(
        motion, spans, original_inputs, stage=args.curriculum_stage, model=model, simulation_config=args.sim_config
    )
    if args.resume is not None:
        previous_metadata = json.loads((args.curriculum_directory / "curriculum.json").read_text())
        previous_parent = previous_metadata["curriculum"]["parent_actor_initialization"]
        parent = parent_actor_contract(
            None if previous_parent is None else Path(previous_parent["checkpoint_path"]), original_inputs
        )
        if parent != previous_parent:
            raise ValueError("resume parent actor binding changed")
    else:
        parent = parent_actor_contract(args.initialize_actor_from, original_inputs)
    contract["parent_actor_initialization"] = parent
    contract["native_reference_model_sha256"] = sha256_file(args.native_model)
    contract["native_reference_sim_config_sha256"] = sha256_file(args.sim_config)
    # Resuming requires the identical derivation. Verify rather than overwrite.
    if args.resume is not None:
        directory = args.curriculum_directory.resolve(strict=True)
        metadata = json.loads((directory / "curriculum.json").read_text())
        if metadata.get("curriculum") != contract:
            parser.error("resume curriculum differs; use explicit parent actor transfer for a new stage")
        generated = directory / "curriculum.npz"
        if metadata["output"]["sha256"] != sha256_file(generated):
            parser.error("resume curriculum payload hash mismatch")
        args.motion_file, args.motion_metadata, args.spans = (
            generated,
            directory / "curriculum.json",
            directory / "curriculum.spans.json",
        )
        if json.loads(args.spans.read_text()) != generated_spans:
            parser.error("resume curriculum spans mismatch")
    else:
        args.motion_file, args.motion_metadata, args.spans = write_curriculum_bundle(
            args.curriculum_directory.resolve(), derived, generated_spans, contract
        )
    del model, motion, derived
    gc.collect()
    with ieee_training_precision() as (precision, guard):
        runner_base = install_curriculum_hooks(args, original_inputs, contract, precision, guard)
        if args.mode == "preflight":
            report = runner_base.preflight(args)
            print(json.dumps(report, indent=2, sort_keys=True))
            if args.json_output:
                with args.json_output.open("x") as stream:
                    json.dump(report, stream, indent=2, sort_keys=True)
            return 0 if report["ready"] else 2
        print(runner_base.run_training(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
