"""Simulation-only full-decoder SONIC training with nominal native23 motors.

Smoke runs may use the existing regression corpus, explicitly without a
generalization claim. Training runs require provenance and per-span proof that
no validation/test recording was included in the concatenated training NPZ.
Run this launcher in its own process: legacy corpus hooks are process-local.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_generalist_corpus import audit_manifest, sha256_file
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile

ROOT = Path(__file__).resolve().parents[2]
SIM_CONFIG = ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
MOTION_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def validate_training_inputs(motion: Path, spans: Path, manifest: Path | None, *, smoke_only: bool):
    payload = json.loads(spans.read_text())
    if payload.get("kind") != "g1_true23_motion_corpus_spans_v1" or payload.get("fps") != 50:
        raise ValueError("generalist requires exact 50-Hz corpus spans")
    rows = payload.get("spans")
    if not isinstance(rows, list) or not rows:
        raise ValueError("corpus spans must be nonempty")
    cursor = 0
    for row in rows:
        if (
            type(row.get("start")) is not int
            or row["start"] != cursor
            or type(row.get("length")) is not int
            or row["length"] <= 15
        ):
            raise ValueError("spans must be contiguous and include causal history margins")
        cursor += row["length"]
    if payload.get("total_frames") != cursor or payload.get("clip_count") != len(rows):
        raise ValueError("span total/count mismatch")
    with np.load(motion, allow_pickle=False) as dataset:
        if float(np.asarray(dataset["fps"]).reshape(-1)[0]) != 50:
            raise ValueError("training payload must be 50 Hz")
        for key in MOTION_KEYS:
            if key not in dataset or dataset[key].shape[0] != cursor or not np.isfinite(dataset[key]).all():
                raise ValueError(f"invalid training payload {key}")
        if dataset["joint_pos"].shape != (cursor, 23) or dataset["joint_vel"].shape != (cursor, 23):
            raise ValueError("training motion requires native23 positions/velocities")
        corpus_audit = None
        if manifest is None:
            if not smoke_only:
                raise ValueError("train requires --corpus-manifest; existing regression clips are smoke-only")
        else:
            corpus_audit = audit_manifest(json.loads(manifest.read_text()), manifest.parent)
            for row in rows:
                asset = row.get("asset_id")
                if corpus_audit["asset_splits"].get(asset) != "train":
                    raise ValueError("every training span must identify a train-split asset")
                path = Path(corpus_audit["asset_bindings"][asset]["path"])
                with np.load(path, allow_pickle=False) as clip:
                    for key in MOTION_KEYS:
                        if key not in clip or not np.array_equal(
                            dataset[key][row["start"] : row["start"] + row["length"]], clip[key]
                        ):
                            raise ValueError(
                                f"training span does not exactly match its audited asset: {asset}/{key}"
                            )
    return {
        "kind": "g1_native23_generalist_training_inputs_v1",
        "motion_sha256": sha256_file(motion),
        "spans_sha256": sha256_file(spans),
        "clip_count": len(rows),
        "total_frames": cursor,
        "smoke_only": smoke_only,
        "corpus_audit": corpus_audit,
        "declared_recording_split_integrity_verified": corpus_audit is not None,
        "held_out_policy_generalization_verified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def install_hooks(args, input_contract, precision, guard):
    from gear_sonic.envs.mjlab import sonic_true23_causal_history as task
    from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import make_causal_multimotion_v14_env_cfg
    from gear_sonic.envs.mjlab.sonic_true23_native_model_actuation import apply_native_model_actuation_profile
    from gear_sonic.scripts import train_g1_23dof_mjlab_causal_history as base
    from gear_sonic.scripts.train_g1_23dof_mjlab_teleop_v13 import _install_corpus_command
    from gear_sonic.trl.mjlab import causal_history_runner, config
    from gear_sonic.trl.mjlab.native23_generalist_actor import Native23GeneralistCore
    from gear_sonic.trl.mjlab.native23_generalist_runner import Native23GeneralistRunner
    from gear_sonic.utils.g1_true23_training_precision import guard_algorithm, write_runtime

    profile = NativeModelActuationProfile.from_sim_config(args.sim_config)
    _install_corpus_command(args.spans)

    def build_env(**kwargs):
        return apply_native_model_actuation_profile(
            make_causal_multimotion_v14_env_cfg(**kwargs),
            profile,
            effort_penalty_weight=args.effort_penalty_weight,
        )

    task.make_causal_history_recovery_env_cfg = build_env

    @dataclass
    class GeneralistActorCfg(config.True23SonicActorCfg):
        class_name: str = "gear_sonic.trl.mjlab.native23_generalist_actor:True23Native23GeneralistActorModel"
        source_checkpoint_path: str = str(args.source_checkpoint)
        std_min: float = 0.02
        std_max: float = 0.5

    original_cfg = config.true23_mjlab_ppo_runner_cfg

    def generalist_cfg():
        cfg = original_cfg()
        cfg.actor = GeneralistActorCfg(
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"}
        )
        cfg.experiment_name = "sonic_native23_generalist_simulation"
        cfg.run_name = "nominal_native_model"
        cfg.algorithm.schedule = "fixed"
        cfg.algorithm.desired_kl = None
        return cfg

    config.true23_mjlab_ppo_runner_cfg = generalist_cfg

    class GuardedGeneralistRunner(Native23GeneralistRunner):
        def __init__(self, *values, **kwargs):
            guard()
            super().__init__(*values, **kwargs)
            guard_algorithm(self.alg, guard)
            actor = self.alg.get_policy()
            write_runtime(
                self.checkpoint_dir.parent / "generalist_runtime.json",
                {
                    "actor": actor.artifact_contract(),
                    "actuation": profile.contract(),
                    "training_inputs": input_contract,
                    "precision": precision,
                    "optimizer_groups": [
                        {"name": g["name"], "elements": sum(p.numel() for p in g["params"])}
                        for g in self.alg.optimizer.param_groups
                    ],
                    "full_lifecycle_training": False,
                    "hardware_authorized": False,
                },
            )

    causal_history_runner.CausalHistoryMjlabOnPolicyRunner = GuardedGeneralistRunner
    original_resolved = base._resolved_training_config

    def resolved(*values, **kwargs):
        result = original_resolved(*values, **kwargs)
        result["native23_generalist"] = {
            "source_checkpoint": str(args.source_checkpoint),
            "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
            "trainable": "all_native23_decoder_affine_layers_plus_bounded_exploration_and_critic",
            "encoder_and_fsq_frozen": True,
            "actuation": profile.contract(),
            "requested_effort_penalty_weight": args.effort_penalty_weight,
            "training_inputs": input_contract,
            "precision": precision,
            "current_curriculum": "clip_contained_reference_reset_nominal_acquisition",
            "full_lifecycle_training": False,
            "checkpoint_pattern": "native23_generalist_model_N.pt",
            "deployment_ready": False,
        }
        return result

    base._resolved_training_config = resolved
    original_preflight = base.preflight

    def preflight(values):
        result = original_preflight(values)
        if result["ready"]:
            core = Native23GeneralistCore(
                warm_start_path=args.warm_start, source_checkpoint_path=args.source_checkpoint
            )
            result["native23_generalist"] = core.artifact_contract()
        result["native_model_actuation"] = profile.contract()
        result["training_inputs"] = input_contract
        return result

    base.preflight = preflight
    from gear_sonic.utils.g1_true23_generalist_source_closure import generalist_source_files

    original_source_files = base._source_files

    def source_files():
        # Full package-relative keys retain every __init__.py and similarly
        # named helper; appending to CAUSAL_SOURCE_FILES flattens basenames.
        result = original_source_files()
        result.update(generalist_source_files(ROOT))
        result["native23_generalist/actuation_config.json"] = args.sim_config
        return result

    base._source_files = source_files
    return base


def main(argv=None):
    from gear_sonic.scripts import train_g1_23dof_mjlab_causal_history as base
    from gear_sonic.utils.g1_true23_training_precision import ieee_training_precision

    parser = base._parser()
    # Add common options to each inherited mode without changing old launchers.
    for subparser in parser._subparsers._group_actions[0].choices.values():
        subparser.add_argument("--source-checkpoint", type=Path, required=True)
        subparser.add_argument("--spans", type=Path, required=True)
        subparser.add_argument("--sim-config", type=Path, default=SIM_CONFIG)
        subparser.add_argument("--corpus-manifest", type=Path)
        subparser.add_argument("--effort-penalty-weight", type=float, default=0.1)
        subparser.set_defaults(learning_rate=5e-6, save_interval=100, session_updates=100)
    args = parser.parse_args(argv)
    if args.mode == "smoke" and (args.iterations > 2 or args.num_envs > 16):
        parser.error("unaudited smoke is bounded to 2 updates and 16 environments")
    if args.mode != "smoke" and args.save_interval > 100:
        parser.error("generalist checkpoints must be produced at least every 100 updates")
    if args.session_updates > 100:
        parser.error("run at most 100 updates before independent CPU evaluation")
    for name in ("source_checkpoint", "warm_start", "motion_file", "motion_metadata", "spans", "sim_config"):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    input_contract = validate_training_inputs(
        args.motion_file, args.spans, args.corpus_manifest, smoke_only=args.mode in {"smoke", "preflight"}
    )
    with ieee_training_precision() as (precision, guard):
        base = install_hooks(args, input_contract, precision, guard)
        if args.mode == "preflight":
            report = base.preflight(args)
            print(json.dumps(report, indent=2, sort_keys=True))
            if args.json_output:
                args.json_output.parent.mkdir(parents=True, exist_ok=True)
                with args.json_output.open("x") as stream:
                    json.dump(report, stream, indent=2, sort_keys=True)
            return 0 if report["ready"] else 2
        print(base.run_training(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
