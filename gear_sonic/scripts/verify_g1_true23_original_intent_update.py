"""Verify actual original-intent PPO updates, never motion qualification."""

import argparse
import json
from pathlib import Path

import torch

from gear_sonic.scripts.export_g1_true23_root_feedback import validate_export_semantics as old_export_semantics
from gear_sonic.scripts.verify_g1_true23_root_feedback_update import (
    _load,
    finite_training_state_summary,
    root_action_sensitivity,
    summarize_state_changes,
)
from gear_sonic.trl.mjlab.native23_root_feedback_runner import validate_root_feedback_checkpoint
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_original_intent_checkpoint import construct_actor, validate_semantics


def audit(initial_path, trained_path, warm_start_path, source_checkpoint_path):
    paths = [
        Path(p).resolve(strict=True) for p in (initial_path, trained_path, warm_start_path, source_checkpoint_path)
    ]
    inputs = {str(p): sha256_file(p) for p in paths}
    from gear_sonic.utils import g1_true23_original_intent_checkpoint as reader

    inputs[str(Path(reader.__file__).resolve())] = sha256_file(Path(reader.__file__))
    inputs[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    _, _, initial = _load(paths[0])
    _, _, trained = _load(paths[1])
    before_semantics, after_semantics = validate_semantics(initial), validate_semantics(trained)
    if before_semantics != after_semantics or initial["lineage_sha256"] != trained["lineage_sha256"]:
        raise ValueError("original-intent update requires the same training run")
    inputs.update(before_semantics["reverified_repository_sources"])
    for row in before_semantics["spec"]["files"].values():
        inputs[row["path"]] = row["sha256"]
    actor = construct_actor(initial, before_semantics, warm_start_path=paths[2], source_checkpoint_path=paths[3])
    validate_root_feedback_checkpoint(initial, actor=actor, lineage=initial["lineage"])
    validate_root_feedback_checkpoint(trained, actor=actor, lineage=initial["lineage"], minimum_update_count=1)
    original = actor.export_training_artifact()["state_dict"]
    before, after = initial["actor"]["state_dict"], trained["actor"]["state_dict"]
    changes = summarize_state_changes(before, after, trained["actor"]["contract"]["exploration"])
    actor.load_training_artifact(initial["actor"])
    sensitivity0 = root_action_sensitivity(actor)
    actor.load_training_artifact(trained["actor"])
    sensitivity1 = root_action_sensitivity(actor)
    old_rejections = []
    for value in (initial, trained):
        try:
            old_export_semantics(value)
        except ValueError as error:
            old_rejections.append(str(error))
        else:
            raise ValueError("old exporter unexpectedly accepts the new recipe")
    cfg = before_semantics["resolved"]
    actual_steps = trained["trainer_state"]["env_common_step_counter"]
    updates = trained["trainer_state"]["completed_update_count"]
    checks = changes.pop("checks")
    checks.update(
        initial_actor_exact_row_trimmed_release=all(torch.equal(v, before[k]) for k, v in original.items()),
        initial_counters_zero=initial["trainer_state"]["completed_update_count"] == 0
        and initial["trainer_state"]["env_common_step_counter"] == 0,
        initial_optimizer_empty=not initial["optimizer_state_dict"]["state"],
        actual_counter_matches_resolved_rollout=actual_steps == updates * cfg["agent"]["num_steps_per_env"],
        initial_feedback_exact_zero=not sensitivity0["any_root_feature_changes_action"],
        trained_feedback_changes_action=sensitivity1["any_root_feature_changes_action"],
        old_exporter_rejects_both_checkpoints=len(old_rejections) == 2,
        critic_source_input_dimension_286=trained["critic_state_dict"]["mlp.0.weight"].shape == (512, 286),
    )
    finite = finite_training_state_summary(trained)
    checks["optimizer_finite_nonempty"] = finite["optimizer_nonempty"]
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"audit input changed: {path}")
    return dict(
        kind="native23_original_intent_actual_update_audit_v1",
        checks=checks,
        update_verification_passed=all(checks.values()),
        completed_updates=updates,
        actual_vector_environment_steps=actual_steps,
        actual_transitions=actual_steps * cfg["num_envs"],
        resolved_rollout_steps_per_env=cfg["agent"]["num_steps_per_env"],
        resolved_ppo_epochs=cfg["agent"]["algorithm"]["num_learning_epochs"],
        resolved_ppo_minibatches=cfg["agent"]["algorithm"]["num_mini_batches"],
        actor_contract=trained["actor"]["contract"],
        lineage_sha256=trained["lineage_sha256"],
        finite_training_state=finite,
        initial_sensitivity=sensitivity0,
        trained_sensitivity=sensitivity1,
        old_export_rejection_reasons=old_rejections,
        inputs=inputs,
        **changes,
        motion_qualification_proven=False,
        hardware_authorized=False,
        deployment_ready=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("initial-checkpoint", "trained-checkpoint", "warm-start", "source-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("original-intent audit refuses overwrite")
    report = audit(args.initial_checkpoint, args.trained_checkpoint, args.warm_start, args.source_checkpoint)
    report["inputs"][str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("update_verification_passed", "completed_updates", "actual_transitions", "checks")
            }
        ),
        flush=True,
    )
    return 0 if report["update_verification_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
