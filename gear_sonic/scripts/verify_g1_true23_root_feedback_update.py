"""Independent root-feedback update evidence, never motion qualification."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from gear_sonic.scripts.export_g1_true23_generalist import validate_export_semantics as validate_base_semantics
from gear_sonic.scripts.export_g1_true23_root_feedback import validate_export_semantics
from gear_sonic.trl.mjlab.native23_generalist_runner import validate_generalist_checkpoint
from gear_sonic.trl.mjlab.native23_root_feedback_actor import True23RootFeedbackActorModel
from gear_sonic.trl.mjlab.native23_root_feedback_runner import validate_root_feedback_checkpoint
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

ENCODER_PREFIX = "core.actor_module.encoders.teleop."
DECODER_PREFIX = "core.actor_module.decoders.g1_dyn."


def finite_training_state_summary(checkpoint):
    """Explicit critic/optimizer finiteness; hashes alone cannot prove this."""
    from gear_sonic.utils.g1_23dof_mjlab_training import _validate_critic_state, _validate_optimizer_state

    critic, _ = _validate_critic_state(checkpoint["critic_state_dict"])
    optimizer = _validate_optimizer_state(checkpoint["optimizer_state_dict"])
    tensors = [
        value
        for state in optimizer["state"].values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    return {
        "kind": "g1_native23_root_feedback_finite_training_state_v1",
        "critic_tensor_count": len(critic),
        "critic_total_elements": sum(value.numel() for value in critic.values()),
        "critic_all_tensors_finite": True,
        "critic_tensor_max_abs": {name: float(value.abs().max()) for name, value in critic.items()},
        "optimizer_parameter_state_count": len(optimizer["state"]),
        "optimizer_state_tensor_count": len(tensors),
        "optimizer_all_numeric_values_finite": True,
        "optimizer_group_names": [group["name"] for group in optimizer["param_groups"]],
        "optimizer_nonempty": bool(optimizer["state"]),
        "motion_qualification_proven": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def summarize_state_changes(before, after, exploration):
    encoder = [key for key in before if key.startswith(ENCODER_PREFIX)]
    decoder = [key for key in before if key.startswith(DECODER_PREFIX)]
    if (
        len(encoder) != 10
        or len(decoder) != 18
        or set(before) != {*encoder, *decoder, "root_conditioner.weight", "distribution.raw_std"}
        or set(before) != set(after)
    ):
        raise ValueError("root-feedback state partition must be encoder10/decoder18/conditioner1/noise1")
    if any(not torch.isfinite(value).all() for state in (before, after) for value in state.values()):
        raise ValueError("root-feedback update audit requires finite state tensors")
    std = (
        exploration["std_min"]
        + (exploration["std_max"] - exploration["std_min"]) * after["distribution.raw_std"].sigmoid()
    )
    changed = {key: not torch.equal(before[key], after[key]) for key in decoder}
    return {
        "checks": {
            "all_18_decoder_tensors_changed": all(changed.values()),
            "all_10_encoder_tensors_unchanged": all(torch.equal(before[key], after[key]) for key in encoder),
            "initial_root_conditioner_exactly_zero": torch.count_nonzero(before["root_conditioner.weight"]).item()
            == 0,
            "root_conditioner_changed": not torch.equal(
                before["root_conditioner.weight"], after["root_conditioner.weight"]
            ),
            "bounded_exploration_parameter_changed": not torch.equal(
                before["distribution.raw_std"], after["distribution.raw_std"]
            ),
            "std_finite_and_within_bounds": bool(
                torch.isfinite(std).all()
                and (std >= exploration["std_min"]).all()
                and (std <= exploration["std_max"]).all()
            ),
        },
        "decoder_tensor_changed": changed,
        "conditioner_changed_elements": int(
            torch.count_nonzero(before["root_conditioner.weight"] - after["root_conditioner.weight"]).item()
        ),
        "conditioner_total_elements": after["root_conditioner.weight"].numel(),
        "conditioner_max_abs": float(after["root_conditioner.weight"].abs().max()),
        "std_min_observed": float(std.min()),
        "std_max_observed": float(std.max()),
    }


def root_action_sensitivity(actor):
    """Isolated unit feedback perturbations, fixed semantic/proprio inputs."""
    obs = {"tokenizer": torch.zeros(10, 267), "policy": torch.zeros(10, 930), "root_feedback": torch.zeros(10, 9)}
    obs["root_feedback"][1:] = torch.eye(9)
    with torch.no_grad():
        output = actor(obs)
    if output.shape != (10, 23) or not torch.isfinite(output).all():
        raise ValueError("root-feedback sensitivity probe produced invalid actions")
    differences = (output[1:] - output[:1]).abs().amax(dim=-1)
    return {
        "probe_definition": "nine_positive_unit_basis_feedback_vectors_vs_zero_same_zero267_and_zero930",
        "per_feature_action_max_abs_delta": differences.tolist(),
        "any_root_feature_changes_action": bool((differences > 0).any()),
        "all_nine_root_features_change_action": bool((differences > 0).all()),
        "maximum_action_delta": float(differences.max()),
        "root_tracking_quality_established": False,
    }


def _load(path):
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError("root-feedback update checkpoint may not be a symlink")
    path = path.resolve(strict=True)
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if sha256_file(path) != digest:
        raise ValueError("root-feedback update checkpoint changed during read")
    return path, digest, checkpoint


def audit_update(*, initial_path, trained_path, parent_path, warm_start_path, source_checkpoint_path):
    initial_path, initial_hash, initial = _load(initial_path)
    trained_path, trained_hash, trained = _load(trained_path)
    parent_path, parent_hash, parent = _load(parent_path)
    initial_semantics = validate_export_semantics(initial)
    trained_semantics = validate_export_semantics(trained)
    validate_base_semantics(parent)
    if initial["lineage_sha256"] != trained["lineage_sha256"] or initial_semantics != trained_semantics:
        raise ValueError("root-feedback update audit requires identical run lineage")
    bound_parent = initial_semantics["root_feedback_training_configuration"]["curriculum"][
        "parent_actor_initialization"
    ]
    if not isinstance(bound_parent, dict) or (
        bound_parent.get("checkpoint_sha256") != parent_hash
        or bound_parent.get("actor_state_sha256") != parent["actor"]["state_sha256"]
    ):
        raise ValueError("root-feedback initial checkpoint does not bind supplied old parent")
    exploration = trained["actor"]["contract"]["exploration"]
    actor = True23RootFeedbackActorModel(
        {"tokenizer": torch.zeros(1, 267), "policy": torch.zeros(1, 930), "root_feedback": torch.zeros(1, 9)},
        {"actor": ["tokenizer", "policy", "root_feedback"]},
        "actor",
        23,
        warm_start_path=str(warm_start_path),
        source_checkpoint_path=str(source_checkpoint_path),
        std_min=exploration["std_min"],
        std_max=exploration["std_max"],
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "std_type": "scalar",
            "init_std": exploration["init_std"],
        },
    )
    proxy = SimpleNamespace(validate_training_artifact=actor.validate_base_training_artifact)
    validate_generalist_checkpoint(parent, actor=proxy, lineage=parent["lineage"])
    validate_root_feedback_checkpoint(initial, actor=actor, lineage=initial["lineage"])
    validate_root_feedback_checkpoint(trained, actor=actor, lineage=initial["lineage"], minimum_update_count=1)
    before, after, original = (
        initial["actor"]["state_dict"],
        trained["actor"]["state_dict"],
        parent["actor"]["state_dict"],
    )
    changes = summarize_state_changes(before, after, exploration)
    actor.load_training_artifact(initial["actor"])
    initial_sensitivity = root_action_sensitivity(actor)
    actor.load_training_artifact(trained["actor"])
    trained_sensitivity = root_action_sensitivity(actor)
    checks = changes.pop("checks")
    checks.update(
        {
            "all_29_old_parent_tensors_exact_in_model0": len(original) == 29
            and all(torch.equal(value, before[key]) for key, value in original.items()),
            "initial_root_feedback_has_exact_zero_effect": not initial_sensitivity[
                "any_root_feature_changes_action"
            ],
            "trained_root_feedback_changes_action": trained_sensitivity["any_root_feature_changes_action"],
            "initial_optimizer_empty": not initial["optimizer_state_dict"]["state"],
            "trained_optimizer_finite_valid_and_nonempty": bool(trained["optimizer_state_dict"]["state"]),
            "initial_counters_zero": initial["trainer_state"]["completed_update_count"] == 0
            and initial["trainer_state"]["env_common_step_counter"] == 0,
            "positive_completed_ppo_updates": trained["trainer_state"]["completed_update_count"]
            > initial["trainer_state"]["completed_update_count"],
            "environment_counter_advanced": trained["trainer_state"]["env_common_step_counter"]
            > initial["trainer_state"]["env_common_step_counter"],
        }
    )
    report = {
        "schema_version": 2,
        "kind": "g1_native23_root_feedback_actual_update_audit",
        "initial_checkpoint_sha256": initial_hash,
        "trained_checkpoint_sha256": trained_hash,
        "parent_checkpoint_sha256": parent_hash,
        "trained_actor_state_sha256": trained["actor"]["state_sha256"],
        "lineage_sha256": trained["lineage_sha256"],
        "completed_update_count": trained["trainer_state"]["completed_update_count"],
        **changes,
        "checks": checks,
        "update_verification_passed": all(checks.values()),
        "initial_sensitivity": initial_sensitivity,
        "trained_sensitivity": trained_sensitivity,
        "motion_qualification_proven": False,
        "physical_estimator_qualified": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    del actor, initial, trained, parent
    gc.collect()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "initial-checkpoint",
        "trained-checkpoint",
        "parent-checkpoint",
        "warm-start",
        "source-checkpoint",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite root-feedback update receipt")
    report = audit_update(
        initial_path=args.initial_checkpoint,
        trained_path=args.trained_checkpoint,
        parent_path=args.parent_checkpoint,
        warm_start_path=args.warm_start,
        source_checkpoint_path=args.source_checkpoint,
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["update_verification_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
