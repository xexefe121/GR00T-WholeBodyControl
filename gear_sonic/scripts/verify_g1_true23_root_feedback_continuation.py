"""Audit real checkpoint continuation bytes without mistaking it for promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gear_sonic.scripts.export_g1_true23_root_feedback import validate_export_semantics
from gear_sonic.scripts.verify_g1_true23_root_feedback_update import (
    _load,
    finite_training_state_summary,
    summarize_state_changes,
)
from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256


def exact_tree_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def audit_continuation(parent_path, initial_path, trained_path):
    parent_path, parent_hash, parent = _load(parent_path)
    initial_path, initial_hash, initial = _load(initial_path)
    trained_path, trained_hash, trained = _load(trained_path)
    semantics = [validate_export_semantics(value) for value in (parent, initial, trained)]
    continuation = semantics[1]["root_feedback_training_configuration"]["continuation"]
    if continuation["checkpoint_sha256"] != parent_hash:
        raise ValueError("initial continuation does not bind supplied parent bytes")
    if initial["lineage_sha256"] != trained["lineage_sha256"]:
        raise ValueError("trained checkpoint changed continuation lineage")
    for value in (parent, initial, trained):
        if _state_sha256(value["actor"]["state_dict"]) != value["actor"]["state_sha256"]:
            raise ValueError("actor state bytes differ from stored hash")
        if _state_sha256(value["critic_state_dict"]) != value["critic_state_sha256"]:
            raise ValueError("critic state bytes differ from stored hash")
    checks = {
        "actor_exact_at_continuation": exact_tree_equal(parent["actor"], initial["actor"]),
        "critic_exact_at_continuation": exact_tree_equal(
            parent["critic_state_dict"], initial["critic_state_dict"]
        ),
        "optimizer_exact_at_continuation": exact_tree_equal(
            parent["optimizer_state_dict"], initial["optimizer_state_dict"]
        ),
        "trainer_counters_exact_at_continuation": exact_tree_equal(
            parent["trainer_state"], initial["trainer_state"]
        ),
        "new_continuation_lineage_not_old_lineage": initial["lineage_sha256"] != parent["lineage_sha256"],
        "updates_advanced_within_evaluated_block": 0
        < trained["trainer_state"]["completed_update_count"] - initial["trainer_state"]["completed_update_count"]
        <= 100,
        "environment_counter_advanced": trained["trainer_state"]["env_common_step_counter"]
        > initial["trainer_state"]["env_common_step_counter"],
    }
    changes = summarize_state_changes(
        initial["actor"]["state_dict"], trained["actor"]["state_dict"], trained["actor"]["contract"]["exploration"]
    )
    update_checks = changes.pop("checks")
    # A continuation's conditioner is already trained, unlike fresh model0.
    initial_was_zero = update_checks.pop("initial_root_conditioner_exactly_zero")
    checks.update(update_checks)
    report = {
        "kind": "g1_native23_root_feedback_actual_continuation_audit_v1",
        "parent_checkpoint_sha256": parent_hash,
        "initial_checkpoint_sha256": initial_hash,
        "trained_checkpoint_sha256": trained_hash,
        "lineage_sha256": trained["lineage_sha256"],
        "initial_update_count": initial["trainer_state"]["completed_update_count"],
        "completed_update_count": trained["trainer_state"]["completed_update_count"],
        "initial_conditioner_was_zero": initial_was_zero,
        "checks": checks,
        "continuation_update_verification_passed": all(checks.values()),
        "finite_state": finite_training_state_summary(trained),
        **changes,
        "simulator_or_rng_state_resumed": False,
        "motion_qualification_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-checkpoint", "initial-checkpoint", "trained-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = audit_continuation(args.parent_checkpoint, args.initial_checkpoint, args.trained_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["continuation_update_verification_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
