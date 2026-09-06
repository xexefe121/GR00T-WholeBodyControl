"""Inspect actual checkpoint deltas; never equate an update with motion success."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import _tensor_state_sha256
from gear_sonic.trl.mjlab.native23_generalist_runner import CHECKPOINT_HEADER
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file
from gear_sonic.utils.g1_23dof_mjlab_training import validate_mjlab_training_lineage


def _load(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("header") != CHECKPOINT_HEADER:
        raise ValueError("update audit requires generalist training checkpoints")
    lineage = validate_mjlab_training_lineage(checkpoint["lineage"])
    if checkpoint["lineage_sha256"] != lineage["lineage_sha256"]:
        raise ValueError("update audit lineage hash mismatch")
    state = checkpoint["actor"]["state_dict"]
    if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("update audit actor state contains non-finite values")
    if checkpoint["actor"]["state_sha256"] != _tensor_state_sha256(state):
        raise ValueError("update audit actor state hash mismatch")
    return checkpoint


def audit_update(initial_path: str | Path, trained_path: str | Path) -> dict[str, Any]:
    initial_path, trained_path = Path(initial_path).resolve(strict=True), Path(trained_path).resolve(strict=True)
    initial, trained = _load(initial_path), _load(trained_path)
    if (
        initial["lineage_sha256"] != trained["lineage_sha256"]
        or initial["actor"]["contract"] != trained["actor"]["contract"]
    ):
        raise ValueError("update audit requires identical actor contract and training lineage")
    before, after = initial["actor"]["state_dict"], trained["actor"]["state_dict"]
    if set(before) != set(after):
        raise ValueError("update audit actor state keys differ")
    contract = trained["actor"]["contract"]
    encoder_prefix, decoder_prefix = "core.actor_module.encoders.teleop.", "core.actor_module.decoders.g1_dyn."
    encoder = [name for name in before if name.startswith(encoder_prefix)]
    decoder = [name for name in before if name.startswith(decoder_prefix)]
    if len(encoder) != 10 or len(decoder) != 18 or set(before) != {*encoder, *decoder, "distribution.raw_std"}:
        raise ValueError("update audit actor parameter partition mismatch")
    changed = {name: not torch.equal(before[name], after[name]) for name in decoder}
    frozen = {name.removeprefix(encoder_prefix): after[name] for name in encoder}
    std_cfg = contract["exploration"]
    std = std_cfg["std_min"] + (std_cfg["std_max"] - std_cfg["std_min"]) * after["distribution.raw_std"].sigmoid()
    policy = {
        name.removeprefix("core."): value for name, value in after.items() if name.startswith("core.actor_module.")
    }
    policy["std"] = std
    policy_hash = inspect_true23_policy_state(
        {"policy_state_dict": policy}, reference_profile=contract["reference_profile"]
    )
    start, end = initial["trainer_state"], trained["trainer_state"]
    checks = {
        "all_18_decoder_tensors_changed": all(changed.values()),
        "all_10_encoder_tensors_unchanged": all(torch.equal(before[name], after[name]) for name in encoder),
        "encoder_matches_pinned_contract": _tensor_state_sha256(frozen) == contract["frozen_encoder_sha256"],
        "std_finite_and_within_bounds": bool(
            torch.isfinite(std).all() and (std >= std_cfg["std_min"]).all() and (std <= std_cfg["std_max"]).all()
        ),
        "bounded_exploration_parameter_changed": not torch.equal(
            before["distribution.raw_std"], after["distribution.raw_std"]
        ),
        "positive_completed_ppo_updates": isinstance(end["completed_update_count"], int)
        and end["completed_update_count"] > start["completed_update_count"],
        "trainer_counters_coherent": end["completed_update_count"] == end["current_learning_iteration"],
        "environment_counter_advanced": end["env_common_step_counter"] > start["env_common_step_counter"],
        "optimizer_state_nonempty": bool(trained["optimizer_state_dict"]["state"]),
    }
    return {
        "schema_version": 1,
        "kind": "g1_native23_generalist_actual_update_audit",
        "initial_checkpoint_sha256": sha256_file(initial_path),
        "trained_checkpoint_sha256": sha256_file(trained_path),
        "lineage_sha256": trained["lineage_sha256"],
        "trained_actor_state_sha256": trained["actor"]["state_sha256"],
        "trained_policy_state_sha256": policy_hash,
        "initial_completed_update_count": start["completed_update_count"],
        "trained_completed_update_count": end["completed_update_count"],
        "std_min_observed": float(std.min()),
        "std_max_observed": float(std.max()),
        "decoder_tensor_changed": changed,
        "checks": checks,
        "update_verification_passed": all(checks.values()),
        "motion_qualification_proven": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--trained-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite update receipt: {args.output}")
    report = audit_update(args.initial_checkpoint, args.trained_checkpoint)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["update_verification_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
