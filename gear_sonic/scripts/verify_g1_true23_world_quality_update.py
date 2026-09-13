"""Independent reconstruction of joint-task bonus and actual stored PPO rewards."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23_world_tracking_termination import termination_contract
from gear_sonic.scripts import verify_g1_true23_decoder_lora_update as decoder_audit
from gear_sonic.scripts.verify_g1_true23_bounded_progress_update import audit_reward_arrays
from gear_sonic.scripts.verify_g1_true23_world_tracking_update import WORLD_KEYS, audit_world_arrays
from gear_sonic.trl.mjlab.native23_world_quality_runner import validate_checkpoint
from gear_sonic.trl.mjlab.native23_world_tracking_runner import CHECKPOINT_HEADER as BASELINE_HEADER
from gear_sonic.utils import g1_true23_world_quality_checkpoint as reader
from gear_sonic.utils.g1_true23_bounded_progress import GAMMA, reward_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_world_quality import quality_contract

QUALITY_KEYS = {"pre_quality_returned_reward", "world_quality_bonus"}


def audit_quality_arrays(arrays, metadata, updates, rollout_steps, num_envs):
    shape = (updates * rollout_steps, num_envs)
    for name in QUALITY_KEYS:
        value = arrays.get(name)
        if value is None or value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("incomplete/nonfinite actual quality reward capture: " + name)
    # The old independent audit still verifies every original component, Phi,
    # boundary rule and base/shaping sum. Subtract only the captured bonus from
    # actual PPO storage for that audit's explicitly pre-bonus arithmetic view.
    base_view = {key: value for key, value in arrays.items() if key not in WORLD_KEYS | QUALITY_KEYS}
    base_view["returned_reward"] = arrays["pre_quality_returned_reward"]
    base_view["stored_reward_with_timeout_bootstrap"] = (
        arrays["stored_reward_with_timeout_bootstrap"] - arrays["world_quality_bonus"]
    )
    base_report = audit_reward_arrays(base_view, metadata, updates, rollout_steps, num_envs)
    errors = {}

    def close(name, actual, expected, tolerance):
        error = float(np.max(np.abs(actual.astype(np.float64) - expected)))
        errors[name] = error
        if error > tolerance:
            raise ValueError(f"quality arithmetic differs for {name}: {error} > {tolerance}")

    done = arrays["terminated"] | arrays["timeouts"]
    parts = arrays["world_cost_parts_after_including_reset_states"].astype(np.float64)
    # Independent rational formula: no call to the runtime exp(Phi) function.
    expected_bonus = np.where(done, 0.0, 2.0 / (1.0 + parts.sum(-1)))
    bonus = arrays["world_quality_bonus"]
    close("independent_joint_task_bonus", bonus, expected_bonus, 5e-7)
    if np.any(bonus[done] != 0) or np.any(bonus < 0) or np.any(bonus > 2):
        raise ValueError("quality reward pays a reset state or exceeds bounds")
    returned = arrays["returned_reward"]
    expected_returned = arrays["pre_quality_returned_reward"] + bonus
    if not np.array_equal(returned, expected_returned):
        raise ValueError("actual returned reward is not exactly old reward plus one bonus")
    expected_stored = returned.astype(np.float64) + GAMMA * arrays["timeouts"] * arrays[
        "critic_value_before_bootstrap"
    ].astype(np.float64)
    close(
        "actual_augmented_PPO_timeout_bootstrap",
        arrays["stored_reward_with_timeout_bootstrap"],
        expected_stored,
        5e-5,
    )
    return dict(
        all_original_base_and_shaping_equations_unchanged=True,
        original_arithmetic_view=base_report,
        original_storage_view_is_actual_storage_minus_captured_bonus=True,
        bonus_reconstructed_independently_from_cost_parts=True,
        zero_bonus_on_every_failure_and_timeout=True,
        actual_returned_reward_exact_single_bonus_sum=True,
        actual_augmented_PPO_storage_reconstructed=True,
        actual_transitions=int(np.prod(shape)),
        maximum_absolute_reconstruction_errors=errors,
        bonus_min=float(bonus.min()),
        bonus_max=float(bonus.max()),
        bonus_mean=float(bonus.mean()),
        pre_quality_returned_reward_mean=float(arrays["pre_quality_returned_reward"].mean()),
        actual_returned_reward_mean=float(returned.mean()),
        physical_tracking_or_convergence_proved=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "initial-checkpoint",
        "trained-checkpoint",
        "baseline-initial-checkpoint",
        "warm-start",
        "source-checkpoint",
        "reward-capture",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("world quality audit refuses overwrite")
    old_semantics, old_validator = decoder_audit.validate_semantics, decoder_audit.validate_decoder_lora_checkpoint
    decoder_audit.validate_semantics = reader.validate_semantics
    decoder_audit.validate_decoder_lora_checkpoint = validate_checkpoint
    try:
        report = decoder_audit.audit(
            args.initial_checkpoint, args.trained_checkpoint, args.warm_start, args.source_checkpoint
        )
    finally:
        decoder_audit.validate_semantics = old_semantics
        decoder_audit.validate_decoder_lora_checkpoint = old_validator
    inputs = report["inputs"]

    def bind(path):
        path = Path(path).resolve(strict=True)
        inputs[str(path)] = sha256_file(path)
        return path

    baseline = torch.load(bind(args.baseline_initial_checkpoint), map_location="cpu", weights_only=True)
    initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
    if baseline["header"] != BASELINE_HEADER or baseline["trainer_state"]["completed_update_count"] != 0:
        raise ValueError("comparison requires the actual world-tracking predecessor initialization")
    cfg0 = baseline["lineage"]["materials"]["resolved_config"]["payload"]
    cfg1 = initial["lineage"]["materials"]["resolved_config"]["payload"]
    for key in ("seed", "num_envs", "agent"):
        if cfg0[key] != cfg1[key]:
            raise ValueError("initial comparison changes batch/seed/PPO recipe: " + key)
    for label, before, after in (
        ("actor", baseline["actor"]["state_dict"], initial["actor"]["state_dict"]),
        ("critic", baseline["critic_state_dict"], initial["critic_state_dict"]),
    ):
        if set(before) != set(after) or any(not torch.equal(before[key], after[key]) for key in before):
            raise ValueError("initial " + label + " differs from matched world-tracking predecessor")
    report["initial_actor_and_critic_equal_matched_world_predecessor"] = True
    metadata = json.loads(bind(args.reward_capture).read_text())
    if (
        metadata.get("kind") != "native23_world_quality_actual_reward_capture_v1"
        or metadata.get("reward_contract") != reward_contract()
        or metadata.get("world_quality_bonus") != quality_contract()
        or metadata.get("world_tracking_termination") != termination_contract()
        or metadata.get("lineage_sha256") != report["lineage_sha256"]
        or metadata.get("completed_updates") != report["completed_updates"]
        or metadata.get("num_envs") != cfg1["num_envs"]
        or metadata.get("training_state_poisoned") is not False
        or metadata.get("world_and_reward_capture_counts_equal") is not True
        or metadata.get("hardware_authorized") is not False
        or metadata.get("deployment_ready") is not False
    ):
        raise ValueError("training capture contract/counters/lineage incomplete")
    del baseline, initial
    arrays_path = bind(metadata["arrays_path"])
    if inputs[str(arrays_path)] != metadata["arrays_sha256"]:
        raise ValueError("training arrays changed")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    controls = report["completed_updates"] * report["resolved_rollout_steps_per_env"]
    if metadata["actual_world_termination_calls"] != controls or metadata["actual_controls_per_env"] != controls:
        raise ValueError("capture count differs from executed controls")
    report["world_tracking"] = audit_world_arrays(arrays, controls, metadata["num_envs"])
    report["quality_reward_equations"] = audit_quality_arrays(
        arrays,
        metadata,
        report["completed_updates"],
        report["resolved_rollout_steps_per_env"],
        metadata["num_envs"],
    )
    if report["world_tracking"]["actual_transitions"] != report["actual_transitions"]:
        raise ValueError("world-failure capture differs from actual transition count")
    for path in (
        Path(__file__),
        Path(reader.__file__),
        Path(audit_reward_arrays.__code__.co_filename),
        Path(audit_world_arrays.__code__.co_filename),
    ):
        bind(path)
    report.update(
        kind="native23_world_quality_actual_update_audit_v1",
        additional_training_failure=termination_contract(),
        world_quality_bonus=quality_contract(),
        passed=report["update_verification_passed"],
        hardware_authorized=False,
        deployment_ready=False,
    )
    for path, expected in inputs.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("audit input changed")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "passed",
                    "completed_updates",
                    "actual_transitions",
                    "world_tracking",
                    "quality_reward_equations",
                )
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
