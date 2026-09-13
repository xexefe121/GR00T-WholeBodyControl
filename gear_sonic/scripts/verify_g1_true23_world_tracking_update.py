"""Verify actual world-failure calls, PPO rewards, frozen base and updates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23_world_tracking_termination import THRESHOLD_M, termination_contract
from gear_sonic.scripts import verify_g1_true23_decoder_lora_update as decoder_audit
from gear_sonic.scripts.verify_g1_true23_bounded_progress_update import audit_reward_arrays
from gear_sonic.trl.mjlab.native23_bounded_progress_runner import CHECKPOINT_HEADER as OLD_HEADER
from gear_sonic.trl.mjlab.native23_world_tracking_runner import validate_checkpoint
from gear_sonic.utils import g1_true23_world_tracking_checkpoint as reader
from gear_sonic.utils.g1_true23_bounded_progress import reward_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

WORLD_KEYS = {
    "world_tracking_desired_position_w", "world_tracking_measured_position_w", "world_tracking_error_m",
    "world_tracking_failure", "world_tracking_reference_q0", "world_tracking_common_step_counter",
}


def audit_world_arrays(arrays, controls, num_envs):
    shape = (controls, num_envs)
    for key, dimensions in {
        "world_tracking_desired_position_w": (*shape, 3),
        "world_tracking_measured_position_w": (*shape, 3),
        "world_tracking_error_m": shape, "world_tracking_failure": shape,
        "world_tracking_reference_q0": shape, "world_tracking_common_step_counter": (controls,),
    }.items():
        if key not in arrays or arrays[key].shape != dimensions or not np.isfinite(arrays[key]).all():
            raise ValueError("incomplete/nonfinite world-failure capture: " + key)
    desired = arrays["world_tracking_desired_position_w"]
    measured = arrays["world_tracking_measured_position_w"]
    error = arrays["world_tracking_error_m"]
    if any(value.dtype != np.float32 for value in (desired, measured, error)):
        raise ValueError("actual world position/error capture must be float32")
    failure = arrays["world_tracking_failure"]
    if failure.dtype != np.bool_:
        raise ValueError("world-failure flags must be boolean")
    expected_error = np.linalg.norm(desired.astype(np.float64) - measured.astype(np.float64), axis=-1)
    reconstruction_error = float(np.max(np.abs(error - expected_error)))
    if reconstruction_error > 5e-7:
        raise ValueError("world-root error does not match actual recorded positions")
    expected_failure = error > np.float32(THRESHOLD_M)
    if not np.array_equal(failure, expected_failure):
        raise ValueError("actual world-root flags differ from the declared strict threshold")
    if np.any(failure & ~arrays["terminated"]) or np.any(failure & ~arrays["stored_done"]):
        raise ValueError("world failure omitted from true-terminal or PPO done flags")
    counters, anchors = arrays["world_tracking_common_step_counter"], arrays["world_tracking_reference_q0"]
    if counters.dtype != np.int64 or anchors.dtype != np.int64 or np.any(anchors < 0):
        raise ValueError("world-failure control counters/anchors invalid")
    np.testing.assert_array_equal(counters, np.arange(1, controls + 1, dtype=np.int64))
    return dict(
        every_actual_world_error_reconstructed=True,
        actual_true_failure_flags_equal_declared_condition=True,
        world_failures_in_actual_termination_and_PPO_done=True,
        actual_transitions=controls * num_envs,
        world_failure_transitions=int(failure.sum()),
        other_failure_transitions_without_world_failure=int((arrays["terminated"] & ~failure).sum()),
        simultaneous_world_failure_and_timeout=int((failure & arrays["timeouts"]).sum()),
        error_reconstruction_max_abs_m=reconstruction_error,
        actual_error_max_m=float(error.max()),
        actual_error_p95_m=float(np.percentile(error, 95)),
        shorter_training_episodes_are_not_motion_success=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("initial-checkpoint", "trained-checkpoint", "baseline-initial-checkpoint",
                 "warm-start", "source-checkpoint", "reward-capture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("world tracking audit refuses overwrite")
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
    if baseline["header"] != OLD_HEADER or baseline["trainer_state"]["completed_update_count"] != 0:
        raise ValueError("comparison requires the actual bounded-predecessor initialization")
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
            raise ValueError("initial " + label + " differs from matched bounded predecessor")
    report["initial_actor_and_critic_equal_matched_bounded_predecessor"] = True
    metadata_path = bind(args.reward_capture)
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get("reward_contract") != reward_contract()
            or metadata.get("world_tracking_termination") != termination_contract()
            or metadata.get("lineage_sha256") != report["lineage_sha256"]
            or metadata.get("completed_updates") != report["completed_updates"]
            or metadata.get("training_state_poisoned") is not False
            or metadata.get("world_and_reward_capture_counts_equal") is not True):
        raise ValueError("training capture contract/counters/lineage incomplete")
    arrays_path = bind(metadata["arrays_path"])
    if inputs[str(arrays_path)] != metadata["arrays_sha256"]:
        raise ValueError("training arrays changed")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    controls = report["completed_updates"] * report["resolved_rollout_steps_per_env"]
    if metadata["actual_world_termination_calls"] != controls:
        raise ValueError("world failure capture count differs from executed controls")
    report["world_tracking"] = audit_world_arrays(arrays, controls, metadata["num_envs"])
    report["reward_equations"] = audit_reward_arrays(
        {key: value for key, value in arrays.items() if key not in WORLD_KEYS},
        metadata, report["completed_updates"], report["resolved_rollout_steps_per_env"], metadata["num_envs"]
    )
    if report["world_tracking"]["actual_transitions"] != report["actual_transitions"]:
        raise ValueError("world-failure capture differs from actual transition count")
    for path in (Path(__file__), Path(reader.__file__)):
        bind(path)
    report.update(
        kind="native23_world_tracking_actual_update_audit_v1",
        additional_training_failure=termination_contract(),
        old_reward_contract_describes_component_arithmetic_not_new_total_termination_flags=True,
        passed=report["update_verification_passed"],
    )
    for path, expected in inputs.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("audit input changed")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({key: report[key] for key in
                      ("passed", "completed_updates", "actual_transitions", "world_tracking")}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
