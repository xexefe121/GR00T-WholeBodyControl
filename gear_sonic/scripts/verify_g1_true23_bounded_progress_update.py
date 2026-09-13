"""Independent NumPy reward reconstruction plus existing frozen-base PPO audit."""

import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.scripts import verify_g1_true23_decoder_lora_update as decoder_audit
from gear_sonic.trl.mjlab.native23_bounded_progress_runner import validate_checkpoint
from gear_sonic.utils import g1_true23_bounded_progress_checkpoint as reader
from gear_sonic.utils.g1_true23_bounded_progress import (
    ALIVE_WEIGHT,
    FAILURE_WEIGHT,
    GAMMA,
    NEGATIVE_WEIGHTS,
    POSITIVE_WEIGHTS,
    STEP_DT,
    reward_contract,
)
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def audit_reward_arrays(arrays, metadata, updates, rollout_steps, num_envs):
    shape = (updates * rollout_steps, num_envs)
    scalar_names = (
        "phi_before",
        "phi_after_including_reset_states",
        "base_reward",
        "shaping_reward",
        "returned_reward",
        "critic_value_before_bootstrap",
        "stored_reward_with_timeout_bootstrap",
    )
    flag_names = ("terminated", "timeouts", "stored_done", "ppo_recorded")
    dimensions = {
        **{name: shape for name in scalar_names + flag_names},
        "raw_costs": (*shape, len(NEGATIVE_WEIGHTS)),
        "weighted_base_components": (*shape, 21),
        "world_cost_parts_before": (*shape, 3),
        "world_cost_parts_after_including_reset_states": (*shape, 3),
    }
    if set(arrays) != set(dimensions) or any(arrays[k].shape != v for k, v in dimensions.items()):
        raise ValueError("reward capture is incomplete or has unexpected arrays")
    for name, value in arrays.items():
        if name in flag_names:
            if value.dtype != np.bool_:
                raise ValueError("reward audit requires actual boolean flags")
        elif value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("reward audit requires finite actual float32 training tensors")
    names = metadata["base_component_names"]
    expected_weights = {
        **NEGATIVE_WEIGHTS,
        **POSITIVE_WEIGHTS,
        "alive": ALIVE_WEIGHT,
        "non_timeout_termination": FAILURE_WEIGHT,
    }
    if len(names) != len(set(names)) or set(names) != set(expected_weights):
        raise ValueError("reward component inventory differs")
    if metadata["raw_cost_names"] != list(NEGATIVE_WEIGHTS):
        raise ValueError("raw-cost capture order differs")
    raw = arrays["raw_costs"].astype(np.float64)
    components = arrays["weighted_base_components"].astype(np.float64)
    if (raw < 0).any():
        raise ValueError("negative raw cost in capture")
    errors = {}

    def close(name, actual, expected, tolerance=3e-5):
        error = float(np.max(np.abs(actual - expected)))
        errors[name] = error
        if error > tolerance:
            raise ValueError(f"reward arithmetic differs for {name}: {error} > {tolerance}")

    for index, (name, weight) in enumerate(NEGATIVE_WEIGHTS.items()):
        expected = weight * STEP_DT * raw[..., index] / (1 + raw[..., index])
        close(name, components[..., names.index(name)], expected, 5e-7)
    for name, weight in POSITIVE_WEIGHTS.items():
        actual = components[..., names.index(name)]
        if (actual < -1e-7).any() or (actual > weight * STEP_DT + 1e-7).any():
            raise ValueError("raw positive term exceeds declared0..1 range")
    terminated, timeouts = arrays["terminated"], arrays["timeouts"]
    close("alive", components[..., names.index("alive")], (~terminated) * ALIVE_WEIGHT * STEP_DT, 5e-7)
    close(
        "termination",
        components[..., names.index("non_timeout_termination")],
        terminated * FAILURE_WEIGHT * STEP_DT,
    )
    close("component_sum", arrays["base_reward"], components.sum(-1))
    lower, upper = np.where(terminated, -102.206, 0.1), np.where(terminated, -99.9, 2.406)
    if (arrays["base_reward"] < lower - 5e-5).any() or (arrays["base_reward"] > upper + 5e-5).any():
        raise ValueError("actual base reward violates proved bounds")
    before, after = (
        arrays["phi_before"].astype(np.float64),
        arrays["phi_after_including_reset_states"].astype(np.float64),
    )
    for name, phi in (("before", before), ("after_including_reset_states", after)):
        parts = arrays["world_cost_parts_" + name].astype(np.float64)
        if (parts < 0).any():
            raise ValueError("negative world potential cost")
        close("potential_" + name, phi, -np.log1p(parts.sum(-1)), 3e-6)
    shaping = GAMMA * after - before
    shaping[terminated] = -before[terminated]
    shaping[timeouts] = (GAMMA - 1) * before[timeouts]
    close("boundary_correct_shaping", arrays["shaping_reward"], shaping, 3e-6)
    close("returned_reward", arrays["returned_reward"], arrays["base_reward"].astype(np.float64) + shaping)
    stored = arrays["returned_reward"].astype(np.float64) + GAMMA * timeouts * arrays[
        "critic_value_before_bootstrap"
    ].astype(np.float64)
    close("actual_PPO_timeout_bootstrap", arrays["stored_reward_with_timeout_bootstrap"], stored, 5e-5)
    if not arrays["ppo_recorded"].all() or not np.array_equal(arrays["stored_done"], terminated | timeouts):
        raise ValueError("actual PPO storage is incomplete or changes termination flags")
    return {
        "all_reward_and_PPO_storage_equations_reconstructed": True,
        "actual_transitions": int(np.prod(shape)),
        "true_terminal_transitions": int(terminated.sum()),
        "timeout_transitions": int(timeouts.sum()),
        "simultaneous_failure_and_timeout_transitions": int((terminated & timeouts).sum()),
        "maximum_absolute_reconstruction_errors": errors,
        "base_reward_min": float(arrays["base_reward"].min()),
        "base_reward_max": float(arrays["base_reward"].max()),
        "shaping_reward_min": float(arrays["shaping_reward"].min()),
        "shaping_reward_max": float(arrays["shaping_reward"].max()),
        "returned_reward_mean": float(arrays["returned_reward"].mean()),
        "new_reward_proves_physical_tracking_or_no_failure_preference": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "initial-checkpoint",
        "trained-checkpoint",
        "warm-start",
        "source-checkpoint",
        "reward-capture",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("bounded reward audit refuses overwrite")
    # Keep all existing tensor, source, frozen-base, optimizer and counter checks.
    # Only the explicit new checkpoint validator replaces the old header reader.
    previous_semantics, previous_validator = (
        decoder_audit.validate_semantics,
        decoder_audit.validate_decoder_lora_checkpoint,
    )
    decoder_audit.validate_semantics = reader.validate_semantics
    decoder_audit.validate_decoder_lora_checkpoint = validate_checkpoint
    try:
        report = decoder_audit.audit(
            args.initial_checkpoint, args.trained_checkpoint, args.warm_start, args.source_checkpoint
        )
    finally:
        decoder_audit.validate_semantics = previous_semantics
        decoder_audit.validate_decoder_lora_checkpoint = previous_validator
    metadata_path = args.reward_capture.resolve(strict=True)
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("reward_contract") != reward_contract()
        or metadata.get("lineage_sha256") != report["lineage_sha256"]
        or metadata.get("training_state_poisoned") is not False
        or metadata.get("completed_updates") != report["completed_updates"]
    ):
        raise ValueError("reward capture contract/counters/lineage differ")
    arrays_path = Path(metadata["arrays_path"]).resolve(strict=True)
    if sha256_file(arrays_path) != metadata["arrays_sha256"]:
        raise ValueError("reward arrays changed")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    report["reward_equations"] = audit_reward_arrays(
        arrays,
        metadata,
        report["completed_updates"],
        report["resolved_rollout_steps_per_env"],
        metadata["num_envs"],
    )
    if report["reward_equations"]["actual_transitions"] != report["actual_transitions"]:
        raise ValueError("reward capture disagrees with checkpoint transition count")
    for path in (Path(__file__), Path(reader.__file__), metadata_path, arrays_path):
        report["inputs"][str(path.resolve())] = sha256_file(path)
    report.update(
        kind="native23_bounded_progress_actual_update_and_reward_audit_v1", reward_contract=reward_contract()
    )
    report["passed"] = (
        report["update_verification_passed"]
        and report["reward_equations"]["all_reward_and_PPO_storage_equations_reconstructed"]
    )
    for path, expected in report["inputs"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("audit input changed during execution")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("passed", "completed_updates", "actual_transitions", "reward_equations", "checks")
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
