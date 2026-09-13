"""Independent captured-mean/loss/reward and real-checkpoint audit."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
import torch

from gear_sonic.scripts import verify_g1_true23_decoder_lora_update as decoder_audit
from gear_sonic.scripts.verify_g1_true23_world_quality_update import audit_quality_arrays
from gear_sonic.scripts.verify_g1_true23_world_tracking_update import audit_world_arrays
from gear_sonic.trl.mjlab.native23_world_projection_runner import (
    ALGORITHM,
    training_contract,
    validate_checkpoint,
)
from gear_sonic.trl.mjlab.native23_world_quality_runner import CHECKPOINT_HEADER as BASELINE_HEADER
from gear_sonic.utils import g1_true23_world_projection_checkpoint as reader
from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF, NATIVE_IL23_ACTION_SCALE
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE,
)
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_SCALE_NATIVE_IL23


def independent_bounds():
    order = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
    scales = np.asarray(NATIVE_IL23_ACTION_SCALE)
    source = np.asarray(SOURCE_SCALE_NATIVE_IL23)
    limits = []
    for capacity in (SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE, SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE):
        cap = np.asarray(capacity)[order]
        limits.append(cap * np.minimum(np.tanh(9.999 * scales / cap), 1 - 1e-6) / source)
    return -limits[0], limits[1]


def audit_projection_arrays(arrays, metadata, updates, rollout_steps, num_envs, epochs, batches):
    if (
        metadata.get("training_contract") != training_contract()
        or metadata.get("training_state_poisoned") is not False
    ):
        raise ValueError("projection capture contract invalid or poisoned")
    if metadata.get("completed_updates") != updates:
        raise ValueError("projection capture update counter differs")
    rollout_count, minibatch_count = updates * rollout_steps, updates * epochs * batches
    batch_size = num_envs * rollout_steps // batches
    if metadata["actual_rollout_calls"] != rollout_count or metadata["actual_minibatch_calls"] != minibatch_count:
        raise ValueError("projection capture missing actual model calls")
    for key, shape in (
        ("rollout_mean", (rollout_count, num_envs, 23)),
        ("rollout_std", (rollout_count, num_envs, 23)),
        ("rollout_sample", (rollout_count, num_envs, 23)),
        ("minibatch_mean", (minibatch_count, batch_size, 23)),
        ("minibatch_std", (minibatch_count, batch_size, 23)),
    ):
        value = arrays.get(key)
        if value is None or value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("incomplete/nonfinite projection capture: " + key)
    counters = arrays["rollout_common_step_counter"]
    if counters.shape != (rollout_count,) or not np.array_equal(counters, np.arange(rollout_count)):
        raise ValueError("projection rollout counters skip or repeat a physical control")
    if any(np.any(arrays[key] <= 0.02) or np.any(arrays[key] >= 0.5) for key in ("rollout_std", "minibatch_std")):
        raise ValueError("exploration standard deviation outside unchanged bounds")
    receipt = metadata["algorithm_receipt"]
    if (
        receipt["completed_ppo_updates"] != updates
        or receipt["completed_minibatches"] != minibatch_count
        or receipt["coefficient"] != 0.05
    ):
        raise ValueError("actual auxiliary algorithm count or coefficient differs")
    low, high = (x.astype(np.float32).astype(np.float64) for x in independent_bounds())
    scales = np.asarray(SOURCE_SCALE_NATIVE_IL23, dtype=np.float32).astype(np.float64)
    mean = arrays["minibatch_mean"].astype(np.float64)
    projected = np.clip(mean, low, high)
    loss = np.sum(((mean - projected) * scales) ** 2, axis=-1).mean(-1)
    fraction = ((mean < low) | (mean > high)).mean(axis=(1, 2))
    per_update = loss.reshape(updates, epochs * batches).mean(-1)
    per_update_fraction = fraction.reshape(updates, epochs * batches).mean(-1)
    logged = metadata["update_losses"]
    if len(logged) != updates:
        raise ValueError("missing executed update losses")
    loss_error = float(np.max(np.abs(per_update - np.array([r["source_projection"] for r in logged]))))
    fraction_error = float(
        np.max(np.abs(per_update_fraction - np.array([r["source_mean_projected_fraction"] for r in logged])))
    )
    if loss_error > 1e-5 or fraction_error > 1e-7:
        raise ValueError("actual auxiliary loss differs from independent captured-mean calculation")
    raw = arrays["rollout_mean"].astype(np.float64)
    std = arrays["rollout_std"].astype(np.float64)
    zlo, zhi = (low - raw) / std, (high - raw) / std
    probability = np.where(zlo > 0, ndtr(-zlo) - ndtr(-zhi), ndtr(zhi) - ndtr(zlo))
    return dict(
        actual_rollout_calls=rollout_count,
        actual_transitions=rollout_count * num_envs,
        actual_ppo_minibatches=minibatch_count,
        actual_minibatch_policy_mean_rows=minibatch_count * batch_size,
        independently_reconstructed_projection_loss_max_error=loss_error,
        independently_reconstructed_projected_fraction_max_error=fraction_error,
        original_gaussian_std_min=float(std.min()),
        original_gaussian_std_max=float(std.max()),
        actual_training_any_joint_mean_projected_fraction=float(
            np.any((raw < low) | (raw > high), axis=-1).mean()
        ),
        actual_training_any_joint_under_one_percent_unprojected_probability_fraction=float(
            np.any(probability < 0.01, axis=-1).mean()
        ),
        first_update_projection_loss=float(per_update[0]),
        last_update_projection_loss=float(per_update[-1]),
        tracking_improvement_proven=False,
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
        "projection-capture",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("world-projection audit refuses overwrite")
    old_semantics, old_validator = decoder_audit.validate_semantics, decoder_audit.validate_decoder_lora_checkpoint
    decoder_audit.validate_semantics, decoder_audit.validate_decoder_lora_checkpoint = (
        reader.validate_semantics,
        validate_checkpoint,
    )
    try:
        report = decoder_audit.audit(
            args.initial_checkpoint, args.trained_checkpoint, args.warm_start, args.source_checkpoint
        )
    finally:
        decoder_audit.validate_semantics, decoder_audit.validate_decoder_lora_checkpoint = (
            old_semantics,
            old_validator,
        )
    inputs = report["inputs"]

    def bind(path):
        path = Path(path).resolve(strict=True)
        inputs[str(path)] = sha256_file(path)
        return path

    baseline = torch.load(bind(args.baseline_initial_checkpoint), map_location="cpu", weights_only=True)
    initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
    if baseline["header"] != BASELINE_HEADER or baseline["trainer_state"]["completed_update_count"] != 0:
        raise ValueError("requires actual matched world-quality initial checkpoint")
    cfg0 = baseline["lineage"]["materials"]["resolved_config"]["payload"]
    cfg1 = initial["lineage"]["materials"]["resolved_config"]["payload"]
    for key in ("seed", "num_envs"):
        if cfg0[key] != cfg1[key]:
            raise ValueError("initial batch/seed changed")
    expected_agent = copy.deepcopy(cfg0["agent"])
    if expected_agent["algorithm"]["class_name"] != "PPO":
        raise ValueError("unexpected prior algorithm")
    expected_agent["algorithm"]["class_name"] = ALGORITHM
    if expected_agent != cfg1["agent"]:
        raise ValueError("agent config changed beyond declared projection algorithm")
    for label, before, after in (
        ("actor", baseline["actor"]["state_dict"], initial["actor"]["state_dict"]),
        ("critic", baseline["critic_state_dict"], initial["critic_state_dict"]),
    ):
        if set(before) != set(after) or any(not torch.equal(before[k], after[k]) for k in before):
            raise ValueError("initial " + label + " differs from quality predecessor")
    report["initial_actor_and_critic_equal_matched_quality_predecessor"] = True
    del baseline, initial
    updates, steps, envs = report["completed_updates"], report["resolved_rollout_steps_per_env"], cfg1["num_envs"]
    metadata = json.loads(bind(args.reward_capture).read_text())
    projection = json.loads(bind(args.projection_capture).read_text())
    for row in (metadata, projection):
        if (
            row["lineage_sha256"] != report["lineage_sha256"]
            or row["completed_updates"] != updates
            or row["training_state_poisoned"] is not False
        ):
            raise ValueError("captured lineage/update/state mismatch")
        path = bind(row["arrays_path"])
        if inputs[str(path)] != row["arrays_sha256"]:
            raise ValueError("capture arrays changed")
    with np.load(metadata["arrays_path"], allow_pickle=False) as archive:
        reward_arrays = {k: archive[k].copy() for k in archive.files}
    report["world_tracking"] = audit_world_arrays(reward_arrays, updates * steps, envs)
    report["quality_reward_equations"] = audit_quality_arrays(reward_arrays, metadata, updates, steps, envs)
    del reward_arrays
    with np.load(projection["arrays_path"], allow_pickle=False) as archive:
        projection_arrays = {k: archive[k].copy() for k in archive.files}
    alg = cfg1["agent"]["algorithm"]
    report["projection_loss"] = audit_projection_arrays(
        projection_arrays, projection, updates, steps, envs, alg["num_learning_epochs"], alg["num_mini_batches"]
    )
    report.update(
        kind="native23_world_projection_actual_update_audit_v1",
        world_projection=training_contract(),
        passed=report["update_verification_passed"],
        hardware_authorized=False,
        deployment_ready=False,
    )
    bind(__file__)
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("audit input changed")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {key: report[key] for key in ("passed", "completed_updates", "actual_transitions", "projection_loss")}
        ),
        flush=True,
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
