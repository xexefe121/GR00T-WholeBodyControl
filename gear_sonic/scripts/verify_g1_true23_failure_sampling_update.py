"""Independent NumPy reset replay plus actual PPO weight/optimizer verification."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.scripts import verify_g1_true23_decoder_lora_update as decoder_audit
from gear_sonic.scripts.verify_g1_true23_world_quality_update import audit_quality_arrays
from gear_sonic.scripts.verify_g1_true23_world_tracking_update import audit_world_arrays
from gear_sonic.trl.mjlab.native23_failure_sampling_runner import validate_checkpoint
from gear_sonic.trl.mjlab.native23_world_quality_runner import CHECKPOINT_HEADER as BASELINE_HEADER
from gear_sonic.utils import g1_true23_failure_sampling_checkpoint as reader
from gear_sonic.utils.g1_true23_failure_sampling import sampling_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def independent_probabilities(ema, lengths):
    width = ema.shape[1]
    bin_lengths = np.clip(lengths[:, None] - np.arange(width)[None] * 50, 0, 50)
    uniform = bin_lengths.astype(np.float64) / lengths[:, None]
    score = np.zeros_like(ema, dtype=np.float64)
    for clip in range(len(lengths)):
        for index in range(width):
            if bin_lengths[clip, index] == 0:
                continue
            for ahead, weight in enumerate((1.0, 0.5, 0.25)):
                if index + ahead < width:
                    score[clip, index] += weight * ema[clip, index + ahead]
    result = uniform.copy()
    for clip in range(len(lengths)):
        if score[clip].sum() > 0:
            result[clip] = 0.2 * uniform[clip] + 0.8 * score[clip] / score[clip].sum()
    return result, bin_lengths


def audit_sampling(arrays, rewards, checkpoint):
    cfg = checkpoint["lineage"]["materials"]["resolved_config"]["payload"]
    state = checkpoint["failure_sampling_state"]
    first, lengths = state["first"].numpy(), state["lengths"].numpy()
    spans = cfg["native23_root_feedback"]["curriculum"]["derived_spans"]["spans"]
    standing = np.array([s["start"] + s["timeline"]["prehistory_frames"] - 1 for s in spans])
    controls, num_envs = state["controls"], cfg["num_envs"]
    anchors, terminated = arrays["observation_anchors"], arrays["observation_terminated"]
    if anchors.shape != (controls, num_envs) or terminated.shape != anchors.shape:
        raise ValueError("sampler control capture shape differs")
    np.testing.assert_array_equal(arrays["observation_control"], np.arange(1, controls + 1))
    np.testing.assert_array_equal(anchors, rewards["world_tracking_reference_q0"])
    np.testing.assert_array_equal(terminated, rewards["terminated"])
    width = arrays["observation_counts"].shape[-1]
    current = np.zeros((len(first), width), dtype=np.float64)
    total = np.zeros_like(current, dtype=np.int64)
    history = [current.copy()]
    for t in range(controls):
        counts = np.zeros_like(total)
        for clip in range(len(first)):
            offset = anchors[t] - first[clip]
            eligible = terminated[t] & (offset >= 0) & (offset < lengths[clip])
            counts[clip] = np.bincount(offset[eligible] // 50, minlength=width)
        np.testing.assert_array_equal(counts, arrays["observation_counts"][t])
        current = current * 0.99 + counts * 0.01
        total += counts
        np.testing.assert_allclose(current, arrays["observation_ema"][t], atol=1e-12, rtol=0)
        history.append(current.copy())
    np.testing.assert_array_equal(total, state["total_failures"].numpy())
    np.testing.assert_allclose(current, state["ema"].numpy(), atol=1e-12, rtol=0)
    offsets = arrays["sample_offsets"]
    sample_controls = arrays["sample_control"]
    if offsets.shape != (len(sample_controls) + 1,) or offsets[0] != 0 or (np.diff(offsets) <= 0).any():
        raise ValueError("sampler reset event offsets incomplete")
    if offsets[-1] != len(arrays["sample_anchors"]) or (np.diff(sample_controls) < 0).any():
        raise ValueError("sampler reset rows missing or out of order")
    nonuniform_source_samples = changed_anchors = source_samples = standing_samples = 0
    maximum_probability_error = 0.0
    for index, control in enumerate(sample_controls):
        if control < 0 or control > controls:
            raise ValueError("sample consumes unavailable future failure observations")
        ema = history[int(control)]
        probabilities, bin_lengths = independent_probabilities(ema, lengths)
        captured = arrays["sample_probabilities"][index]
        maximum_probability_error = max(maximum_probability_error, float(np.abs(captured - probabilities).max()))
        np.testing.assert_allclose(captured, probabilities, atol=1e-12, rtol=0)
        for row in range(offsets[index], offsets[index + 1]):
            env, clip = int(arrays["sample_env_ids"][row]), int(arrays["sample_choices"][row])
            fraction = arrays["sample_fractions"][row]
            if not 0 <= env < num_envs or not 0 <= clip < len(first) or not 0 <= fraction < 1:
                raise ValueError("invalid actual reset draw")
            is_standing = env % 4 == 0
            if bool(arrays["sample_standing"][row]) != is_standing:
                raise ValueError("sampler changed standing-start allocation")
            uniform = first[clip] + int(np.floor(np.float32(fraction) * np.float32(lengths[clip])))
            if is_standing:
                expected = standing[clip]
                standing_samples += 1
            elif not ema[clip].any():
                expected = uniform
                source_samples += 1
            else:
                p = probabilities[clip]
                cdf = np.cumsum(p)
                last = int(np.ceil(lengths[clip] / 50)) - 1
                cdf[last:] = 1.0
                chosen = int(np.searchsorted(cdf, float(fraction), side="right"))
                before = cdf[chosen - 1] if chosen else 0.0
                within = int(np.floor((float(fraction) - before) / p[chosen] * bin_lengths[clip, chosen]))
                within = min(max(within, 0), bin_lengths[clip, chosen] - 1)
                expected = first[clip] + chosen * 50 + within
                source_samples += 1
                nonuniform_source_samples += 1
                changed_anchors += expected != uniform
            if int(arrays["sample_anchors"][row]) != int(expected):
                raise ValueError("actual reset anchor does not match independent inverse CDF")
    if total.sum() <= 0 or nonuniform_source_samples <= 0 or changed_anchors <= 0:
        raise ValueError("smoke did not exercise real failure feedback and changed reset selection")
    return dict(
        actual_sampling_controls=controls,
        actual_sampling_source_failures=int(total.sum()),
        independently_reconstructed_reset_events=len(sample_controls),
        actual_standing_start_samples=standing_samples,
        actual_source_start_samples=source_samples,
        actual_failure_weighted_source_samples=nonuniform_source_samples,
        actual_source_anchors_different_from_uniform=int(changed_anchors),
        maximum_probability_reconstruction_error=maximum_probability_error,
        standing_start_allocation_unchanged=True,
        all_sampled_anchors_within_original_source=True,
        weights_uses_only_already_observed_failures=True,
        physical_control_improvement_proven=False,
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
        "sampling-capture",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("failure sampling audit refuses overwrite")
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
    trained = torch.load(args.trained_checkpoint, map_location="cpu", weights_only=True)
    if baseline["header"] != BASELINE_HEADER or baseline["trainer_state"]["completed_update_count"] != 0:
        raise ValueError("requires original Q0 comparison, not a failed trained policy")
    cfg0, cfg1 = [c["lineage"]["materials"]["resolved_config"]["payload"] for c in (baseline, initial)]
    for key in (
        "seed",
        "num_envs",
        "action_count",
        "history_length",
        "semantic_profile",
        "recovery",
        "domain_randomization",
    ):
        if cfg0[key] != cfg1[key]:
            raise ValueError("matched initial recipe changed: " + key)
    for key in ("actor", "critic", "algorithm", "obs_groups", "num_steps_per_env"):
        if cfg0["agent"][key] != cfg1["agent"][key]:
            raise ValueError("PPO/actor/critic configuration changed: " + key)
    for label, before, after in (
        ("actor", baseline["actor"]["state_dict"], initial["actor"]["state_dict"]),
        ("critic", baseline["critic_state_dict"], initial["critic_state_dict"]),
    ):
        if set(before) != set(after) or any(not torch.equal(before[key], after[key]) for key in before):
            raise ValueError("initial " + label + " differs from original Q0")
    report["initial_actor_and_critic_equal_Q0"] = True
    del baseline, initial
    captures = []
    for path in (args.reward_capture, args.sampling_capture):
        metadata = json.loads(bind(path).read_text())
        if (
            metadata.get("lineage_sha256") != report["lineage_sha256"]
            or metadata.get("training_state_poisoned") is not False
        ):
            raise ValueError("capture lineage changed or training state poisoned")
        arrays_path = bind(metadata["arrays_path"])
        if inputs[str(arrays_path)] != metadata["arrays_sha256"]:
            raise ValueError("capture arrays changed")
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        captures.append((metadata, arrays))
    (reward_metadata, rewards), (sampling_metadata, samples) = captures
    if sampling_metadata.get("contract") != sampling_contract():
        raise ValueError("actual sampling contract missing")
    if (
        sampling_metadata.get("kind") != "native23_actual_failure_sampling_capture_v1"
        or sampling_metadata.get("completed_updates") != report["completed_updates"]
        or sampling_metadata.get("actual_controls") != report["actual_vector_environment_steps"]
        or sampling_metadata.get("actual_source_failures") != int(trained["failure_sampling_state"]["total_failures"].sum())
        or reward_metadata.get("kind") != "native23_world_quality_actual_reward_capture_v1"
        or reward_metadata.get("completed_updates") != report["completed_updates"]
        or reward_metadata.get("num_envs") != cfg1["num_envs"]
    ):
        raise ValueError("capture kind or executed counters differ")
    controls = report["actual_vector_environment_steps"]
    report["world_tracking"] = audit_world_arrays(rewards, controls, cfg1["num_envs"])
    report["quality_reward_equations"] = audit_quality_arrays(
        rewards,
        reward_metadata,
        report["completed_updates"],
        report["resolved_rollout_steps_per_env"],
        cfg1["num_envs"],
    )
    report.update(audit_sampling(samples, rewards, trained))
    report.update(
        kind="native23_failure_sampling_actual_update_audit_v1", passed=report["update_verification_passed"]
    )
    for path in (
        Path(__file__),
        Path(reader.__file__),
        Path(audit_world_arrays.__code__.co_filename),
        Path(audit_quality_arrays.__code__.co_filename),
    ):
        bind(path)
    for path, expected in inputs.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError("audit input changed")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "passed",
                    "completed_updates",
                    "actual_transitions",
                    "actual_sampling_source_failures",
                    "actual_failure_weighted_source_samples",
                    "actual_source_anchors_different_from_uniform",
                    "checks",
                )
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
