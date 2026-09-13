"""Read-only training exposure and reward-signal audit, not a policy evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256


def describe(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return None
    if not np.isfinite(values).all():
        raise ValueError("nonfinite learning evidence")
    return dict(
        count=int(values.size),
        mean=float(values.mean()),
        p50=float(np.median(values)),
        p95=float(np.percentile(values, 95)),
    )


def classify_frames(anchors, spans):
    # Termination/reward held q1 is exactly one frame after the captured q0.
    frames = np.asarray(anchors) + 1
    if frames.dtype.kind not in "iu":
        raise ValueError("reference frame indices must be integers")
    labels = np.full(frames.shape, -1, dtype=np.int32)
    names = []
    for span in spans:
        for phase in span["timeline"]["phases"]:
            index = len(names)
            names.append((span["name"], phase["name"]))
            mask = (frames >= span["start"] + phase["frame_start"]) & (
                frames < span["start"] + phase["frame_stop"]
            )
            if np.any(labels[mask] != -1):
                raise ValueError("overlapping training reference phases")
            labels[mask] = index
    if (labels < 0).any():
        raise ValueError("training reference frame lacks an explicit phase")
    return labels, names


def summarize_arrays(arrays, metadata, spans, block_controls=640):
    root_error = arrays["world_tracking_error_m"]
    labels, names = classify_frames(arrays["world_tracking_reference_q0"], spans)
    if root_error.shape != labels.shape or len(root_error) < block_controls * 2:
        raise ValueError("insufficient or mismatched training blocks")
    if not np.isfinite(arrays["raw_costs"]).all() or (arrays["raw_costs"] < 0).any():
        raise ValueError("raw cost capture must be finite and nonnegative")
    sections = {"first10_updates": slice(0, block_controls), "last10_updates": slice(-block_controls, None)}
    outputs = {}
    for title, section in sections.items():

        def aggregate(mask):
            count = int(mask.sum())
            if not count:
                return dict(transitions=0)
            terminated = arrays["terminated"][section][mask]
            costs = arrays["raw_costs"][section][mask]
            return dict(
                transitions=count,
                root_error_m=describe(root_error[section][mask]),
                returned_reward=describe(arrays["returned_reward"][section][mask]),
                failure_fraction=float(terminated.mean()),
                world_failure_fraction=float(arrays["world_tracking_failure"][section][mask].mean()),
                timeout_fraction=float(arrays["timeouts"][section][mask].mean()),
                costs={
                    name: dict(
                        raw=describe(costs[:, j]),
                        mean_bounded_slope=float(np.mean(1 / (1 + costs[:, j].astype(float)) ** 2)),
                        bounded_slope_below_1_percent_fraction=float(np.mean(costs[:, j] > 9)),
                    )
                    for j, name in enumerate(metadata["raw_cost_names"])
                },
                mean_reward_components={
                    name: float(arrays["weighted_base_components"][section][..., j][mask].mean())
                    for j, name in enumerate(metadata["base_component_names"])
                },
            )

        outputs[title] = dict(
            all=aggregate(np.ones(labels[section].shape, dtype=bool)),
            by_clip_phase=[
                dict(clip=clip, phase=phase, **aggregate(labels[section] == j))
                for j, (clip, phase) in enumerate(names)
            ],
        )
    # Exact identity of the existing bounded base: alive=5+sum negative weights.
    # Thus each bounded cost is equivalently a weighted quality 1/(1+x), not
    # proof that PPO differentiates this reward or cannot learn from saturation.
    done = arrays["terminated"] | arrays["timeouts"]
    quality = np.exp(arrays["phi_after_including_reset_states"].astype(float))
    quality[done] = 0
    hypothetical_bonus = 2 * quality
    return dict(
        total_transitions=int(root_error.size),
        windows=outputs,
        exposure=[
            dict(clip=clip, phase=phase, transitions=int((labels == j).sum()))
            for j, (clip, phase) in enumerate(names)
        ],
        hypothetical_world_quality_bonus_per_control=describe(hypothetical_bonus),
        hypothetical_bonus_formula="100*0.02*exp(phi_after) for non-done transitions, zero for failures/timeouts",
        hypothetical_bonus_changes_old_objective=True,
        hypothetical_bonus_is_actual_trained_reward=False,
        post_reset_states_used_for_hypothetical_bonus=False,
        saturation_is_reward_sensitivity_not_backpropagated_actor_gradient=True,
        changing_policy_and_reset_distribution_confounds_early_late_comparison=True,
        fixed_policy_matched_source_tracking_improvement_proven=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("learning audit refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("learning evidence hash mismatch: " + str(path))
        inputs[str(path)] = digest
        return path

    bind(__file__)
    training = args.run / "train"
    previous_audit = json.loads(bind(args.run / "update_audit.json").read_text())
    if previous_audit.get("passed") is not True or previous_audit.get("completed_updates") != 100:
        raise ValueError("requires completed independently audited100-update run")
    metadata = json.loads(bind(training / "reward_steps_after_100_updates.json").read_text())
    with np.load(bind(metadata["arrays_path"], metadata["arrays_sha256"]), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    spans = json.loads(bind(args.run / "curriculum/curriculum.spans.json").read_text())["spans"]
    with np.load(bind(args.run / "curriculum/curriculum.npz"), allow_pickle=False) as source:
        target = source["body_pos_w"][arrays["world_tracking_reference_q0"] + 1, 0].astype(np.float32)
    offsets = arrays["world_tracking_desired_position_w"] - target
    # Independent source-frame proof, allowing GPU float32 addition roundoff.
    np.testing.assert_allclose(offsets, np.broadcast_to(offsets[0], offsets.shape), atol=4e-6, rtol=0)
    result = summarize_arrays(arrays, metadata, spans)
    result["held_q1_target_reconstructed_with_constant_environment_origins"] = True
    runtime = json.loads(bind(training / "curriculum_runtime_100.json").read_text())
    result["training_complete_timelines"] = runtime["completed_reference_timelines"]
    result["training_completed_suffixes"] = runtime["completed_reference_suffixes_not_full_lifecycles"]
    files = list(training.glob("events.out.tfevents.*"))
    if len(files) != 1:
        raise ValueError("requires one unambiguous TensorBoard event file")
    events = EventAccumulator(str(bind(files[0])))
    events.Reload()
    tags = (
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Loss/value",
        "Policy/mean_std",
        "Metrics/motion/error_anchor_pos",
        "Metrics/motion/error_joint_pos",
    )
    result["logged_scalar_windows"] = {
        tag: dict(
            events=len(events.Scalars(tag)),
            first10_events=describe([item.value for item in events.Scalars(tag)[:10]]),
            last10_events=describe([item.value for item in events.Scalars(tag)[-10:]]),
            event_steps=[item.step for item in events.Scalars(tag)],
            event_windows_are_not_necessarily_same_update_windows=True,
        )
        for tag in tags
    }
    for path, digest in inputs.items():
        if file_sha256(path) != digest:
            raise ValueError("learning evidence changed during audit: " + path)
    result.update(
        kind="native23_world_tracking_learning_signal_audit_v1",
        inputs=inputs,
        new_policy_updates=0,
        no_hardware_connection=True,
        hardware_authorized=False,
        deployment_ready=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    for name, window in result["windows"].items():
        row = window["all"]
        print(
            json.dumps(
                dict(
                    window=name,
                    root=row["root_error_m"],
                    reward=row["returned_reward"],
                    failure_fraction=row["failure_fraction"],
                    feet=row["costs"]["measured_feet_world_position_l2"],
                    hands_head=row["costs"]["original_task_position_l2"],
                )
            ),
            flush=True,
        )
    print(json.dumps(dict(output=str(args.output), sha256=file_sha256(args.output))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
