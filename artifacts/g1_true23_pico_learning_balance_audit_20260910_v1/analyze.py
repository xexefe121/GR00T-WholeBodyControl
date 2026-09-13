"""Saved training coverage and reward allocation; not causal policy qualification."""

import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_23dof_contract import (
    ISAACLAB_TO_MUJOCO_DOF,
    NATIVE_IL23_TO_CANONICAL_IL29,
    SONIC_HARDWARE_DEFAULT_Q,
)
from gear_sonic.utils.g1_true23_bounded_progress import NEGATIVE_WEIGHTS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "audit_v1.json"
NAMES = list(NEGATIVE_WEIGHTS)


def stats(value):
    x = np.asarray(value, dtype=float).reshape(-1)
    assert x.size and np.isfinite(x).all()
    return dict(
        count=int(x.size),
        mean=float(x.mean()),
        min=float(x.min()),
        p05_p50_p95_p99=np.percentile(x, [5, 50, 95, 99]).tolist(),
        max=float(x.max()),
    )


def tilt(quat):
    assert np.max(np.abs(np.linalg.norm(quat, axis=-1) - 1)) < 1e-5
    return np.arccos(np.clip(1 - 2 * (quat[..., 1] ** 2 + quat[..., 2] ** 2), -1, 1))


def features(history, root9, quaternion, desired_joints):
    shape = history.shape[:-1]
    q29 = history[..., 30:320].reshape(*shape, 10, 29)[..., -1, :]
    v29 = history[..., 320:610].reshape(*shape, 10, 29)[..., -1, :]
    missing = sorted(set(range(29)) - set(NATIVE_IL23_TO_CANONICAL_IL29))
    assert not np.any(q29[..., missing]) and not np.any(v29[..., missing])
    q = q29[..., list(NATIVE_IL23_TO_CANONICAL_IL29)][..., list(ISAACLAB_TO_MUJOCO_DOF)].astype(float)
    q += np.asarray(SONIC_HARDWARE_DEFAULT_Q, dtype=np.float32).astype(float)
    velocity = v29[..., list(NATIVE_IL23_TO_CANONICAL_IL29)][..., list(ISAACLAB_TO_MUJOCO_DOF)]
    return dict(
        root_position_error_m=np.linalg.norm(root9[..., :3], axis=-1),
        root_velocity_error_m_s=np.linalg.norm(root9[..., 3:6] - root9[..., 6:9], axis=-1),
        root_speed_m_s=np.linalg.norm(root9[..., 6:9], axis=-1),
        root_tilt_rad=tilt(quaternion),
        lower_posture_rms_rad=np.sqrt(np.mean((q[..., :13] - desired_joints[..., :13]) ** 2, axis=-1)),
        legs_velocity_rms_rad_s=np.sqrt(np.mean(velocity[..., :12] ** 2, axis=-1)),
        right_ankle_roll_rad=q[..., 11],
        right_ankle_roll_velocity_rad_s=velocity[..., 11],
    )


def main():
    if OUT.exists():
        raise FileExistsError("saved learning audit refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        value = sha256_file(path)
        if expected is not None:
            assert value == expected, path
        inputs[str(path)] = value
        return path

    def read(path, expected=None):
        with np.load(bind(path, expected), allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    bind(__file__)
    bind(HERE / "EXPERIMENT.md")
    for module in ("g1_true23_bounded_progress.py", "g1_23dof_contract.py"):
        bind(ROOT / "gear_sonic/utils" / module)
    bank_root = ROOT / "artifacts/g1_true23_pico_training_20260910_v1/bank_v1"
    bank = json.loads(
        bind(
            bank_root / "report.json", "1ed51fb48efa5ee6ec2e8c32d0fe58d5455ac1b77ce001abb1f034b65df09291"
        ).read_text()
    )
    spans_path = Path(bank["files"]["spans"])
    spans = json.loads(bind(spans_path, bank["outputs"][str(spans_path)]).read_text())["spans"]
    spec_path = Path(bank["files"]["spec"])
    spec = json.loads(bind(spec_path, bank["outputs"][str(spec_path)]).read_text())
    native = spec["files"]["native_motion"]
    motion = read(native["path"], native["sha256"])
    pico = next(span for span in spans if span["name"] == "pico")
    source = next(phase for phase in pico["timeline"]["phases"] if phase["name"] == "source_motion")
    samples = []
    failure_dir = ROOT / "artifacts/g1_true23_pico_ankle_feedforward_20260910_v1/eval1000_v1/pico"
    failure_report = json.loads(bind(failure_dir / "report.json").read_text())
    assert failure_report["result"]["completed_controls"] == 2004
    failure = read(failure_dir / "attempts.npz", failure_report["attempts_sha256"])
    for control in (1830, 1853, 1880, 1900, 1910, 1920):
        # Received normal q0 is9+control; desired root/joints are q1=10+control.
        anchor = pico["start"] + 9 + control
        np.testing.assert_allclose(failure["timestamps"][control, 1], (9 + control) * 0.02, rtol=0, atol=1e-12)
        f = features(
            failure["history930"][control],
            failure["root_feedback9"][control],
            failure["measured_qpos"][control, 3:7],
            motion["joint_pos"][anchor + 1],
        )
        samples.append(
            dict(
                control_index=control,
                bank_anchor=anchor,
                executed_source_seconds=(control - source["control_start"]) * 0.02,
                features={key: float(value) for key, value in f.items()},
            )
        )
    trainings = []
    for name, family in (
        ("parent500", "g1_true23_pico_training_20260910_v1"),
        ("foot1000", "g1_true23_pico_foot_precision_20260910_v1"),
    ):
        directory = ROOT / "artifacts" / family / "train500_v1"
        arithmetic_path = directory / "arithmetic_audit.json"
        arithmetic = json.loads(bind(arithmetic_path).read_text())
        assert arithmetic["passed"] and arithmetic["all_transition_rewards_verified"] == 1024000
        for field in ("reward_capture.npz", "sampled_actual_inputs.npz", "world_failure_capture.npz"):
            bind(directory / field, arithmetic["inputs"][str(directory / field)])
        reward = read(directory / "reward_capture.npz")
        observations = read(directory / "sampled_actual_inputs.npz")
        failure_capture = read(directory / "world_failure_capture.npz")
        raw = reward["raw_costs"]
        assert raw.shape == (8000, 128, 13) and np.isfinite(raw).all() and raw.min() >= 0
        anchor = failure_capture["reference_q0"]
        selected = observations["control_index"]
        np.testing.assert_array_equal(anchor[selected], observations["reference_q0"])
        done = reward["terminated"] | reward["timeouts"]
        negative = raw.astype(float) / (1 + raw.astype(float)) * np.asarray(list(NEGATIVE_WEIGHTS.values())) * 0.02
        reward_terms = []
        for j, term in enumerate(NAMES):
            values = raw[..., j][~done]
            reward_terms.append(
                dict(
                    name=term,
                    raw=stats(values),
                    bounded_reward_contribution=stats(negative[..., j][~done]),
                    fraction_raw_cost_at_least9=float((values >= 9).mean()),
                    transform_derivative_per_raw_unit=stats(1 / (1 + values.astype(float)) ** 2),
                )
            )
        exposure = []
        for span in spans:
            mask = (anchor >= span["start"]) & (anchor < span["start"] + span["length"])
            phase = next(p for p in span["timeline"]["phases"] if p["name"] == "source_motion")
            source_mask = (
                mask
                & (anchor + 1 >= span["start"] + phase["frame_start"])
                & (anchor + 1 < span["start"] + phase["frame_stop"])
            )
            exposure.append(
                dict(
                    name=span["name"],
                    all_controls=int(mask.sum()),
                    source_controls=int(source_mask.sum()),
                    sampled_source_anchors=int(np.unique(anchor[source_mask]).size),
                    true_terminal_source_fraction=float(reward["terminated"][source_mask].mean()),
                )
            )
        assert sum(item["all_controls"] for item in exposure) == 1024000
        obs_features = features(
            observations["policy"],
            observations["root_feedback"],
            observations["measured_root_quaternion_wxyz"],
            motion["joint_pos"][observations["reference_q0"] + 1],
        )
        local = []
        for sample in samples:
            center = sample["bank_anchor"]
            full_mask = np.abs(anchor - center) <= 25
            sample_mask = np.abs(observations["reference_q0"] - center) <= 25
            non_done = full_mask & ~done
            feature_stats = {}
            for field, values in obs_features.items():
                field_values = values[sample_mask]
                feature_stats[field] = (
                    dict(
                        distribution=stats(field_values),
                        evaluation_value=sample["features"][field],
                        empirical_fraction_at_or_below_evaluation=float(
                            (field_values <= sample["features"][field]).mean()
                        ),
                    )
                    if field_values.size
                    else None
                )
            local.append(
                dict(
                    control_index=sample["control_index"],
                    bank_anchor=center,
                    actual_nearby_training_transitions=int(full_mask.sum()),
                    exact_anchor_transitions=int((anchor == center).sum()),
                    recorded_input_states_near_phase=int(sample_mask.sum()),
                    terminated_fraction=float(reward["terminated"][full_mask].mean()),
                    negative_reward_by_term={
                        term: float(negative[..., j][non_done].mean()) for j, term in enumerate(NAMES)
                    },
                    nonterminal_base_reward=stats(reward["base_reward"][non_done]),
                    nonterminal_world_quality_bonus=stats(reward["world_quality_bonus"][non_done]),
                    nonterminal_foot_bonus=stats(reward["foot_precision_bonus"][non_done])
                    if "foot_precision_bonus" in reward
                    else None,
                    marginal_feature_comparison=feature_stats,
                )
            )
        trainings.append(
            dict(
                name=name,
                reward_terms=reward_terms,
                exposure=exposure,
                phase_neighborhoods=local,
                sampled_observation_states=int(observations["reference_q0"].size),
                learned_population_is_nonstationary=True,
            )
        )
        print(
            json.dumps(
                dict(
                    name=name,
                    exposure=exposure,
                    nearby_counts=[row["actual_nearby_training_transitions"] for row in local],
                    sampled_counts=[row["recorded_input_states_near_phase"] for row in local],
                )
            ),
            flush=True,
        )
    for path, digest in inputs.items():
        assert sha256_file(Path(path)) == digest, path
    report = dict(
        inputs=inputs,
        training=trainings,
        evaluation_states=samples,
        phase_join_exact_at_all_captured_inputs=True,
        joint_observation_reconstruction_has_float32_roundoff=True,
        reward_transform_derivative_is_not_physics_or_ppo_gradient=True,
        marginal_coverage_is_not_joint_state_coverage=True,
        novel_failure_state_or_reward_conflict_claimed=False,
        no_training_or_dynamics_executed=True,
        deployment_ready=False,
        hardware_authorized=False,
    )
    with OUT.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
