"""Recount completed/censored episodes and check initial pairing, without Torch."""

import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "sensor_exploration_20260906_v1"
inputs = {}


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"reset diagnostic evidence changed: {path}")
    inputs[str(path)] = digest
    return path


bind(Path(__file__))
for name in ("experiment_report.json", "reset_controls_report.json"):
    report = json.loads(bind(SOURCE / name).read_text())
    assert all(row["return_code"] == 0 for row in report["stages"])
    for path, expected in report["inputs"].items():
        bind(path, expected)

names = ("original_noise", "zero_action", "zero_sensor", "zero_both", "nominal_reset", "nominal_reset_lift")
rows, initials = [], {}
for name in names:
    report = json.loads(bind(SOURCE / name / "ieee_sensor_report.json").read_text())
    for path, expected in report["inputs"].items():
        bind(path, expected)
    assert report["weights_updated"] is False and report["optimizer_steps"] == 0
    assert report["checkpoint_sha256"] == "f20f82385dd7c652a7a74b6103a4753f0317b6fb10055452862d647e7dc14de5"
    assert report["guarded_actor_calls"] == 128 and report["later_adaptive_resets_paired"] is False
    with np.load(SOURCE / name / "rollout.npz", allow_pickle=False) as data:
        done = data["done"]
        assert done.shape == (128, 32) and done.dtype == bool
        counts, censored, starts = [], [], np.zeros_like(done)
        for env in range(32):
            ends = np.flatnonzero(done[:, env])
            boundaries = np.r_[-1, ends]
            counts.extend(np.diff(boundaries).tolist())
            censored.append(int(127 - boundaries[-1]))
            reset_steps = np.r_[0, ends + 1]
            reset_steps = reset_steps[reset_steps < 128]
            starts[reset_steps, env] = data["invalid_before_step"][reset_steps, env]
        terms = {key.removeprefix("termination_"): data[key] for key in data.files if key.startswith("termination_")}
        np.testing.assert_array_equal(np.logical_or.reduce(list(terms.values())), done)
        guard = terms["stage_one_actuation_guard"]
        actual = dict(
            control_steps_per_environment=128,
            sampled_transitions=4096,
            completed_episodes=len(counts),
            completed_episode_mean_steps=statistics.mean(counts),
            completed_episode_median_steps=statistics.median(counts),
            completed_episode_maximum_steps=max(counts),
            one_step_completed_episodes=counts.count(1),
            unfinished_episode_lengths_steps=censored,
            termination_counts_nonexclusive={key: int(value.sum()) for key, value in terms.items()},
            invalid_controller_at_episode_start=int(starts.sum()),
            guard_terminations_from_invalid_episode_start=int((guard & starts).sum()),
            guard_terminations_without_invalid_start_that_step=int((guard & ~starts).sum()),
        )
        assert actual == report["episode_audit"]
        assert sum(counts) + sum(censored) == 4096
        initials[name] = {key: data[key].copy() if key.startswith("initial_") else data[key][0].copy()
                          for key in ("initial_qpos", "initial_qvel", "phase", "q", "dq", "target")}
    with np.load(SOURCE / name / "actor_observations.npz", allow_pickle=False) as data:
        assert data["tokenizer"].shape == (128, 32, 267)
        assert data["policy"].shape == (128, 32, 930)
        assert all(np.isfinite(data[key]).all() for key in data.files)
    rows.append(dict(name=name, action_noise_scale=report["action_noise_scale"],
                     sensor_noise_scale=report["sensor_noise"]["amplitude_scale"], **actual))

for name in names[1:4]:
    for key, expected in initials[names[0]].items():
        np.testing.assert_array_equal(initials[name][key], expected)
for name in names[4:]:
    np.testing.assert_array_equal(initials[name]["phase"], initials["zero_both"]["phase"])
assert all(row["completed_episode_median_steps"] == 3 for row in rows[:4])
assert rows[4]["completed_episode_median_steps"] == 25
assert rows[5]["completed_episode_median_steps"] == 30
result = dict(
    inputs=inputs, cases=rows, sampled_transitions=24576, guarded_actor_calls=768,
    four_noise_cases_initial_physics_phase_joints_previous_target_exactly_paired=True,
    all_six_initial_source_phases_exactly_paired=True,
    later_adaptive_resets_paired=False, unfinished_episodes_censored=True,
    reset_controls_also_disable_sensor_and_action_noise=True,
    statistical_generalization_proven=False, full_motion_qualified=False,
    hardware_authorized=False, deployment_ready=False,
)
with (HERE / "reset_lead_verified.json").open("x") as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({key: value for key, value in result.items() if key not in ("inputs", "cases")}))
