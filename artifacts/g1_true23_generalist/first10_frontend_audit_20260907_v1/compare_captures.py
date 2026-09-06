"""Compare bounded passive library captures; never integrate or command a robot."""

import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(value):
    value = np.array(value, dtype=np.float64, copy=True)
    w, x, y, z = value / np.linalg.norm(value)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def compare(left, right):
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    row_max = delta.reshape(len(left), -1).max(axis=1)
    unequal = np.flatnonzero(row_max)
    return dict(shape=list(left.shape), exactly_equal=bool(np.array_equal(left, right)),
                max_abs_difference=float(delta.max()),
                first_different_row=None if len(unequal) == 0 else int(unequal[0]),
                per_row_max_abs_difference=row_max.tolist())


folder = Path(__file__).resolve().parent
repo = folder.parents[2]
original_root = Path("/mnt/z/codex/GR00T-WholeBodyControl")
original_path = folder / "original/trace.npz"
current_path = folder / "current/trace.npz"
saved_path = original_root / "artifacts/g1_true23/pico_internet_fullbody_v14_100_eval/model_100/walk001.trajectory.npz"
new_path = repo / "artifacts/g1_true23_generalist/historical_walk_reproduction_20260907_v1/historical_released_gains.000.original_walk_v14_100.npz"
original, current, saved, new = [np.load(path, allow_pickle=False) for path in (original_path, current_path, saved_path, new_path)]
original_report = json.loads((folder / "original/report.json").read_text())
current_report = json.loads((folder / "current/report.json").read_text())
same_fields = ("encoder267", "history930", "decoder994", "raw23", "physics_pre_qpos", "physics_post_qpos", "physics_pre_qvel", "physics_post_qvel", "physics_time")
current_new = {key: compare(current[key], new[key][:len(current[key])]) for key in same_fields}
current_new["physics_command"] = compare(current["physics_command"], new["applied_torque23"][:100])
original_current = {key: compare(original[key], current[key]) for key in (*same_fields, "physics_command")}
old_qpos = compare(original["original_qpos"], saved["qpos"][:11])
gains_equal = {key: original_report[key] == current_report[key] for key in ("kp", "kd", "effort")}
changed_history = np.flatnonzero(original["history930"][1] != current["history930"][1])
stale_reconstructed = (
    rotation(original["physics_post_qpos"][9, 3:7]).T
    @ rotation(original["physics_pre_qpos"][9, 3:7])
    @ original["physics_pre_qvel"][9, 3:6]
).astype(np.float32)
fresh_reconstructed = original["physics_post_qvel"][9, 3:6].astype(np.float32)
assert old_qpos["exactly_equal"]
assert all(item["exactly_equal"] for item in current_new.values())
assert all(gains_equal.values())
assert np.array_equal(changed_history, [27, 28, 29])
assert np.array_equal(original["encoder267"][:3], current["encoder267"][:3])
assert np.array_equal(original["decoder994"][:3, :64], current["decoder994"][:3, :64])
assert np.array_equal(original["physics_post_qpos"][:10], current["physics_post_qpos"][:10])
assert np.array_equal(stale_reconstructed, original["history930"][1, 27:30])
assert np.array_equal(fresh_reconstructed, current["history930"][1, 27:30])
inputs = {str(path): sha(path) for path in (
    original_path, current_path, saved_path, new_path, Path(__file__),
    folder / "original/report.json", folder / "current/report.json",
)}
receipt = dict(
    schema_version=1,
    kind="original_vs_refreshed_frontend_first10_control_diagnostic_v1",
    scope="Exactly ten actual policy controls per original/current source capture; this analysis only reads traces. No additional integration.",
    outcome="Frontend timing mismatch, not floating-point precision: newest angular velocity reads stale MuJoCo cvel in original controller; current controller refreshes kinematics before measured history.",
    original_actual_vs_historical_saved_first11_qpos=old_qpos,
    current_actual_vs_shared_benchmark_first10=current_new,
    original_actual_vs_current_actual_first10=original_current,
    gains_and_effort_exactly_equal=gains_equal,
    first_difference=dict(
        control_index_zero_based=1,
        history930_indices=changed_history.tolist(),
        term="newest base angular velocity, term-major 10x3 angular block",
        original_stale_angular_body_radps=original["history930"][1, 27:30].tolist(),
        current_refreshed_angular_body_radps=current["history930"][1, 27:30].tolist(),
        stale_formula="R(post_qpos[9]).T @ R(pre_qpos[9]) @ pre_qvel[9,3:6]",
        stale_formula_matches_original_float32_exactly=True,
        fresh_formula="post_qvel[9,3:6]",
        fresh_formula_matches_current_float32_exactly=True,
        initial_encoder_history_raw_exactly_equal=True,
        initial_entire_10_physics_substeps_exactly_equal=True,
        first_three_encoder_and_token_calls_exactly_equal=True,
        second_raw_action_max_abs_difference=float(np.max(np.abs(original["raw23"][1]-current["raw23"][1]))),
        first_changed_physics_substep_zero_based=10,
        first_changed_physics_command_max_abs_nm=float(np.max(np.abs(original["physics_command"][10]-current["physics_command"][10]))),
    ),
    first_two_integration_steps=dict(
        times=original["physics_time"][:2].tolist(),
        command_nm=original["physics_command"][:2].tolist(),
        pre_qpos=original["physics_pre_qpos"][:2].tolist(),
        post_qpos=original["physics_post_qpos"][:2].tolist(),
        original_current_shared_benchmark_identical=True,
    ),
    interpretation=[
        "Current shared benchmark correctly reproduces the current actual library frontend and native integration for these ten controls.",
        "historical_released_gains restores gain arrays only, not original stale-cvel observation timing. Do not label this profile exact historical-policy reproduction.",
        "Keep common nominal native-actuator comparison on synchronous measured observations; preserve original frontend separately as a historical diagnostic control.",
        "Exact first-ten historical reproduction does not establish full-horizon equality or dance qualification. No full historical campaign was rerun here.",
        "The source also differs in quaternion copy safety; this is not the first divergence demonstrated by these traces.",
    ],
    inputs=inputs,
    hardware_authorized=False,
    deployment_ready=False,
    simulator_qualified=False,
)
with (folder / "comparison.json").open("x") as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({"receipt": str(folder / "comparison.json"), "sha256": sha(folder / "comparison.json"), "outcome": receipt["outcome"]}))
