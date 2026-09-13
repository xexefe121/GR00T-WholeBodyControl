"""Identify hypothetical missing-axis dynamics from existing source SIM data."""

from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.utils.g1_23dof_contract import (
    NATIVE_IL23_JOINT_NAMES,
    NATIVE_IL23_TO_CANONICAL_IL29,
    SOURCE_IL29_EXCLUDED_INDICES,
)
from gear_sonic.utils.g1_29dof_normal_teacher import ExactNormal29Teacher
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_SCALE_NATIVE_IL23

HERE = Path(__file__).parent
BASE = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")
SOURCE = Path("/mnt/z/codex/GR00T-WholeBodyControl/sonic_release/last.pt")
MISSING = np.asarray(SOURCE_IL29_EXCLUDED_INDICES)
KEEP = np.asarray(NATIVE_IL23_TO_CANONICAL_IL29)
Q_SLOTS = 30 + np.arange(10)[:, None] * 29 + MISSING
DQ_SLOTS = 320 + np.arange(10)[:, None] * 29 + MISSING
A_SLOTS = 610 + np.arange(10)[:, None] * 29 + MISSING
LEGS = np.asarray(
    [
        i
        for i, n in enumerate(NATIVE_IL23_JOINT_NAMES)
        if n.startswith(("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"))
    ]
)
SCALE = np.asarray(SOURCE_SCALE_NATIVE_IL23)


def channels(history):
    q = history[:, 30:320].reshape(-1, 10, 29)[:, -1]
    dq = history[:, 320:610].reshape(-1, 10, 29)[:, -1]
    state = np.c_[q[:, MISSING], dq[:, MISSING]].astype(np.float64)
    external = np.c_[q[:, KEEP], dq[:, KEEP], history[:, 27:30], history[:, 927:930]].astype(np.float64)
    return state, external


def rms(value):
    return float(np.sqrt(np.mean(np.square(value))))


def main():
    output = HERE / "calibration_v1"
    if output.exists():
        raise FileExistsError("calibration refuses overwrite or an implicit sweep")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("bound input changed: " + str(path))
        inputs[str(path)] = digest
        return path

    def load(directory):
        report = json.loads(bind(directory / "report.json").read_text())
        trace_path = (directory / "trace.npz").resolve(strict=True)
        expected_trace = report.get("trace_sha256") or report["inputs"][str(trace_path)]
        with np.load(bind(trace_path, expected_trace), allow_pickle=False) as z:
            result = {
                k: z[k].copy()
                for k in (
                    "attempt_encoder267",
                    "attempt_history930",
                    "attempt_raw29",
                    "attempt_token64",
                    "attempt_source_timestamps_s",
                )
            }
        n = len(result["attempt_raw29"])
        for key, width in (
            ("attempt_encoder267", 267),
            ("attempt_history930", 930),
            ("attempt_raw29", 29),
            ("attempt_token64", 64),
        ):
            a = result[key]
            if a.shape != (n, width) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError("source input shape/dtype changed: " + key)
        np.testing.assert_allclose(
            np.diff(result["attempt_source_timestamps_s"], axis=0), 0.02, atol=1e-10, rtol=0
        )
        result["source_controller_report"] = dict(
            kind=report["kind"],
            actual_joint_range_excess_max_rad=report.get("actual_joint_range_excess_max_rad"),
            deployment_ready=report.get("deployment_ready"),
        )
        return result

    bind(__file__)
    bind(HERE / "EXPERIMENT.md")
    recordings = {"pico_fit": load(BASE / "normal29_upstream_action_v1/actual_v1/frozen29_2ms/pico")}
    for name in ("walk002", "walk003", "walk008"):
        recordings[name] = load(BASE / "original29_normal_v1/actual_v1" / name)
    train = recordings["pico_fit"]
    state, external = channels(train["attempt_history930"])
    features = np.c_[state[:-1], train["attempt_raw29"][:-1], external[:-1]]
    labels = state[1:]
    mean_x, mean_y = features.mean(0), labels.mean(0)
    scale_x = np.maximum(features.std(0), 1e-5)
    scale_y = np.maximum(labels.std(0), 1e-5)
    x, y = (features - mean_x) / scale_x, (labels - mean_y) / scale_y
    weight = np.linalg.solve(x.T @ x + 0.001 * np.eye(x.shape[1]), x.T @ y)
    matrix = weight / scale_x[:, None] * scale_y[None, :]
    offset = mean_y - mean_x @ matrix
    eigenvalues = np.linalg.eigvals(matrix[:12].T)
    radius = float(np.max(np.abs(eigenvalues)))
    output.mkdir()
    with (output / "model.npz").open("xb") as stream:
        np.savez_compressed(
            stream, matrix=matrix, offset=offset, mean_x=mean_x, scale_x=scale_x, mean_y=mean_y, scale_y=scale_y
        )
    bind(output / "model.npz")
    torch.set_num_threads(1)
    teacher = ExactNormal29Teacher(bind(SOURCE))
    identity = teacher.descriptor()
    results = {}
    for name, recording in recordings.items():
        truth, known = channels(recording["attempt_history930"])
        predicted = np.zeros_like(truth)
        for i in range(len(truth) - 1):
            predicted[i + 1] = np.r_[predicted[i], recording["attempt_raw29"][i], known[i]] @ matrix + offset
        finite = bool(np.isfinite(predicted).all())
        result = dict(
            samples=len(truth),
            optimizer_excluded=name != "pico_fit",
            initialized_with_true_missing_state=False,
            internal_state_resets=0,
            finite=finite,
            source_controller_report=recording["source_controller_report"],
            missing_position_rmse_rad=rms(predicted[:, :6] - truth[:, :6]),
            zero_fill_position_rmse_rad=rms(truth[:, :6]),
            missing_velocity_rmse_rad_s=rms(predicted[:, 6:] - truth[:, 6:]),
            zero_fill_velocity_rmse_rad_s=rms(truth[:, 6:]),
        )
        histories = deque((np.zeros(12) for _ in range(9)), maxlen=10)
        predicted_history = []
        for s in predicted:
            histories.append(s.copy())
            predicted_history.append(np.asarray(histories).copy())
        selected = np.linspace(0, len(truth) - 1, 96, dtype=int)
        variants = {name: [] for name in ("zero_fill", "actions_only", "predicted_state_and_commands")}
        actual = []
        for i in selected:
            e, h = recording["attempt_encoder267"][i], recording["attempt_history930"][i]
            raw, token = teacher.infer_with_token(e, h)
            np.testing.assert_array_equal(raw, recording["attempt_raw29"][i])
            np.testing.assert_array_equal(token, recording["attempt_token64"][i])
            actual.append(raw)
            for variant, rows in variants.items():
                altered = h.copy()
                if variant == "predicted_state_and_commands":
                    altered[Q_SLOTS] = predicted_history[i][:, :6]
                    altered[DQ_SLOTS] = predicted_history[i][:, 6:]
                else:
                    altered[Q_SLOTS] = 0
                    altered[DQ_SLOTS] = 0
                    if variant == "zero_fill":
                        altered[A_SLOTS] = 0
                value, same_token = teacher.infer_with_token(e, altered)
                np.testing.assert_array_equal(same_token, token)
                rows.append(value)
        actual = np.asarray(actual)
        result["decoder_comparison"] = {}
        for variant, rows in variants.items():
            raw = np.asarray(rows)
            delta = (raw[:, KEEP] - actual[:, KEEP]) * SCALE
            result["decoder_comparison"][variant] = dict(
                source_affine_retained_target_difference_rms_rad=rms(delta),
                source_affine_leg_target_difference_rms_rad=rms(delta[:, LEGS]),
                maximum_retained_raw_magnitude=float(np.max(np.abs(raw[:, KEEP]))),
                counterfactual_only_not_actual_tracking=True,
            )
        with (output / (name + ".npz")).open("xb") as stream:
            np.savez_compressed(
                stream,
                truth=truth,
                predicted=predicted,
                selected=selected,
                actual_raw29=actual,
                **{name: np.asarray(rows) for name, rows in variants.items()},
            )
        bind(output / (name + ".npz"))
        results[name] = result
        print(json.dumps(dict(recording=name, **result)), flush=True)
    checks = {}
    for name, result in results.items():
        comparison = result["decoder_comparison"]
        value = comparison["predicted_state_and_commands"]["source_affine_leg_target_difference_rms_rad"]
        checks[name] = bool(
            result["finite"]
            and result["missing_position_rmse_rad"] <= 0.75 * result["zero_fill_position_rmse_rad"]
            and result["missing_velocity_rmse_rad_s"] <= result["zero_fill_velocity_rmse_rad_s"]
            and all(
                value < comparison[v]["source_affine_leg_target_difference_rms_rad"]
                for v in ("zero_fill", "actions_only")
            )
        )
    assert teacher.descriptor() == identity
    report = dict(
        kind="hypothetical_source_missing_axis_linear_dynamics_calibration_v1",
        trained_transitions=len(features),
        regularization=0.001,
        internal_spectral_radius=radius,
        stable_internal_recurrence=radius < 1,
        missing_il29_indices=MISSING.tolist(),
        feature_order=["internal_q6_dq6", "current_raw29", "retained_q23_dq23_gyro3_gravity3"],
        results=results,
        development_checks=checks,
        guarded_native23_experiment_justified=bool(
            radius < 1 and all(checks[n] for n in ("walk002", "walk003", "walk008"))
        ),
        frozen_source_identity=identity,
        inputs=inputs,
        source_dynamics_fit_not_tracking_policy_training=True,
        source_actions_drive_predictions_during_this_open_loop_test=True,
        new_simulation_steps=0,
        physical_23_dof_trial_performed=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            dict(
                internal_spectral_radius=radius,
                development_checks=checks,
                guarded_native23_experiment_justified=report["guarded_native23_experiment_justified"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
