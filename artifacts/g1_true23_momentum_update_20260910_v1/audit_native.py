"""Reconstruct the new observer and compare measured tracking to both baselines.

No new native policy rollout. Full saved-target native physics verification
is supplied independently by this experiment's physics_audit.json.
"""

from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.utils.g1_29dof_normal_teacher import ExactNormal29Teacher
from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters, file_sha256
from gear_sonic.utils.g1_true23_discarded_action_memory import KEEP, MISSING, validate_history
from gear_sonic.utils.g1_true23_momentum_source_model import MomentumSourceModel
from gear_sonic.utils.g1_true23_source_action_codec import source_action_history_numpy

HERE = Path(__file__).resolve().parent
ACTUAL = HERE / "native_actual_v1"
OLD_VIRTUAL = HERE.parent / "g1_true23_virtual_state_20260910_v1/native_actual_v1"
BASE = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")
ASSETS = Path("/mnt/z/codex/GR00T-WholeBodyControl")


def main():
    output = ACTUAL / "history_tracking_audit.json"
    if output.exists():
        raise FileExistsError("audit refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("audit input changed: " + str(path))
        inputs[str(path)] = digest
        return path

    def read(path):
        return json.loads(bind(path).read_text())

    def arrays(path):
        with np.load(bind(path), allow_pickle=False) as z:
            return {key: z[key].copy() for key in z.files}

    bind(__file__)
    report = read(ACTUAL / "report.json")
    for path, digest in report["inputs"].items():
        bind(path, digest)
    physics = read(ACTUAL / "physics_audit.json")
    assert physics["physics_reintegration"]["maximum_qpos_difference"] == 0
    assert physics["physics_reintegration"]["maximum_qvel_difference"] == 0
    trace, attempt, assimilation = (
        arrays(ACTUAL / name) for name in ("trace.npz", "attempts.npz", "assimilation.npz")
    )
    old_report = read(OLD_VIRTUAL / "report.json")
    bind(OLD_VIRTUAL / "trace.npz", old_report["inputs"][str(OLD_VIRTUAL / "trace.npz")])
    old_trace = arrays(OLD_VIRTUAL / "trace.npz")
    baseline_report = read(BASE / "normal_core_pico_v1/report.json")
    baseline_path = BASE / "normal_core_pico_v1/trace.npz"
    bind(baseline_path, baseline_report["inputs"][str(baseline_path)])
    baseline = arrays(baseline_path)
    motion = arrays(report["timeline"]["timeline_path"])
    source = read(BASE / "normal29_upstream_action_v1/actual_v1/frozen29_2ms/pico/report.json")
    p = CppParameters(read(source["parameter_path"]))
    predictor = MomentumSourceModel(
        bind(source["model_path"], source["model_sha256"]),
        p,
        source_effort29=source["actuation"]["effective_sim_effort29"],
    )
    torch.set_num_threads(1)
    teacher = ExactNormal29Teacher(bind(ASSETS / "sonic_release/last.pt"))
    count = len(attempt["released_raw29"])
    predicted_count = report["source_predictor"]["completed_predictions"]
    selected = set(np.linspace(0, count - 1, min(96, count), dtype=int).tolist())
    rows = deque((np.zeros(18, np.float32) for _ in range(10)), maxlen=10)
    accepted_index = 0
    for i in range(count):
        native_h = trace["native_precodec_history930"][i]
        validate_history(native_h, require_zero_action_slots=True)
        expected = source_action_history_numpy(native_h)
        model_rows = np.asarray(rows)
        for k, start in enumerate((30, 320, 610)):
            expected[start : start + 290].reshape(10, 29)[:, MISSING] = model_rows[:, k * 6 : k * 6 + 6]
        np.testing.assert_array_equal(expected, attempt["history930"][i])
        if i in selected:
            raw29, token64 = teacher.infer_with_token(attempt["encoder267"][i], expected)
            np.testing.assert_array_equal(raw29, attempt["released_raw29"][i])
            np.testing.assert_array_equal(raw29[KEEP], attempt["released_raw23"][i])
            np.testing.assert_array_equal(token64, attempt["decoder994"][i, :64])
        if i < predicted_count:
            raw = attempt["released_raw29"][i]
            prediction = predictor.advance(attempt["measured_qpos"][i], attempt["measured_qvel"][i], raw)
            if attempt["virtual_forecast_accepted"][i]:
                np.testing.assert_array_equal(
                    prediction, attempt["predicted_next_missing_state12"][accepted_index]
                )
                accepted_index += 1
                rows.append(np.r_[prediction, raw[MISSING]])
            else:
                assert i == count - 1 and predictor.maximum_virtual_joint_excess > 1e-6
        else:
            assert i == count - 1 and not attempt["virtual_forecast_accepted"][i]
    for key, value in predictor.assimilation_arrays().items():
        np.testing.assert_array_equal(value, assimilation[key])
    assert predictor.completed == predicted_count
    assert (
        predictor.maximum_virtual_joint_excess
        == report["source_predictor"]["maximum_predicted_missing_joint_range_excess_rad"]
    )
    phase = next(p for p in report["timeline"]["phases"] if p["name"] == "source_motion")
    all_traces = dict(momentum_model=trace, previous_velocity_copy_model=old_trace, zero_model_baseline=baseline)
    common = min(len(t["qpos"]) - 1 for t in all_traces.values())

    def tracking(saved, stop):
        start, stop = phase["control_start"], min(stop, phase["control_stop"])
        if stop <= start:
            return dict(measured=False, source_controls=0)
        qpos = saved["qpos"][1 : stop + 1]
        err = qpos[:, 7:] - motion["joint_pos"][11 : 11 + stop]
        return dict(
            measured=True,
            source_controls=stop - start,
            source_duration_s=(stop - start) * 0.02,
            leg_rmse_rad=float(np.sqrt(np.mean(err[start:stop, :12] ** 2))),
            waist_rmse_rad=float(np.sqrt(np.mean(err[start:stop, 12] ** 2))),
            arms_rmse_rad=float(np.sqrt(np.mean(err[start:stop, 13:] ** 2))),
            foot_ankle_origin_world_p95_m=np.percentile(
                saved["landmark_error_m"][start:stop, :2], 95, axis=0
            ).tolist(),
            foot_ankle_origin_pelvis_centered_world_axes_p95_m=np.percentile(
                saved["relative_landmark_error_m"][start:stop, :2], 95, axis=0
            ).tolist(),
            root_world_p95_m=float(np.percentile(saved["pelvis_error_m"][start:stop], 95)),
            foot_contact_or_slip_qualification=False,
        )

    same_prefix = {name: tracking(saved, common) for name, saved in all_traces.items()}
    entire = {name: tracking(saved, len(saved["qpos"]) - 1) for name, saved in all_traces.items()}
    completed = report["result"]["completed_controls"]
    result = dict(
        kind="momentum_source_native23_history_and_tracking_audit_v1",
        inputs=inputs,
        histories_reconstructed_bit_exact=count,
        causal_source_predictions_recomputed=predicted_count,
        all_momentum_assimilation_records_bit_exact=True,
        separate_frozen_source_decoder_reinference_bit_exact=len(selected),
        reinference_indices=sorted(selected),
        source_teacher=teacher.descriptor(),
        same_prefix_controls=common,
        identical_source_prefix_metrics=same_prefix,
        each_executed_source_metrics_not_equal_horizon_comparison=entire,
        full_source_completed=completed == report["result"]["requested_controls"]
        and report["result"]["failure"] is None,
        candidate_promoted=False,
        no_new_native23_rollout=True,
        hardware_authorized=False,
        deployment_ready=False,
    )
    for path, digest in inputs.items():
        assert file_sha256(Path(path)) == digest
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            dict(
                passed=True,
                histories=count,
                predictions=predicted_count,
                reinference=len(selected),
                identical_source_prefix_metrics=same_prefix,
                candidate_source_metrics=entire["momentum_model"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
