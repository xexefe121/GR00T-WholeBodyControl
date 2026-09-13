"""Reconstruct all internal history, localize virtual failure, compare feet.

No new native23 control rollout. The existing physics_audit.json independently
reintegrates all16290 physical substeps with saved applied targets exactly.
This file replays the separate internal model, not the actual23 plant.
"""

from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.scripts import simulate_g1_sonic_library_motions as stock
from gear_sonic.utils.g1_29dof_normal_teacher import ExactNormal29Teacher
from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters, file_sha256
from gear_sonic.utils.g1_true23_discarded_action_memory import KEEP, MISSING, validate_history
from gear_sonic.utils.g1_true23_source_action_codec import source_action_history_numpy
from gear_sonic.utils.g1_true23_virtual_source_model import MISSING_HW, VirtualSourceModel

HERE = Path(__file__).resolve().parent
ACTUAL = HERE / "native_actual_v1"
BASE = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")
ASSETS = Path("/mnt/z/codex/GR00T-WholeBodyControl")


def main():
    output = ACTUAL / "history_and_tracking_audit.json"
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
    trace, attempts = arrays(ACTUAL / "trace.npz"), arrays(ACTUAL / "attempts.npz")
    baseline = arrays(BASE / "normal_core_pico_v1/trace.npz")
    motion = arrays(report["timeline"]["timeline_path"])
    source = read(BASE / "normal29_upstream_action_v1/actual_v1/frozen29_2ms/pico/report.json")
    p = CppParameters(read(source["parameter_path"]))
    predictor = VirtualSourceModel(
        bind(source["model_path"], source["model_sha256"]),
        p,
        source_effort29=source["actuation"]["effective_sim_effort29"],
    )
    torch.set_num_threads(1)
    teacher = ExactNormal29Teacher(bind(ASSETS / "sonic_release/last.pt"))
    count = len(attempts["released_raw29"])
    selected = set(np.linspace(0, count - 1, 96, dtype=int).tolist())
    rows = deque((np.zeros(18, np.float32) for _ in range(10)), maxlen=10)
    last_before = None
    last_target = None
    # Independent history construction: no adapter or VirtualSourceHistory calls.
    for i in range(count):
        native_h = trace["native_precodec_history930"][i]
        validate_history(native_h, require_zero_action_slots=True)
        expected = source_action_history_numpy(native_h)
        model_rows = np.asarray(rows)
        for k, start in enumerate((30, 320, 610)):
            expected[start : start + 290].reshape(10, 29)[:, MISSING] = model_rows[:, k * 6 : k * 6 + 6]
        np.testing.assert_array_equal(expected, attempts["history930"][i])
        if i in selected:
            raw29, token64 = teacher.infer_with_token(attempts["encoder267"][i], expected)
            np.testing.assert_array_equal(raw29, attempts["released_raw29"][i])
            np.testing.assert_array_equal(raw29[KEEP], attempts["released_raw23"][i])
            np.testing.assert_array_equal(token64, attempts["decoder994"][i, :64])
        last_before = predictor.internal_state12().copy()
        raw = attempts["released_raw29"][i]
        last_target = (
            p.default_angles + raw.astype(np.float64)[stock.MUJOCO_TO_ISAAC_INDEX] * p.action_scale
        ).astype(np.float32)
        prediction = predictor.advance(attempts["measured_qpos"][i], attempts["measured_qvel"][i], raw)
        if attempts["virtual_forecast_accepted"][i]:
            np.testing.assert_array_equal(prediction, attempts["predicted_next_missing_state12"][i])
            rows.append(np.r_[prediction, raw[MISSING]])
        else:
            assert i == count - 1 and predictor.maximum_virtual_joint_excess > 1e-6
    assert predictor.completed == report["source_predictor"]["completed_predictions"]
    assert (
        predictor.maximum_virtual_joint_excess
        == report["source_predictor"]["maximum_predicted_missing_joint_range_excess_rad"]
    )
    q = predictor.data.qpos[7:]
    limits = predictor.model.jnt_range[1:]
    excess = np.maximum(np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0)
    last = []
    for k, joint in enumerate(MISSING_HW):
        last.append(
            dict(
                name=predictor.model.joint(int(joint) + 1).name,
                source_hardware_index=int(joint),
                prior_q=float(last_before[k] + p.default_angles[joint]),
                prior_dq=float(last_before[k + 6]),
                requested_source_target=float(last_target[joint]),
                predicted_q=float(q[joint]),
                predicted_dq=float(predictor.data.qvel[6 + joint]),
                source_joint_range=limits[joint].tolist(),
                end_prediction_excess_rad=float(excess[joint]),
                final_predicted_source_motor_force_nm=float(predictor.data.qfrc_actuator[6 + joint]),
            )
        )
    phase = next(p for p in report["timeline"]["phases"] if p["name"] == "source_motion")
    start = phase["control_start"]
    stop = min(report["common_prefix_controls"], phase["control_stop"])
    metrics = {}
    for name, saved in (("internal_source_model", trace), ("zero_model_baseline", baseline)):
        qpos = saved["qpos"][1 : stop + 1]
        wanted = motion["joint_pos"][11 : 11 + stop]
        err = qpos[:, 7:] - wanted
        metrics[name] = dict(
            source_controls=stop - start,
            source_duration_s=(stop - start) * 0.02,
            leg_rmse_rad=float(np.sqrt(np.mean(err[start:stop, :12] ** 2))),
            waist_rmse_rad=float(np.sqrt(np.mean(err[start:stop, 12] ** 2))),
            arms_rmse_rad=float(np.sqrt(np.mean(err[start:stop, 13:] ** 2))),
            foot_ankle_origin_world_position_p95_m=np.percentile(
                saved["landmark_error_m"][start:stop, :2], 95, axis=0
            ).tolist(),
            foot_ankle_origin_pelvis_centered_world_axes_p95_m=np.percentile(
                saved["relative_landmark_error_m"][start:stop, :2], 95, axis=0
            ).tolist(),
            root_world_position_p95_m=float(np.percentile(saved["pelvis_error_m"][start:stop], 95)),
            foot_contact_or_slip_qualification=False,
        )
        assert (
            metrics[name]["leg_rmse_rad"]
            == report["common_prefix_only_metrics"][name]["unchanged_referee_post_control_q2"][
                "leg_joint_rmse_rad"
            ]
        )
    result = dict(
        kind="internal_source_native23_history_and_tracking_audit_v1",
        inputs=inputs,
        histories_reconstructed_bit_exact=count,
        causal_source_predictions_recomputed=count,
        separate_frozen_source_decoder_reinference_bit_exact=len(selected),
        reinference_indices=sorted(selected),
        source_teacher=teacher.descriptor(),
        missing_joint_state_at_failure=last,
        identical_source_prefix_metrics=metrics,
        full_source_completed=False,
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
                reinference=len(selected),
                failure_joints=[r for r in last if r["end_prediction_excess_rad"] > 0],
                metrics=metrics,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
