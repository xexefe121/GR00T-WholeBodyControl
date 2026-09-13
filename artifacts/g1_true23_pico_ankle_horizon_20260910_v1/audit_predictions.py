"""Independent accepted prediction/codec checks and fresh nominal-model copies."""

import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
EVAL = HERE / "eval1000_v1"
OUT = EVAL / "independent_predictions_audit.json"


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def main():
    if OUT.exists():
        raise FileExistsError("prediction audit refuses overwrite")
    summary = json.loads((EVAL / "summary.json").read_text())
    rows = []
    for case in summary["cases"]:
        directory = EVAL / case["name"]
        report = json.loads((directory / "report.json").read_text())
        assert sha256_file(directory / "report.json") == case["report_sha256"]
        for path, expected in report["inputs"].items():
            assert sha256_file(Path(path)) == expected, path
        for filename, field in (
            ("preview_predictions.npz", "preview_predictions_sha256"),
            ("preview_records.json", "preview_records_sha256"),
            ("trace.npz", "trace_sha256"),
            ("attempts.npz", "attempts_sha256"),
        ):
            assert sha256_file(directory / filename) == report[field]
        predictions = read(directory / "preview_predictions.npz")["qpos"]
        trace, attempts = read(directory / "trace.npz"), read(directory / "attempts.npz")
        records = json.loads((directory / "preview_records.json").read_text())
        n = case["completed"]
        assert predictions.shape == (n, 50, 30)
        assert len(records["accepted"]) == n
        assert len(attempts["accepted23"]) == n
        c = CleanTrue23MujocoController(model_path=ASSETS / MODEL, physics_path=ROOT / PHYSICS, policy=None)
        low, high = c.model.jnt_range[1:, 0] + 0.0019, c.model.jnt_range[1:, 1] - 0.0019
        q = predictions[:, :10, 7:]
        next_excess = float(np.maximum(np.maximum(low - q, q - high), 0).max())
        q = predictions[:, :, 7:][:, :, [4, 5, 10, 11]]
        ankle_excess = float(np.maximum(np.maximum(low[[4, 5, 10, 11]] - q, q - high[[4, 5, 10, 11]]), 0).max())
        assert next_excess == ankle_excess == 0
        for i, raw in enumerate(attempts["accepted23"]):
            _, target = safe_target_transform_numpy(raw)
            np.testing.assert_array_equal(target, trace["target23"][i])
            if not records["accepted"][i]["intervened"]:
                np.testing.assert_array_equal(raw, attempts["inverse23"][i])
        selected = np.unique(np.linspace(0, n - 1, min(96, n), dtype=int))
        for i in selected:
            q, v = attempts["measured_qpos"][i], attempts["measured_qvel"][i]
            mujoco.mj_resetData(c.model, c.data)
            c.reset(
                base_position=q[:3],
                base_quaternion_wxyz=q[3:7],
                joint_position_hardware=q[7:],
                root_velocity=v[:6],
                joint_velocity_hardware=v[6:],
            )
            target = trace["target23"][i].astype(float)
            for step in range(50):
                request = c.physics.kp * (target - c.data.qpos[7:]) - c.physics.kd * c.data.qvel[6:]
                c.data.ctrl[:] = np.clip(request, -c.physics.effort, c.physics.effort)
                mujoco.mj_step(c.model, c.data)
                np.testing.assert_array_equal(c.data.qpos, predictions[i, step])
        times = np.asarray([record["elapsed_s"] for record in records["accepted"]])
        row = dict(
            name=case["name"],
            all_accepted_checks_recomputed=n,
            fresh_prediction_controls_exact=len(selected),
            fresh_prediction_substeps_exact=50 * len(selected),
            next20ms_all_joint_reserve_excess=next_excess,
            extended100ms_ankle_reserve_excess=ankle_excess,
            filter_only_time_p50_p95_p99_max_s=np.percentile(times, [50, 95, 99, 100]).tolist(),
            filter_alone_over20ms=int((times > 0.02).sum()),
            measured_filter_timing_is_not_actual_clock_qualification=True,
            closed_loop_source_tracking_passed=case["source_tracking_passed"],
            deployment_ready=False,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    with OUT.open("x") as stream:
        json.dump(
            dict(
                cases=rows,
                predictor_implementation_imported=False,
                evidence_consistency_passed=True,
                auditor_sha256=sha256_file(Path(__file__)),
                deployment_ready=False,
                hardware_authorized=False,
            ),
            stream,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
