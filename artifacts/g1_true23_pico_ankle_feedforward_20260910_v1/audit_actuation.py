"""Independent total-torque math, unchanged prefix, and fresh accepted previews."""

import ast
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
EVAL = HERE / "eval1000_v1"
OUT = EVAL / "independent_actuation_audit.json"


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def main():
    if OUT.exists():
        raise FileExistsError("actuation audit refuses overwrite")
    summary = json.loads((EVAL / "summary.json").read_text())
    old = ast.parse((ROOT / "gear_sonic/utils/g1_true23_generalist_benchmark.py").read_text())
    new = ast.parse((ROOT / "gear_sonic/utils/g1_true23_ankle_feedforward_benchmark.py").read_text())
    old_helpers = {
        node.name: ast.dump(node) for node in old.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    new_helpers = {
        node.name: ast.dump(node) for node in new.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert old_helpers.keys() == new_helpers.keys()
    assert all(old_helpers[key] == new_helpers[key] for key in old_helpers if key != "run_reference_diagnostic")
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
        n, ff = case["completed"], trace["feedforward_torque23"]
        assert predictions.shape == (n, 10, 30) and ff.shape == (n, 23)
        assert len(records["accepted"]) == len(attempts["accepted23"]) == n
        assert ff.dtype == np.float64 and np.isfinite(ff).all() and np.max(np.abs(ff)) <= 10
        other = [j for j in range(23) if j not in (4, 5, 10, 11)]
        assert not np.any(ff[:, other])
        ff_indices = np.flatnonzero(np.any(ff, axis=1))
        assert all(
            record["feedforward_used"] == bool(np.any(ff[i])) for i, record in enumerate(records["accepted"])
        )
        profile = NativeModelActuationProfile.from_sim_config(ROOT / PHYSICS)
        target = np.repeat(trace["target23"].astype(float), 10, axis=0)
        pd = (
            np.asarray(profile.kp) * (target - trace["physics_pre_qpos"][:, 7:])
            - np.asarray(profile.kd) * trace["physics_pre_qvel"][:, 6:]
        )
        total = pd + np.repeat(ff, 10, axis=0)
        np.testing.assert_array_equal(total, trace["requested_torque23"])
        np.testing.assert_array_equal(
            np.clip(total, -np.asarray(profile.effort), profile.effort), trace["applied_torque23"]
        )
        np.testing.assert_array_equal(trace["applied_torque23"], trace["engine_actuator_force23"])
        assert not np.any(trace["physics_external_force_world_n"])
        for i, raw in enumerate(attempts["accepted23"]):
            _, decoded = safe_target_transform_numpy(raw)
            np.testing.assert_array_equal(decoded, trace["target23"][i])
            if np.any(ff[i]):
                np.testing.assert_array_equal(raw, attempts["inverse23"][i])
        earlier = ROOT / "artifacts/g1_true23_pico_foot_precision_20260910_v1/eval1000_v1" / case["name"]
        earlier_report = json.loads((earlier / "report.json").read_text())
        assert sha256_file(earlier / "trace.npz") == earlier_report["trace_sha256"]
        assert sha256_file(earlier / "attempts.npz") == earlier_report["attempts_sha256"]
        baseline, baseline_attempts = read(earlier / "trace.npz"), read(earlier / "attempts.npz")
        first = int(ff_indices[0]) if len(ff_indices) else n
        if len(ff_indices):
            assert first == earlier_report["result"]["completed_controls"]
        for field in ("qpos", "qvel"):
            np.testing.assert_array_equal(trace[field][: first + 1], baseline[field][: first + 1])
        for field in (
            "physics_pre_qpos",
            "physics_post_qpos",
            "physics_pre_qvel",
            "physics_post_qvel",
            "requested_torque23",
            "applied_torque23",
            "engine_actuator_force23",
        ):
            np.testing.assert_array_equal(trace[field][: first * 10], baseline[field][: first * 10])
        count = first + int(len(ff_indices) > 0)
        for field in ("encoder267", "history930", "decoder994", "released_raw23", "inverse23", "root_feedback9"):
            np.testing.assert_array_equal(attempts[field][:count], baseline_attempts[field][:count])
        c = CleanTrue23MujocoController(model_path=ASSETS / MODEL, physics_path=ROOT / PHYSICS, policy=None)
        low, high = c.model.jnt_range[1:, 0] + 0.0019, c.model.jnt_range[1:, 1] - 0.0019
        poses = predictions[:, :, 7:]
        reserve_excess = float(np.maximum(np.maximum(low - poses, poses - high), 0).max())
        assert reserve_excess == 0
        selected = np.unique(np.r_[np.linspace(0, n - 1, min(96, n), dtype=int), ff_indices])
        selected_ff_velocity_ratio = 0.0
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
            position_target = trace["target23"][i].astype(float)
            for step in range(10):
                pd_request = c.physics.kp * (position_target - c.data.qpos[7:]) - c.physics.kd * c.data.qvel[6:]
                c.data.ctrl[:] = np.clip(pd_request + ff[i], -c.physics.effort, c.physics.effort)
                mujoco.mj_step(c.model, c.data)
                np.testing.assert_array_equal(c.data.qpos, predictions[i, step])
                if np.any(ff[i]):
                    selected_ff_velocity_ratio = max(
                        selected_ff_velocity_ratio, float((np.abs(c.data.qvel[6:]) / profile.velocity).max())
                    )
        assert selected_ff_velocity_ratio <= 1
        times = np.asarray([record["elapsed_s"] for record in records["accepted"]])
        row = dict(
            name=case["name"],
            actual_pd_plus_feedforward_steps_verified=n * 10,
            original_position_only_prefix_controls_bit_exact=first,
            first_feedforward_control=int(ff_indices[0]) if len(ff_indices) else None,
            feedforward_controls=len(ff_indices),
            all_accepted_predicted_range_checks=n,
            fresh_prediction_controls_bit_exact=len(selected),
            all_feedforward_predictions_reintegrated=True,
            accepted_feedforward_prediction_velocity_ratio=selected_ff_velocity_ratio,
            filter_p50_p95_p99_max_s=np.percentile(times, [50, 95, 99, 100]).tolist(),
            filter_alone_over20ms=int((times > 0.02).sum()),
            source_tracking_passed=case["source_tracking_passed"],
            deployment_ready=False,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    with OUT.open("x") as stream:
        json.dump(
            dict(
                cases=rows,
                other_benchmark_helper_asts_unchanged=True,
                feedforward_implementation_imported=False,
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
