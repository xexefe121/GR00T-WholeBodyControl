"""Independent saved-torque and observation reconstruction; no new controller trial."""

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    ACTION_SCALE_HARDWARE,
    HOME_Q_HARDWARE,
    Native124Checkpoint21204Policy,
    load_checkpoint21204_binding,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
BASE = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def metrics(trace, motion, phase, stop):
    start, end = phase["control_start"], min(stop, phase["control_stop"])
    if end <= start:
        return dict(source_controls=0)
    q = trace["qpos"][start + 1 : end + 1]
    err = q[:, 7:] - motion["joint_pos"][start + 11 : end + 11]
    root = np.linalg.norm(q[:, :3] - motion["body_pos_w"][start + 11 : end + 11, 0], axis=-1)
    return dict(
        source_controls=end - start,
        leg_rmse_rad=float(np.sqrt(np.mean(err[:, :12] ** 2))),
        arm_rmse_rad=float(np.sqrt(np.mean(err[:, 13:] ** 2))),
        root_p95_m=float(np.percentile(root, 95)),
        relative_foot_p95_m=np.percentile(trace["relative_landmark_error_m"][start:end, :2], 95, axis=0).tolist(),
    )


def main():
    output = HERE / "actual_v1/audit.json"
    if output.exists():
        raise FileExistsError("audit refuses overwrite")
    summary = json.loads((HERE / "actual_v1/summary.json").read_text())
    policy = Native124Checkpoint21204Policy(load_checkpoint21204_binding(ASSETS))
    rows = []
    for case in summary["cases"]:
        directory = HERE / "actual_v1" / case["case"]
        report = json.loads((directory / "report.json").read_text())
        assert sha256_file(directory / "report.json") == case["report_sha256"]
        for path, expected in report["inputs"].items():
            assert sha256_file(Path(path)) == expected, path
        for name, key in (("trace.npz", "trace_sha256"), ("attempts.npz", "attempts_sha256")):
            assert sha256_file(directory / name) == report[key]
        trace, attempts = read(directory / "trace.npz"), read(directory / "attempts.npz")
        motion = read(Path(report["timeline"]["timeline_path"]))
        n = len(trace["qpos"]) - 1
        count = len(attempts["observation124"])
        assert n == case["completed_controls"] and n <= count <= n + 1
        assert len(trace["physics_time"]) == n * 10
        np.testing.assert_array_equal(attempts["measured_qpos"], trace["qpos"][:count])
        np.testing.assert_array_equal(attempts["measured_qvel"], trace["qvel"][:count])
        np.testing.assert_array_equal(attempts["applied_target23"], trace["target23"])
        expected_previous = np.concatenate(
            (np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, np.float32)[None], trace["target23"])
        )[:count]
        np.testing.assert_array_equal(attempts["previous_target23"], expected_previous)
        c = CleanTrue23MujocoController(model_path=ASSETS / MODEL, physics_path=ROOT / PHYSICS, policy=None)
        assert compiled_model_sha256(c.model) == report["result"]["compiled_model_sha256"]
        first_q, first_v = trace["qpos"][0], trace["qvel"][0]
        c.reset(
            base_position=first_q[:3],
            base_quaternion_wxyz=first_q[3:7],
            joint_position_hardware=first_q[7:],
            root_velocity=first_v[:6],
            joint_velocity_hardware=first_v[6:],
        )
        max_q = max_v = 0.0
        for step, torque in enumerate(trace["applied_torque23"]):
            np.testing.assert_array_equal(c.data.qpos, trace["physics_pre_qpos"][step])
            np.testing.assert_array_equal(c.data.qvel, trace["physics_pre_qvel"][step])
            c.data.ctrl[:] = torque
            mujoco.mj_step(c.model, c.data)
            max_q = max(max_q, float(np.max(np.abs(c.data.qpos - trace["physics_post_qpos"][step]))))
            max_v = max(max_v, float(np.max(np.abs(c.data.qvel - trace["physics_post_qvel"][step]))))
        assert max_q == max_v == 0.0
        probe = mujoco.MjData(c.model)
        max_obs = 0.0
        for i in range(count):
            q9, q10 = i + 9, i + 10
            anchor = q9 if report["phase"] == "causal_q9" else q10
            if report["phase"] == "current_q10":
                pose = trace["qpos"][i]
            elif i:
                pose = trace["qpos"][i - 1]
            else:
                pose = np.r_[motion["body_pos_w"][9, 0], motion["body_quat_w"][9, 0], motion["joint_pos"][9]]
            probe.qpos[:] = pose
            mujoco.mj_kinematics(c.model, probe)
            robot = Rotation.from_quat(probe.xquat[c.model.body("torso_link").id][[1, 2, 3, 0]])
            reference = Rotation.from_quat(motion["body_quat_w"][anchor, 13, [1, 2, 3, 0]])
            ori = (robot.inv() * reference).as_matrix()[:, :2].reshape(6).astype(np.float32)
            expected = np.r_[
                motion["joint_pos"][anchor].astype(np.float32),
                (motion["joint_pos"][q10].astype(np.float32) - motion["joint_pos"][q9].astype(np.float32))
                / np.float32(0.02),
                ori,
                trace["qvel"][i, 3:6].astype(np.float32),
                trace["qpos"][i, 7:].astype(np.float32) - HOME_Q_HARDWARE,
                trace["qvel"][i, 6:].astype(np.float32),
                (expected_previous[i] - HOME_Q_HARDWARE) / ACTION_SCALE_HARDWARE,
            ].astype(np.float32)
            obs = attempts["observation124"][i]
            max_obs = max(max_obs, float(np.max(np.abs(expected - obs))))
            np.testing.assert_allclose(expected, obs, atol=2e-7, rtol=0)
            np.testing.assert_array_equal(policy.run(obs[None]), attempts["selected_raw_hw23"][i])
            np.testing.assert_array_equal(attempts["source_indices"][i], [q9, q10, anchor])
        ranges = c.model.jnt_range[1:]
        q, v = trace["physics_post_qpos"][:, 7:], trace["physics_post_qvel"][:, 6:]
        excess = float(np.maximum(np.maximum(ranges[:, 0] - q, q - ranges[:, 1]), 0).max())
        velocity_limit = np.asarray(
            json.loads((ROOT / PHYSICS).read_text())["physics"]["velocity_limit_hardware_radps"]
        )
        speed = float((np.abs(v) / velocity_limit).max())
        effort = float((np.abs(trace["applied_torque23"]) / c.physics.effort).max())
        assert excess == case["hard_range_excess_rad"] and speed == case["velocity_ratio"]
        old_dir = BASE / "released_core_comparison_v1/normal" / report["name"]
        if report["name"] == "pico":
            old_dir = BASE / "normal_core_pico_v1"
        old_report = json.loads((old_dir / "report.json").read_text())
        old_trace_path = old_dir / "trace.npz"
        assert sha256_file(old_trace_path) == old_report["inputs"][str(old_trace_path)]
        baseline = read(old_trace_path)
        np.testing.assert_array_equal(first_q, baseline["qpos"][0])
        np.testing.assert_array_equal(first_v, baseline["qvel"][0])
        source = next(row for row in report["timeline"]["phases"] if row["name"] == "source_motion")
        common = min(n, len(baseline["qpos"]) - 1)
        row = dict(
            case=case["case"],
            source_pins=len(report["inputs"]),
            controls=n,
            substeps=n * 10,
            saved_torque_reintegration_exact=True,
            observations_reconstructed=count,
            observation_reconstruction_max_abs_error=max_obs,
            independent_onnx_reinference_exact=count,
            physical_range_excess_rad=excess,
            velocity_ratio=speed,
            effort_ratio=effort,
            physical_bounds_passed=excess == 0 and speed <= 1 and effort <= 1,
            matched_source_prefix=dict(
                native124=metrics(trace, motion, source, common), sonic=metrics(baseline, motion, source, common)
            ),
            compared_baseline_trace_sha256=sha256_file(old_trace_path),
            original_full_result=case,
            deployment_ready=False,
            teacher_label_admitted=False,
        )
        rows.append(row)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "case",
                        "controls",
                        "observations_reconstructed",
                        "physical_bounds_passed",
                        "matched_source_prefix",
                    )
                }
            ),
            flush=True,
        )
    with output.open("x") as stream:
        json.dump(
            dict(
                cases=rows,
                auditor_sha256=sha256_file(Path(__file__)),
                passed=True,
                means_evidence_consistency_not_controller_acceptance=True,
                deployment_ready=False,
                teacher_label_admitted=False,
            ),
            stream,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
