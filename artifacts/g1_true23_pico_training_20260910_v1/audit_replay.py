"""Fresh saved-torque replay and milestone inference audit; never live control."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import CleanTrue23MujocoController
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS, task_points
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_normal_lora_replay import NormalLoraPolicy
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    output = args.directory / "independent_audit.json"
    if output.exists():
        raise FileExistsError("checkpoint replay audit refuses overwrite")
    summary = json.loads((args.directory / "summary.json").read_text())
    request = json.loads((args.directory / "started.json").read_text())
    checkpoints = [
        Path(p) for p, digest in request["inputs"].items() if digest == summary["identity"]["checkpoint_sha256"]
    ]
    assert len(checkpoints) == 1
    torch.set_num_threads(1)
    policy = NormalLoraPolicy(
        checkpoints[0],
        warm_start_path=ASSETS / "sonic_release/g1_23dof_rev_1_0_init.pt",
        source_checkpoint_path=ASSETS / "sonic_release/last.pt",
    )
    rows = []
    for case in summary["cases"]:
        directory = args.directory / case["name"]
        report = json.loads((directory / "report.json").read_text())
        assert sha256_file(directory / "report.json") == case["report_sha256"]
        for path, expected in report["inputs"].items():
            assert sha256_file(Path(path)) == expected, path
        for filename, key in (("trace.npz", "trace_sha256"), ("attempts.npz", "attempts_sha256")):
            assert sha256_file(directory / filename) == report[key]
        trace, attempts = read(directory / "trace.npz"), read(directory / "attempts.npz")
        motion = read(Path(report["timeline"]["timeline_path"]))
        n = len(trace["qpos"]) - 1
        assert n == report["result"]["completed_controls"]
        assert len(trace["physics_time"]) == n * 10
        assert all(np.isfinite(v).all() for group in (trace, attempts) for v in group.values())
        c = CleanTrue23MujocoController(model_path=ASSETS / MODEL, physics_path=ROOT / PHYSICS, policy=None)
        assert compiled_model_sha256(c.model) == report["result"]["compiled_model_sha256"]
        q, v = trace["qpos"][0], trace["qvel"][0]
        c.reset(
            base_position=q[:3],
            base_quaternion_wxyz=q[3:7],
            joint_position_hardware=q[7:],
            root_velocity=v[:6],
            joint_velocity_hardware=v[6:],
        )
        for i, tau in enumerate(trace["applied_torque23"]):
            np.testing.assert_array_equal(c.data.qpos, trace["physics_pre_qpos"][i])
            np.testing.assert_array_equal(c.data.qvel, trace["physics_pre_qvel"][i])
            c.data.ctrl[:] = tau
            mujoco.mj_step(c.model, c.data)
            np.testing.assert_array_equal(c.data.qpos, trace["physics_post_qpos"][i])
            np.testing.assert_array_equal(c.data.qvel, trace["physics_post_qvel"][i])
            if i % 10 == 9:
                np.testing.assert_array_equal(c.data.qpos, trace["qpos"][i // 10 + 1])
                np.testing.assert_array_equal(c.data.qvel, trace["qvel"][i // 10 + 1])
        probe = mujoco.MjData(c.model)
        for i, q in enumerate(trace["qpos"][1:]):
            probe.qpos[:], probe.qvel[:] = q, trace["qvel"][i + 1]
            mujoco.mj_forward(c.model, probe)
            actual = task_points(probe.xpos[1:], probe.xquat[1:])
            reference = task_points(motion["body_pos_w"][i + 11], motion["body_quat_w"][i + 11])
            np.testing.assert_array_equal(np.linalg.norm(actual - reference, axis=1), trace["landmark_error_m"][i])
            relative = (actual - q[:3]) - (reference - motion["body_pos_w"][i + 11, 0])
            np.testing.assert_array_equal(np.linalg.norm(relative, axis=1), trace["relative_landmark_error_m"][i])
        count = len(attempts["released_raw23"])
        assert n <= count <= n + 1
        np.testing.assert_array_equal(attempts["measured_qpos"], trace["qpos"][:count])
        np.testing.assert_array_equal(attempts["measured_qvel"], trace["qvel"][:count])
        selected = np.unique(np.linspace(0, count - 1, min(96, count), dtype=int))
        for i in selected:
            raw, decoder = policy.infer(
                attempts["encoder267"][i], attempts["history930"][i], attempts["root_feedback9"][i]
            )
            np.testing.assert_array_equal(raw, attempts["released_raw23"][i])
            np.testing.assert_array_equal(decoder, attempts["decoder994"][i])
        q = trace["physics_post_qpos"][:, 7:]
        lower, upper = c.model.jnt_range[1:, 0], c.model.jnt_range[1:, 1]
        excess = float(np.maximum(np.maximum(lower - q, q - upper), 0).max())
        assert excess == case["hard_range_excess"]
        row = dict(
            name=case["name"],
            physical_substeps_bit_exact=n * 10,
            post_control_landmarks_recomputed_exact=n,
            independent_policy_reinference_exact=len(selected),
            joint_range_excess_rad=excess,
            velocity_ratio=case["velocity_ratio"],
            effort_ratio=float(np.max(np.abs(trace["applied_torque23"]) / c.physics.effort)),
            full_source_tracking_passed=case["source_tracking_passed"],
            source_metrics=case["source_metrics"],
            deployment_ready=False,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    with output.open("x") as stream:
        json.dump(
            dict(
                cases=rows,
                evidence_consistency_passed=True,
                controller_qualification_claimed=False,
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
