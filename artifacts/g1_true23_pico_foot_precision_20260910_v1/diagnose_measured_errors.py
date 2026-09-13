"""Locate measured drift/limit approach without executing another controller."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT.parent / "GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"


def yaw(quaternion):
    w, x, y, z = quaternion.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", type=int, choices=(600, 1000), required=True)
    args = parser.parse_args()
    output = HERE / f"measured_errors_{args.update}.json"
    if output.exists():
        raise FileExistsError("measured diagnosis refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None:
            assert digest == expected, path
        inputs[str(path)] = digest
        return path

    bind(__file__)
    model = mujoco.MjModel.from_xml_path(str(bind(MODEL)))
    joints = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    assert len(joints) == 23
    np.testing.assert_array_equal(model.jnt_qposadr[joints], np.arange(7, 30))
    np.testing.assert_array_equal(model.jnt_dofadr[joints], np.arange(6, 29))
    rows = []
    for name in ("walk002", "walk003", "walk008", "pico"):
        directory = HERE / f"eval{args.update}_v1" / name
        report = json.loads(bind(directory / "report.json").read_text())
        assert report["identity"]["completed_training_updates"] == args.update
        assert sha256_file(MODEL) == report["result"]["model_sha256"]
        with np.load(bind(directory / "trace.npz", report["trace_sha256"]), allow_pickle=False) as bundle:
            qpos, qvel = bundle["qpos"].copy(), bundle["qvel"].copy()
        timeline = report["timeline"]
        with np.load(bind(timeline["timeline_path"], timeline["timeline_sha256"]), allow_pickle=False) as bundle:
            reference_q = bundle["joint_pos"].copy()
            reference_p = bundle["body_pos_w"][:, 0].copy()
            reference_quat = bundle["body_quat_w"][:, 0].copy()
        phase = next(row for row in timeline["phases"] if row["name"] == "source_motion")
        start, stop = phase["control_start"], min(phase["control_stop"], len(qpos) - 1)
        assert stop > start
        measured = qpos[start + 1 : stop + 1]
        root_error = np.linalg.norm(measured[:, :3] - reference_p[start + 11 : stop + 11], axis=1)
        heading = yaw(measured[:, 3:7]) - yaw(reference_quat[start + 11 : stop + 11])
        heading = np.abs(np.arctan2(np.sin(heading), np.cos(heading)))
        outside = np.flatnonzero(root_error > 0.30)
        per_joint_rmse = np.sqrt(np.mean((measured[:, 7:] - reference_q[start + 11 : stop + 11]) ** 2, axis=0))
        closest = []
        for index, joint in enumerate(joints):
            lower, upper = model.jnt_range[joint]
            q, dq = qpos[-1, 7 + index], qvel[-1, 6 + index]
            margin = min(q - lower, upper - q)
            closest.append(
                dict(
                    joint=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
                    position_rad=float(q),
                    velocity_rad_s=float(dq),
                    lower_rad=float(lower),
                    upper_rad=float(upper),
                    nearest_margin_rad=float(margin),
                    velocity_toward_nearest_limit=bool(dq < 0 if q - lower < upper - q else dq > 0),
                    source_position_rmse_rad=float(per_joint_rmse[index]),
                )
            )
        row = dict(
            name=name,
            source_controls=stop - start,
            first_post_source_root_error_above_training_30cm_s=(
                float((outside[0] + 1) / 50) if len(outside) else None
            ),
            source_root_error_above_training_30cm_fraction=float((root_error > 0.30).mean()),
            source_root_heading_error_p95_rad=float(np.percentile(heading, 95)),
            final_root_height_m=float(qpos[-1, 2]),
            final_closest_joints=sorted(closest, key=lambda item: item["nearest_margin_rad"])[:5],
            largest_source_joint_position_errors=sorted(
                closest, key=lambda item: -item["source_position_rmse_rad"]
            )[:5],
            failure=report["result"]["failure"],
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(
        kind="foot_precision_measured_error_localization_v1",
        cases=rows,
        inputs=inputs,
        source_rate_hz=50,
        training_root_threshold_is_not_a_new_runtime_guard=True,
        causal_failure_attribution_claimed=False,
        no_controller_or_physics_execution=True,
        deployment_ready=False,
        hardware_authorized=False,
    )
    for path, expected in inputs.items():
        assert sha256_file(Path(path)) == expected, path
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
