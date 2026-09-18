"""Rank recorded native23 rollouts by tracking fidelity against the packet reference.

Scores the quantities the packets themselves carry: the native23 joint reference
`q_ref23_native`, and the pelvis-local three-point VR targets
`vr_3point_local_target`, whose order is (left hand, right hand, head) with the
same fixed body offsets the source adapter applies.

This is not the pinned `measure_g1_true23_saved_teleop_tracking` qualification
path, which accepts only a public TWIST2 import report. It reuses the same
reference quantities and the same physical model so that candidates can be
ranked against each other on one clip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_23dof_native124_21204_adapter import native_to_hardware_compact
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

# Body offsets applied by gear_sonic/utils/g1_23dof_xr24_soma_adapter.py when it
# builds vr_3point_local_target, in the order the packet stores them.
POINTS = (
    ("left_wrist_roll_rubber_hand", (0.18, -0.025, 0.0)),
    ("right_wrist_roll_rubber_hand", (0.18, 0.025, 0.0)),
    ("torso_link", (0.0, 0.0, 0.35)),
)
POINT_NAMES = ("left_hand", "right_hand", "head")
LEG_JOINTS = 12  # first twelve native23 joints are the two legs


def local_points(model, poses):
    """Pelvis-local positions of the three VR reference points, per pose."""
    data = mujoco.MjData(model)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name, _ in POINTS]
    if pelvis < 0 or any(index < 0 for index in bodies):
        raise ValueError("model lacks a required reference body")
    out = np.empty((len(poses), 3, 3), dtype=np.float64)
    for row, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_fwdPosition(model, data)
        origin = data.xpos[pelvis]
        rotation = data.xmat[pelvis].reshape(3, 3)
        for slot, (index, (_, offset)) in enumerate(zip(bodies, POINTS)):
            point = data.xpos[index] + data.xmat[index].reshape(3, 3) @ np.asarray(offset)
            out[row, slot] = rotation.T @ (point - origin)
    return out


def percentile95(values):
    return float(np.percentile(values, 95))


def score(trace_path: Path, packets_path: Path, model):
    with np.load(trace_path, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)

    bundle = json.loads(packets_path.read_text(encoding="utf-8"))
    packets = bundle["robot_independent_reference_packets"]

    # qpos holds the initial state plus one state per control; packet k pairs with
    # state k, so the last state has no packet and is not scored.
    controls = min(len(qpos), len(packets)) - 1
    if controls < 1:
        raise ValueError(f"{trace_path}: {len(qpos)} states against {len(packets)} packets")

    # Packet 0 carries control q10, which is the initial recorded state qpos[0].
    # State qpos[k] is therefore scored against packet k, verified by the initial
    # state matching packet 0 to 0.032 rad maximum.
    #
    # q_ref23_native is stored in NATIVE_IL23 order; MuJoCo qpos is in hardware
    # order. Convert with the repository's own mapping rather than assuming the
    # two orders agree - they do not.
    measured = qpos[1 : controls + 1]
    reference_joints = np.asarray(
        [native_to_hardware_compact(packets[k]["q_ref23_native"]) for k in range(1, controls + 1)],
        dtype=np.float64,
    )
    reference_points = np.asarray(
        [packets[k]["vr_3point_local_target"] for k in range(1, controls + 1)], dtype=np.float64
    ).reshape(controls, 3, 3)

    joint_error = measured[:, 7:] - reference_joints
    measured_points = local_points(model, measured)
    point_error = np.linalg.norm(measured_points - reference_points, axis=2)

    result = {
        "controls_scored": controls,
        "leg_rmse_rad": float(np.sqrt(np.mean(joint_error[:, :LEG_JOINTS] ** 2))),
        "all_joint_rmse_rad": float(np.sqrt(np.mean(joint_error**2))),
        "max_abs_joint_error_rad": float(np.max(np.abs(joint_error))),
    }
    for slot, name in enumerate(POINT_NAMES):
        result[f"{name}_p95_m"] = percentile95(point_error[:, slot])
    result["worst_point_p95_m"] = max(result[f"{name}_p95_m"] for name in POINT_NAMES)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-directory", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.repository_root.resolve()
    _, model, _ = prepare_true23_model(root / MODEL, root / PHYSICS)

    rows = []
    for directory in sorted(p for p in args.sweep_directory.iterdir() if p.is_dir()):
        trace = directory / "measured_trace.npz"
        report = directory / "report.json"
        if not trace.exists() or not report.exists():
            print(f"skip (incomplete): {directory.name}")
            continue
        recorded = json.loads(report.read_text(encoding="utf-8"))
        row = {"checkpoint": directory.name, "passed": recorded.get("passed")}
        row.update(score(trace, args.packets, model))
        rows.append(row)
        print(f"scored: {directory.name}")

    rows.sort(key=lambda r: r["worst_point_p95_m"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    header = (
        f"{'checkpoint':58} {'legRMSE':>8} {'allRMSE':>8} "
        f"{'Lhand95':>8} {'Rhand95':>8} {'head95':>8}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['checkpoint']:58} {row['leg_rmse_rad']:8.4f} {row['all_joint_rmse_rad']:8.4f} "
            f"{row['left_hand_p95_m']:8.4f} {row['right_hand_p95_m']:8.4f} {row['head_p95_m']:8.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
