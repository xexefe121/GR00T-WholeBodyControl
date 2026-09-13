"""Read-only native geometry diagnostics; support-box tests are not feasibility proofs."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, motion_states, sha256


def run(bundle, clip, output):
    model, _, motion, timeline, _ = load_native_bundle(bundle, clip)
    states = motion_states(motion)
    feet = [model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    spheres = [
        index
        for index in range(model.ngeom)
        if model.geom_bodyid[index] in feet
        and model.geom_type[index] == mujoco.mjtGeom.mjGEOM_SPHERE
        and model.geom_contype[index] != 0
    ]
    assert len(spheres) == 8
    data = mujoco.MjData(model)
    com, positions = [], []
    for state in states:
        data.qpos[:] = state[:30]
        mujoco.mj_kinematics(model, data)
        com.append(np.sum(model.body_mass[:, None] * data.xipos, axis=0) / model.body_mass.sum())
        positions.append(data.geom_xpos[spheres].copy())
    com, positions = np.asarray(com), np.asarray(positions)
    clearance = positions[:, :, 2] - model.geom_size[spheres, 0]
    phase = next(item for item in timeline["phases"] if item["name"] == "source_motion")
    first = phase["control_start"] + 11
    last = phase["control_stop"] + 11
    summary = {}
    for name, ids in (
        ("standing", np.arange(11, 261)),
        ("transition", np.arange(261, first)),
        ("source_first3s", np.arange(first, min(first + 150, last))),
        ("full_source", np.arange(first, last)),
    ):
        outside, no_support = [], 0
        for frame in ids:
            candidates = positions[frame, clearance[frame] <= 0.01, :2]
            if not len(candidates):
                no_support += 1
                continue
            lower, upper = candidates.min(axis=0), candidates.max(axis=0)
            outside.append(
                float(np.linalg.norm(np.maximum(np.maximum(lower - com[frame, :2], com[frame, :2] - upper), 0)))
            )
        summary[name] = dict(
            frames=len(ids),
            frames_without_sphere_within_10mm=no_support,
            lowest_sphere_clearance_min_p50_p95_max=np.percentile(
                clearance[ids].min(axis=1), [0, 50, 95, 100]
            ).tolist(),
            com_xy_outside_near_floor_sphere_axis_box_p50_p95_max=np.percentile(outside, [50, 95, 100]).tolist()
            if outside
            else None,
        )
    report = dict(
        kind="native_reference_geometry_support_diagnostic",
        source_sha256=sha256(Path(bundle) / clip / "native_original.npz"),
        code_sha256=sha256(Path(__file__)),
        mujoco=mujoco.__version__,
        interpretation=(
            "Reference geometry only. A static COM test ignores acceleration and angular momentum; "
            "this is not a dynamic-feasibility verdict."
        ),
        summary=summary,
        selected_frames=[
            dict(
                frame=frame,
                root=states[frame, :3].tolist(),
                com=com[frame].tolist(),
                foot_clearance=clearance[frame].reshape(2, 4).min(axis=1).tolist(),
            )
            for frame in (10, 300, 350, 360, 380, 400, 420, 500)
        ],
    )
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--clip", default="walk002")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.bundle, args.clip, args.output)
