"""Read-only source-clock root, relative-foot and collision-geometry decomposition."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mjbatch_mpc import load_motion_override, load_native_bundle, motion_states, sha256


def run(args):
    native, contract, motion, timeline, manifest = load_native_bundle(args.bundle, args.clip)
    override = None
    if args.motion_override:
        motion, override = load_motion_override(
            args.motion_override, args.bundle, args.clip, native, contract, motion, timeline, manifest
        )
    phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    feet = [native.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    spheres = [
        i
        for i in range(native.ngeom)
        if native.geom_bodyid[i] in feet
        and native.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
        and native.geom_contype[i] != 0
    ]
    assert len(spheres) == 8
    floor = native.geom("floor").id
    assert native.geom_type[floor] == mujoco.mjtGeom.mjGEOM_PLANE
    with np.load(args.trace, allow_pickle=False) as archive:
        frames = archive["source_frame"].copy()
        actual_qpos = archive["qpos"][1:].copy()
    selected = (frames >= phase["control_start"] + 11) & (frames < phase["control_stop"] + 11)
    frames, actual_qpos = frames[selected], actual_qpos[selected]
    reference_qpos = motion_states(motion)[frames, :30]
    poses, clearances = [], []
    data = mujoco.MjData(native)
    for qposes in (reference_qpos, actual_qpos):
        body, clearance = [], []
        for qpos in qposes:
            data.qpos[:] = qpos
            mujoco.mj_kinematics(native, data)
            body.append(data.xpos[[1, *feet]].copy())
            normal = data.geom_xmat[floor].reshape(3, 3)[:, 2]
            clearance.append(
                (data.geom_xpos[spheres] - data.geom_xpos[floor]) @ normal - native.geom_size[spheres, 0]
            )
        poses.append(np.asarray(body))
        clearances.append(np.asarray(clearance))
    reference, actual = poses
    root_error = actual[:, 0] - reference[:, 0]
    world_foot_error = actual[:, 1:] - reference[:, 1:]
    relative_foot_error = world_foot_error - root_error[:, None]
    report = dict(
        kind="source_control_endpoints_native_geometry_decomposition",
        clip=args.clip,
        trace_sha256=sha256(args.trace),
        code_sha256=sha256(__file__),
        motion_override=override,
        mujoco=mujoco.__version__,
        source_controls=len(frames),
        floor_position_m=data.geom_xpos[floor].tolist(),
        floor_unit_normal=normal.tolist(),
        full_source_completed=len(frames) == phase["requested_controls"],
        root_abs_component_p95_m=np.percentile(np.abs(root_error), 95, axis=0).tolist(),
        root_xy_error_p95_m=float(np.percentile(np.linalg.norm(root_error[:, :2], axis=-1), 95)),
        root_3d_error_p95_m=float(np.percentile(np.linalg.norm(root_error, axis=-1), 95)),
        reference_root_z_min_p50_max_m=np.percentile(reference[:, 0, 2], [0, 50, 100]).tolist(),
        actual_root_z_min_p50_max_m=np.percentile(actual[:, 0, 2], [0, 50, 100]).tolist(),
        feet_world_error_p95_m=np.percentile(np.linalg.norm(world_foot_error, axis=-1), 95, axis=0).tolist(),
        feet_root_relative_error_p95_m=np.percentile(
            np.linalg.norm(relative_foot_error, axis=-1), 95, axis=0
        ).tolist(),
        feet_root_relative_abs_components_p95_m=np.percentile(np.abs(relative_foot_error), 95, axis=0).tolist(),
        reference_lowest_collision_clearance_min_p50_p95_max_m=np.percentile(
            clearances[0].min(axis=1), [0, 50, 95, 100]
        ).tolist(),
        actual_lowest_collision_clearance_min_p50_p95_max_m=np.percentile(
            clearances[1].min(axis=1), [0, 50, 95, 100]
        ).tolist(),
        reference_frames_all_spheres_above_10mm=int(np.sum(clearances[0].min(axis=1) > 0.01)),
        selected=[
            dict(
                source_frame=int(frames[i]),
                source_elapsed_seconds=float((frames[i] - phase["control_start"] - 10) * 0.02),
                root_reference=reference[i, 0].tolist(),
                root_actual=actual[i, 0].tolist(),
                foot_reference_clearance_m=clearances[0][i].reshape(2, 4).min(axis=1).tolist(),
                foot_actual_clearance_m=clearances[1][i].reshape(2, 4).min(axis=1).tolist(),
            )
            for i in sorted({0, len(frames) // 2, len(frames) - 1})
        ],
        interpretation=(
            "Endpoint geometry on unchanged source clock. Clearance alone is not dynamic-feasibility proof; "
            "original29 intent is independently assessed by the referee."
        ),
    )
    args.output.with_suffix(".source.py").write_bytes(Path(__file__).read_bytes())
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("bundle", "trace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--motion-override", type=Path)
    parser.add_argument("--clip", required=True)
    run(parser.parse_args())
