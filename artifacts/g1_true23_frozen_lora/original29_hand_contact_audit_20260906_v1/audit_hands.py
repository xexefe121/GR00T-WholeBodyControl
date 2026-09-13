"""Inspect hand collision masks, visual geometry and virtual SONIC landmarks."""

import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils import g1_23dof_task_space_retarget as retarget
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent / "original29_recorded_baseline_20260906_v1"
FIT = HERE.parent / "original29_recorded_native23_fit_20260906_v1"


def main():
    if (HERE / "report.json").exists():
        raise FileExistsError("use a new evidence directory")
    identities = {}

    def bind(path):
        identities[str(path)] = file_sha256(path)
        return path

    bind(Path(__file__))
    bind(Path(retarget.__file__))
    bind(BASE / "report.json")
    bind(FIT / "report.json")
    bind(ROOT / "gear_sonic_deploy/g1/scene_29dof.xml")
    bind(ROOT / "gear_sonic_deploy/g1/g1_29dof_old.xml")
    bind(ROOT / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp")
    models = {
        "original29": mujoco.MjModel.from_binary_path(str(bind(BASE / "original29.mjb"))),
        "native23": mujoco.MjModel.from_binary_path(str(bind(FIT / "native23.mjb"))),
    }
    records = []
    for name in ("hand_crawling", "happy_dance"):
        with np.load(bind(BASE / f"{name}.cpp_parameters_and_float32_targets.npz")) as archive:
            original = archive["pre_qpos"].copy()
        with np.load(bind(FIT / f"{name}.native23.npz")) as archive:
            native = np.column_stack(
                (archive["body_pos_w"][:, 0], archive["body_quat_w"][:, 0], archive["joint_pos"])
            )
        for label, poses in (("original29", original), ("native23", native)):
            model, rows = models[label], []
            data = mujoco.MjData(model)
            geoms = [
                g
                for g in range(model.ngeom)
                if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                and "rubber_hand" in model.mesh(int(model.geom_dataid[g])).name
            ]
            metadata = [
                {
                    "geom_id": g,
                    "body": model.body(int(model.geom_bodyid[g])).name,
                    "mesh": model.mesh(int(model.geom_dataid[g])).name,
                    "contype": int(model.geom_contype[g]),
                    "conaffinity": int(model.geom_conaffinity[g]),
                }
                for g in geoms
            ]
            for frame, pose in enumerate(poses):
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                task_positions, _ = retarget._task_pose_arrays(
                    model, data, retarget.DEFAULT_TASKS, source=label == "original29"
                )
                hands = {
                    task.name: position.tolist()
                    for task, position in zip(retarget.DEFAULT_TASKS, task_positions, strict=True)
                    if task.name in ("left_hand", "right_hand")
                }
                lowest = []
                for geom in geoms:
                    mesh = int(model.geom_dataid[geom])
                    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                    vertices = (
                        model.mesh_vert[start : start + count] @ data.geom_xmat[geom].reshape(3, 3).T
                        + data.geom_xpos[geom]
                    )
                    lowest.append(float(vertices[:, 2].min()))
                rows.append(
                    {
                        "frame": frame,
                        "virtual_hand_positions": hands,
                        "minimum_rubber_hand_vertex_z_m": min(lowest),
                    }
                )
            worst = min(rows, key=lambda row: row["minimum_rubber_hand_vertex_z_m"])
            records.append(
                {
                    "clip": name,
                    "model": label,
                    "frames": len(poses),
                    "hand_geoms": metadata,
                    "explicit_contact_pair_count": int(model.npair),
                    "frames_with_virtual_hand_point_below_floor": sum(
                        any(value[2] < 0 for value in row["virtual_hand_positions"].values()) for row in rows
                    ),
                    "frames_with_rubber_hand_mesh_below_floor_including_visual_only": sum(
                        row["minimum_rubber_hand_vertex_z_m"] < 0 for row in rows
                    ),
                    "worst": worst,
                    "frame191": rows[191],
                }
            )
    if any(file_sha256(path) != digest for path, digest in identities.items()):
        raise ValueError("hand audit input changed")
    dump(
        HERE / "report.json",
        {
            "kind": "g1_original29_native23_hand_contact_semantics_audit_v1",
            "inputs": identities,
            "records": records,
            "virtual_vr_offset_is_not_a_physical_contact_landmark": True,
            "native23_collisions_disabled_or_geometry_modified": False,
            "source_baseline_is_not_full_physical_hand_collision_qualification": True,
            "teacher_accepted": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        },
    )


if __name__ == "__main__":
    main()
