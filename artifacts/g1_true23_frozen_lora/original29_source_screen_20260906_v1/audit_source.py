"""Separate nominal planner geometry from archived original29 controller motion.

No controller execution or pose repair. Raw planner poses are not assumed to
have valid contacts just because they came from the released planner.
"""

import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.scripts import build_g1_true23_sonic_library_motions as library
from gear_sonic.scripts.refine_g1_true23_reference_forces import bind_identity, dump
from gear_sonic.scripts.refine_g1_true23_stance_contacts import load_motion
from gear_sonic.utils.g1_23dof_task_space_retarget import DEFAULT_SOURCE_MODEL
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_reference_lineage import root_path_repair_bounds

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
PLANNER = ASSETS / "artifacts/g1_true23/sonic_library_motion_suite_v2"
STOCK = ASSETS / "artifacts/g1_true23/sonic_policy_physical_suite_v2"


def floor_audit(model, poses):
    data = mujoco.MjData(model)
    floors = set(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist())
    if len(floors) != 1:
        raise ValueError("source audit requires one explicit floor")
    rows = []
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        contacts = []
        for contact in data.contact[: data.ncon]:
            if any(int(g) in floors for g in contact.geom):
                body = next(
                    model.body(int(model.geom_bodyid[int(g)])).name for g in contact.geom if int(g) not in floors
                )
                contacts.append({"body": body, "distance_m": float(contact.dist)})
        contacts.sort(key=lambda item: item["distance_m"])
        rows.append(
            {
                "frame": frame,
                "minimum_distance_m": contacts[0]["distance_m"] if contacts else None,
                "worst_contacts": contacts[:3],
                "foot_origin_heights_m": {
                    name: float(data.xpos[model.body(name).id, 2])
                    for name in ("left_ankle_roll_link", "right_ankle_roll_link")
                },
            }
        )
    return {
        "frames_checked": len(rows),
        "floor_overlap_frames": sum(
            row["minimum_distance_m"] is not None and row["minimum_distance_m"] < 0 for row in rows
        ),
        "worst_overlap_m": max(
            [0] + [-row["minimum_distance_m"] for row in rows if row["minimum_distance_m"] is not None]
        ),
        "rows": rows,
        "contact_validity_or_dynamic_feasibility_proven": False,
    }


def main():
    if (HERE / "report.json").exists():
        raise FileExistsError("use a new source audit directory")
    identities = {}
    retarget_path = ROOT / DEFAULT_SOURCE_MODEL
    stock_scene = ASSETS / "gear_sonic_deploy/g1/scene_29dof.xml"
    for path in (
        Path(__file__),
        retarget_path,
        stock_scene,
        stock_scene.parent / "g1_29dof.xml",
        PLANNER / "report.json",
        STOCK / "report.json",
        Path(library.__file__),
        ROOT / "gear_sonic/utils/g1_true23_reference_lineage.py",
        ROOT / "gear_sonic/scripts/simulate_g1_sonic_library_motions.py",
    ):
        bind_identity(identities, path)
    stock_report = json.loads((STOCK / "report.json").read_text())
    if identities[str(stock_scene)] != stock_report["scene_sha256"]:
        raise ValueError("stock scene wrapper does not match its archived report")
    stock_records = {r["name"]: r for r in stock_report["records"]}
    planner_records = {r["name"]: r for r in json.loads((PLANNER / "report.json").read_text())["records"]}
    models = {
        "retarget_source29": mujoco.MjModel.from_xml_path(str(retarget_path)),
        "current_stock_scene29": mujoco.MjModel.from_xml_path(str(stock_scene)),
    }
    hashes = {name: compiled_model_sha256(model) for name, model in models.items()}
    records = []
    for name in ("hand_crawling", "elbow_crawling", "happy_dance"):
        path = PLANNER / f"{name}.npz"
        bind_identity(identities, path)
        if identities[str(path)] != planner_records[name]["npz_sha256"]:
            raise ValueError("planner source hash mismatch")
        raw = load_motion(path)
        xyz, quat, joints = library._resample_qpos(raw["qpos"])
        qpos = np.column_stack((xyz, quat, joints))
        record = {
            "name": name,
            "raw_planner_frames": len(qpos),
            "raw_planner_geometry": {key: floor_audit(model, qpos) for key, model in models.items()},
        }
        if name in stock_records:
            old = stock_records[name]
            path = STOCK / old["physical_npz"]
            bind_identity(identities, path)
            if (
                identities[str(path)] != old["physical_npz_sha256"]
                or old["source_planner_npz_sha256"] != planner_records[name]["npz_sha256"]
            ):
                raise ValueError("archived stock trajectory provenance mismatch")
            physical = load_motion(path)
            if physical["qpos"].shape != qpos.shape or float(physical["control_dt"][0]) != 0.02:
                raise ValueError("stock trajectory does not cover the same full command count")
            record["archived_stock29"] = {
                "root_path": root_path_repair_bounds(xyz, physical["qpos"][:, :3], maximum_offset_m=0),
                "comparison_timing": "post_control_pose_for_each_same_index_planner_command; no phase shift or crop",
                "sample_is_one_control_interval_after_its_command": True,
                "archive_metrics": old["metrics"],
                "geometry_in_current_stock_scene": floor_audit(models["current_stock_scene29"], physical["qpos"]),
                "original_artifact_lacks_full_compiled_model_and_source_closure": True,
                "new_stock_policy_execution_or_parity_qualification": False,
            }
        else:
            record["archived_stock29"] = None
        dump(HERE / f"{name}.report.json", record)
        bind_identity(identities, HERE / f"{name}.report.json")
        records.append(record)
        print(
            json.dumps(
                {
                    "clip": name,
                    "planner_overlaps": {
                        key: {field: value[field] for field in ("floor_overlap_frames", "worst_overlap_m")}
                        for key, value in record["raw_planner_geometry"].items()
                    },
                    "stock29_root_comparison": record["archived_stock29"]["root_path"]
                    if record["archived_stock29"]
                    else None,
                }
            ),
            flush=True,
        )
    for path in list(identities):
        bind_identity(identities, path)
    if hashes != {name: compiled_model_sha256(model) for name, model in models.items()}:
        raise ValueError("source audit changed compiled models")
    dump(
        HERE / "report.json",
        {
            "kind": "g1_true23_original29_source_geometry_and_lineage_screen_v1",
            "records": records,
            "files": identities,
            "compiled_models": hashes,
            "raw_planner_is_not_an_accepted_physics_teacher": True,
            "original29_and_original_planner_fidelity_are_separate_comparisons": True,
            "teacher_accepted": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        },
    )


if __name__ == "__main__":
    main()
