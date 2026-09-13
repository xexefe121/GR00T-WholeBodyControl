"""Independent full-eight-clip geometry/force and original-source FK audit."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.scripts import (
    build_g1_true23_sonic_library_motions as library,
    condition_g1_true23_reference_floor as geometry,
)
from gear_sonic.scripts.refine_g1_true23_reference_forces import bind_identity, dump
from gear_sonic.scripts.refine_g1_true23_stance_contacts import load_motion
from gear_sonic.scripts.retarget_g1_true23_stance import audit_rebuilt_causal_terms
from gear_sonic.utils import g1_true23_actuation_profile, g1_true23_reference_support
from gear_sonic.utils.g1_23dof_task_space_retarget import DEFAULT_SOURCE_MODEL, DEFAULT_TASKS
from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos, reference_geometry
from gear_sonic.utils.g1_true23_reference_lineage import root_path_repair_bounds
from gear_sonic.utils.g1_true23_reference_support import audit_reference_support
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
FITTED = HERE.parent / "original_planner_se3_fit_20260906_v2"
PLANNER = ASSETS / "artifacts/g1_true23/sonic_library_motion_suite_v2"
RAW = {
    "original_sonic_hand_crawl": "hand_crawling",
    "original_sonic_elbow_crawl": "elbow_crawling",
    "original_sonic_happy_dance": "happy_dance",
}
FLAGS = dict(teacher_accepted=False, hardware_authorized=False, deployment_ready=False)


def original_task_audit(source, target, raw, native):
    xyz, quat, joints = library._resample_qpos(raw["qpos"])
    source_path, native_path = np.column_stack((xyz, quat, joints)), motion_qpos(target, native)
    if len(source_path) != len(native_path):
        raise ValueError("cannot compare cropped or retimed original motion")
    source_data, target_data = mujoco.MjData(source), mujoco.MjData(target)
    errors, relative_errors, orientation = [], [], []
    for source_pose, native_pose in zip(source_path, native_path, strict=True):
        source_data.qpos[:] = source_pose
        target_data.qpos[:] = native_pose
        mujoco.mj_forward(source, source_data)
        mujoco.mj_forward(target, target_data)
        frame_error, frame_relative, frame_orientation = [], [], []
        for task in DEFAULT_TASKS:
            positions, rotations = [], []
            for model, data, body_name, point in (
                (source, source_data, task.source_body, task.source_point),
                (target, target_data, task.target_body, task.target_point),
            ):
                body = model.body(body_name).id
                rotation = np.eye(3) if task.kind == "subtree_com" else data.xmat[body].reshape(3, 3)
                position = (
                    data.subtree_com[body].copy()
                    if task.kind == "subtree_com"
                    else data.xpos[body] + rotation @ np.asarray(point)
                )
                positions.append(position)
                rotations.append(rotation)
            frame_error.append(np.linalg.norm(positions[1] - positions[0]))
            frame_relative.append(
                np.linalg.norm((positions[1] - native_pose[:3]) - (positions[0] - source_pose[:3]))
            )
            frame_orientation.append(Rotation.from_matrix(rotations[1] @ rotations[0].T).magnitude())
        errors.append(frame_error)
        relative_errors.append(frame_relative)
        orientation.append(frame_orientation)
    errors, relative_errors, orientation = np.array(errors), np.array(relative_errors), np.array(orientation)
    root_rotation = (
        Rotation.from_quat(native_path[:, [4, 5, 6, 3]]) * Rotation.from_quat(source_path[:, [4, 5, 6, 3]]).inv()
    ).magnitude()
    return {
        "frames_checked": len(source_path),
        "frames_dropped": 0,
        "time_warp_applied": False,
        "root_path": root_path_repair_bounds(xyz, native_path[:, :3], maximum_offset_m=0),
        "maximum_root_orientation_error_rad": float(root_rotation.max()),
        "maximum_root_absolute_position_error_m": float(np.linalg.norm(native_path[:, :3] - xyz, axis=1).max()),
        "per_task": {
            task.name: {
                "position_max_m": float(errors[:, i].max()),
                "position_mean_m": float(errors[:, i].mean()),
                "relative_position_max_m": float(relative_errors[:, i].max()),
                "orientation_max_rad": float(orientation[:, i].max()),
            }
            for i, task in enumerate(DEFAULT_TASKS)
        },
        "both_feet_within_existing_5mm_screen": bool(errors[:, :2].max() <= 0.005),
        "policy_or_hardware_parity_proven": False,
    }


def main():
    if (HERE / "started.json").exists():
        raise FileExistsError("audit needs a fresh immutable output directory")
    fit_report = json.loads((FITTED / "report.json").read_text())
    if not fit_report["all_attempts_completed"] or fit_report["clips_written"] != 3:
        raise ValueError("all three original fitted outputs must exist")
    identities = dict(fit_report["inputs"])
    for path in list(identities):
        bind_identity(identities, path)
    manifest = json.loads((FITTED / "motions.json").read_text())
    original_manifest_path = (
        ROOT / "gear_sonic/config/sim_validation/g1_true23_frozen_lora_original_sonic_rehearsal_v1.json"
    )
    expected = json.loads(original_manifest_path.read_text())["motions"]
    entries = manifest["motions"]
    if [(e["name"], e["weight"]) for e in entries] != [(e["name"], e["weight"]) for e in expected] or len(
        entries
    ) != 8:
        raise ValueError("all eight original clip identities, order and weights must remain")
    for new, old in zip(entries, expected, strict=True):
        if new["name"] not in RAW and Path(new["path"]).resolve() != (ASSETS / old["path"]).resolve():
            raise ValueError("PICO reference was substituted")
        bind_identity(identities, new["path"])
        if identities[str(Path(new["path"]).resolve())] != new["sha256"]:
            raise ValueError("candidate motion differs from its manifest hash")
    profile = NativeSupportActuationProfile.from_sim_config(ROOT / SIM_CONFIG)
    limits = np.asarray(profile.effort) * 0.95 * 0.25
    source = mujoco.MjModel.from_xml_path(str(ROOT / DEFAULT_SOURCE_MODEL))
    mesh = mujoco.MjModel.from_xml_path(str(ROOT / geometry.DEFAULT_TARGET_MODEL))
    training, training_sources = geometry.build_training_geometry()
    models = {"retarget_mesh": mesh, "training_capsules": training}
    hashes = {name: compiled_model_sha256(model) for name, model in {"source29": source, **models}.items()}
    for path in (
        Path(__file__),
        FITTED / "report.json",
        original_manifest_path,
        ROOT / SIM_CONFIG,
        *(
            Path(inspect.getfile(module))
            for module in (geometry, g1_true23_actuation_profile, g1_true23_reference_support)
        ),
        ROOT / "gear_sonic/utils/g1_true23_contact_geometry.py",
        ROOT / "gear_sonic/utils/g1_true23_reference_floor.py",
        ROOT / "gear_sonic/utils/g1_true23_reference_lineage.py",
        *training_sources,
    ):
        bind_identity(identities, path)
    dump(HERE / "started.json", {"inputs": identities, "compiled_models": hashes, "complete": False, **FLAGS})
    records = []
    for entry in entries:
        name = entry["name"]
        arrays = load_motion(entry["path"])
        print(json.dumps({"starting_clip": name, "frames": len(arrays["joint_pos"])}), flush=True)
        record = {
            "name": name,
            "weight": entry["weight"],
            "frames": len(arrays["joint_pos"]),
            "geometry": {key: reference_geometry(model, arrays) for key, model in models.items()},
            "fk": audit_reference_kinematics(SimpleNamespace(module=mujoco, model=mesh), arrays),
            "causal": audit_rebuilt_causal_terms(arrays),
            **FLAGS,
        }
        if name in RAW:
            record["original_source_fidelity"] = original_task_audit(
                source, mesh, load_motion(PLANNER / f"{RAW[name]}.npz"), arrays
            )
        record["conditional_reference_force"] = {
            key: audit_reference_support(model, arrays, limits, gap_tolerance_m=0.002, reference_dynamics=True)
            for key, model in models.items()
        }
        dump(HERE / f"{name}.report.json", record)
        bind_identity(identities, HERE / f"{name}.report.json")
        records.append(record)
        print(
            json.dumps(
                {
                    "clip": name,
                    "floor_overlap_frames": {
                        key: value["frames_with_floor_overlap"] for key, value in record["geometry"].items()
                    },
                    "conditional_force_frames": {
                        key: value["frames_with_conditional_solution_within_effort_limits"]
                        for key, value in record["conditional_reference_force"].items()
                    },
                    "original_source_fidelity": record.get("original_source_fidelity"),
                }
            ),
            flush=True,
        )
    for path in list(identities):
        bind_identity(identities, path)
    if hashes != {name: compiled_model_sha256(model) for name, model in {"source29": source, **models}.items()}:
        raise ValueError("independent audit mutated compiled models")
    dump(
        HERE / "report.json",
        {
            "kind": "g1_true23_original_planner_se3_independent_corpus_screen_v1",
            "records": records,
            "files": identities,
            "compiled_models": hashes,
            "all_eight_clips_audited": True,
            "total_frames": sum(r["frames"] for r in records),
            "causal_packets_rebuilt": sum(r["causal"]["packets_rebuilt_and_validated"] for r in records),
            "policy_rollouts_or_training_performed": False,
            "dynamic_feasibility_proven": False,
            **FLAGS,
        },
    )


if __name__ == "__main__":
    main()
