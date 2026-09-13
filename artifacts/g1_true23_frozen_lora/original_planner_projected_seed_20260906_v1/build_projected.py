"""Whole-horizon joint projection baseline, not a dynamic or task-space teacher.

All original 30-Hz planner samples define the unchanged 50-Hz time grid. This
diagnostic tests whether offline anticipation reduces the sequential IK seed's
foot error. It does not claim that projecting joint coordinates compensates
for absent waist/wrist joints. No root translation/rotation or tempo changes.
"""

from dataclasses import asdict
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.scripts import build_g1_true23_sonic_library_motions as library
from gear_sonic.scripts.refine_g1_true23_reference_forces import bind_identity, dump
from gear_sonic.scripts.refine_g1_true23_stance_contacts import load_motion
from gear_sonic.scripts.retarget_g1_true23_stance import audit_rebuilt_causal_terms
from gear_sonic.utils import g1_23dof_task_space_retarget as retarget, g1_23dof_trajectory_projection as projection
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
PLANNER = ASSETS / "artifacts/g1_true23/sonic_library_motion_suite_v2"
NAMES = {
    "original_sonic_hand_crawl": "hand_crawling",
    "original_sonic_elbow_crawl": "elbow_crawling",
    "original_sonic_happy_dance": "happy_dance",
}
FLAGS = dict(teacher_accepted=False, hardware_authorized=False, deployment_ready=False)


def task_errors(source_model, target_model, source_qpos, native):
    source_data, target_data = mujoco.MjData(source_model), mujoco.MjData(target_model)
    position, angle = [], []
    for index, qpos in enumerate(source_qpos):
        source_data.qpos[:] = qpos
        target_data.qpos[:] = np.r_[
            native["body_pos_w"][index, 0], native["body_quat_w"][index, 0], native["joint_pos"][index]
        ]
        mujoco.mj_forward(source_model, source_data)
        mujoco.mj_forward(target_model, target_data)
        source_pos, source_quat = retarget._task_pose_arrays(
            source_model, source_data, retarget.DEFAULT_TASKS, source=True
        )
        target_pos, target_quat = retarget._task_pose_arrays(
            target_model, target_data, retarget.DEFAULT_TASKS, source=False
        )
        position.append(np.linalg.norm(source_pos - target_pos, axis=1))
        angle.append(
            (
                Rotation.from_quat(source_quat[:, [1, 2, 3, 0]]).inv()
                * Rotation.from_quat(target_quat[:, [1, 2, 3, 0]])
            ).magnitude()
        )
    position, angle = np.asarray(position), np.asarray(angle)
    return {
        task.name: {
            "position_mean_m": float(position[:, index].mean()),
            "position_p95_m": float(np.percentile(position[:, index], 95)),
            "position_max_m": float(position[:, index].max()),
            "orientation_mean_rad": float(angle[:, index].mean()),
            "orientation_max_rad": float(angle[:, index].max()),
        }
        for index, task in enumerate(retarget.DEFAULT_TASKS)
    }


def main():
    if (HERE / "started.json").exists():
        raise FileExistsError("use a new immutable experiment directory")
    identities = {}
    source_path, target_path = ROOT / retarget.DEFAULT_SOURCE_MODEL, ROOT / retarget.DEFAULT_TARGET_MODEL
    manifest_path = (
        ROOT / "gear_sonic/config/sim_validation/g1_true23_frozen_lora_original_sonic_rehearsal_v1.json"
    )
    for path in (
        Path(__file__),
        source_path,
        target_path,
        manifest_path,
        PLANNER / "report.json",
        *(PLANNER / f"{name}.npz" for name in NAMES.values()),
        *(Path(inspect.getfile(module)) for module in (library, retarget, projection)),
    ):
        bind_identity(identities, path)
    source_model, target_model = retarget.load_models(source_path, target_path)
    model_hashes = {"source": compiled_model_sha256(source_model), "target": compiled_model_sha256(target_model)}
    source_layout, target_layout = retarget._model_layout(source_model), retarget._model_layout(target_model)
    indices = [source_layout.joint_names.index(name) for name in target_layout.joint_names]
    low, high = retarget.safe_target_joint_bounds(target_model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    source_records = {r["name"]: r for r in json.loads((PLANNER / "report.json").read_text())["records"]}
    dump(
        HERE / "started.json", {"inputs": identities, "compiled_models": model_hashes, "complete": False, **FLAGS}
    )
    records = []
    for name, raw_name in NAMES.items():
        raw_path = PLANNER / f"{raw_name}.npz"
        if identities[str(raw_path)] != source_records[raw_name]["npz_sha256"]:
            raise ValueError("original planner hash changed")
        raw = load_motion(raw_path)
        assert float(raw["fps"][0]) == 30 and int(raw["mode"][0]) == library.PLANNER_MODES[raw_name]
        xyz, quat, joints = library._resample_qpos(raw["qpos"])
        direct = np.clip(joints[:, indices], low, high)
        initial_velocity = np.clip((direct[1] - direct[0]) / 0.02, -4.975, 4.975)
        print(json.dumps({"starting_original_clip": name, "frames": len(xyz)}), flush=True)
        record = {
            "name": name,
            "frames": len(xyz),
            "frames_dropped": 0,
            "time_scale": 1.0,
            "raw_source_sha256": identities[str(raw_path)],
            **FLAGS,
        }
        try:
            result = projection.project_nearest_trajectory(
                direct,
                lower_bounds=low,
                upper_bounds=high,
                dt=0.02,
                max_velocity=4.975,
                max_acceleration=79.6,
                initial_velocity=initial_velocity,
            )
            arrays = retarget.build_mjlab_motion_arrays(
                target_model,
                SimpleNamespace(
                    joint_pos_hardware=result.projected_path,
                    root_pos_w=xyz,
                    root_quat_wxyz=quat,
                    fps=50,
                ),
            )
            output = HERE / f"{name}.npz"
            with output.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            arrays = load_motion(output)
            audit = projection.audit_trajectory_constraints(
                arrays["joint_pos"],
                lower_bounds=low,
                upper_bounds=high,
                dt=0.02,
                max_velocity=5,
                max_acceleration=80,
                initial_velocity=initial_velocity,
                tolerance=2e-7,
            )
            fk = audit_reference_kinematics(SimpleNamespace(module=mujoco, model=target_model), arrays)
            if not audit.passed or not fk["position_fk_consistent"] or not fk["orientation_fk_consistent"]:
                raise ValueError("saved whole-path projection failed temporal/FK audit")
            errors = task_errors(source_model, target_model, np.column_stack((xyz, quat, joints)), arrays)
            bind_identity(identities, output)
            record.update(
                output=str(output),
                output_sha256=identities[str(output)],
                projection={
                    "iterations": result.iterations,
                    "objective": result.objective,
                    "primal_residual": result.primal_residual_max,
                    "dual_residual": result.dual_residual_max,
                },
                temporal=asdict(audit),
                initial_velocity=initial_velocity.tolist(),
                fk=fk,
                causal=audit_rebuilt_causal_terms(arrays),
                task_errors=errors,
                failure=None,
                both_feet_within_existing_5mm_screen=all(
                    errors[foot]["position_max_m"] <= 0.005 for foot in ("left_foot", "right_foot")
                ),
            )
        except (ValueError, RuntimeError) as error:
            record.update(output=None, failure={"type": type(error).__name__, "message": str(error)})
        records.append(record)
        dump(HERE / f"{name}.report.json", record)
        print(json.dumps(record), flush=True)
    entries = json.loads(manifest_path.read_text())["motions"]
    outputs = {row["name"]: row.get("output") for row in records}
    if all(outputs.values()):
        corpus = []
        for entry in entries:
            path = (
                Path(outputs[entry["name"]])
                if entry["name"] in outputs
                else (ASSETS / entry["path"]).resolve(strict=True)
            )
            bind_identity(identities, path)
            corpus.append({**entry, "path": str(path), "sha256": identities[str(path)]})
        dump(
            HERE / "motions.json",
            {
                "kind": "g1_true23_original_planner_projection_diagnostic_manifest_v1",
                "motions": corpus,
                "unchanged_pico_clip_count": len(entries) - len(NAMES),
                "diagnostic_only": True,
                **FLAGS,
            },
        )
        bind_identity(identities, HERE / "motions.json")
    for path in list(identities):
        bind_identity(identities, path)
    assert model_hashes == {
        "source": compiled_model_sha256(source_model),
        "target": compiled_model_sha256(target_model),
    }
    dump(
        HERE / "report.json",
        {
            "kind": "g1_true23_original_planner_whole_path_projection_diagnostic_v1",
            "records": records,
            "inputs": identities,
            "compiled_models": model_hashes,
            "all_attempts_completed": True,
            "clips_written": sum(bool(r.get("output")) for r in records),
            "contact_force_qualification_deferred": True,
            "kinematic_or_controller_parity_proven": False,
            **FLAGS,
        },
    )


if __name__ == "__main__":
    main()
