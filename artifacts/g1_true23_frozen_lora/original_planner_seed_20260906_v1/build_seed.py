"""All-three original planner task-space IK seeds; no rollout teacher or promotion.

The failed lower/root initializer is not used as a feasibility certificate.
All 23 joints instead participate in local task-space IK with hard temporal
bounds. Whole-path geometry/force restoration and original-source fidelity
must still be qualified independently. Every failed clip is recorded.
"""

from dataclasses import asdict
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from gear_sonic.scripts import build_g1_true23_sonic_library_motions as library
from gear_sonic.scripts.refine_g1_true23_reference_forces import bind_identity, dump
from gear_sonic.scripts.refine_g1_true23_stance_contacts import load_motion
from gear_sonic.scripts.retarget_g1_true23_stance import audit_rebuilt_causal_terms
from gear_sonic.utils import g1_23dof_task_space_retarget as retarget
from gear_sonic.utils import g1_23dof_trajectory_projection as projection
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_reference_lineage import root_path_repair_bounds
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"
PLANNER = ASSETS / "artifacts/g1_true23/sonic_library_motion_suite_v2"
NAMES = ("hand_crawling", "elbow_crawling", "happy_dance")


def main():
    if (HERE / "started.json").exists():
        raise FileExistsError("seed attempt already started; use a separate immutable output directory")
    identities = {}
    source_path, target_path = ROOT / retarget.DEFAULT_SOURCE_MODEL, ROOT / retarget.DEFAULT_TARGET_MODEL
    for path in (
        Path(__file__),
        source_path,
        target_path,
        PLANNER / "report.json",
        *(PLANNER / f"{name}.npz" for name in NAMES),
        *(Path(inspect.getfile(module)) for module in (library, retarget, projection)),
    ):
        bind_identity(identities, path)
    source_model, target_model = retarget.load_models(source_path, target_path)
    model_hashes = {"source": compiled_model_sha256(source_model), "target": compiled_model_sha256(target_model)}
    config = retarget.RetargetConfig(
        optimize_lower_body=True,
        enable_lower_root_feasibility=False,
        max_velocity_rad_s=5,
        max_acceleration_rad_s2=80,
        native_action_clip=9.5,
        safe_limit_guard_rad=0.05,
    )
    dump(
        HERE / "started.json",
        {
            "config": asdict(config),
            "inputs": identities,
            "compiled_models": model_hashes,
            "complete": False,
            "teacher_accepted": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        },
    )
    original_records = {row["name"]: row for row in json.loads((PLANNER / "report.json").read_text())["records"]}
    records = []
    for name in NAMES:
        source_file = PLANNER / f"{name}.npz"
        if identities[str(source_file.resolve())] != original_records[name]["npz_sha256"]:
            raise ValueError("original planner identity changed")
        raw = load_motion(source_file)
        assert float(raw["fps"][0]) == 30 and int(raw["mode"][0]) == library.PLANNER_MODES[name]
        xyz, quat, joints = library._resample_qpos(raw["qpos"])
        print(
            json.dumps(
                {
                    "starting_original_clip": name,
                    "source_30hz_frames": len(raw["qpos"]),
                    "target_50hz_frames": len(xyz),
                }
            ),
            flush=True,
        )
        source50 = HERE / f"{name}.source50.npz"
        with source50.open("xb") as stream:
            np.savez_compressed(stream, qpos=np.column_stack((xyz, quat, joints)), fps=np.array([50.0]))
        bind_identity(identities, source50)
        record = {
            "name": name,
            "source_sha256": original_records[name]["npz_sha256"],
            "source50": str(source50),
            "frames_requested": len(xyz),
            "controlled_joints": 23,
            "time_scale": 1.0,
            "retimed": False,
            "frames_dropped": 0,
            "output": None,
            "whole_path_contact_force_qualification_deferred": True,
            "teacher_accepted": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        }
        try:
            result = retarget.retarget_trajectory(
                source_model=source_model,
                target_model=target_model,
                root_pos_w=xyz,
                root_quat_wxyz=quat,
                source_joint_pos_hardware=joints,
                fps=50,
                config=config,
            )
            arrays = retarget.build_mjlab_motion_arrays(target_model, result)
            output = HERE / f"{name}.true23.npz"
            with output.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            arrays = load_motion(output)
            fk = audit_reference_kinematics(SimpleNamespace(module=mujoco, model=target_model), arrays)
            if not fk["position_fk_consistent"] or not fk["orientation_fk_consistent"]:
                raise ValueError("serialized seed FK audit failed")
            lower, upper = retarget.safe_target_joint_bounds(
                target_model, native_action_clip=9.5, safe_limit_guard_rad=0.05
            )
            temporal = projection.audit_trajectory_constraints(
                arrays["joint_pos"],
                lower_bounds=lower,
                upper_bounds=upper,
                dt=0.02,
                max_velocity=5,
                max_acceleration=80,
                initial_velocity=(arrays["joint_pos"][1] - arrays["joint_pos"][0]) / 0.02,
                tolerance=2e-7,
            )
            if not temporal.passed:
                raise ValueError("serialized seed violates unchanged position/velocity/acceleration limits")
            causal = audit_rebuilt_causal_terms(arrays)
            bind_identity(identities, output)
            record.update(
                output=str(output),
                output_sha256=identities[str(output.resolve())],
                frames_out=len(arrays["joint_pos"]),
                summary=result.summary(),
                temporal=asdict(temporal),
                fk=fk,
                causal=causal,
                original_root_path=root_path_repair_bounds(xyz, arrays["body_pos_w"][:, 0], maximum_offset_m=0),
                failure=None,
            )
        except (ValueError, RuntimeError) as error:
            record["failure"] = {"type": type(error).__name__, "message": str(error)}
        dump(HERE / f"{name}.report.json", record)
        records.append(record)
        print(
            json.dumps({"clip": name, "written": record["output"] is not None, "failure": record["failure"]}),
            flush=True,
        )
    for path in list(identities):
        bind_identity(identities, path)
    assert model_hashes == {
        "source": compiled_model_sha256(source_model),
        "target": compiled_model_sha256(target_model),
    }
    dump(
        HERE / "report.json",
        {
            "kind": "g1_true23_original_planner_task_space_seed_attempt_v1",
            "records": records,
            "inputs": identities,
            "config": asdict(config),
            "compiled_models": model_hashes,
            "clips_requested": len(NAMES),
            "clips_written": sum(row["output"] is not None for row in records),
            "all_attempts_completed": True,
            "teacher_accepted": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        },
    )


if __name__ == "__main__":
    main()
