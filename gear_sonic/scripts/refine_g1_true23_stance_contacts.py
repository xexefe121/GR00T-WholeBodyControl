"""Refine every full stance candidate with joint contact/time constraints.

Writes new diagnostic candidates even when restoration remains infeasible;
every failure remains in the corpus report. Does not train or deploy policies.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import osqp
import scipy
from scipy.spatial.transform import Rotation

from gear_sonic.scripts import condition_g1_true23_reference_floor as geometry
from gear_sonic.scripts.retarget_g1_true23_stance import audit_rebuilt_causal_terms
from gear_sonic.utils import g1_true23_contact_trajectory
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
)
from gear_sonic.utils.g1_23dof_task_space_retarget import build_mjlab_motion_arrays, safe_target_joint_bounds
from gear_sonic.utils.g1_23dof_trajectory_projection import audit_trajectory_constraints
from gear_sonic.utils.g1_true23_contact_trajectory import (
    ContactLinearization,
    ContactTrajectoryConfig,
    refine_contact_trajectory,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos, reference_geometry
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics
from gear_sonic.utils.g1_true23_stance_retarget import StanceRetargetConfig


def load_motion(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def root_orientation_error(left, right):
    """Compare rotations, not quaternion signs or float32 normalization bits."""
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 4:
        raise ValueError("root quaternion arrays must have equal [frames,4] shape")
    return float(
        (Rotation.from_quat(left[:, [1, 2, 3, 0]]).inv() * Rotation.from_quat(right[:, [1, 2, 3, 0]]))
        .magnitude()
        .max(initial=0)
    )


def refinement_problem(model, source_motion, candidate, evidence):
    original = motion_qpos(model, source_motion)
    current = motion_qpos(model, candidate)
    if original.shape != current.shape or root_orientation_error(original[:, 3:7], current[:, 3:7]) > 2e-7:
        raise ValueError("contact refinement cannot change original frame count/root orientation")
    rows = evidence["solver_rows"]
    if [row["frame"] for row in rows] != list(range(len(original))):
        raise ValueError("support hypotheses must cover every original frame in order")
    cfg = StanceRetargetConfig(**evidence["config"])
    low, high = safe_target_joint_bounds(model, native_action_clip=9.5)
    low = np.maximum(low, np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + 0.05)
    high = np.minimum(high, np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) - 0.05)
    lower = np.column_stack(
        (
            np.full((len(original), 3), -cfg.maximum_root_offset_m),
            np.maximum(original[:, 7:] - cfg.maximum_joint_change_rad, low),
        )
    )
    upper = np.column_stack(
        (
            np.full((len(original), 3), cfg.maximum_root_offset_m),
            np.minimum(original[:, 7:] + cfg.maximum_joint_change_rad, high),
        )
    )
    return {
        "source_qpos": original,
        "desired": np.column_stack((current[:, :3] - original[:, :3], current[:, 7:])),
        "lower": lower,
        "upper": upper,
        "velocity": np.r_[
            np.full(3, cfg.maximum_offset_velocity_m_s), np.full(23, cfg.maximum_joint_velocity_rad_s)
        ],
        "acceleration": np.r_[
            np.full(3, cfg.maximum_offset_acceleration_m_s2), np.full(23, cfg.maximum_joint_acceleration_rad_s2)
        ],
        "initial_velocity": np.asarray(evidence["initial_projection_velocity"]),
        "supports": [row["support_bodies"] for row in rows],
        "input_root_orientation_serialization_error_rad": root_orientation_error(
            original[:, 3:7], current[:, 3:7]
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stance-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-iterations", type=int, default=20)
    parser.add_argument("--qp-maximum-iterations", type=int, default=100000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source_dir, output = args.stance_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = ContactTrajectoryConfig(
        maximum_iterations=args.maximum_iterations, qp_maximum_iterations=args.qp_maximum_iterations
    )
    manifest = json.loads((source_dir / "motions.json").read_text())
    if manifest.get("kind") != "g1_true23_stance_candidate_manifest_v1" or any(
        manifest.get(key) is not False for key in ("teacher_accepted", "hardware_authorized", "deployment_ready")
    ):
        raise ValueError("source must be a complete unaccepted stance candidate manifest")
    entries = manifest["motions"]
    names = [entry["name"] for entry in entries]
    if (
        not names
        or len(set(names)) != len(names)
        or any(
            not isinstance(name, str)
            or not name
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name)
            for name in names
        )
    ):
        raise ValueError("source must contain unique safe motion names")
    model_path = root / geometry.DEFAULT_TARGET_MODEL
    model = mujoco.MjModel.from_xml_path(str(model_path))
    training, runtime_sources = geometry.build_training_geometry()
    models = {"retarget_mesh": model, "training_capsules": training}
    model_hashes = {name: compiled_model_sha256(value) for name, value in models.items()}
    paths = [
        source_dir / "motions.json",
        model_path,
        Path(__file__),
        Path(inspect.getfile(g1_true23_contact_trajectory)),
        Path(inspect.getfile(geometry)),
        *runtime_sources,
    ]
    for name in (
        "g1_23dof_task_space_retarget.py",
        "g1_23dof_trajectory_projection.py",
        "g1_23dof_safe_target_transform.py",
        "g1_23dof_contract.py",
        "g1_true23_stance_retarget.py",
        "g1_true23_contact_geometry.py",
        "g1_true23_reference_floor.py",
        "g1_true23_sim_acquisition.py",
        "g1_true23_clean_mujoco_teleop.py",
        "g1_true23_sonic_library_replay.py",
    ):
        paths.append(root / "gear_sonic/utils" / name)
    paths.append(root / "gear_sonic/scripts/retarget_g1_true23_stance.py")
    inputs = {str(path.resolve()): geometry.sha256(path) for path in paths}
    loaded = []
    for entry in entries:
        path = Path(entry["path"]).resolve(strict=True)
        if path.parent != source_dir:
            raise ValueError("stance candidate must remain beside its report")
        report_path = source_dir / (entry["name"] + ".report.json")
        old = json.loads(report_path.read_text())
        original_path = Path(old["source"])
        if (
            geometry.sha256(path) != old["output_sha256"]
            or geometry.sha256(original_path) != old["source_sha256"]
            or old["retarget"]["collision_model_sha256"] != model_hashes
        ):
            raise ValueError("source candidate, original motion or collision model identity changed")
        for item in (path, report_path, original_path):
            inputs[str(item.resolve())] = geometry.sha256(item)
        loaded.append((entry, old, load_motion(original_path), load_motion(path)))
    output.mkdir(parents=True)
    records = []
    for entry, old, original, candidate in loaded:
        print(json.dumps({"starting_clip": entry["name"], "frames": len(candidate["joint_pos"])}), flush=True)
        problem = refinement_problem(model, original, candidate, old["retarget"])
        contacts = ContactLinearization(models, problem["source_qpos"], problem["supports"], config)
        refined, report = refine_contact_trajectory(
            *[
                problem[key]
                for key in ("desired", "lower", "upper", "velocity", "acceleration", "initial_velocity")
            ],
            contacts,
            config=config,
            progress=lambda status: print(json.dumps({"clip": entry["name"], **status}), flush=True),
        )
        source = problem["source_qpos"]
        arrays = build_mjlab_motion_arrays(
            model,
            SimpleNamespace(
                root_pos_w=(source[:, :3] + refined[:, :3]).astype(np.float32),
                root_quat_wxyz=source[:, 3:7],
                joint_pos_hardware=refined[:, 3:].astype(np.float32),
                fps=50.0,
            ),
        )
        serialized_qpos = motion_qpos(model, arrays)
        orientation_error = root_orientation_error(source[:, 3:7], serialized_qpos[:, 3:7])
        if orientation_error > 2e-7:
            raise ValueError("contact refinement changed original root rotation beyond serialization tolerance")
        serialized_path = np.column_stack((serialized_qpos[:, :3] - source[:, :3], serialized_qpos[:, 7:]))
        final_contact = contacts.audit(serialized_path)
        final_temporal = audit_trajectory_constraints(
            serialized_path,
            lower_bounds=problem["lower"],
            upper_bounds=problem["upper"],
            dt=0.02,
            max_velocity=problem["velocity"],
            max_acceleration=problem["acceleration"],
            initial_velocity=problem["initial_velocity"],
            tolerance=config.audit_tolerance,
        )
        if not final_temporal.passed:
            raise ValueError("serialized contact refinement failed immutable derivative/position bounds")
        fk = audit_reference_kinematics(SimpleNamespace(module=mujoco, model=model), arrays)
        if not fk["position_fk_consistent"] or not fk["orientation_fk_consistent"]:
            raise ValueError("serialized contact refinement failed native23 FK consistency")
        causal = audit_rebuilt_causal_terms(arrays)
        candidate_path = output / (entry["name"] + ".npz")
        with candidate_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        record = {
            "name": entry["name"],
            "source_sha256": old["source_sha256"],
            "previous_candidate_sha256": old["output_sha256"],
            "weight": entry["weight"],
            "output": str(candidate_path),
            "output_sha256": geometry.sha256(candidate_path),
            "frames_in": len(source),
            "frames_out": len(arrays["joint_pos"]),
            "frames_dropped": 0,
            "root_orientation_changed": False,
            "input_root_orientation_serialization_error_rad": problem[
                "input_root_orientation_serialization_error_rad"
            ],
            "output_root_orientation_serialization_error_rad": orientation_error,
            "root_orientation_serialization_tolerance_rad": 2e-7,
            "time_scale": 1.0,
            "controlled_joint_count": 23,
            "refinement": report,
            "serialized_contact_audit": final_contact,
            "serialized_temporal_audit": asdict(final_temporal),
            "serialized_fk_audit": fk,
            "geometry_after": {name: reference_geometry(value, arrays) for name, value in models.items()},
            "causal_terms": causal,
            "maximum_joint_change_from_original_rad": float(np.abs(serialized_qpos[:, 7:] - source[:, 7:]).max()),
            "maximum_root_offset_per_axis_m": np.abs(serialized_path[:, :3]).max(axis=0).tolist(),
            "contact_and_derivative_constraints_passed": final_contact["passed"] and final_temporal.passed,
            "dynamic_feasibility_proven": False,
            "teacher_accepted": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        }
        with (output / (entry["name"] + ".report.json")).open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
        records.append(record)
        print(
            json.dumps(
                {
                    "clip": entry["name"],
                    "contact_constraints_passed": final_contact["passed"],
                    "violated_frames": final_contact["violated_frames"],
                    "failure": report["failure"],
                }
            ),
            flush=True,
        )
    if any(geometry.sha256(Path(path)) != digest for path, digest in inputs.items()):
        raise RuntimeError("contact refinement input/source bytes changed during execution")
    if model_hashes != {name: compiled_model_sha256(value) for name, value in models.items()}:
        raise RuntimeError("contact refinement changed supplied models")
    summary = {
        "kind": "g1_true23_full_corpus_contact_refinement_v1",
        "records": records,
        "inputs": inputs,
        "compiled_models": model_hashes,
        "clips_in": len(entries),
        "clips_out": len(records),
        "contact_constraint_passes": sum(
            record["contact_and_derivative_constraints_passed"] for record in records
        ),
        "versions": {
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "osqp": osqp.__version__,
        },
        "production_training_changed": False,
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    with (output / "report.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    refined_manifest = {
        "kind": "g1_true23_stance_candidate_manifest_v1",
        "refinement_kind": summary["kind"],
        "motions": [
            {"name": record["name"], "path": record["output"], "weight": record["weight"]} for record in records
        ],
        "contact_constraints_passed_for_all_clips": summary["contact_constraint_passes"] == len(entries),
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    with (output / "motions.json").open("x") as stream:
        json.dump(refined_manifest, stream, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
