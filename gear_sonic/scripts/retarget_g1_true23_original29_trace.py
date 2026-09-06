"""Fit complete recorded original29 motion using all native23 joints.

The source is explicitly the parameter-corrected offline policy execution,
not the nominal planner or a previous native23 controller rollout. Every
requested original clip remains in the report; incomplete sources cannot be
cropped, padded, repeated or silently replaced. No robot I/O or training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.record_g1_sonic_original29_baseline import CLIPS, dump
from gear_sonic.scripts.retarget_g1_true23_stance import audit_rebuilt_causal_terms
from gear_sonic.utils import g1_23dof_task_space_retarget as retarget
from gear_sonic.utils.g1_23dof_trajectory_projection import project_nearest_trajectory
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_original_task_trajectory import (
    OriginalTaskConfig,
    OriginalTaskPath,
    fit_original_task_path,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos, reference_geometry
from gear_sonic.utils.g1_true23_reference_lineage import root_path_repair_bounds
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics

PROFILE = "cpp_parameters_and_float32_targets"
FLAGS = dict(teacher_accepted=False, hardware_authorized=False, deployment_ready=False)


def completed_trace_source(record, arrays):
    """Require full, contiguous, same-command-time source states, not survival labels."""
    details = record.get("details", {})
    if (
        record.get("failure") is not None
        or record.get("profile") != PROFILE
        or details.get("profile") != PROFILE
        or details.get("complete_original_clip") is not True
        or details.get("pre_qpos_is_same_command_time_state") is not True
        or any(details.get(key) is not False for key in FLAGS)
    ):
        raise ValueError("requires a complete, explicitly unaccepted C++-parameter simulation source")
    count = details.get("frames_requested")
    if type(count) is not int or count < 3 or details.get("frames_completed") != count:
        raise ValueError("incomplete source clips cannot be retargeted as a whole original clip")
    source = np.asarray(arrays["pre_qpos"])
    post, planned = np.asarray(arrays["qpos"]), np.asarray(arrays["planned_qpos50"])
    if any(value.shape != (count, 36) or not np.isfinite(value).all() for value in (source, post, planned)):
        raise ValueError("every planned and executed source frame must remain present")
    if not np.array_equal(source[1:], post[:-1]):
        raise ValueError("source trace has a state gap or hidden phase change")
    if not np.allclose(np.linalg.norm(source[:, 3:7], axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("source quaternions must be finite unit rotations")
    if not np.array_equal(arrays["control_dt"], [0.02]) or not np.array_equal(
        arrays["command_time_s"], np.arange(count) * 0.02
    ):
        raise ValueError("source timing changed; no time warp is allowed")
    if not np.array_equal(arrays["post_control_time_s"], (np.arange(count) + 1) * 0.02):
        raise ValueError("source post-control sampling offset is missing or wrong")
    if arrays["physics_post_qpos"].shape != (count * 10, 36) or not np.array_equal(
        arrays["physics_post_qpos"][9::10], post
    ):
        raise ValueError("the full physics trace including final state must remain present")
    return np.array(source, dtype=float, copy=True)


def task_errors(source_model, target_model, source_qpos, native_qpos):
    source_data, target_data = mujoco.MjData(source_model), mujoco.MjData(target_model)
    positions, rotations = [], []
    for source_pose, target_pose in zip(source_qpos, native_qpos, strict=True):
        source_data.qpos[:] = source_pose
        target_data.qpos[:] = target_pose
        mujoco.mj_forward(source_model, source_data)
        mujoco.mj_forward(target_model, target_data)
        source_pos, source_quat = retarget._task_pose_arrays(
            source_model, source_data, retarget.DEFAULT_TASKS, source=True
        )
        target_pos, target_quat = retarget._task_pose_arrays(
            target_model, target_data, retarget.DEFAULT_TASKS, source=False
        )
        positions.append(np.linalg.norm(source_pos - target_pos, axis=1))
        rotations.append(
            (
                Rotation.from_quat(source_quat[:, [1, 2, 3, 0]]).inv()
                * Rotation.from_quat(target_quat[:, [1, 2, 3, 0]])
            ).magnitude()
        )
    positions, rotations = np.asarray(positions), np.asarray(rotations)
    return {
        task.name: {
            "position_max_m": float(positions[:, i].max()),
            "position_mean_m": float(positions[:, i].mean()),
            "orientation_max_rad": float(rotations[:, i].max()),
        }
        for i, task in enumerate(retarget.DEFAULT_TASKS)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-iterations", type=int, default=24)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    baseline, output = args.baseline_dir.resolve(strict=True), args.output_dir.resolve()
    config = OriginalTaskConfig(maximum_iterations=args.maximum_iterations)
    original = json.loads((baseline / "report.json").read_text())
    if original.get("kind") != "g1_sonic_original29_recorded_comparison_v1" or any(
        original.get(key) is not False for key in FLAGS
    ):
        raise ValueError("requires the unaccepted recorded original29 comparison")
    selected = [row for row in original["records"] if row["profile"] == PROFILE]
    if [row["name"] for row in selected] != list(CLIPS):
        raise ValueError("all three original clip requests must remain in order")
    inputs = dict(original["inputs"])

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        actual = file_sha256(path)
        if actual != inputs.get(str(path), actual) or (expected is not None and actual != expected):
            raise ValueError(f"recorded-source dependency changed: {path}")
        inputs[str(path)] = actual
        return path

    for path in list(inputs):
        bind(path)
    bind(baseline / "report.json")
    source = mujoco.MjModel.from_binary_path(
        str(bind(baseline / "original29.mjb", original["compiled_model_sha256"]))
    )
    target = mujoco.MjModel.from_xml_path(str(bind(root / retarget.DEFAULT_TARGET_MODEL)))
    hashes = {"source": compiled_model_sha256(source), "target": compiled_model_sha256(target)}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    bind(Path(__file__))
    source_layout, target_layout = retarget._model_layout(source), retarget._model_layout(target)
    indices = [source_layout.joint_names.index(name) for name in target_layout.joint_names]
    low, high = retarget.safe_target_joint_bounds(target, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    output.mkdir(parents=True, exist_ok=False)
    mujoco.mj_saveModel(target, filename=str(output / "native23.mjb"))
    bind(output / "native23.mjb", hashes["target"])
    dump(
        output / "started.json",
        {"inputs": dict(inputs), "config": asdict(config), "compiled_models": hashes, **FLAGS},
    )
    records = []
    for row in selected:
        record = {"name": row["name"], "source_profile": PROFILE, "output": None, "failure": None, **FLAGS}
        print(
            json.dumps(
                {"starting_clip": row["name"], "source_complete": row["details"]["complete_original_clip"]}
            ),
            flush=True,
        )
        try:
            with np.load(bind(row["trace_path"], row["trace_sha256"]), allow_pickle=False) as archive:
                trace = {key: archive[key].copy() for key in archive.files}
            source_qpos = completed_trace_source(row, trace)
            direct = np.clip(source_qpos[:, 7:][:, indices], low, high)
            seed = project_nearest_trajectory(
                direct,
                lower_bounds=low,
                upper_bounds=high,
                dt=0.02,
                max_velocity=4.975,
                max_acceleration=79.6,
                initial_velocity=np.clip((direct[1] - direct[0]) / 0.02, -4.975, 4.975),
            )
            problem = OriginalTaskPath(source, target, source_qpos, seed.projected_path, config=config)

            def progress(event):
                print(
                    json.dumps(
                        {
                            "clip": row["name"],
                            "iteration": event["iteration"],
                            "accepted": event["accepted"],
                            "merit": event.get("merit"),
                            "qp_status": event["qp"]["status"],
                        }
                    ),
                    flush=True,
                )

            variables, fitting = fit_original_task_path(problem, progress=progress)
            arrays = problem.serialize(variables)
            path = output / f"{row['name']}.native23.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            with np.load(path, allow_pickle=False) as archive:
                saved = {key: archive[key].copy() for key in archive.files}
            actual_variables = problem.serialized_variables(saved)
            audit = problem.audit(actual_variables)
            fk = audit_reference_kinematics(SimpleNamespace(module=mujoco, model=target), saved)
            if not audit["passed"] or not fk["position_fk_consistent"] or not fk["orientation_fk_consistent"]:
                raise ValueError("serialized native23 reference failed path or FK checks")
            actual_qpos = motion_qpos(target, saved)
            policy_errors = task_errors(source, target, source_qpos, actual_qpos)
            planner_errors = task_errors(source, target, trace["planned_qpos50"], actual_qpos)
            record.update(
                output=str(path),
                output_sha256=file_sha256(path),
                frames=len(source_qpos),
                frames_dropped=0,
                time_scale=1.0,
                controlled_joint_count=23,
                fit=fitting,
                temporal=audit,
                fk=fk,
                causal=audit_rebuilt_causal_terms(saved),
                policy_execution_relative_task_errors=policy_errors,
                nominal_planner_relative_task_errors=planner_errors,
                policy_execution_relative_root=root_path_repair_bounds(
                    source_qpos[:, :3], actual_qpos[:, :3], maximum_offset_m=0
                ),
                nominal_planner_relative_root=root_path_repair_bounds(
                    trace["planned_qpos50"][:, :3], actual_qpos[:, :3], maximum_offset_m=0
                ),
                geometry=reference_geometry(target, saved),
                both_feet_within_existing_5mm_policy_relative_screen=all(
                    policy_errors[foot]["position_max_m"] <= 0.005 for foot in ("left_foot", "right_foot")
                ),
            )
            bind(path)
        except (ValueError, RuntimeError, KeyError) as exc:
            record["failure"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        dump(output / f"{row['name']}.report.json", record)
        print(
            json.dumps({"clip": row["name"], "failure": record["failure"], "output": record["output"]}), flush=True
        )
    for path in list(inputs):
        bind(path)
    if hashes != {"source": compiled_model_sha256(source), "target": compiled_model_sha256(target)}:
        raise ValueError("retargeting modified its source or target model")
    dump(
        output / "report.json",
        {
            "kind": "g1_true23_original29_recorded_task_fit_diagnostic_v1",
            "inputs": inputs,
            "records": records,
            "compiled_models": hashes,
            "config": asdict(config),
            "all_requested_clips_attempted": len(records) == 3,
            "all_requested_clips_written": all(row["output"] and row["failure"] is None for row in records),
            "contact_force_qualification_performed": False,
            "controller_replay_or_training_performed": False,
            "pico_references_modified": False,
            "accepted_training_manifest_written": False,
            **FLAGS,
        },
    )


if __name__ == "__main__":
    main()
