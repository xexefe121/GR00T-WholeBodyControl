"""Compare explicit neutral-wrist proxies on complete original29 recordings.

Accept either the untouched recorded baseline or the separately labelled
hand-collision experiment. Preserve all three requests and reject incomplete
sources. This writes diagnostic references only, never a training manifest,
controller, network packet or deployment authorization.
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

from gear_sonic.scripts.record_g1_sonic_original29_baseline import CLIPS, dump
from gear_sonic.scripts.retarget_g1_true23_original29_trace import FLAGS, PROFILE, completed_trace_source
from gear_sonic.scripts.retarget_g1_true23_stance import audit_rebuilt_causal_terms
from gear_sonic.utils import g1_23dof_task_space_retarget as retarget
from gear_sonic.utils.g1_23dof_trajectory_projection import project_nearest_trajectory
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_hand_frame_tasks import (
    HAND_FRAME_CONVENTION,
    NeutralWristHandTaskPath,
    task_pose_errors,
)
from gear_sonic.utils.g1_true23_original_task_trajectory import OriginalTaskConfig, fit_original_task_path
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos, reference_geometry
from gear_sonic.utils.g1_true23_reference_lineage import root_path_repair_bounds
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics


def select_source_records(report):
    """Retain model-experiment identity; profile normalization is explicit only."""
    if any(report.get(key) is not False for key in FLAGS):
        raise ValueError("requires an explicitly unaccepted source recording")
    kind = report.get("kind")
    if kind == "g1_sonic_original29_recorded_comparison_v1":
        rows = [row for row in report["records"] if row.get("profile") == PROFILE]
        model_file, model_hash = "original29.mjb", report["compiled_model_sha256"]
    elif kind == "g1_sonic_original29_full_hand_collision_recorded_diagnostic_v1":
        variant = report["model_variant"]
        if (
            variant.get("kind") != "g1_sonic_original29_hand_collision_variant_v1"
            or variant.get("original_source_unchanged") is not True
            or variant.get("native23_model_modified") is not False
            or any(variant.get(key) is not False for key in FLAGS)
        ):
            raise ValueError("unverified or promoted hand-collision model experiment")
        rows = []
        for row in report["records"]:
            if row.get("details", {}).get("profile") != PROFILE or row.get("profile", PROFILE) != PROFILE:
                raise ValueError("collision source profile changed or is missing")
            rows.append({**row, "profile": PROFILE})
        model_file = "original29_with_hand_collisions.mjb"
        model_hash = variant["variant_compiled_sha256"]
    else:
        raise ValueError("unknown source recording kind; no automatic teacher or model substitution")
    if [row["name"] for row in rows] != list(CLIPS):
        raise ValueError("all three original clip requests must remain once and in order")
    return rows, model_file, model_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison-fit-dir", type=Path)
    parser.add_argument("--maximum-iterations", type=int, default=24)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source_dir, output = args.source_dir.resolve(strict=True), args.output_dir.resolve()
    config = OriginalTaskConfig(maximum_iterations=args.maximum_iterations)
    original = json.loads((source_dir / "report.json").read_text())
    selected, model_file, model_hash = select_source_records(original)
    inputs = dict(original["inputs"])

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
            raise ValueError(f"recorded-source dependency changed: {path}")
        inputs[str(path)] = digest
        return path

    for path in list(inputs):
        bind(path)
    bind(source_dir / "report.json")
    source = mujoco.MjModel.from_binary_path(str(bind(source_dir / model_file, model_hash)))
    target = mujoco.MjModel.from_xml_path(str(bind(root / retarget.DEFAULT_TARGET_MODEL)))
    hashes = {"source": compiled_model_sha256(source), "target": compiled_model_sha256(target)}
    if hashes["source"] != model_hash:
        raise ValueError("loaded source model identity changed")
    comparison = None
    if args.comparison_fit_dir:
        comparison = json.loads(bind(args.comparison_fit_dir / "report.json").read_text())
        if (
            comparison.get("kind") != "g1_true23_original29_recorded_task_fit_diagnostic_v1"
            or comparison.get("compiled_models") != hashes
            or any(comparison.get(key) is not False for key in FLAGS)
            or [row["name"] for row in comparison["records"]] != list(CLIPS)
        ):
            raise ValueError("comparison fit must retain the same source and target models and all requests")
        for path, digest in comparison["inputs"].items():
            bind(path, digest)
        for row in selected:
            if comparison["inputs"].get(row["trace_path"]) != row["trace_sha256"]:
                raise ValueError("comparison must use the exact same original trace, not another rollout")
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
    dump(output / "started.json", {"inputs": dict(inputs), "config": asdict(config), **FLAGS})
    records = []
    for row in selected:
        record = {
            "name": row["name"],
            "source_kind": original["kind"],
            "source_profile": PROFILE,
            "source_trace_sha256": row["trace_sha256"],
            "output": None,
            "failure": None,
            **FLAGS,
        }
        print(json.dumps({"starting_hand_frame_fit": row["name"]}), flush=True)
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
            problem = NeutralWristHandTaskPath(source, target, source_qpos, seed.projected_path, config=config)

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
            fitting["objective_task_point_convention"] = HAND_FRAME_CONVENTION
            arrays = problem.serialize(variables)
            path = output / f"{row['name']}.native23.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            with np.load(path, allow_pickle=False) as archive:
                saved = {key: archive[key].copy() for key in archive.files}
            actual_variables = problem.serialized_variables(saved)
            audit = problem.audit(actual_variables)
            fk = audit_reference_kinematics(SimpleNamespace(module=mujoco, model=target), saved)
            causal = audit_rebuilt_causal_terms(saved)
            if not audit["passed"] or not fk["position_fk_consistent"] or not fk["orientation_fk_consistent"]:
                raise ValueError("serialized native23 reference failed path or FK checks")
            actual_qpos = motion_qpos(target, saved)

            def both_conventions(poses):
                return {
                    "neutral_wrist_proxy": task_pose_errors(
                        source, target, source_qpos, poses, tasks=problem.tasks
                    ),
                    "legacy_numeric_offsets": task_pose_errors(
                        source, target, source_qpos, poses, tasks=retarget.DEFAULT_TASKS
                    ),
                }

            previous = None
            if comparison:
                old = next(entry for entry in comparison["records"] if entry["name"] == row["name"])
                if old["failure"] is not None or not old["output"]:
                    raise ValueError("comparison source has no complete old fit")
                with np.load(bind(old["output"], old["output_sha256"]), allow_pickle=False) as archive:
                    old_motion = {key: archive[key].copy() for key in archive.files}
                previous = {
                    "same_source_trace_and_models_verified": True,
                    "output_sha256": old["output_sha256"],
                    "task_errors": both_conventions(motion_qpos(target, old_motion)),
                    "geometry": reference_geometry(target, old_motion),
                }
            record.update(
                output=str(path),
                output_sha256=file_sha256(path),
                frames=len(source_qpos),
                frames_dropped=0,
                time_scale=1.0,
                controlled_joint_count=23,
                hand_frame_derivation=problem.hand_frame_evidence,
                fit=fitting,
                temporal=audit,
                fk=fk,
                causal=causal,
                causal_features_rebuilt_in_unchanged_legacy_wire_convention=True,
                new_hand_task_convention_not_promoted_to_policy_inputs=True,
                policy_relative_task_errors=both_conventions(actual_qpos),
                nominal_planner_relative_neutral_wrist_proxy_errors=task_pose_errors(
                    source, target, trace["planned_qpos50"], actual_qpos, tasks=problem.tasks
                ),
                policy_relative_root=root_path_repair_bounds(
                    source_qpos[:, :3], actual_qpos[:, :3], maximum_offset_m=0
                ),
                geometry=reference_geometry(target, saved),
                same_source_old_fit_comparison=previous,
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
        raise ValueError("hand-frame fitting modified its source or target model")
    dump(
        output / "report.json",
        {
            "kind": "g1_true23_neutral_wrist_hand_frame_fit_diagnostic_v1",
            "source_kind": original["kind"],
            "task_point_convention": HAND_FRAME_CONVENTION,
            "inputs": inputs,
            "records": records,
            "compiled_models": hashes,
            "config": asdict(config),
            "all_requested_clips_attempted": len(records) == 3,
            "all_requested_clips_written": all(row["output"] and row["failure"] is None for row in records),
            "contact_force_qualification_performed": False,
            "controller_replay_or_training_performed": False,
            "pico_references_or_causal_wire_convention_modified": False,
            "accepted_training_manifest_written": False,
            **FLAGS,
        },
    )


if __name__ == "__main__":
    main()
