"""Compare recorded native23 dynamics with the complete original29 choreography.

The original source is resampled using its already declared duration map, never
amplitude-reduced or aligned to the measured robot. This separates retargeting
distortion from policy error. Source FK is not an original29 dynamics baseline.
No simulator stepping, robot transport, or qualification-rule changes occur.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_curriculum import MOTION_KEYS
from gear_sonic.utils.g1_true23_generalist_retarget import (
    AdaptationLimits,
    _resample,
    _source_task_poses,
    validate_named_motion,
)
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


def source_phase_poses(trace, timeline, result):
    phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    start, stop = phase["control_start"], phase["control_stop"]
    count = timeline["total_requested_controls"]
    poses = np.asarray(trace["qpos"])
    if (
        result["completed_controls"] != count
        or result["requested_controls"] != count
        or result["failure"] is not None
        or poses.shape != (count + 1, 30)
        or not np.isfinite(poses).all()
        or not 0 <= start < stop <= count
        or phase["frame_start"] != start + 11
        or phase["frame_stop"] != stop + 11
    ):
        raise ValueError("original-source comparison requires the complete unshifted lifecycle")
    return poses[start + 1 : stop + 1].copy(), phase


def point_statistics(actual, reference, actual_root, reference_root, names):
    actual, reference = np.asarray(actual), np.asarray(reference)
    actual_root, reference_root = np.asarray(actual_root), np.asarray(reference_root)
    if (
        actual.ndim != 3
        or actual.shape != reference.shape
        or actual.shape[1:] != (len(names), 3)
        or len(actual) == 0
        or actual_root.shape != (len(actual), 3)
        or reference_root.shape != actual_root.shape
        or any(not np.isfinite(a).all() for a in (actual, reference, actual_root, reference_root))
    ):
        raise ValueError("task statistics require complete finite matching point/root arrays")
    errors = {
        "world": np.linalg.norm(actual - reference, axis=-1),
        "pelvis_relative": np.linalg.norm(
            (actual - actual_root[:, None]) - (reference - reference_root[:, None]), axis=-1
        ),
    }
    return {
        frame: {
            name: {"mean_m": float(v.mean()), "p95_m": float(np.percentile(v, 95)), "max_m": float(v.max())}
            for name, v in zip(names, values.T)
        }
        for frame, values in errors.items()
    }


def native_points(model, poses):
    data = mujoco.MjData(model)
    points = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_fwdPosition(model, data)
        points.append(ik._task_pose_arrays(model, data, ik.DEFAULT_TASKS, source=False)[0])
    return np.asarray(points)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation-directory", "retarget-directory", "source-model", "target-model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--retarget-model", type=Path, help="Historical retarget XML if its bytes differ from the evaluation XML"
    )
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite original-source evidence")
    directory = args.retarget_directory.resolve(strict=True)
    paths = {
        "evaluation": args.evaluation_directory.resolve(strict=True) / "report.json",
        "retarget": directory / "report.json",
        "adapted": directory / "adapted.true23.npz",
        "named_source": directory / "planned.named29.npz",
        "source_model": args.source_model.resolve(strict=True),
        "target_model": args.target_model.resolve(strict=True),
        "retarget_model": (args.retarget_model or args.target_model).resolve(strict=True),
    }
    bindings = {str(p): sha256_file(p) for p in paths.values()}
    for path in collect_local_source_closure(
        Path(__file__).resolve().parents[2], [Path(__file__).resolve()]
    ).files:
        bindings[str(path)] = sha256_file(path)
    evaluation = json.loads(paths["evaluation"].read_text())
    fit = json.loads(paths["retarget"].read_text())
    if (
        evaluation.get("kind") != "g1_true23_root_feedback_single_policy_lifecycle_campaign_v1"
        or evaluation.get("no_postinitial_robot_pose_rewrites") is not True
        or evaluation.get("no_fallback_controller") is not True
        or [row.get("case") for row in evaluation.get("records", [])]
        != ["nominal", "standing_push_x", "standing_push_y"]
    ):
        raise ValueError("original-source measurement requires all three no-reset CPU cases")
    if (
        fit.get("accepted") is not True
        or fit.get("source_field") != "planned_qpos50"
        or fit.get("recorded_policy_pose_used_as_choreography") is not False
        or fit.get("adapted_motion_sha256") != bindings[str(paths["adapted"])]
        or fit.get("named_source_sha256") != bindings[str(paths["named_source"])]
        or any(
            fit["input_bindings"].get(str(paths[k])) != bindings[str(paths[k])]
            for k in ("source_model", "retarget_model")
        )
    ):
        raise ValueError("requires the exact accepted original-planner retarget lineage")
    source_model = (
        mujoco.MjModel.from_binary_path(str(paths["source_model"]))
        if paths["source_model"].suffix == ".mjb"
        else mujoco.MjModel.from_xml_path(str(paths["source_model"]))
    )
    target_model = mujoco.MjModel.from_xml_path(str(paths["target_model"]))
    historical_target = mujoco.MjModel.from_xml_path(str(paths["retarget_model"]))
    if compiled_model_sha256(historical_target) != compiled_model_sha256(target_model):
        raise ValueError("retarget and evaluation compiled FK models differ")
    if (source_model.nq, target_model.nq, target_model.nu) != (36, 30, 23) or fit["compiled_models"] != {
        "source": compiled_model_sha256(source_model),
        "target": compiled_model_sha256(target_model),
    }:
        raise ValueError("original-source FK model identity differs")
    with np.load(paths["named_source"], allow_pickle=False) as archive:
        original = validate_named_motion(dict(archive), source_model, allow_source_limit_excess=True)
    with np.load(paths["adapted"], allow_pickle=False) as archive:
        adapted = dict(archive)
    attempt = fit["attempts"][fit["selected_attempt"]]
    if attempt["accepted"] is not True or attempt["failures"]:
        raise ValueError("selected retarget attempt did not pass")
    sampled, times, duration = _resample(
        original, attempt["requested_duration_scale"], AdaptationLimits(**fit["limits"])
    )
    if not np.array_equal(times, adapted["source_time_map_s"]):
        raise ValueError("original-source timestamp map changed")
    for key in ("joint_pos", "root_pos_w", "root_quat_wxyz"):
        if not np.array_equal(sampled[key], adapted["source_" + key + "_resampled"]):
            raise ValueError("stored original source was changed or amplitude-reduced")
    original_points, _ = _source_task_poses(source_model, sampled)
    np.testing.assert_allclose(original_points, adapted["source_task_pos_w"], atol=1e-9, rtol=0)
    reference_path = Path(evaluation["timeline"]["timeline_path"])
    if sha256_file(reference_path) != evaluation["timeline"]["timeline_sha256"]:
        raise ValueError("lifecycle reference hash differs")
    bindings[str(reference_path)] = sha256_file(reference_path)
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = {key: archive[key].copy() for key in MOTION_KEYS}
    names = [task.name for task in ik.DEFAULT_TASKS]
    adapted_poses = np.column_stack(
        (adapted["body_pos_w"][:, 0], adapted["body_quat_w"][:, 0], adapted["joint_pos"])
    )
    adapted_points = native_points(target_model, adapted_poses)
    rows = []
    for row in evaluation["records"]:
        if (
            row["result"]["model_sha256"] != bindings[str(paths["target_model"])]
            or row["result"]["state_pose_writes_after_reset"] != 0
            or row["result"]["history_resets_during_motion"] != 0
        ):
            raise ValueError("recorded dynamics used another model or reset during motion")
        trace = Path(row["trace_path"])
        if sha256_file(trace) != row["trace_sha256"]:
            raise ValueError("measured native23 trace changed")
        bindings[str(trace)] = row["trace_sha256"]
        with np.load(trace, allow_pickle=False) as archive:
            poses, phase = source_phase_poses(archive, evaluation["timeline"], row["result"])
        for key in MOTION_KEYS:
            if not np.array_equal(reference[key][phase["frame_start"] : phase["frame_stop"]], adapted[key]):
                raise ValueError("evaluated source phase differs from the complete adapted motion")
        if len(poses) != len(times):
            raise ValueError("original-source comparison cannot drop or duplicate source controls")
        actual = native_points(target_model, poses)
        rows.append(
            {
                "case": row["case"],
                "policy_source": row["policy_identity"]["source"],
                "executed_to_original29": point_statistics(
                    actual, original_points, poses[:, :3], sampled["root_pos_w"], names
                ),
                "executed_to_adapted23": point_statistics(
                    actual, adapted_points, poses[:, :3], adapted_poses[:, :3], names
                ),
            }
        )
    if any(sha256_file(Path(path)) != expected for path, expected in bindings.items()):
        raise ValueError("original-source comparison input changed during measurement")
    result = {
        "kind": "g1_true23_full_original_planner_source_tracking_v1",
        "inputs": bindings,
        "cases": rows,
        "task_convention": [asdict(task) for task in ik.DEFAULT_TASKS],
        "source_original_frames": len(original["joint_pos"]),
        "retarget_and_evaluation_compiled_fk_models_identical": True,
        "compared_source_controls": len(times),
        "actual_duration_scale": duration,
        "retarget_excursion_scale": attempt["requested_excursion_scale"],
        "original29_comparison_source_amplitude_reduced": False,
        "adapted23_reference_to_original29": point_statistics(
            adapted_points, original_points, adapted_poses[:, :3], sampled["root_pos_w"], names
        ),
        "reference_or_measured_state_reanchored": False,
        "lag_alignment_used": False,
        "original29_closed_loop_dynamics_compared": False,
        "original_speed_policy_parity_proven": False,
        "pose_playback_or_simulator_stepping": False,
        "acceptance_rules_changed": False,
        "contact_or_generalization_qualified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "cases": len(rows), "source_controls": len(times)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
