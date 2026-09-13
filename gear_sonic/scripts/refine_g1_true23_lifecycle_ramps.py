"""Bounded offline shoulder-clearance repair of generated entry/exit references.

Never changes requested source motion, standing poses, root trajectory or timing.
Default searches smooth endpoint-flat shoulder offsets only. An explicit rate
projection may also adjust generated-ramp joints by at most0.1rad before the
shoulder search, preserving the same full-path rate, collision and1mm COM gates.
Every full-path safe joint/rate bound and actual geometric contact is rechecked.
Reference geometry is not learned-policy tracking, contact-force or hardware proof.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import measure_self_contacts
from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_trajectory_projection import audit_trajectory_constraints
from gear_sonic.utils.g1_true23_collision_sampling import interpolate_original_poses
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def shoulder_clearance_bump(count):
    if type(count) is not int or not 25 <= count <= 200:
        raise ValueError("ramp bump requires 25..200 existing interior samples")
    time = np.arange(1, count + 1) / (count + 1)
    # Value, first and second derivative vanish at both external endpoints.
    return 64 * time**3 * (1 - time) ** 3


def ramp_candidates():
    levels = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
    return sorted(product(levels, repeat=2), key=lambda pair: (pair[0] ** 2 + pair[1] ** 2, pair))


def ramp_geometry(model, poses, baseline_com=None):
    data = mujoco.MjData(model)
    com = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        com.append(data.subtree_com[model.body("pelvis").id].copy())
    com = np.asarray(com)
    return {
        "self_contacts": measure_self_contacts(model, poses),
        "maximum_com_change_m": float(np.linalg.norm(com - baseline_com, axis=1).max())
        if baseline_com is not None
        else 0.0,
    }, com


def validate_ramp_input_model(report, identity, *, reference_only=False):
    """Accept explicitly identified pre-replay geometry, never fabricate an evaluation."""
    if reference_only:
        if (
            report.get("kind") != "g1_true23_registered_source_lifecycle_geometry_diagnostic_v1"
            or report.get("compiled_physics_model_sha256") != identity
            or report.get("policy_evaluation_performed") is not False
            or report.get("training_reference_accepted") is not False
        ):
            raise ValueError("reference-only ramp repair requires explicit same-model unevaluated geometry")
    elif (
        report.get("kind") != "g1_true23_root_feedback_single_policy_lifecycle_campaign_v1"
        or not report.get("records")
        or any(row["result"]["compiled_model_sha256"] != identity for row in report["records"])
    ):
        raise ValueError("ramp repair requires an intact evaluated lifecycle on the same physical model")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs_group = parser.add_mutually_exclusive_group(required=True)
    inputs_group.add_argument("--evaluation-directory", type=Path)
    inputs_group.add_argument("--reference-diagnostic-directory", type=Path)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--project-generated-ramp-rates", action="store_true")
    parser.add_argument("--optimize-generated-arm-path", action="store_true")
    args = parser.parse_args(argv)
    if args.optimize_generated_arm_path and not args.project_generated_ramp_rates:
        parser.error("multi-joint arm paths require explicit generated-ramp rate projection")
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("ramp repair refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and expected != digest:
            raise ValueError(f"ramp repair input changed: {path}")
        inputs[str(path)] = digest
        return path

    reference_only = args.reference_diagnostic_directory is not None
    directory = args.reference_diagnostic_directory if reference_only else args.evaluation_directory
    evaluation_path = bind(directory / "report.json")
    evaluation = json.loads(evaluation_path.read_text())
    timeline = evaluation["timeline"]
    reference_path = bind(timeline["timeline_path"], timeline["timeline_sha256"])
    bind(timeline["source_motion_path"], timeline["source_motion_sha256"])
    root = Path(__file__).resolve().parents[2]
    for path in (
        args.asset_root / MODEL,
        root / PHYSICS,
        *collect_local_source_closure(root, [Path(__file__)]).files,
    ):
        bind(path)
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    identity = compiled_model_sha256(model)
    validate_ramp_input_model(evaluation, identity, reference_only=reference_only)
    with np.load(reference_path, allow_pickle=False) as archive:
        motion = {key: archive[key].copy() for key in archive.files}
    original_pose = motion_qpos(model, motion)
    repaired = original_pose.copy()
    layout = ik._model_layout(model)
    low, high = ik.safe_target_joint_bounds(model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    rate_projection = None
    if args.project_generated_ramp_rates:
        from gear_sonic.utils.g1_true23_ramp_rate_projection import project_generated_ramp_rates

        repaired, rate_projection = project_generated_ramp_rates(original_pose, timeline, low, high)
    columns = [7 + layout.joint_names.index(side + "_shoulder_roll_joint") for side in ("left", "right")]
    source_phase = next(row for row in timeline["phases"] if row["name"] == "source_motion")
    source_span = slice(source_phase["frame_start"], source_phase["frame_stop"])
    if measure_self_contacts(model, original_pose[source_span])["frames_with_robot_robot_penetration"] != 0:
        raise ValueError("repair generated ramps only after full source controls are collision-clear")
    reports, passed = [], True
    arm_path = None
    if args.optimize_generated_arm_path:
        from gear_sonic.utils.g1_true23_ramp_clearance_path import fit_generated_arm_paths

        repaired, arm_path = fit_generated_arm_paths(
            model,
            original_pose,
            repaired,
            timeline,
            layout.joint_names,
            low,
            high,
            progress=lambda row: print(json.dumps(row), flush=True),
        )
        reports, passed = arm_path["phases"], arm_path["accepted"]
    for phase in () if arm_path is not None else timeline["phases"]:
        if phase["name"] not in {"acquisition_ramp", "return_ramp"}:
            continue
        start, stop = phase["frame_start"], phase["frame_stop"]
        times = np.arange(stop - start + 2, dtype=float)
        half_times = np.arange(2 * (len(times) - 1) + 1) / 2
        baseline = interpolate_original_poses(original_pose[start - 1 : stop + 1], times, half_times)
        before, com = ramp_geometry(model, baseline)
        bump = shoulder_clearance_bump(stop - start)
        attempts, selected = [], None
        for left, right in ramp_candidates():
            candidate = repaired.copy()
            candidate[start:stop, columns[0]] += left * bump
            candidate[start:stop, columns[1]] -= right * bump
            bounds = audit_trajectory_constraints(
                candidate[:, 7:],
                lower_bounds=low,
                upper_bounds=high,
                dt=0.02,
                max_velocity=5.0,
                max_acceleration=80.0,
                tolerance=2e-7,
            )
            row = {
                "left_outward_rad": left,
                "right_outward_rad": right,
                "path_bounds": asdict(bounds),
                "accepted": False,
            }
            if bounds.passed:
                samples = interpolate_original_poses(candidate[start - 1 : stop + 1], times, half_times)
                geometry, _ = ramp_geometry(model, samples, com)
                row["geometry"] = geometry
                row["accepted"] = (
                    geometry["self_contacts"]["frames_with_robot_robot_penetration"] == 0
                    and geometry["maximum_com_change_m"] <= 0.001
                )
            attempts.append(row)
            if row["accepted"]:
                selected = len(attempts) - 1
                repaired = candidate
                break
        passed = passed and selected is not None
        reports.append(
            {"phase": phase["name"], "before": before, "attempts": attempts, "selected_attempt": selected}
        )
        print(
            json.dumps(
                {
                    "phase": phase["name"],
                    "passed": selected is not None,
                    "attempts": len(attempts),
                    "selected_offsets_rad": [
                        attempts[selected]["left_outward_rad"],
                        attempts[selected]["right_outward_rad"],
                    ]
                    if selected is not None
                    else None,
                }
            ),
            flush=True,
        )
    contacts = measure_self_contacts(model, repaired)
    bounds = audit_trajectory_constraints(
        repaired[:, 7:],
        lower_bounds=low,
        upper_bounds=high,
        dt=0.02,
        max_velocity=5.0,
        max_acceleration=80.0,
        tolerance=2e-7,
    )
    passed = passed and contacts["frames_with_robot_robot_penetration"] == 0 and bounds.passed
    allowed = np.zeros_like(original_pose, dtype=bool)
    for phase in timeline["phases"]:
        if phase["name"] in {"acquisition_ramp", "return_ramp"}:
            selected_columns = slice(7, None) if rate_projection is not None else columns
            allowed[phase["frame_start"] : phase["frame_stop"], selected_columns] = True
    if not np.array_equal(repaired[~allowed], original_pose[~allowed]):
        raise ValueError("ramp repair changed source, standing, root, legs or unselected joints")
    report = {
        "kind": (
            "g1_true23_bounded_rate_and_arm_path_ramps_v3"
            if arm_path is not None
            else "g1_true23_bounded_rate_and_shoulder_clearance_ramps_v2"
            if rate_projection is not None
            else "g1_true23_bounded_generated_shoulder_clearance_ramps_v1"
        ),
        "accepted": bool(passed),
        "input_is_unevaluated_registered_reference": reference_only,
        "policy_evaluation_fabricated_or_assumed": False,
        "phases": reports,
        "full_path_bounds": asdict(bounds),
        "full_reference_self_contacts": contacts,
        "com_change_limit_m": 0.001,
        "all_unchanged_pose_components_bit_exact": True,
        "generated_ramp_100hz_geometry_checked": True,
        "source_motion_modified": False,
        "source_timing_modified": False,
        "compiled_physics_model_sha256": identity,
        "input_bindings": inputs,
        "continuous_between_sample_clearance_proven": False,
        "dynamic_feasibility_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    if rate_projection is not None:
        report["generated_ramp_rate_projection"] = rate_projection
    if arm_path is not None:
        report["generated_arm_path"] = arm_path
    if any(sha256_file(Path(p)) != h for p, h in inputs.items()) or compiled_model_sha256(model) != identity:
        raise ValueError("ramp repair input or physical model changed")
    output.mkdir(parents=True, exist_ok=False)
    if passed:
        result = deepcopy(motion)
        result["joint_pos"] = repaired[:, 7:].astype(motion["joint_pos"].dtype)
        data = mujoco.MjData(model)
        changed = np.flatnonzero(np.any(repaired != original_pose, axis=1))
        for frame in changed:
            data.qpos[:] = repaired[frame]
            mujoco.mj_forward(model, data)
            result["body_pos_w"][frame] = data.xpos[1:]
            result["body_quat_w"][frame] = data.xquat[1:]
        result["joint_vel"] = np.gradient(result["joint_pos"], 0.02, axis=0)
        result["body_lin_vel_w"] = np.gradient(result["body_pos_w"], 0.02, axis=0)
        for body in range(model.nbody - 1):
            rotations = Rotation.from_quat(result["body_quat_w"][:, body, [1, 2, 3, 0]])
            result["body_ang_vel_w"][1:, body] = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / 0.02
        for key in result:
            if key != "fps":
                result[key][source_span] = motion[key][source_span]
                np.testing.assert_array_equal(result[key][source_span], motion[key][source_span])
        serialized_pose = motion_qpos(model, result)
        serialized_bounds = audit_trajectory_constraints(
            serialized_pose[:, 7:],
            lower_bounds=low,
            upper_bounds=high,
            dt=0.02,
            max_velocity=5.0,
            max_acceleration=80.0,
            tolerance=2e-7,
        )
        serialized_contacts = measure_self_contacts(model, serialized_pose)
        if (
            not serialized_bounds.passed
            or serialized_contacts["frames_with_robot_robot_penetration"] != 0
            or not np.array_equal(serialized_pose[~allowed], original_pose[~allowed])
        ):
            raise ValueError("actual serialized ramp poses fail full bounds, collision or preservation audit")
        serialized_phases = []
        for phase in timeline["phases"]:
            if phase["name"] not in {"acquisition_ramp", "return_ramp"}:
                continue
            start, stop = phase["frame_start"], phase["frame_stop"]
            times = np.arange(stop - start + 2, dtype=float)
            half_times = np.arange(2 * (len(times) - 1) + 1) / 2
            baseline_samples = interpolate_original_poses(original_pose[start - 1 : stop + 1], times, half_times)
            _, baseline_com = ramp_geometry(model, baseline_samples)
            samples = interpolate_original_poses(serialized_pose[start - 1 : stop + 1], times, half_times)
            geometry, _ = ramp_geometry(model, samples, baseline_com)
            if (
                geometry["self_contacts"]["frames_with_robot_robot_penetration"]
                or geometry["maximum_com_change_m"] > 0.001
            ):
                raise ValueError("serialized 100-Hz ramp geometry or COM gate failed")
            serialized_phases.append({"phase": phase["name"], **geometry})
        report["serialized_full_path_bounds"] = asdict(serialized_bounds)
        report["serialized_full_reference_self_contacts"] = serialized_contacts
        report["serialized_100hz_ramp_geometry"] = serialized_phases
        destination = output / "lifecycle_reference.npz"
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **result)
        with np.load(destination, allow_pickle=False) as reloaded:
            if set(reloaded.files) != set(result) or any(
                not np.array_equal(reloaded[key], result[key]) for key in result
            ):
                raise ValueError("saved generated-reference arrays differ from verified values")
        report["output"] = {"path": destination.name, "sha256": sha256_file(destination)}
        report["all_six_source_arrays_bit_exact"] = True
        report["generated_velocity_channels_recomputed_source_channels_preserved"] = True
        updated_timeline = deepcopy(timeline)
        updated_timeline.update(
            timeline_path=str(destination),
            timeline_sha256=report["output"]["sha256"],
            generated_ramp=(
                "quintic_slerp_with_bounded_rate_and_multi_joint_arm_path_v3"
                if arm_path is not None
                else "quintic_slerp_with_bounded_rate_and_shoulder_clearance_v2"
                if rate_projection is not None
                else "quintic_slerp_with_bounded_endpoint_flat_shoulder_clearance_v1"
            ),
        )
        report["timeline"] = updated_timeline
    else:
        with (output / "ramps.rejected.npz").open("xb") as stream:
            np.savez_compressed(
                stream, diagnostic_only_not_accepted_motion=np.array([True]), qpos_native23=repaired
            )
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "output": str(output),
                "accepted": bool(passed),
                "remaining_penetrating_frames": contacts["frames_with_robot_robot_penetration"],
            }
        ),
        flush=True,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
