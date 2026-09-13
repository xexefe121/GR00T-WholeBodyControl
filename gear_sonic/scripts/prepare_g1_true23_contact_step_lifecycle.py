"""Build and audit explicit stepping entry/return; preserve every source sample."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import measure_self_contacts
from gear_sonic.scripts.evaluate_g1_true23_generalist_baselines import write_json
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_task_space_retarget import safe_target_joint_bounds
from gear_sonic.utils.g1_23dof_trajectory_projection import (
    audit_trajectory_constraints,
    project_nearest_trajectory,
)
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_collision_sampling import interpolate_original_poses
from gear_sonic.utils.g1_true23_contact_step_transition import (
    PROFILE,
    solve_return_step_transition,
    solve_step_transition,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import FLAGS, MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_ramp_clearance_path import fit_generated_arm_paths
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos
from gear_sonic.utils.g1_true23_reference_support import audit_reference_support
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

REPORT_KIND = "g1_true23_contact_step_lifecycle_reference_diagnostic_v2"


def motion_from_poses(model, poses):
    data = mujoco.MjData(model)
    positions, quaternions = [], []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[1:].copy())
        quaternions.append(data.xquat[1:].copy())
    positions, quaternions = np.asarray(positions), np.asarray(quaternions)
    angular = np.zeros_like(positions)
    for body in range(24):
        rotations = Rotation.from_quat(quaternions[:, body, [1, 2, 3, 0]])
        angular[1:, body] = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / 0.02
    return dict(
        fps=np.array([50.0]),
        joint_pos=poses[:, 7:].copy(),
        joint_vel=np.gradient(poses[:, 7:], 0.02, axis=0),
        body_pos_w=positions,
        body_quat_w=quaternions,
        body_lin_vel_w=np.gradient(positions, 0.02, axis=0),
        body_ang_vel_w=angular,
    )


def audit_step_geometry(model, motion, timeline):
    poses = motion_qpos(model, motion)
    low, high = safe_target_joint_bounds(model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    bounds = audit_trajectory_constraints(
        poses[:, 7:],
        lower_bounds=low,
        upper_bounds=high,
        dt=0.02,
        max_velocity=5,
        max_acceleration=80,
        tolerance=2e-7,
    )
    contacts = measure_self_contacts(model, poses)
    feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    geoms = [
        int(g)
        for body in feet
        for g in np.flatnonzero(model.geom_bodyid == body)
        if model.geom_contype[g] or model.geom_conaffinity[g]
    ]
    data, phases = mujoco.MjData(model), []
    for phase in timeline["phases"]:
        if phase["name"] not in ("acquisition_ramp", "return_ramp"):
            continue
        span = poses[phase["frame_start"] - 1 : phase["frame_stop"] + 1]
        samples = interpolate_original_poses(span, np.arange(len(span)), np.arange(2 * len(span) - 1) / 2)
        gaps = []
        for pose in samples:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            gaps.append(data.geom_xpos[geoms, 2] - model.geom_size[geoms, 0])
        gaps = np.asarray(gaps)
        phases.append(
            dict(
                phase=phase["name"],
                samples_100hz=len(samples),
                minimum_sole_gap_m=float(gaps.min()),
                floor_penetration_sample_count=int(np.any(gaps < -2e-7, axis=1).sum()),
                self_contacts=measure_self_contacts(model, samples),
            )
        )
    passed = (
        bounds.passed
        and contacts["frames_with_robot_robot_penetration"] == 0
        and all(
            row["minimum_sole_gap_m"] >= -2e-7 and row["self_contacts"]["frames_with_robot_robot_penetration"] == 0
            for row in phases
        )
    )
    return dict(
        provisional_geometry_screen_passed=bool(passed),
        full_path_bounds=asdict(bounds),
        full_reference_self_contacts=contacts,
        generated_phases=phases,
        continuous_clearance_or_contact_dynamics_proven=False,
    )


def preserve_nongenerated(original, old_timeline, motion, timeline):
    old = {phase["name"]: phase for phase in old_timeline["phases"]}
    for phase in timeline["phases"]:
        if phase["name"] in ("acquisition_ramp", "return_ramp"):
            continue
        before = old[phase["name"]]
        if phase["requested_controls"] != before["requested_controls"]:
            raise ValueError("step transitions changed source or standing duration")
        for key in motion:
            if key != "fps":
                motion[key][phase["frame_start"] : phase["frame_stop"]] = original[key][
                    before["frame_start"] : before["frame_stop"]
                ]
    for key in motion:
        if key != "fps":
            motion[key][:11] = original[key][:11]


def project_step_arm_join_rates(poses, timeline, lower, upper):
    """Correct only generated arms at source joins, never the contact/leg plan."""
    desired = poses[:, 20:].copy()
    low, high = desired.copy(), desired.copy()
    editable = np.zeros(len(poses), dtype=bool)
    for phase in timeline["phases"]:
        if phase["name"] in ("acquisition_ramp", "return_ramp"):
            editable[phase["frame_start"] : phase["frame_stop"]] = True
    low[editable] = np.maximum(lower[13:], desired[editable] - 0.025)
    high[editable] = np.minimum(upper[13:], desired[editable] + 0.025)
    projected = project_nearest_trajectory(
        desired, lower_bounds=low, upper_bounds=high, dt=0.02, max_velocity=5, max_acceleration=80
    )
    result = poses.copy()
    result[editable, 20:] = projected.projected_path[editable]
    np.testing.assert_array_equal(result[:, :20], poses[:, :20])
    np.testing.assert_array_equal(result[~editable], poses[~editable])
    delta = float(np.abs(result[:, 20:] - desired).max())
    if delta > 0.025 + 2e-7:
        raise ValueError("generated arm join correction exceeded its explicit bound")
    return result, dict(
        maximum_arm_change_rad=delta,
        original_source_and_standing_exact=True,
        root_legs_and_waist_exact=True,
        geometry_or_dynamics_qualified=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-campaign", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists() or args.output_directory.is_symlink():
        raise FileExistsError("step reference preparation refuses overwrite")
    baseline_path = args.baseline_campaign.resolve(strict=True)
    baseline = json.loads(baseline_path.read_text())
    if baseline.get("kind") != "g1_true23_root_feedback_single_policy_lifecycle_campaign_v1":
        raise ValueError("step reference requires a normal full lifecycle baseline")
    old_timeline = baseline["timeline"]
    old_path = Path(old_timeline["timeline_path"]).resolve(strict=True)
    if sha256_file(old_path) != old_timeline["timeline_sha256"]:
        raise ValueError("baseline reference bytes changed")
    root = Path(__file__).resolve().parents[2]
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    identity = compiled_model_sha256(model)
    if any(row["result"]["compiled_model_sha256"] != identity for row in baseline["records"]):
        raise ValueError("step reference may not change physical model")
    paths = [baseline_path, old_path, args.asset_root / MODEL, root / PHYSICS]
    paths.extend(collect_local_source_closure(root, [Path(__file__)]).files)
    inputs = {str(Path(path).resolve()): sha256_file(path) for path in paths}
    with np.load(old_path, allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    poses = motion_qpos(model, original)
    old_phases = {phase["name"]: phase for phase in old_timeline["phases"]}
    plans, reports, target_arrays = {}, {}, {}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "started.json", dict(kind=REPORT_KIND, inputs=inputs, profile=PROFILE, **FLAGS))
    for name in ("acquisition_ramp", "return_ramp"):
        phase = old_phases[name]
        solve = solve_return_step_transition if name == "return_ramp" else solve_step_transition
        plan, targets, report = solve(
            model,
            poses[phase["frame_start"] - 1],
            poses[phase["frame_stop"]],
            progress=lambda row: print(json.dumps(dict(phase=name, **row)), flush=True),
        )
        plans[name], reports[name] = plan[1:-1], report
        for key in ("positions", "rotations", "com", "root_z", "support"):
            target_arrays[name + "_" + key] = np.asarray([target[key] for target in targets])
        target_arrays[name + "_qpos"] = plan
        write_json(output / (name + ".plan.json"), report)
    timeline = deepcopy(old_timeline)
    timeline.update(
        kind=REPORT_KIND,
        generated_ramp=PROFILE,
        generated_transition_profile=PROFILE,
        original_ramp_timing_preserved=False,
        generated_ramp_contact_or_force_feasibility_qualified=False,
        source_timing_scale=1.0,
        all_source_channels_samples_preserved=True,
        **FLAGS,
    )
    sections, phases, cursor = [poses[:11].copy()], [], 0
    for before in old_timeline["phases"]:
        name = before["name"]
        section = plans[name] if name in plans else poses[before["frame_start"] : before["frame_stop"]].copy()
        count = len(section)
        sections.append(section)
        phases.append(
            dict(
                name=name,
                control_start=cursor,
                control_stop=cursor + count,
                frame_start=11 + cursor,
                frame_stop=11 + cursor + count,
                requested_controls=count,
            )
        )
        cursor += count
    timeline.update(phases=phases, total_requested_controls=cursor, total_frames=cursor + 11)
    phase = next(row for row in phases if row["name"] == "source_motion")
    timeline.update(source_start_frame=phase["frame_start"], source_stop_frame_exclusive=phase["frame_stop"])
    planned = np.concatenate(sections)
    low, high = safe_target_joint_bounds(model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    rate_corrected, arm_join_repair = project_step_arm_join_rates(planned, timeline, low, high)
    repaired, arm_repair = fit_generated_arm_paths(
        model,
        planned,
        rate_corrected,
        timeline,
        HARDWARE_23_JOINT_NAMES,
        low,
        high,
        progress=lambda row: print(json.dumps(dict(arm_repair=row)), flush=True),
    )
    maximum_total_arm_change = float(np.abs(repaired[:, 20:] - planned[:, 20:]).max())
    if maximum_total_arm_change > 0.3 + 2e-7:
        raise ValueError("combined generated-arm repairs exceed the original 0.3 rad bound")
    motion = motion_from_poses(model, repaired)
    preserve_nongenerated(original, old_timeline, motion, timeline)
    geometry = audit_step_geometry(model, motion, timeline)
    profile = NativeModelActuationProfile.from_sim_config(root / PHYSICS)
    support = {}
    for phase in phases:
        if phase["name"] not in plans:
            continue
        span = slice(phase["frame_start"] - 1, phase["frame_stop"] + 1)
        subset = {key: value.copy() if key == "fps" else value[span].copy() for key, value in motion.items()}
        support[phase["name"]] = audit_reference_support(
            model, subset, profile.effort, reference_dynamics=True, record_solver_failures=True
        )
    reference_path = output / "lifecycle_reference.diagnostic.npz"
    with reference_path.open("xb") as stream:
        np.savez_compressed(stream, **motion)
    with (output / "contact_plan.diagnostic.npz").open("xb") as stream:
        np.savez_compressed(stream, **target_arrays)
    timeline.update(timeline_path=str(reference_path), timeline_sha256=sha256_file(reference_path))
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"step reference input changed: {path}")
    if compiled_model_sha256(model) != identity:
        raise ValueError("step generator changed physical model")
    report = dict(
        kind=REPORT_KIND,
        inputs=inputs,
        timeline=timeline,
        plans=reports,
        generated_arm_repair=arm_repair,
        generated_arm_join_repair=arm_join_repair,
        maximum_total_generated_arm_change_rad=maximum_total_arm_change,
        geometry=geometry,
        reference_support=support,
        compiled_physics_model_sha256=identity,
        original_source_and_standing_channels_preserved=True,
        source_timing_modified=False,
        generated_transition_timing_modified=True,
        training_reference_accepted=False,
        dynamic_feasibility_qualified=False,
        **FLAGS,
    )
    write_json(output / "report.json", report)
    print(
        json.dumps(
            dict(
                output=str(output),
                geometry_passed=geometry["provisional_geometry_screen_passed"],
                joint_bounds=geometry["full_path_bounds"],
                self_collision_frames=geometry["full_reference_self_contacts"][
                    "frames_with_robot_robot_penetration"
                ],
                phase_floor_gaps_m={
                    row["phase"]: row["minimum_sole_gap_m"] for row in geometry["generated_phases"]
                },
                support_failures={name: row["frames_with_no_support_solution"] for name, row in support.items()},
            )
        ),
        flush=True,
    )
    return 0 if geometry["provisional_geometry_screen_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
