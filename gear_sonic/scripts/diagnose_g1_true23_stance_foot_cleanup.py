"""SIM-only contact-conditioned source experiment with full independent audits.

Raw original-fidelity qualification is not inherited. This is an explicitly
changed reference candidate, not an accepted bank member or deployable policy.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import measure_self_contacts
from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_trajectory_projection import audit_trajectory_constraints
from gear_sonic.utils.g1_true23_collision_sampling import interpolate_original_poses
from gear_sonic.utils.g1_true23_generalist_benchmark import FLAGS, MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_source_closure import collect_local_source_closure
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos
from gear_sonic.utils.g1_true23_reference_support import audit_reference_support
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics
from gear_sonic.utils.g1_true23_stance_foot_cleanup import clean_stance_feet
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def sole_gaps(model, poses):
    data = mujoco.MjData(model)
    ids = [
        int(g)
        for side in ("left", "right")
        for g in np.flatnonzero(model.geom_bodyid == model.body(side + "_ankle_roll_link").id)
        if model.geom_contype[g] or model.geom_conaffinity[g]
    ]
    rows = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        rows.append(data.geom_xpos[ids, 2] - model.geom_size[ids, 0])
    return np.asarray(rows).reshape(len(poses), 2, 4)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--allow-bounded-root-translation", action="store_true")
    parser.add_argument("--whole-path-contact-conditioning", action="store_true")
    parser.add_argument(
        "--source-initial-joint-velocity-diagnostic",
        action="store_true",
        help="Explicit moving-source boundary from first two poses; does not qualify standing acquisition",
    )
    parser.add_argument(
        "--record-support-solver-failures-diagnostic",
        action="store_true",
        help="Retain numerically indeterminate support rows as failures, never as feasibility certificates",
    )
    args = parser.parse_args(argv)
    if args.whole_path_contact_conditioning and args.allow_bounded_root_translation:
        parser.error("whole-path conditioning is its own explicit coupled-root method")
    if args.source_initial_joint_velocity_diagnostic and not args.whole_path_contact_conditioning:
        parser.error("moving-source boundary requires explicit whole-path contact conditioning")
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("contact cleanup refuses overwrite")
    root = Path(__file__).resolve().parents[2]
    inputs = {
        str(p.resolve(strict=True)): sha256_file(p)
        for p in (
            args.motion,
            args.asset_root / MODEL,
            root / PHYSICS,
            *collect_local_source_closure(root, [Path(__file__)]).files,
        )
    }
    with np.load(args.motion, allow_pickle=False) as archive:
        motion = {
            key: archive[key].copy()
            for key in (
                "fps",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            )
        }
    _, model, _ = prepare_true23_model(args.asset_root / MODEL, root / PHYSICS)
    identity = compiled_model_sha256(model)
    poses = motion_qpos(model, motion)
    initial_joint_velocity = (
        (poses[1, 7:] - poses[0, 7:]) / 0.02 if args.source_initial_joint_velocity_diagnostic else None
    )
    if args.whole_path_contact_conditioning:
        from gear_sonic.utils.g1_true23_contact_conditioned_path import fit_contact_conditioned_path

        fitted, contacts, solver = fit_contact_conditioned_path(
            model,
            poses,
            progress=lambda row: print(json.dumps(row), flush=True),
            initial_joint_velocity=initial_joint_velocity,
        )
    else:
        fitted, contacts, solver = clean_stance_feet(
            model,
            poses,
            allow_bounded_root_translation=args.allow_bounded_root_translation,
            progress=lambda row: print(json.dumps(row), flush=True),
        )
    # Fit FK to the exact stored float32 joints, not to higher-precision values
    # that disappear on serialization. Preserve all unaffected source channels.
    joints = fitted[:, 7:].astype(motion["joint_pos"].dtype)
    result = ik.build_mjlab_motion_arrays(
        model,
        SimpleNamespace(
            root_pos_w=fitted[:, :3],
            root_quat_wxyz=poses[:, 3:7],
            joint_pos_hardware=joints.astype(float),
            fps=50.0,
        ),
    )
    leg_bodies = model.jnt_bodyid[1:13] - 1
    unchanged_bodies = np.setdiff1d(np.arange(model.nbody - 1), leg_bodies)
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        if not args.whole_path_contact_conditioning and (
            not args.allow_bounded_root_translation or key in ("body_quat_w", "body_ang_vel_w")
        ):
            result[key][:, unchanged_bodies] = motion[key][:, unchanged_bodies]
    if not args.whole_path_contact_conditioning:
        result["joint_pos"][:, 12:] = motion["joint_pos"][:, 12:]
        result["joint_vel"][:, 12:] = motion["joint_vel"][:, 12:]
    result["body_quat_w"][:, 0] = motion["body_quat_w"][:, 0]
    result["body_ang_vel_w"][:, 0] = motion["body_ang_vel_w"][:, 0]
    serialized = motion_qpos(model, result)
    fk = audit_reference_kinematics(SimpleNamespace(model=model, module=mujoco), result)
    low, high = ik.safe_target_joint_bounds(model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    bounds = audit_trajectory_constraints(
        serialized[:, 7:],
        lower_bounds=low,
        upper_bounds=high,
        dt=0.02,
        max_velocity=5,
        max_acceleration=80,
        initial_velocity=0.0 if initial_joint_velocity is None else initial_joint_velocity,
        tolerance=2e-7,
    )
    gaps = sole_gaps(model, serialized)
    half = interpolate_original_poses(
        serialized, np.arange(len(serialized)), np.arange(2 * len(serialized) - 1) / 2
    )
    half_gaps = sole_gaps(model, half)
    control_collision, half_collision = [measure_self_contacts(model, p) for p in (serialized, half)]
    from scipy.spatial.transform import Rotation

    from gear_sonic.utils.g1_true23_stance_foot_cleanup import foot_frames

    original_half = interpolate_original_poses(poses, np.arange(len(poses)), np.arange(2 * len(poses) - 1) / 2)
    original_feet, original_foot_rotations = foot_frames(model, original_half)
    conditioned_feet, conditioned_rotations = foot_frames(model, half)
    shift = float(np.linalg.norm(conditioned_feet - original_feet, axis=2).max())
    rotation_shift = float(
        Rotation.from_matrix(
            (conditioned_rotations @ original_foot_rotations.transpose(0, 1, 3, 2)).reshape(-1, 3, 3)
        )
        .magnitude()
        .max()
    )
    correction_bounds_passed = bool(shift <= 0.02 + 1e-6 and rotation_shift <= 0.15 + 1e-6)
    geometry = dict(
        fk=fk,
        bounds=asdict(bounds),
        control_self_contacts=control_collision,
        half_control_self_contacts=half_collision,
        minimum_sole_gap_m=float(gaps.min()),
        minimum_100hz_sole_gap_m=float(half_gaps.min()),
        maximum_inferred_stance_sole_gap_m=float(gaps[contacts].max()) if contacts.any() else None,
        maximum_inferred_stance_absolute_sole_gap_m=float(np.abs(gaps[contacts]).max())
        if contacts.any()
        else None,
        maximum_inferred_stance_minimum_sole_gap_m=float(gaps.min(axis=2)[contacts].max())
        if contacts.any()
        else None,
        serialized_correction_bounds_passed=correction_bounds_passed,
        maximum_serialized100hz_ankle_shift_m=shift,
        maximum_serialized100hz_ankle_rotation_change_rad=rotation_shift,
        no_contact_complementarity_claimed=True,
    )
    print(
        json.dumps(
            {
                "stage": "serialized_geometry",
                "bounds": bounds.passed,
                "minimum_100hz_sole_gap_m": geometry["minimum_100hz_sole_gap_m"],
                "self_penetration_frames_100hz": half_collision["frames_with_robot_robot_penetration"],
            }
        ),
        flush=True,
    )
    if compiled_model_sha256(model) != identity or any(sha256_file(Path(p)) != h for p, h in inputs.items()):
        raise ValueError("contact cleanup source code, inputs or physical model changed")
    # Preserve the completed reference/geometry stage before a separate support
    # LP can fail. This file explicitly cannot be read as a completed support
    # audit or an accepted training/deployment reference.
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "contact_conditioned.diagnostic.npz"
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **result)
    with (output / "contact_hypothesis.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            inferred_contact_flags=contacts,
            original_qpos=poses,
            conditioned_qpos=serialized,
            sole_gaps=gaps,
        )
    with (output / "geometry_only.json").open("x") as stream:
        json.dump(
            dict(
                kind="g1_true23_contact_cleanup_geometry_stage_v1",
                inputs=inputs,
                output=dict(path=destination.name, sha256=sha256_file(destination)),
                hypothesis=dict(
                    path="contact_hypothesis.npz", sha256=sha256_file(output / "contact_hypothesis.npz")
                ),
                compiled_physics_model_sha256=identity,
                solver=solver,
                geometry=geometry,
                support_audit_complete=False,
                training_reference_accepted=False,
                standing_acquisition_qualified=False,
                **FLAGS,
            ),
            stream,
            indent=2,
            allow_nan=False,
        )
    profile = NativeModelActuationProfile.from_sim_config(root / PHYSICS)
    support = audit_reference_support(
        model,
        result,
        profile.effort,
        reference_dynamics=True,
        record_solver_failures=args.record_support_solver_failures_diagnostic,
    )
    if compiled_model_sha256(model) != identity or any(sha256_file(Path(p)) != h for p, h in inputs.items()):
        raise ValueError("contact cleanup source code, inputs or physical model changed during support audit")
    report = dict(
        kind=(
            "g1_true23_moving_source_contact_cleanup_experiment_v2"
            if args.source_initial_joint_velocity_diagnostic
            else "g1_true23_stance_foot_cleanup_experiment_v1"
        ),
        source_initial_velocity_diagnostic=args.source_initial_joint_velocity_diagnostic,
        support_solver_failure_recording_diagnostic=args.record_support_solver_failures_diagnostic,
        declared_initial_joint_velocity_rad_s=(
            None if initial_joint_velocity is None else initial_joint_velocity.tolist()
        ),
        standing_acquisition_qualified=False,
        inputs=inputs,
        output=dict(path=destination.name, sha256=sha256_file(destination)),
        solver=solver,
        geometry=geometry,
        reference_support=support,
        compiled_physics_model_sha256=identity,
        training_reference_accepted=False,
        provisional_geometry_screen_passed=bool(
            correction_bounds_passed
            and bounds.passed
            and fk["position_fk_consistent"]
            and fk["orientation_fk_consistent"]
            and not half_collision["frames_with_robot_robot_penetration"]
            and half_gaps.min() >= -2e-7
            and contacts.any()
            and gaps.min(axis=2)[contacts].max() <= 0.002
        ),
        raw_original_fidelity_acceptance_inherited=False,
        **FLAGS,
    )
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "output": str(output),
                "frames": len(serialized),
                "support_failures": support["frames_with_no_support_solution"],
                "effort_failures": support["frames_with_solution_above_effort_limits"],
                "maximum_ankle_shift_m": solver["maximum_ankle_translation_m"],
                "training_reference_accepted": False,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
