"""Explicit bounded contact-reference experiment, never original-motion parity.

The immutable raw bank remains the provenance anchor. A new SIM curriculum may
use separately bounded contact-conditioned poses, followed by one fixed offline
start calibration. Historical geometry claims are independently rechecked.
Remaining inverse-dynamics support failures are disclosed, not waived away.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_curriculum import MOTION_KEYS, array_digest
from gear_sonic.utils.g1_true23_registered_bank_reference import START_REGISTRATION_PROFILE
from gear_sonic.utils.g1_true23_start_registration import register_motion_start

CONTACT_PROFILE = "bounded_contact_conditioned_source_v1"
CONTACT_REPAIR_INDEX_KIND = "g1_true23_contact_conditioned_bank_lifecycle_repairs_v1"


def _motion(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _equal_motion(actual, expected):
    return set(actual) == set(expected) and all(
        actual[key].dtype == expected[key].dtype and np.array_equal(actual[key], expected[key]) for key in expected
    )


def load_contact_repair_member(row, clip):
    """Bind raw -> once registered -> conditioned -> final fixed registration."""
    inputs = {}

    def bind(path, expected):
        path = Path(path).resolve(strict=True)
        if sha256_file(path) != expected:
            raise ValueError("contact reference material bytes differ")
        inputs[str(path)] = expected
        return path

    if row.get("original_source_motion_sha256") != clip["source_sha256"]:
        raise ValueError("contact reference must retain the original audited member")
    original = _motion(bind(clip["source_path"], clip["source_sha256"]))
    if tuple(original.get("joint_names", [])) != tuple(HARDWARE_23_JOINT_NAMES):
        raise ValueError("contact reference requires exact physical native23 joint identity")
    before, _ = register_motion_start(original)
    before_path = bind(row["registered_input_path"], row["registered_input_sha256"])
    if not _equal_motion(_motion(before_path), before):
        raise ValueError("contact input is not the independently registered original bank member")
    report_path = bind(row["contact_report_path"], row["contact_report_sha256"])
    report = json.loads(report_path.read_text())
    solver = report.get("solver", {})
    if (
        report.get("kind") != "g1_true23_stance_foot_cleanup_experiment_v1"
        or report.get("provisional_geometry_screen_passed") is not True
        or report.get("training_reference_accepted") is not False
        or report.get("raw_original_fidelity_acceptance_inherited") is not False
        or report.get("hardware_authorized") is not False
        or report.get("deployment_ready") is not False
        or report.get("inputs", {}).get(str(before_path)) != inputs[str(before_path)]
        or solver.get("kind") != "g1_true23_joint_contact_clearance_whole_path_hypothesis_v1"
        or solver.get("selected_contact_feature") != "nearest_original_native_sole_sphere_v2"
        or solver.get("source_frames_removed") != 0
        or solver.get("source_retimed") is not False
        or solver.get("dynamic_feasibility_proven") is not False
    ):
        raise ValueError("contact reference must disclose its bounded, unqualified changed-source experiment")
    contact_path = bind(report_path.parent / report["output"]["path"], report["output"]["sha256"])
    if contact_path.parent != report_path.parent:
        raise ValueError("contact output must be a sibling of its bound report")
    conditioned = _motion(contact_path)
    if set(conditioned) != {*MOTION_KEYS, "fps"} or len(conditioned["joint_pos"]) != clip["length"]:
        raise ValueError("contact reference must retain every original source frame and channel")
    final, registration = register_motion_start(
        {**conditioned, "joint_names": np.asarray(HARDWARE_23_JOINT_NAMES)}
    )
    final_path = bind(row["registered_source_path"], row["source_motion_sha256"])
    if not _equal_motion(_motion(final_path), final):
        raise ValueError("final contact reference differs from independently recomputed fixed registration")
    support = report["reference_support"]
    result = {
        **row,
        "contact_report_path": str(report_path),
        "conditioned_source_path": str(contact_path),
        "conditioned_source_sha256": inputs[str(contact_path)],
        "original_source_arrays_sha256": array_digest(original),
        "registered_source_arrays_sha256": array_digest(final),
        "source_start_registration": registration,
        "remaining_conditional_force_support_failures": support["frames_with_no_support_solution"],
        "remaining_conditional_effort_failures": support["frames_with_solution_above_effort_limits"],
    }
    return result, inputs


def audit_contact_motion(model, original, conditioned):
    """Check actual serialized arrays, not optimizer status or saved pass flags."""
    from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import measure_self_contacts
    from gear_sonic.scripts.diagnose_g1_true23_stance_foot_cleanup import sole_gaps
    from gear_sonic.utils import g1_23dof_task_space_retarget as ik
    from gear_sonic.utils.g1_23dof_trajectory_projection import audit_trajectory_constraints
    from gear_sonic.utils.g1_true23_collision_sampling import interpolate_original_poses
    from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos
    from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics
    from gear_sonic.utils.g1_true23_stance_foot_cleanup import foot_frames

    if set(conditioned) != {*MOTION_KEYS, "fps"}:
        raise ValueError("contact output must contain only the complete physical motion channels")
    # Named raw artifacts may carry additional immutable provenance metadata.
    original = {key: original[key] for key in (*MOTION_KEYS, "fps")}
    for key in (*MOTION_KEYS, "fps"):
        if (
            conditioned[key].shape != original[key].shape
            or conditioned[key].dtype != original[key].dtype
            or not np.isfinite(conditioned[key]).all()
        ):
            raise ValueError("contact correction cannot change channels, shapes, dtypes or frame coverage")
    if (
        not np.array_equal(conditioned["fps"], np.array([50.0]))
        or not np.array_equal(conditioned["joint_pos"][:, 12], original["joint_pos"][:, 12])
        or not np.array_equal(conditioned["body_quat_w"][:, 0], original["body_quat_w"][:, 0])
        or not np.array_equal(conditioned["body_ang_vel_w"][:, 0], original["body_ang_vel_w"][:, 0])
    ):
        raise ValueError("contact correction cannot retime, change waist yaw or rotate the root")
    identity = compiled_model_sha256(model)
    before, after = (motion_qpos(model, motion) for motion in (original, conditioned))
    delta = after[:, :3] - before[:, :3]
    joint_delta = np.abs(after[:, 7:] - before[:, 7:])
    root_velocity = float(np.abs(np.diff(delta, axis=0) / 0.02).max())
    root_acceleration = float(np.abs(np.diff(delta, n=2, axis=0) / 0.02**2).max())
    if (
        np.abs(delta).max() > 0.03 + 2e-7
        or root_velocity > 0.75 + 2e-7
        or root_acceleration > 0.5 + 2e-7
        or np.any(joint_delta > np.r_[np.full(12, 0.2), 0.0, np.full(10, 0.3)] + 2e-7)
    ):
        raise ValueError("contact correction exceeds root/joint displacement or root temporal bounds")
    low, high = ik.safe_target_joint_bounds(model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    bounds = audit_trajectory_constraints(
        after[:, 7:],
        lower_bounds=low,
        upper_bounds=high,
        dt=0.02,
        max_velocity=5,
        max_acceleration=80,
        tolerance=2e-7,
    )
    fk = audit_reference_kinematics(SimpleNamespace(model=model, module=mujoco), conditioned)
    rebuilt = ik.build_mjlab_motion_arrays(
        model,
        SimpleNamespace(
            root_pos_w=after[:, :3], root_quat_wxyz=after[:, 3:7], joint_pos_hardware=after[:, 7:], fps=50.0
        ),
    )
    # Root angular velocity is an explicitly unchanged original input. Remaining
    # derivatives must describe the stored path within float32 FK roundoff.
    if (
        not bounds.passed
        or not fk["position_fk_consistent"]
        or not fk["orientation_fk_consistent"]
        or not np.array_equal(rebuilt["joint_vel"], conditioned["joint_vel"])
        or not np.allclose(rebuilt["body_lin_vel_w"], conditioned["body_lin_vel_w"], atol=1e-5, rtol=1e-5)
        or not np.allclose(
            rebuilt["body_ang_vel_w"][:, 1:], conditioned["body_ang_vel_w"][:, 1:], atol=1e-4, rtol=1e-5
        )
    ):
        raise ValueError("contact reference FK, stored derivatives or unchanged joint/rate limits fail")
    times, query = np.arange(len(before)), np.arange(2 * len(before) - 1) / 2
    samples = [interpolate_original_poses(p, times, query) for p in (before, after)]
    (old_pos, old_rot), (new_pos, new_rot) = [foot_frames(model, p) for p in samples]
    ankle_shift = float(np.linalg.norm(new_pos - old_pos, axis=2).max())
    ankle_rotation = float(
        Rotation.from_matrix((new_rot @ old_rot.transpose(0, 1, 3, 2)).reshape(-1, 3, 3)).magnitude().max()
    )
    tasks = [t for t in ik.DEFAULT_TASKS if t.name in ("left_hand", "right_hand", "head_proxy")]
    upper = []
    data = mujoco.MjData(model)
    for poses in samples:
        points = []
        for pose in poses:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            points.append(
                [
                    data.xpos[model.body(t.target_body).id]
                    + data.xmat[model.body(t.target_body).id].reshape(3, 3) @ np.asarray(t.target_point)
                    for t in tasks
                ]
            )
        upper.append(np.asarray(points))
    upper_shift = float(np.linalg.norm(upper[1] - upper[0], axis=2).max())
    original_gaps, gaps = [sole_gaps(model, poses) for poses in samples]
    contact_controls = ik.infer_foot_contacts(
        old_pos[::2], fps=50, height_tolerance_m=0.035, speed_tolerance_m_s=0.45
    )
    contacts = np.empty((len(query), 2), dtype=bool)
    contacts[::2] = contact_controls
    contacts[1::2] = contact_controls[:-1] & contact_controls[1:]
    selected_gaps = np.take_along_axis(gaps, original_gaps.argmin(axis=2)[..., None], axis=2)[..., 0][contacts]
    collision = measure_self_contacts(model, samples[1])
    if (
        ankle_shift > 0.02 + 2e-7
        or ankle_rotation > 0.15 + 2e-7
        or upper_shift > 0.05 + 2e-7
        or gaps.min() < -2e-7
        or not selected_gaps.size
        or selected_gaps.max() > 0.002 + 2e-7
        or collision["frames_with_robot_robot_penetration"] != 0
        or compiled_model_sha256(model) != identity
    ):
        raise ValueError("contact correction fails independent100Hz landmark, floor or self-collision bounds")
    return dict(
        independent_serialized_geometry_passed=True,
        frame_count=len(after),
        samples_100hz=len(query),
        compiled_physics_model_sha256=identity,
        maximum_root_translation_per_axis_m=float(np.abs(delta).max()),
        maximum_root_correction_velocity_m_s=root_velocity,
        maximum_root_correction_acceleration_m_s2=root_acceleration,
        maximum_ankle_translation_m=ankle_shift,
        maximum_ankle_rotation_rad=ankle_rotation,
        maximum_upper_landmark_translation_m=upper_shift,
        minimum_sole_gap_m=float(gaps.min()),
        maximum_selected_original_stance_point_gap_m=float(selected_gaps.max()),
        self_penetrating_samples=0,
        full_source_frames_and_timing_preserved=True,
        original_root_attitude_and_waist_yaw_bit_exact=True,
        stored_velocity_channels_checked=True,
        dynamics_or_contact_complementarity_proven=False,
    )


def derive_contact_source(source, material, model):
    before, first_registration = register_motion_start(source)
    report_path = Path(material["contact_report_path"]).resolve(strict=True)
    if sha256_file(report_path) != material["contact_report_sha256"]:
        raise ValueError("contact report changed before independent curriculum derivation")
    report = json.loads(report_path.read_text())
    if (
        report["output"]["sha256"] != material["conditioned_source_sha256"]
        or report["reference_support"]["frames_with_no_support_solution"]
        != material["remaining_conditional_force_support_failures"]
        or report["reference_support"]["frames_with_solution_above_effort_limits"]
        != material["remaining_conditional_effort_failures"]
    ):
        raise ValueError("contact continuation must disclose the bound source and all support failures")
    contact_path = Path(material["conditioned_source_path"]).resolve(strict=True)
    if sha256_file(contact_path) != material["conditioned_source_sha256"]:
        raise ValueError("contact source changed before independent curriculum derivation")
    conditioned = _motion(contact_path)
    geometry = audit_contact_motion(model, before, conditioned)
    if geometry["compiled_physics_model_sha256"] != report["compiled_physics_model_sha256"]:
        raise ValueError("contact support experiment belongs to different physical geometry")
    final, registration = register_motion_start(conditioned)
    proof = {
        "profile": CONTACT_PROFILE,
        **{
            key: material[key]
            for key in (
                "conditioned_source_path",
                "conditioned_source_sha256",
                "contact_report_path",
                "contact_report_sha256",
                "remaining_conditional_force_support_failures",
                "remaining_conditional_effort_failures",
            )
        },
        "original_source_arrays_sha256": array_digest(source),
        "original_registered_source_arrays_sha256": array_digest(before),
        "conditioned_source_arrays_sha256": array_digest(conditioned),
        "final_registered_source_arrays_sha256": array_digest(final),
        "original_source_start_registration": first_registration,
        "final_source_start_registration": registration,
        "geometry": geometry,
        "all_original_frames_retained_without_retiming": True,
        "final_fixed_calibration_after_contact_conditioning": True,
        "raw_original_fidelity_acceptance_inherited": False,
        "same_reference_benchmark_claimed": False,
        "measured_robot_state_used": False,
        "dynamic_feasibility_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    return final, proof


def validate_contact_transition(previous, current, *, explicitly_allowed):
    if (
        not explicitly_allowed
        or previous.get("source_reference_conditioning", "none") != "none"
        or current.get("source_reference_conditioning") != CONTACT_PROFILE
        or previous.get("source_start_registration") != START_REGISTRATION_PROFILE
        or current.get("source_start_registration") != START_REGISTRATION_PROFILE
        or previous["original_training_inputs"]["motion_sha256"]
        != current["original_training_inputs"]["motion_sha256"]
        or previous["derived_arrays_sha256"] == current["derived_arrays_sha256"]
    ):
        raise ValueError(
            "contact conditioning requires an explicit changed-reference continuation of the same raw bank"
        )
    old_rows, new_rows = previous["derived_spans"]["spans"], current["derived_spans"]["spans"]
    if [r["name"] for r in old_rows] != [r["name"] for r in new_rows]:
        raise ValueError("contact conditioning cannot add, omit or reorder source recordings")
    for old, new in zip(old_rows, new_rows, strict=True):
        if any(
            old.get(k) != new.get(k)
            for k in (
                "start",
                "length",
                "asset_id",
                "ownership",
                "original_source_frames",
                "source_arrays_sha256",
                "original_source_indices_requested",
                "every_original_source_frame_requested",
            )
        ) or any(
            old["timeline"].get(k) != new["timeline"].get(k)
            for k in (
                "phases",
                "source_frames",
                "total_frames",
                "total_requested_controls",
                "configured_standing_qpos",
            )
        ):
            raise ValueError("contact conditioning cannot crop, retime or change raw ownership/source/state")
        proof = new.get("source_reference_conditioning", {})
        if (
            proof.get("profile") != CONTACT_PROFILE
            or proof.get("original_source_arrays_sha256") != old["source_arrays_sha256"]
            or proof.get("original_registered_source_arrays_sha256") != old.get("registered_source_arrays_sha256")
            or proof.get("final_registered_source_arrays_sha256") != new.get("registered_source_arrays_sha256")
            or proof.get("final_source_start_registration") != new.get("source_start_registration")
            or proof.get("geometry", {}).get("independent_serialized_geometry_passed") is not True
            or proof.get("all_original_frames_retained_without_retiming") is not True
            or any(
                type(proof.get(k)) is not int or not 0 <= proof[k] <= new["original_source_frames"]
                for k in ("remaining_conditional_force_support_failures", "remaining_conditional_effort_failures")
            )
            or any(
                proof.get(k) is not False
                for k in (
                    "raw_original_fidelity_acceptance_inherited",
                    "same_reference_benchmark_claimed",
                    "measured_robot_state_used",
                    "dynamic_feasibility_proven",
                    "hardware_authorized",
                    "deployment_ready",
                )
            )
            or not new["timeline"].get("generated_reference_repair")
        ):
            raise ValueError(
                "contact transition must disclose all corrections, independent geometry and remaining failures"
            )
    return dict(
        kind="g1_true23_contact_conditioned_reference_bank_continuation_v1",
        previous_clip_count=len(old_rows),
        new_clip_count=len(new_rows),
        reference_arrays_changed=True,
        original_audited_source_arrays_preserved=True,
        previous_source_and_lifecycle_arrays_preserved=False,
        same_reference_resume_claimed=False,
        raw_original_fidelity_acceptance_inherited=False,
        every_bank_clip_requires_separate_complete_scheduled_cpu_comparison=True,
        remaining_conditional_force_support_failures={
            r["name"]: r["source_reference_conditioning"]["remaining_conditional_force_support_failures"]
            for r in new_rows
        },
        old_single_clip_scores_qualify_bank=False,
        held_out_generalization_verified=False,
        actor_critic_optimizer_and_counters_preserved=True,
        dynamic_feasibility_proven=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
