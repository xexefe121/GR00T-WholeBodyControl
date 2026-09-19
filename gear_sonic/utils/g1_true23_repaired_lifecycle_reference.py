"""Explicit, independently verified generated-ramp reference replacement.

Source choreography and its six training arrays stay byte-exact. This replaces
only a bound generated reference, never the actor, measured state or controller.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import measure_self_contacts
from gear_sonic.scripts.refine_g1_true23_lifecycle_ramps import (
    ramp_candidates,
    ramp_geometry,
    shoulder_clearance_bump,
)
from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_trajectory_projection import audit_trajectory_constraints
from gear_sonic.utils.g1_true23_collision_sampling import interpolate_original_poses
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics


def verify_shoulder_pose_replacement(before, after, timeline, repair, joint_names):
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or before.ndim != 2 or before.shape[1] != 30 or not np.isfinite(after).all():
        raise ValueError("repaired lifecycle must preserve every finite native23 pose")
    expected = before.copy()
    columns = [7 + joint_names.index(side + "_shoulder_roll_joint") for side in ("left", "right")]
    phases = [p for p in timeline["phases"] if p["name"] in {"acquisition_ramp", "return_ramp"}]
    if [p["name"] for p in phases] != [p["phase"] for p in repair["phases"]]:
        raise ValueError("ramp repair must preserve both generated phases")
    for phase, proof in zip(phases, repair["phases"], strict=True):
        selected = proof["selected_attempt"]
        if type(selected) is not int or not 0 <= selected < len(proof["attempts"]):
            raise ValueError("ramp replacement needs a selected accepted bounded offset")
        row = proof["attempts"][selected]
        amplitude = (row["left_outward_rad"], row["right_outward_rad"])
        if row.get("accepted") is not True or amplitude not in ramp_candidates():
            raise ValueError("unrecognized or rejected generated shoulder offset")
        start, stop = phase["frame_start"], phase["frame_stop"]
        bump = shoulder_clearance_bump(stop - start)
        expected[start:stop, columns[0]] += amplitude[0] * bump
        expected[start:stop, columns[1]] -= amplitude[1] * bump
    if not np.array_equal(expected, after):
        raise ValueError(
            "repaired lifecycle changes source, standing, root, legs, timing or undeclared shoulder samples"
        )


def load_repaired_lifecycle_reference(path, baseline, timeline, model, source_motion_sha256):
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"generated reference repair input changed: {path}")
        inputs[str(path)] = digest
        return path

    report_path = bind(path)
    report = json.loads(report_path.read_text())
    identity = compiled_model_sha256(model)
    arm_path = report.get("kind") == "g1_true23_bounded_rate_and_arm_path_ramps_v3"
    rate_corrected = arm_path or report.get("kind") == "g1_true23_bounded_rate_and_shoulder_clearance_ramps_v2"
    if (
        report.get("kind")
        not in {
            "g1_true23_bounded_generated_shoulder_clearance_ramps_v1",
            "g1_true23_bounded_rate_and_shoulder_clearance_ramps_v2",
            "g1_true23_bounded_rate_and_arm_path_ramps_v3",
        }
        or report.get("accepted") is not True
        or report.get("all_six_source_arrays_bit_exact") is not True
        or report.get("compiled_physics_model_sha256") != identity
        or report.get("timeline", {}).get("source_motion_sha256") != source_motion_sha256
        or report.get("hardware_authorized") is not False
        or report.get("deployment_ready") is not False
    ):
        raise ValueError("requires accepted same-source same-model offline shoulder-ramp repair")
    originals = [
        (Path(p), h) for p, h in report["input_bindings"].items() if Path(p).name == "lifecycle_reference.npz"
    ]
    if len(originals) != 1:
        raise ValueError("ramp repair must bind one complete original generated timeline")
    original_path = bind(*originals[0])
    with np.load(original_path, allow_pickle=False) as archive:
        if set(archive.files) != set(baseline) or any(
            not np.array_equal(archive[k], baseline[k]) for k in baseline
        ):
            raise ValueError("ramp repair does not belong to this exact default lifecycle")
    reference_path = (report_path.parent / report["output"]["path"]).resolve(strict=True)
    if reference_path.parent != report_path.parent:
        raise ValueError("repaired reference must reside beside its report")
    bind(reference_path, report["output"]["sha256"])
    with np.load(reference_path, allow_pickle=False) as archive:
        motion = {key: archive[key].copy() for key in archive.files}
    if set(motion) != set(baseline) or any(
        motion[key].shape != baseline[key].shape or not np.isfinite(motion[key]).all() for key in motion
    ):
        raise ValueError("repaired reference cannot change complete array schema or omit samples")
    for key in (
        "phases",
        "source_frames",
        "total_frames",
        "total_requested_controls",
        "configured_standing_qpos",
        "returned_standing_qpos",
    ):
        if report["timeline"].get(key) != timeline.get(key):
            raise ValueError("ramp repair cannot change requested source or standing timeline")
    phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    span = slice(phase["frame_start"], phase["frame_stop"])
    if any(not np.array_equal(motion[k][span], baseline[k][span]) for k in motion if k != "fps"):
        raise ValueError("repaired lifecycle must preserve every source array byte-exact")
    before, after = motion_qpos(model, baseline), motion_qpos(model, motion)
    low, high = ik.safe_target_joint_bounds(model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    shoulder_baseline = before
    if rate_corrected:
        from gear_sonic.utils.g1_true23_ramp_rate_projection import project_generated_ramp_rates

        shoulder_baseline, rate_proof = project_generated_ramp_rates(before, timeline, low, high)
        if rate_proof != report.get("generated_ramp_rate_projection"):
            raise ValueError("generated rate projection differs from independent frozen-source recomputation")
    elif report.get("generated_ramp_rate_projection") is not None:
        raise ValueError("rate projection cannot be hidden in a shoulder-only reference")
    if arm_path:
        from gear_sonic.utils.g1_true23_ramp_clearance_path import (
            COM_CHANGE_LIMIT_M,
            MAXIMUM_ARM_CHANGE_RAD,
            PROFILE,
            verify_arm_pose_replacement,
        )

        proof = report.get("generated_arm_path", {})
        if (
            proof.get("kind") != PROFILE
            or proof.get("accepted") is not True
            or proof.get("arm_change_limit_rad") != MAXIMUM_ARM_CHANGE_RAD
            or proof.get("com_change_limit_m") != COM_CHANGE_LIMIT_M
        ):
            raise ValueError("requires declared bounded multi-joint generated-arm path")
        maximum = verify_arm_pose_replacement(
            shoulder_baseline, after, timeline, ik._model_layout(model).joint_names
        )
        if abs(maximum - proof.get("maximum_arm_change_rad", -1)) > 2e-7:
            raise ValueError("generated-arm displacement differs from independent pose audit")
    else:
        if report.get("generated_arm_path") is not None:
            raise ValueError("multi-joint arm path cannot be hidden in shoulder-only repair")
        verify_shoulder_pose_replacement(
            shoulder_baseline, after, timeline, report, ik._model_layout(model).joint_names
        )
    fk = audit_reference_kinematics(SimpleNamespace(model=model, module=mujoco), motion)
    bounds = audit_trajectory_constraints(
        after[:, 7:],
        lower_bounds=low,
        upper_bounds=high,
        dt=0.02,
        max_velocity=5.0,
        max_acceleration=80.0,
        tolerance=2e-7,
    )
    contacts = measure_self_contacts(model, after)
    if (
        not bounds.passed
        or not fk["position_fk_consistent"]
        or not fk["orientation_fk_consistent"]
        or contacts["frames_with_robot_robot_penetration"]
    ):
        raise ValueError("repaired lifecycle fails independent full-path geometry or rate audit")
    phase_audits = []
    for phase in timeline["phases"]:
        if phase["name"] not in {"acquisition_ramp", "return_ramp"}:
            continue
        start, stop = phase["frame_start"], phase["frame_stop"]
        times = np.arange(stop - start + 2, dtype=float)
        half_times = np.arange(2 * (len(times) - 1) + 1) / 2
        original = interpolate_original_poses(before[start - 1 : stop + 1], times, half_times)
        _, com = ramp_geometry(model, original)
        samples = interpolate_original_poses(after[start - 1 : stop + 1], times, half_times)
        geometry, _ = ramp_geometry(model, samples, com)
        if (
            geometry["self_contacts"]["frames_with_robot_robot_penetration"]
            or geometry["maximum_com_change_m"] > 0.001
        ):
            raise ValueError("repaired ramp fails independent 100-Hz collision or COM audit")
        phase_audits.append({"phase": phase["name"], **geometry})
    if compiled_model_sha256(model) != identity or any(sha256_file(Path(p)) != h for p, h in inputs.items()):
        raise ValueError("repaired lifecycle source or model changed during verification")
    updated = dict(timeline)
    updated.update(
        generated_ramp=report["timeline"]["generated_ramp"],
        generated_reference_repair={
            "report_path": str(report_path),
            "report_sha256": inputs[str(report_path)],
            "reference_sha256": inputs[str(reference_path)],
            "all_six_source_arrays_bit_exact": True,
            "full_control_grid_and_generated_100hz_ramp_collision_audits_passed": True,
            "phase_audits": phase_audits,
            "policy_or_robot_state_changed": False,
            "dynamic_feasibility_proven": False,
            "continuous_between_sample_clearance_proven": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        },
    )
    return motion, updated, inputs
