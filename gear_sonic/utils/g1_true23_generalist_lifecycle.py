"""Unqualified full-source standing/motion/standing references for one actor.

Generated ramps are kinematic reference requests, not validated contact plans.
Only the separate benchmark integrates the robot; this builder never actuates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from gear_sonic.utils.g1_true23_generalist_benchmark import FLAGS, summarize_tracking
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion

PREHISTORY_FRAMES = 11
RETURN_TARGETS = ("configured_origin", "planned_endpoint")
PHASE_CONTROLS = (
    ("initial_standing", 250),
    ("acquisition_ramp", 100),
    ("source_motion", None),
    ("return_ramp", 100),
    ("returned_standing", 250),
    ("standing_proof_margin", 50),
)


def _blend_poses(start, finish, count):
    """Interior quintic phase ramp; exact source endpoint remains in source block."""
    alpha = np.arange(1, count + 1, dtype=np.float64) / (count + 1)
    blend = 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5
    poses = start[None] + blend[:, None] * (finish - start)[None]
    rotations = Rotation.from_quat(np.stack((start[3:7], finish[3:7]))[:, [1, 2, 3, 0]])
    poses[:, 3:7] = Slerp([0.0, 1.0], rotations)(blend).as_quat()[:, [3, 0, 1, 2]]
    return poses


def _planned_endpoint_standing(standing, endpoint):
    """Upright posture at the fixed planned endpoint, never at measured robot XY."""
    result = standing.copy()
    result[:2] = endpoint[:2]
    initial = Rotation.from_quat(standing[[4, 5, 6, 3]])
    final = Rotation.from_quat(endpoint[[4, 5, 6, 3]])
    initial_heading, final_heading = initial.as_matrix()[:2, 0], final.as_matrix()[:2, 0]
    if min(np.linalg.norm(initial_heading), np.linalg.norm(final_heading)) < 1e-6:
        raise ValueError("planned endpoint cannot define an upright standing heading")
    initial_yaw = np.arctan2(initial_heading[1], initial_heading[0])
    final_yaw = np.arctan2(final_heading[1], final_heading[0])
    result[3:7] = (Rotation.from_euler("z", final_yaw - initial_yaw) * initial).as_quat()[[3, 0, 1, 2]]
    return result


def build_lifecycle_timeline(source, *, model, simulation_config, return_target="configured_origin"):
    """Retain every source channel/sample exactly between additional phases."""
    import mujoco

    if return_target not in RETURN_TARGETS:
        raise ValueError("unsupported lifecycle standing return target")
    count = validate_library_motion(source)
    if (model.nq, model.nv, model.nu, model.nbody) != (30, 29, 23, 25):
        raise ValueError("lifecycle reference requires exact native23 topology")
    config = json.loads(Path(simulation_config).read_text())
    initial = config["initial_state"]
    standing = np.asarray(
        [*initial["base_position_m"], *initial["base_quaternion_wxyz"], *initial["joint_position_hardware_rad"]],
        dtype=np.float64,
    )
    if standing.shape != (30,) or not np.isfinite(standing).all():
        raise ValueError("invalid configured native23 standing state")
    source_poses = np.concatenate(
        (source["body_pos_w"][:, 0], source["body_quat_w"][:, 0], source["joint_pos"]), axis=1
    )
    returned_standing = (
        _planned_endpoint_standing(standing, source_poses[-1]) if return_target == "planned_endpoint" else standing
    )
    sections = [np.tile(standing, (PREHISTORY_FRAMES, 1))]
    phases, cursor = [], 0
    for name, controls in PHASE_CONTROLS:
        controls = count if controls is None else controls
        if name == "source_motion":
            poses = source_poses.copy()
        elif name == "acquisition_ramp":
            poses = _blend_poses(standing, source_poses[0], controls)
        elif name == "return_ramp":
            poses = _blend_poses(source_poses[-1], returned_standing, controls)
        elif name in {"returned_standing", "standing_proof_margin"}:
            poses = np.tile(returned_standing, (controls, 1))
        else:
            poses = np.tile(standing, (controls, 1))
        sections.append(poses)
        phases.append(
            dict(
                name=name,
                control_start=cursor,
                control_stop=cursor + controls,
                frame_start=PREHISTORY_FRAMES + cursor,
                frame_stop=PREHISTORY_FRAMES + cursor + controls,
                requested_controls=controls,
            )
        )
        cursor += controls
    poses = np.concatenate(sections)
    data = mujoco.MjData(model)
    positions, quaternions = [], []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_fwdPosition(model, data)
        positions.append(data.xpos[1:].copy())
        quaternions.append(data.xquat[1:].copy())
    positions, quaternions = np.asarray(positions), np.asarray(quaternions)
    angular = np.zeros_like(positions)
    for body in range(24):
        rotations = Rotation.from_quat(quaternions[:, body, [1, 2, 3, 0]])
        angular[1:, body] = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / 0.02
    motion = dict(
        fps=np.array([50.0]),
        joint_pos=poses[:, 7:].copy(),
        joint_vel=np.gradient(poses[:, 7:], 0.02, axis=0),
        body_pos_w=positions,
        body_quat_w=quaternions,
        body_lin_vel_w=np.gradient(positions, 0.02, axis=0),
        body_ang_vel_w=angular,
    )
    phase = next(row for row in phases if row["name"] == "source_motion")
    span = slice(phase["frame_start"], phase["frame_stop"])
    source_geometry_max_error = float(np.max(np.abs(motion["body_pos_w"][span] - source["body_pos_w"])))
    # Do not quietly repair a reference whose body channels represent another
    # skeleton: source preservation and native FK consistency must both hold.
    if source_geometry_max_error > 2e-5:
        raise ValueError("source body positions are inconsistent with native23 joint/root FK")
    dot = np.abs(np.sum(motion["body_quat_w"][span] * source["body_quat_w"], axis=-1))
    if not np.allclose(dot, 1.0, atol=2e-5, rtol=0):
        raise ValueError("source body orientations are inconsistent with native23 FK")
    for key in motion:
        if key != "fps":
            motion[key][span] = source[key]
            np.testing.assert_array_equal(motion[key][span], source[key])
    validate_library_motion(motion)
    timeline = dict(
        kind="g1_true23_single_actor_lifecycle_reference_v1",
        source_frames=count,
        total_frames=len(poses),
        total_requested_controls=cursor,
        prehistory_frames=PREHISTORY_FRAMES,
        phases=phases,
        source_frame_indices=list(range(count)),
        source_timing_scale=1.0,
        source_start_frame=phase["frame_start"],
        source_stop_frame_exclusive=phase["frame_stop"],
        all_source_channels_samples_preserved=True,
        source_history_frames_trimmed=False,
        source_geometry_max_error_m=source_geometry_max_error,
        source_world_frame_aligned_or_modified=False,
        configured_standing_qpos=standing.tolist(),
        generated_ramp="interior_quintic_phase_with_quaternion_slerp",
        generated_ramp_contact_or_force_feasibility_qualified=False,
        ramp_endpoint_velocity_matching_qualified=False,
        generated_reference_not_action_teacher=True,
        **FLAGS,
    )
    if return_target == "planned_endpoint":
        start = next(row["frame_start"] for row in phases if row["name"] == "return_ramp")
        endpoint = source_poses[-1]
        root_path = np.concatenate((endpoint[None, :3], poses[start:, :3]))
        root_step = np.diff(root_path, axis=0) / 0.02
        timeline.update(
            kind="g1_true23_single_actor_endpoint_lifecycle_reference_v2",
            return_target="fixed_planned_terminal_xy_and_heading",
            returned_standing_qpos=returned_standing.tolist(),
            return_origin_locomotion_requested=False,
            measured_robot_state_used_for_return_target=False,
            return_root_horizontal_displacement_m=float(np.linalg.norm(returned_standing[:2] - endpoint[:2])),
            return_root_horizontal_speed_max_m_s=float(np.max(np.linalg.norm(root_step[:, :2], axis=1))),
            source_terminal_backward_root_velocity_m_s=(
                (source_poses[-1, :3] - source_poses[-2, :3]) / 0.02
            ).tolist(),
        )
    return motion, timeline


def assess_lifecycle_diagnostic(timeline, report, arrays):
    if report["available_controls"] != timeline["total_requested_controls"]:
        raise ValueError("benchmark shortened the lifecycle timeline")
    completed = report["completed_controls"]
    phases = []
    for phase in timeline["phases"]:
        actual = max(0, min(completed, phase["control_stop"]) - phase["control_start"])
        phases.append(
            {
                **phase,
                "completed_controls": actual,
                "phase_duration_integrated": actual == phase["requested_controls"],
            }
        )
    source = next(row for row in phases if row["name"] == "source_motion")
    start, finish = source["control_start"], source["control_start"] + source["completed_controls"]
    errors = arrays["landmark_error_m"][start:finish]
    source_failure = report["failure"] if completed <= source["control_stop"] else None
    source_screen = summarize_tracking(
        errors,
        completed=source["completed_controls"],
        requested=source["requested_controls"],
        available=source["requested_controls"],
        failure=source_failure,
    )
    proof = next(row for row in phases if row["name"] == "standing_proof_margin")
    proof_start, proof_end = proof["control_start"], min(completed, proof["control_stop"])
    joint_error, root_speed, root_position_error, root_orientation_error = None, None, None, None
    if proof_end > proof_start:
        states = arrays["qpos"][proof_start + 1 : proof_end + 1]
        velocities = arrays["qvel"][proof_start + 1 : proof_end + 1]
        target = np.asarray(timeline.get("returned_standing_qpos", timeline["configured_standing_qpos"]))
        joint_error = float(np.max(np.abs(states[:, 7:] - target[7:])))
        root_speed = float(np.max(np.linalg.norm(velocities[:, :3], axis=1)))
        root_position_error = float(np.max(np.linalg.norm(states[:, :3] - target[:3], axis=1)))
        desired = Rotation.from_quat(target[[4, 5, 6, 3]])
        measured = Rotation.from_quat(states[:, [4, 5, 6, 3]])
        root_orientation_error = float(np.max((desired * measured.inv()).magnitude()))
    return dict(
        kind="g1_true23_single_policy_lifecycle_diagnostic_v1",
        phases=phases,
        single_policy_full_lifecycle_integrated=report["failure"] is None
        and completed == timeline["total_requested_controls"],
        source_motion_tracking=source_screen,
        every_original_source_frame_evaluated=source["completed_controls"] == source["requested_controls"],
        final_proof_standing_joint_error_max_rad=joint_error,
        final_proof_root_speed_max_m_s=root_speed,
        final_proof_root_position_error_max_m=root_position_error,
        final_proof_root_orientation_error_max_rad=root_orientation_error,
        fallback_or_specialist_controller_used=False,
        robot_state_resets_after_initial_standing=0,
        standing_or_contact_qualification=False,
        **FLAGS,
    )
