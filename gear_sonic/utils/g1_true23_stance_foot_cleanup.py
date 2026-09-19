"""Explicit offline contact-conditioned reference hypothesis, not raw-source parity.

Correct near-floor, slow feet with fixed pelvis/upper body. No frame removal,
retiming, actuator changes or artificial root forces. Inferred contacts are a
hypothesis, not measured labels. Every correction is bounded and separately
reported against the input; no old5mm original-fidelity gate is relabeled.
"""

from dataclasses import asdict

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_trajectory_projection import (
    audit_trajectory_constraints,
    project_nearest_trajectory,
)
from gear_sonic.utils.g1_true23_original_task_trajectory import so3_left_jacobian_inverse

MAXIMUM_FOOT_TRANSLATION_M = 0.02
MAXIMUM_FOOT_ROTATION_RAD = 0.15
MAXIMUM_LEG_CHANGE_RAD = 0.2


def bounded_stance_targets(positions, rotations, contacts, sole_heights):
    """Bound corrections, not the requested motion or physical acceptance gates.

    A large toe roll can remain ungrounded after this deliberately partial
    correction. Its residual and force-support failure must remain visible.
    """
    target_pos, target_rot = positions.copy(), rotations.copy()
    clipped_translation, clipped_rotation = [], []
    for side, height in enumerate(sole_heights):
        delta_z = height - positions[:, side, 2]
        yaw = np.arctan2(rotations[:, side, 1, 0], rotations[:, side, 0, 0])
        flat = Rotation.from_euler("z", yaw).as_matrix()
        vectors = Rotation.from_matrix(flat @ rotations[:, side].transpose(0, 2, 1)).as_rotvec()
        angles = np.linalg.norm(vectors, axis=1)
        clipped_translation.append(
            np.flatnonzero(contacts[:, side] & (np.abs(delta_z) > MAXIMUM_FOOT_TRANSLATION_M)).tolist()
        )
        clipped_rotation.append(np.flatnonzero(contacts[:, side] & (angles > MAXIMUM_FOOT_ROTATION_RAD)).tolist())
        selected = contacts[:, side]
        target_pos[selected, side, 2] += np.clip(
            delta_z[selected], -MAXIMUM_FOOT_TRANSLATION_M, MAXIMUM_FOOT_TRANSLATION_M
        )
        vectors *= np.minimum(1, MAXIMUM_FOOT_ROTATION_RAD / np.maximum(angles, 1e-12))[:, None]
        target_rot[selected, side] = (
            Rotation.from_rotvec(vectors[selected]).as_matrix() @ rotations[selected, side]
        )
    return (
        target_pos,
        target_rot,
        dict(
            translation_capped_frames_by_foot=clipped_translation,
            rotation_capped_frames_by_foot=clipped_rotation,
            partial_corrections_are_not_complete_grounding=True,
        ),
    )


def foot_frames(model, poses):
    data = mujoco.MjData(model)
    bodies = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    positions, rotations = [], []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[bodies].copy())
        rotations.append(data.xmat[bodies].reshape(2, 3, 3).copy())
    return np.asarray(positions), np.asarray(rotations)


def flat_sole_height(model, body):
    geoms = [
        int(g)
        for g in np.flatnonzero(model.geom_bodyid == body)
        if model.geom_contype[g] or model.geom_conaffinity[g]
    ]
    if len(geoms) != 4 or any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE for g in geoms):
        raise ValueError("stance cleanup requires the unchanged four-sphere native foot geometry")
    heights = model.geom_size[geoms, 0] - model.geom_pos[geoms, 2]
    if np.ptp(heights) > 1e-10:
        raise ValueError("stance cleanup requires coplanar native sole spheres")
    return float(heights[0])


def foot_task_residual_jacobian(model, data, body, position, rotation, columns):
    error_rot = Rotation.from_matrix(data.xmat[body].reshape(3, 3) @ rotation.T).as_rotvec()
    residual = np.r_[100 * (data.xpos[body] - position), 10 * error_rot]
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jp, jr, body)
    jacobian = np.vstack((100 * jp[:, columns], 10 * so3_left_jacobian_inverse(error_rot) @ jr[:, columns]))
    return residual, jacobian


def clean_stance_feet(model, poses, *, allow_bounded_root_translation=False, progress=None):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 30 or len(poses) < 3 or not np.isfinite(poses).all():
        raise ValueError("stance cleanup requires full finite native23 source poses")
    positions, rotations = foot_frames(model, poses)
    contacts = ik.infer_foot_contacts(positions, fps=50, height_tolerance_m=0.035, speed_tolerance_m_s=0.45)
    bodies = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    target_pos, target_rot, target_proof = bounded_stance_targets(
        positions, rotations, contacts, [flat_sole_height(model, body) for body in bodies]
    )
    low, high = ik.safe_target_joint_bounds(model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    desired = poses.copy()
    data = mujoco.MjData(model)
    unsolved = []
    for frame, pose in enumerate(desired):
        if allow_bounded_root_translation:
            center = poses[frame].copy()
            variables = np.r_[np.zeros(3), center[7:19]]
            columns = np.r_[0:3, 6:18]
            lower = np.r_[np.full(3, -0.03), np.maximum(low[:12], center[7:19] - MAXIMUM_LEG_CHANGE_RAD)]
            upper = np.r_[np.full(3, 0.03), np.minimum(high[:12], center[7:19] + MAXIMUM_LEG_CHANGE_RAD)]

            def evaluate_root(values, *, derivatives=False):
                pose[:3], pose[7:19] = center[:3] + values[:3], values[3:]
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                tasks = [
                    foot_task_residual_jacobian(
                        model, data, body, target_pos[frame, side], target_rot[frame, side], columns
                    )
                    for side, body in enumerate(bodies)
                ]
                regularizer = np.r_[np.full(3, 2.0), np.full(12, 0.02)]
                if derivatives:
                    return np.vstack([*(t[1] for t in tasks), np.diag(regularizer)])
                return np.concatenate([*(t[0] for t in tasks), regularizer * (values - variables)])

            solution = least_squares(
                evaluate_root,
                variables,
                jac=lambda x: evaluate_root(x, derivatives=True),
                bounds=(lower, upper),
                max_nfev=80,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            final_error = evaluate_root(solution.x)
            if np.max(np.abs(final_error[:12])) > 0.01:
                unsolved.append([frame, "coupled_root_legs"])
        for side, body in enumerate(bodies):
            if allow_bounded_root_translation or not contacts[frame, side]:
                continue
            joints = np.arange(6 * side, 6 * side + 6)
            qcolumns, vcolumns = joints + 7, joints + 6
            lo = np.maximum(low[joints], poses[frame, qcolumns] - MAXIMUM_LEG_CHANGE_RAD)
            hi = np.minimum(high[joints], poses[frame, qcolumns] + MAXIMUM_LEG_CHANGE_RAD)

            def evaluate_leg(values, *, derivatives=False):
                pose[qcolumns] = values
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                task = foot_task_residual_jacobian(
                    model, data, body, target_pos[frame, side], target_rot[frame, side], vcolumns
                )
                return task[1 if derivatives else 0]

            solution = least_squares(
                evaluate_leg,
                pose[qcolumns].copy(),
                jac=lambda x: evaluate_leg(x, derivatives=True),
                bounds=(lo, hi),
                max_nfev=64,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            final_error = evaluate_leg(solution.x)
            if np.max(np.abs(final_error)) > 0.01:
                unsolved.append([frame, side])
        if progress and frame % 100 == 0:
            progress({"frame": frame, "total_frames": len(poses), "ik_unconverged": len(unsolved)})
    # Freeze every inactive leg exactly. Project all active leg samples jointly
    # so contact switches cannot hide joint velocity/acceleration discontinuity.
    editable = (
        np.ones((len(poses), 12), dtype=bool) if allow_bounded_root_translation else np.repeat(contacts, 6, axis=1)
    )
    lo, hi = poses[:, 7:19].copy(), poses[:, 7:19].copy()
    lo[editable] = np.maximum(np.broadcast_to(low[:12], lo.shape), poses[:, 7:19] - MAXIMUM_LEG_CHANGE_RAD)[
        editable
    ]
    hi[editable] = np.minimum(np.broadcast_to(high[:12], hi.shape), poses[:, 7:19] + MAXIMUM_LEG_CHANGE_RAD)[
        editable
    ]
    projection = project_nearest_trajectory(
        desired[:, 7:19], lower_bounds=lo, upper_bounds=hi, dt=0.02, max_velocity=5, max_acceleration=80
    )
    result = poses.copy()
    result[:, 7:19][editable] = projection.projected_path[editable]
    root_projection = None
    if allow_bounded_root_translation:
        root_projection = project_nearest_trajectory(
            desired[:, :3] - poses[:, :3],
            lower_bounds=np.full(3, -0.03),
            upper_bounds=np.full(3, 0.03),
            dt=0.02,
            max_velocity=0.75,
            max_acceleration=6,
        )
        result[:, :3] += root_projection.projected_path
    final_positions, final_rotations = foot_frames(model, result)
    displacement = np.linalg.norm(final_positions - positions, axis=2)
    rotation_change = (
        Rotation.from_matrix((final_rotations @ rotations.transpose(0, 1, 3, 2)).reshape(-1, 3, 3))
        .magnitude()
        .reshape(len(poses), 2)
    )
    target_position_error = np.linalg.norm(final_positions - target_pos, axis=2)
    target_orientation_error = (
        Rotation.from_matrix((target_rot @ final_rotations.transpose(0, 1, 3, 2)).reshape(-1, 3, 3))
        .magnitude()
        .reshape(len(poses), 2)
    )
    bounds = audit_trajectory_constraints(
        result[:, 7:],
        lower_bounds=low,
        upper_bounds=high,
        dt=0.02,
        max_velocity=5,
        max_acceleration=80,
        tolerance=2e-7,
    )
    return (
        result,
        contacts,
        dict(
            kind="g1_true23_bounded_contact_root_and_legs_hypothesis_v3"
            if allow_bounded_root_translation
            else "g1_true23_explicit_bounded_partial_stance_foot_cleanup_hypothesis_v2",
            root_translation_enabled=allow_bounded_root_translation,
            root_translation_max_per_axis_m=np.max(np.abs(result[:, :3] - poses[:, :3]), axis=0).tolist(),
            root_translation_projection=asdict(root_projection.audit) if root_projection is not None else None,
            original_root_attitude_and_upper_joints_bit_exact=np.array_equal(
                result[:, np.r_[3:7, 19:30]], poses[:, np.r_[3:7, 19:30]]
            ),
            correction_bounds_passed=bool(
                displacement.max() <= MAXIMUM_FOOT_TRANSLATION_M + 1e-6
                and rotation_change.max() <= MAXIMUM_FOOT_ROTATION_RAD + 1e-6
                and np.max(np.abs(result[:, 7:] - poses[:, 7:])) <= MAXIMUM_LEG_CHANGE_RAD + 2e-7
            ),
            target_proof=target_proof,
            inferred_contacts_not_measured=True,
            inference={"fps": 50, "height_tolerance_m": 0.035, "speed_tolerance_m_s": 0.45},
            frames=len(poses),
            inferred_contact_frames_by_foot=contacts.sum(axis=0).tolist(),
            original_root_and_upper_joint_poses_bit_exact=np.array_equal(
                result[:, np.r_[0:7, 19:30]], poses[:, np.r_[0:7, 19:30]]
            ),
            inactive_leg_poses_bit_exact=np.array_equal(result[:, 7:19][~editable], poses[:, 7:19][~editable]),
            maximum_ankle_translation_m=float(displacement.max()),
            maximum_ankle_rotation_rad=float(rotation_change.max()),
            maximum_joint_change_rad=float(np.max(np.abs(result[:, 7:] - poses[:, 7:]))),
            target_position_error_max_m=float(target_position_error.max()),
            target_orientation_error_max_rad=float(target_orientation_error.max()),
            bounds=asdict(bounds),
            projection_iterations=projection.iterations,
            ik_unconverged=unsolved,
            raw_source_fidelity_gate_redefined=False,
            raw_source5mm_fidelity_claimed=False,
            source_frames_removed=0,
            source_retimed=False,
            training_reference_accepted=False,
            dynamic_feasibility_proven=False,
            hardware_authorized=False,
            deployment_ready=False,
        ),
    )
