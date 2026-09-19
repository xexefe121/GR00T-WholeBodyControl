"""SIM-only multi-joint clearance of existing generated ramp interiors.

The rate-corrected source, standing, root and leg poses remain bit-exact. Only
ten arm joints inside each already scheduled ramp may move, by at most0.3rad.
Full joint/rate limits and the original1mm COM envelope remain hard. Query
margin is not physical acceptance; the unchanged model checks all100Hz poses.
"""

from dataclasses import asdict

import mujoco
import numpy as np
from scipy import sparse

from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import measure_self_contacts
from gear_sonic.utils.g1_23dof_trajectory_projection import audit_trajectory_constraints
from gear_sonic.utils.g1_true23_collision_sampling import interpolate_original_poses, interpolation_weights
from gear_sonic.utils.g1_true23_elastic_collision_step import solve_elastic_collision_step
from gear_sonic.utils.g1_true23_self_collision import SelfCollisionLinearizer

PROFILE = "g1_true23_bounded_generated_arm_path_v1"
MAXIMUM_ARM_CHANGE_RAD = 0.3
COM_CHANGE_LIMIT_M = 0.001


def arm_path_columns(joint_names):
    expected = tuple(
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
    )
    if len(joint_names) != 23 or len(set(joint_names)) != 23:
        raise ValueError("ramp arm path requires unique native23 joints")
    return np.array([tuple(joint_names).index(name) for name in expected])


def verify_arm_pose_replacement(before, after, timeline, joint_names):
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or before.ndim != 2 or before.shape[1] != 30:
        raise ValueError("arm path must preserve complete native23 pose shape")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("arm path poses must be finite")
    phases = [p for p in timeline["phases"] if p["name"] in ("acquisition_ramp", "return_ramp")]
    if [p["name"] for p in phases] != ["acquisition_ramp", "return_ramp"]:
        raise ValueError("arm path requires both existing generated ramps")
    allowed = np.zeros_like(before, dtype=bool)
    columns = 7 + arm_path_columns(joint_names)
    for phase in phases:
        start, stop = phase["frame_start"], phase["frame_stop"]
        if type(start) is not int or type(stop) is not int or not 2 <= start < stop <= len(before) - 2:
            raise ValueError("arm path requires interior ramp boundaries")
        allowed[start:stop, columns] = True
    source = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    if np.any(allowed[source["frame_start"] : source["frame_stop"]]):
        raise ValueError("arm path may never include source frames")
    if not np.array_equal(before[~allowed], after[~allowed]):
        raise ValueError("arm path changed frozen source, standing, root or lower-body poses")
    maximum = float(np.max(np.abs(after - before)))
    if maximum > MAXIMUM_ARM_CHANGE_RAD + 2e-7:
        raise ValueError("arm path exceeds0.3rad generated-arm displacement")
    return maximum


def com_path_jacobian(model, poses, dof_addresses):
    """World COM and analytical joint Jacobian on the exact physical model."""
    data = mujoco.MjData(model)
    pelvis = model.body("pelvis").id
    positions, blocks = [], []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacSubtreeCom(model, data, jacobian, pelvis)
        positions.append(data.subtree_com[pelvis].copy())
        blocks.append(sparse.csc_matrix(jacobian[:, dof_addresses]))
    return np.asarray(positions), sparse.block_diag(blocks, format="csc")


def ramp_linear_constraints(count, width):
    """Position, velocity-step and ordinary second-difference rows."""
    identity = sparse.eye(count, format="csc")
    first = sparse.diags((-np.ones(count - 1), np.ones(count - 1)), (0, 1), shape=(count - 1, count))
    second = sparse.diags(
        (np.ones(count - 2), -2 * np.ones(count - 2), np.ones(count - 2)),
        (0, 1, 2),
        shape=(count - 2, count),
    )
    return sparse.kron(sparse.vstack((identity, first, second)), sparse.eye(width), format="csc")


def fit_generated_arm_paths(
    model, original, rate_corrected, timeline, joint_names, lower, upper, *, progress=None
):
    original, rate_corrected = np.asarray(original), np.asarray(rate_corrected)
    verify_arm_pose_replacement(rate_corrected, rate_corrected, timeline, joint_names)
    if original.shape != rate_corrected.shape or not np.isfinite(original).all():
        raise ValueError("arm path needs original matching full lifecycle for COM comparison")
    joints = arm_path_columns(joint_names)
    columns, dofs = joints + 7, joints + 6
    output = rate_corrected.copy()
    query = SelfCollisionLinearizer(model)
    reports = []
    for phase in timeline["phases"]:
        if phase["name"] not in ("acquisition_ramp", "return_ramp"):
            continue
        start, stop = phase["frame_start"], phase["frame_stop"]
        # Two frozen neighbors on each side capture acceleration at both joins.
        span = slice(start - 2, stop + 2)
        poses = output[span].copy()
        baseline = original[span]
        count, width = len(poses), len(joints)
        times = np.arange(count, dtype=float)
        all_half_times = np.arange(2 * count - 1) / 2
        weights = interpolation_weights(times, all_half_times)[2:-2]
        sample_template = interpolate_original_poses(poses, times, all_half_times)[2:-2]
        baseline_samples = interpolate_original_poses(baseline, times, all_half_times)[2:-2]
        baseline_com, _ = com_path_jacobian(model, baseline_samples, dofs)
        sample_map = sparse.kron(weights, sparse.eye(width), format="csc")
        center = poses[:, columns].copy()
        lo = np.maximum(lower[joints], center - MAXIMUM_ARM_CHANGE_RAD)
        hi = np.minimum(upper[joints], center + MAXIMUM_ARM_CHANGE_RAD)
        lo[:2] = hi[:2] = center[:2]
        lo[-2:] = hi[-2:] = center[-2:]
        matrix = ramp_linear_constraints(count, width)
        rate_low = np.r_[np.full((count - 1) * width, -5 * 0.02), np.full((count - 2) * width, -80 * 0.02**2)]
        rate_high = -rate_low
        groups = [
            {"indices": np.arange(3 * i, 3 * i + 3), "scales": np.ones(3), "radius": 0.00098}
            for i in range(len(sample_template))
        ]

        def measure(values):
            samples = sample_template.copy()
            samples[:, 7:] = weights @ values[:, 7:]
            com, jacobian = com_path_jacobian(model, samples, dofs)
            collision_jacobian, distances, _ = query.path_rows(samples, dofs)
            shortfall = np.maximum(0.001 - distances, 0)
            return (
                samples,
                com,
                jacobian @ sample_map,
                collision_jacobian @ sample_map,
                distances,
                float(shortfall @ shortfall),
            )

        iterations = []
        initial = measure(poses)
        for iteration in range(32):
            samples, com, jacobian, collision_jacobian, distances, merit = measure(poses)
            contacts = measure_self_contacts(model, samples)
            if (
                contacts["frames_with_robot_robot_penetration"] == 0
                and np.max(np.linalg.norm(com - baseline_com, axis=1)) <= COM_CHANGE_LIMIT_M
            ):
                break
            current = poses[:, columns].ravel()
            position_low = np.maximum(lo.ravel(), current - 0.12)
            position_high = np.minimum(hi.ravel(), current + 0.12)
            linear_value = matrix @ current
            step, solve = solve_elastic_collision_step(
                np.ones(len(current)),
                current - center.ravel(),
                (com - baseline_com).ravel(),
                jacobian,
                matrix,
                np.r_[position_low, rate_low] - linear_value,
                np.r_[position_high, rate_high] - linear_value,
                groups,
                collision_jacobian,
                distances,
                objective_profile="minimum_change_v2",
            )
            row = {"iteration": iteration, "solver": solve, "accepted": False, "merit_before": merit}
            if step is not None:
                for fraction in (1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
                    candidate = poses.copy()
                    candidate[2:-2, columns] += fraction * step.reshape(count, width)[2:-2]
                    changed = measure(candidate)
                    actual_com = float(np.max(np.linalg.norm(changed[1] - baseline_com, axis=1)))
                    actual = matrix @ candidate[:, columns].ravel()
                    violation = float(
                        max(
                            np.maximum(np.r_[lo.ravel(), rate_low] - actual, 0).max(),
                            np.maximum(actual - np.r_[hi.ravel(), rate_high], 0).max(),
                        )
                    )
                    if changed[-1] < merit - 1e-15 and actual_com <= COM_CHANGE_LIMIT_M and violation <= 2e-7:
                        poses = candidate
                        row.update(
                            accepted=True,
                            fraction=fraction,
                            merit_after=changed[-1],
                            maximum_com_change_m=actual_com,
                            path_violation=violation,
                        )
                        break
            iterations.append(row)
            if progress:
                progress(
                    {
                        "phase": phase["name"],
                        **{k: v for k, v in row.items() if k != "solver"},
                        "solver_status": solve["status"],
                    }
                )
            if not row["accepted"]:
                break
        final = measure(poses)
        contacts = measure_self_contacts(model, final[0])
        com_change = float(np.max(np.linalg.norm(final[1] - baseline_com, axis=1)))
        accepted = contacts["frames_with_robot_robot_penetration"] == 0 and com_change <= COM_CHANGE_LIMIT_M
        output[span] = poses
        reports.append(
            dict(
                phase=phase["name"],
                accepted=accepted,
                iterations=iterations,
                initial_merit=initial[-1],
                final_merit=final[-1],
                self_contacts=contacts,
                maximum_com_change_m=com_change,
            )
        )
    maximum = verify_arm_pose_replacement(rate_corrected, output, timeline, joint_names)
    bounds = audit_trajectory_constraints(
        output[:, 7:],
        lower_bounds=lower,
        upper_bounds=upper,
        dt=0.02,
        max_velocity=5,
        max_acceleration=80,
        tolerance=2e-7,
    )
    return output, dict(
        kind=PROFILE,
        accepted=bool(bounds.passed and all(p["accepted"] for p in reports)),
        phases=reports,
        full_path_bounds=asdict(bounds),
        maximum_arm_change_rad=maximum,
        arm_change_limit_rad=MAXIMUM_ARM_CHANGE_RAD,
        com_change_limit_m=COM_CHANGE_LIMIT_M,
        source_standing_root_and_rate_corrected_lower_body_bit_exact=True,
        source_timing_changed=False,
        dynamic_feasibility_proven=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
