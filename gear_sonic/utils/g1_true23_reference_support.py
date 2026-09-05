"""Conditional reference support/effort screen; never a motion controller.

At each reference pose, set velocity and acceleration to zero and solve
    J.T @ contact_force + joint_torque + static_friction = bias - passive.
Optional inverse-dynamics mode derives velocity/acceleration from the complete
pose path and adds M @ acceleration. These are finite-difference hypotheses,
not exact dynamics between samples. The floating base has no actuator.
Contact candidates come from the supplied
model within an explicit floor-gap tolerance; candidate forces do NOT prove
actual contact. Pyramidal cones, static friction and normalized torque limits
form a linear program. Dynamic motions need not be statically supportable.

MuJoCo force/Jacobian convention:
https://mujoco.readthedocs.io/en/3.5.0/computation/index.html#contact
https://mujoco.readthedocs.io/en/3.5.0/APIreference/APIfunctions.html#mj-jac
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
from scipy.optimize import linprog

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_contact_geometry import lift_reset_floor_overlap
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos


def minimum_effort_support(required, contact_map, torque_limits, friction_limits):
    """Minimize the largest absolute joint torque / supplied torque limit.

    Contact-map columns are unilateral cone rays with nonnegative weights.
    Friction-loss assistance is bounded but optimally selected: a static,
    optimistic assumption, not measured actuator friction or moving friction.
    """
    required, contact_map, torque_limits, friction_limits = (
        np.asarray(value, dtype=np.float64) for value in (required, contact_map, torque_limits, friction_limits)
    )
    if (
        required.shape != (29,)
        or contact_map.ndim != 2
        or contact_map.shape[0] != 29
        or torque_limits.shape != (23,)
        or friction_limits.shape != (23,)
        or any(not np.isfinite(value).all() for value in (required, contact_map, torque_limits, friction_limits))
        or np.any(torque_limits <= 0)
        or np.any(friction_limits < 0)
    ):
        raise ValueError("support LP requires finite native23 forces and positive torque bounds")
    rays = contact_map.shape[1]
    joint_map = np.vstack((np.zeros((6, 23)), np.eye(23)))
    equality = np.column_stack((contact_map, joint_map, joint_map, np.zeros(29)))
    inequalities = np.zeros((46, rays + 47))
    inequalities[:23, rays : rays + 23] = np.eye(23)
    inequalities[23:, rays : rays + 23] = -np.eye(23)
    inequalities[:, -1] = -np.tile(torque_limits, 2)
    objective = np.zeros(rays + 47)
    objective[-1] = 1
    bounds = (
        [(0, None)] * rays + [(None, None)] * 23 + [(-value, value) for value in friction_limits] + [(0, None)]
    )
    problem = dict(A_ub=inequalities, b_ub=np.zeros(46), A_eq=equality, b_eq=required, bounds=bounds)
    tolerances = {"primal_feasibility_tolerance": 1e-8, "dual_feasibility_tolerance": 1e-8}
    result = linprog(objective, **problem, method="highs", options=tolerances)
    root_wrench_infeasibility_confirmed = False
    if result.status not in (0, 2) and rays:
        # Static support existence depends only on the unactuated six rows;
        # joint torque is unbounded until alpha is minimized. This smaller LP
        # can certify infeasibility even if the full degenerate LP is Unknown.
        root_result = linprog(
            np.zeros(rays),
            A_eq=contact_map[:6],
            b_eq=required[:6],
            bounds=(0, None),
            method="highs",
            options=tolerances,
        )
        if root_result.status == 2:
            result = root_result
            root_wrench_infeasibility_confirmed = True
    retried_without_presolve = result.status not in (0, 2)
    if retried_without_presolve:
        # Degenerate coplanar contact rays can leave HiGHS presolve at Unknown.
        # Re-solve the SAME LP by dual simplex without presolve. Unknown never
        # counts as infeasible, and numerical acceptance bounds stay unchanged.
        result = linprog(objective, **problem, method="highs-ds", options={**tolerances, "presolve": False})
    if result.status == 2:
        return {
            "status": "no_support_wrench_under_candidate_cones",
            "minimum_peak_effort_ratio": None,
            "within_supplied_effort_limits": False,
            "solver_retried_without_presolve": retried_without_presolve,
            "infeasibility_confirmed_by_root_wrench_lp": root_wrench_infeasibility_confirmed,
        }
    if not result.success:
        raise RuntimeError(f"support LP did not solve reliably: {result.message}")
    residual = float(np.max(np.abs(equality @ result.x - required)))
    torque = result.x[rays : rays + 23]
    friction = result.x[rays + 23 : rays + 46]
    ratio = float(np.max(np.abs(torque) / torque_limits))
    if (
        residual > 1e-5
        or np.max(inequalities @ result.x) > 1e-6
        or np.any(result.x[:rays] < -1e-7)
        or np.any(np.abs(friction) > friction_limits + 1e-7)
        or abs(ratio - result.x[-1]) > 1e-6
    ):
        raise RuntimeError("support LP solution failed independent force/bound checks")
    return {
        "status": "conditional_force_balance_solution",
        "minimum_peak_effort_ratio": ratio,
        "within_supplied_effort_limits": bool(ratio <= 1 + 1e-6),
        "solver_retried_without_presolve": retried_without_presolve,
        "infeasibility_confirmed_by_root_wrench_lp": False,
        "maximum_generalized_force_residual": residual,
        "joint_torques_nm": torque.tolist(),
        "static_friction_assistance_nm": friction.tolist(),
        "cone_ray_weights": result.x[:rays].tolist(),
        "limiting_joint": HARDWARE_23_JOINT_NAMES[int(np.argmax(np.abs(torque) / torque_limits))],
    }


def contact_cone_rays(dimension, friction):
    """MuJoCo pyramidal cone edges in contact-frame force:torque ordering."""
    friction = np.asarray(friction, dtype=np.float64)
    if dimension not in (1, 3, 4, 6) or friction.shape != (5,) or not np.isfinite(friction).all():
        raise ValueError("invalid contact dimension or friction coefficients")
    if np.any(friction < 0):
        raise ValueError("friction coefficients must be nonnegative")
    if dimension == 1:
        return np.array([[1, 0, 0, 0, 0, 0]], dtype=np.float64).T
    result = np.zeros((6, 2 * (dimension - 1)))
    result[0] = 1
    for axis in range(1, dimension):
        result[axis, 2 * (axis - 1) : 2 * axis] = [friction[axis - 1], -friction[axis - 1]]
    return result


def floor_contact_map(model, data, plane_id, gap_tolerance_m):
    """Candidate ground wrenches; no self-contact or joint-limit assistance."""
    columns, records = [], []
    first_ray = 0
    for contact in data.contact[: data.ncon]:
        first, second = map(int, contact.geom)
        if plane_id not in (first, second) or contact.dist > gap_tolerance_m + 1e-10:
            continue
        robot_geom = second if first == plane_id else first
        sign = 1 if first == plane_id else -1
        rotation = sign * np.asarray(contact.frame).reshape(3, 3).T
        if not np.allclose(rotation[:, 0], [0, 0, 1], atol=1e-7, rtol=0):
            raise ValueError("support screen requires an upward flat-floor contact normal")
        body_id = int(model.geom_bodyid[robot_geom])
        jacobian_pos, jacobian_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian_pos, jacobian_rot, contact.pos, body_id)
        rays = contact_cone_rays(int(contact.dim), contact.friction)
        world_forces, world_torques = rotation @ rays[:3], rotation @ rays[3:]
        columns.append(jacobian_pos.T @ world_forces + jacobian_rot.T @ world_torques)
        records.append(
            {
                "robot_geom_id": robot_geom,
                "robot_geom": model.geom(robot_geom).name,
                "body": model.body(body_id).name,
                "position_w": contact.pos.tolist(),
                "distance_m": float(contact.dist),
                "dimension": int(contact.dim),
                "friction": contact.friction.tolist(),
                "cone_ray_start": first_ray,
                "cone_ray_count": rays.shape[1],
            }
        )
        first_ray += rays.shape[1]
    return (np.column_stack(columns) if columns else np.zeros((model.nv, 0))), records


def pose_path_derivatives(model, qpos, dt):
    """Independent, unsmoothed pose derivatives; angular rates in root frame.

    Scalar coordinates use central differences and second-order endpoints.
    Quaternion interval rates use MuJoCo's tangent-space subtraction, rotate
    to world before central/one-sided averaging, then back to each root frame.
    The generalized acceleration differentiates these velocity coordinates.
    No archived velocity channel or quaternion-component subtraction is used.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 30 or len(qpos) < 3 or not np.isfinite(qpos).all():
        raise ValueError("pose derivatives require at least three finite native23 positions")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("pose derivative timestep must be finite and positive")
    rotations = np.empty((len(qpos), 3, 3))
    for index, position in enumerate(qpos):
        mujoco.mju_quat2Mat(rotations[index].reshape(9), position[3:7])
    interval_world = np.empty((len(qpos) - 1, 3))
    scratch = np.zeros(model.nv)
    for index in range(len(qpos) - 1):
        mujoco.mj_differentiatePos(model, scratch, dt, qpos[index], qpos[index + 1])
        interval_world[index] = rotations[index] @ scratch[3:6]
    angular_world = np.empty((len(qpos), 3))
    angular_world[1:-1] = (interval_world[:-1] + interval_world[1:]) / 2
    angular_world[0] = 1.5 * interval_world[0] - 0.5 * interval_world[1]
    angular_world[-1] = 1.5 * interval_world[-1] - 0.5 * interval_world[-2]
    qvel = np.zeros((len(qpos), model.nv))
    qvel[:, :3] = np.gradient(qpos[:, :3], dt, axis=0, edge_order=2)
    qvel[:, 3:6] = np.einsum("nji,nj->ni", rotations, angular_world)
    qvel[:, 6:] = np.gradient(qpos[:, 7:], dt, axis=0, edge_order=2)
    return qvel, np.gradient(qvel, dt, axis=0, edge_order=2)


def audit_reference_support(model, motion, torque_limits, *, gap_tolerance_m=0.002, reference_dynamics=False):
    """Audit every pose, without editing or stepping the model.

    A conditional pass is NOT dynamic feasibility or a permitted robot action.
    A static failure is NOT proof that an accelerating dance is impossible.
    """
    if not np.isfinite(gap_tolerance_m) or not 0 <= gap_tolerance_m <= 0.01:
        raise ValueError("candidate floor gap tolerance must be between 0 and 0.01 m")
    if not isinstance(reference_dynamics, bool):
        raise ValueError("reference dynamics selection must be boolean")
    qpos = motion_qpos(model, motion)
    if model.nv != 29 or model.neq or model.ntendon:
        raise ValueError("support screen requires unconstrained native23 articulation")
    if not np.allclose(model.opt.gravity, [0, 0, -9.81], atol=1e-8, rtol=0):
        raise ValueError("support screen requires explicit Earth gravity along negative z")
    # Reuse the independent single-articulation / fixed-horizontal-floor check.
    # Its diagnostic lift is discarded and no reference pose is adjusted here.
    lift_reset_floor_overlap(model, model.qpos0[None, :])
    original_hash = compiled_model_sha256(model)
    candidate_model = copy.copy(model)
    candidate_model.geom_margin[:] = np.maximum(candidate_model.geom_margin, gap_tolerance_m)
    candidate_model.pair_margin[:] = np.maximum(candidate_model.pair_margin, gap_tolerance_m)
    # Runtime margin changes do not enlarge the compiled static BVH: without
    # this diagnostic-only switch, nearby toe geoms can be silently culled.
    # Check every collision-eligible pair; original model/physics stay intact.
    candidate_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    plane = int(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)[0])
    data = mujoco.MjData(candidate_model)
    if reference_dynamics:
        qvel, qacc = pose_path_derivatives(model, qpos, 0.02)
    else:
        qvel = qacc = np.zeros((len(qpos), model.nv))
    rows = []
    for index, position in enumerate(qpos):
        data.qpos[:] = position
        data.qvel[:] = qvel[index]
        mujoco.mj_fwdPosition(candidate_model, data)
        mujoco.mj_fwdVelocity(candidate_model, data)
        inertia_force = np.zeros(model.nv)
        mujoco.mj_mulM(candidate_model, data, inertia_force, qacc[index])
        required = inertia_force + data.qfrc_bias - data.qfrc_passive
        force_map, contacts = floor_contact_map(candidate_model, data, plane, gap_tolerance_m)
        solution = minimum_effort_support(required, force_map, torque_limits, model.dof_frictionloss[6:])
        rows.append(
            {
                "frame": index,
                "candidate_contacts": contacts,
                "candidate_contact_count": len(contacts),
                "center_of_mass_w": data.subtree_com[1].tolist(),
                "required_generalized_force": required.tolist(),
                "generalized_velocity": qvel[index].tolist(),
                "generalized_acceleration": qacc[index].tolist(),
                "floor_overlap_detected": any(row["distance_m"] < 0 for row in contacts),
                "joint_range_violation": bool(
                    np.any(position[7:] < model.jnt_range[1:, 0]) or np.any(position[7:] > model.jnt_range[1:, 1])
                ),
                **solution,
            }
        )
    if compiled_model_sha256(model) != original_hash:
        raise RuntimeError("support audit changed the input model")
    ratios = [row["minimum_peak_effort_ratio"] for row in rows if row["minimum_peak_effort_ratio"] is not None]
    return {
        "kind": "g1_true23_conditional_reference_support_v1",
        "mode": "reference_inverse_dynamics" if reference_dynamics else "quasistatic",
        "compiled_mjb_sha256": original_hash,
        "candidate_model_mjb_sha256": compiled_model_sha256(candidate_model),
        "frames_checked": len(rows),
        "frames_dropped": 0,
        "frames_with_no_candidate_contact": sum(not row["candidate_contact_count"] for row in rows),
        "frames_with_no_support_solution": sum(row["minimum_peak_effort_ratio"] is None for row in rows),
        "frames_with_solution_above_effort_limits": sum(
            row["minimum_peak_effort_ratio"] is not None and not row["within_supplied_effort_limits"]
            for row in rows
        ),
        "frames_with_conditional_solution_within_effort_limits": sum(
            row["within_supplied_effort_limits"] for row in rows
        ),
        "maximum_finite_minimum_effort_ratio": max(ratios) if ratios else None,
        "torque_limits_nm": np.asarray(torque_limits).tolist(),
        "candidate_floor_gap_tolerance_m": gap_tolerance_m,
        "candidate_collision_midphase_disabled_for_runtime_margins": True,
        "cone": "mujoco_pyramidal_edges",
        "cone_matches_model": bool(model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL),
        "elliptic_cones_conservatively_approximated": bool(model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL),
        "velocity_and_acceleration_assumed_zero": not reference_dynamics,
        "reference_derivative_method": "pose_path_derivatives_at_50hz_unsmoothed" if reference_dynamics else None,
        "reference_velocity_channels_used": False,
        "derivative_endpoint_frames_use_one_sided_estimates": reference_dynamics,
        "friction_assistance_optimistically_bounded_without_velocity_sign_constraint": True,
        "target_position_or_target_slew_bounds_enforced": False,
        "floating_base_armature": model.dof_armature[:6].tolist(),
        "floating_base_damping": model.dof_damping[:6].tolist(),
        "floating_base_frictionloss_not_included_in_lp": model.dof_frictionloss[:6].tolist(),
        "dynamics_model_status": "supplied_compiled_reference_model_not_controller_matched_rollout",
        "self_contacts_and_joint_limit_forces_ignored": True,
        "force_residual_units": "mixed_generalized_force_N_and_Nm",
        "contact_complementarity_or_support_kinematics_proven": False,
        "dynamic_feasibility_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
        "rows": rows,
    }
