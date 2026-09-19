"""SIM-only full-body inverse-dynamics tracking diagnostic, not SONIC.

Only returned source-unit PD targets and explicitly selected, bounded JOINT
feedforward may enter the physical referee. Ground forces below are optimization
variables, NEVER forces applied to a simulator.
Private model margins reveal nearby candidate contacts; the integrated model
is neither changed nor stepped here. Successful QPs do not prove real contact.

MuJoCo generalized-force and tangent-space conventions:
https://mujoco.readthedocs.io/en/3.5.0/computation/index.html
https://mujoco.readthedocs.io/en/3.5.0/APIreference/APIfunctions.html
"""

from __future__ import annotations

import copy

import clarabel
import mujoco
import numpy as np
from scipy import sparse

from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF, NATIVE_IL23_ACTION_SCALE
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_RAW_ACTION_CLIP,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_native_model_actuation import native_model_pd_numpy
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_reference_support import floor_contact_map, pose_path_derivatives
from gear_sonic.utils.g1_true23_source_action_codec import (
    SOURCE_SCALE_NATIVE_IL23,
    source_scaled_precompensation,
)

KIND = "g1_true23_full_body_inverse_dynamics_counterfactual_v6"
PD_ONLY = "bounded_pd_only"
PD_FEEDFORWARD = "bounded_target_plus_feedforward_v1"
CANDIDATE_GAP_M = 0.004
POSE_KP = np.r_[np.full(3, 60.0), np.full(3, 100.0), np.full(23, 80.0)]
POSE_KD = np.r_[np.full(3, 15.0), np.full(3, 20.0), np.full(23, 18.0)]
ACCELERATION_WEIGHTS = np.r_[np.full(3, 20.0), np.full(3, 10.0), np.ones(23)]
ACCELERATION_BOUNDS = np.r_[np.full(3, 30.0), np.full(26, 80.0)]
CONTACT_KP, CONTACT_KD, CONTACT_WEIGHT = 100.0, 20.0, 100.0
FOOT_KP, FOOT_KD, FOOT_CORNER_WEIGHT = 80.0, 18.0, 25.0


def tracker_contract(strength, prefix_controls=None, actuation_mode=PD_ONLY):
    if type(strength) not in (int, float) or strength not in (0, 1):
        raise ValueError("inverse-dynamics probe permits only zero control or full controller")
    if prefix_controls is not None and (type(prefix_controls) is not int or prefix_controls != 250):
        raise ValueError("inverse-dynamics prefix is only the explicit 250-control standing smoke")
    if actuation_mode not in (PD_ONLY, PD_FEEDFORWARD):
        raise ValueError("unknown SIM-only tracker actuation mode")
    return dict(
        kind=KIND,
        strength=float(strength),
        prefix_controls=prefix_controls,
        actuation_mode=actuation_mode,
        actuator_command_law_changed=actuation_mode == PD_FEEDFORWARD and strength != 0,
        actuator_law="clip(kp*(bounded_target-q)-kd*dq+held_feedforward,-effort,effort)",
        feedforward_component_bounded_by_same_configured_effort=True,
        feedforward_held_at_50hz_and_total_torque_saturated_at_500hz=True,
        existing_sonic_previous_action_history_does_not_encode_feedforward=actuation_mode == PD_FEEDFORWARD,
        drop_in_existing_sonic_policy_or_hardware_compatibility_proven=False,
        reference="full_native_pose_q1_and_derivatives_from_received_q0_q1_q2_only",
        reference_ages_ms=[200, 180, 160],
        privileged_full_native_reference_and_simulator_state=True,
        pose_kp=POSE_KP.tolist(),
        pose_kd=POSE_KD.tolist(),
        acceleration_weights=ACCELERATION_WEIGHTS.tolist(),
        acceleration_bounds=ACCELERATION_BOUNDS.tolist(),
        contact_kp=CONTACT_KP,
        contact_kd=CONTACT_KD,
        contact_acceleration_weight=CONTACT_WEIGHT,
        foot_task="world_position_velocity_acceleration_of_all_eight_physical_sole_sphere_centers",
        foot_task_reference="received_q0_q1_q2_finite_difference_dynamics_and_exact_reference_FK_Jacobians",
        foot_kp=FOOT_KP,
        foot_kd=FOOT_KD,
        foot_corner_acceleration_weight=FOOT_CORNER_WEIGHT,
        torque_regularizer=0.001,
        contact_force_regularizer=0.000001,
        contact_force_upper_bound_bodyweights=5.0,
        qp_reparameterization="joint_torque_in_effort_units_and_contact_rays_in_bodyweight_units_same_objective_and_original_rows",
        qp_static_kkt_regularization=False,
        qp_dynamic_regularization_and_iterative_refinement=True,
        qp_requested_primal_dual_gap_tolerances=1e-9,
        qp_accepts_reduced_accuracy_status=False,
        qp_independent_original_constraint_tolerance=1e-8,
        candidate_actual_and_reference_floor_gap_m=CANDIDATE_GAP_M,
        candidate_contacts="reference_foot_minimum_gap_nominates_stance_then_each_measured_sole_contact_must_pass_actual_gap",
        reference_corner_height_does_not_exclude_actual_stance_toe_contact=True,
        contact_target="reference_world_xy_ground_z_zero_zero_contact_velocity",
        contact_acceleration_constraints="soft_penalty_not_rigid_contact_complementarity",
        contact_point_approximation="instantaneous_mujoco_contact_point_to_reference_sphere_bottom",
        force_balance="M*qdd+bias-passive=B*tau+candidate_ground_ray_map*nonnegative_weights",
        unactuated_root_generalized_actuator_force_zero=True,
        frictionloss_and_self_contact_and_joint_limit_assistance_ignored=True,
        optimized_contact_forces_never_applied_to_physics=True,
        integrated_model_never_modified=True,
        actuator_gains_limits_and_existing_projection_unchanged=True,
        commanded_torque_converted_to_50hz_held_target_with_500hz_original_pd=actuation_mode == PD_ONLY,
        midrun_state_resets_or_fallbacks=False,
        checkpoint_weights_modified=False,
        modified_controller_not_pure_sonic_policy=strength != 0,
        eligible_parent_for_training_continuation=False,
        dynamic_feasibility_proven=False,
        hardware_authorized=False,
        deployment_ready=False,
        simulator_qualified=False,
    )


def pd_plus_feedforward_numpy(target, q, dq, profile, feedforward):
    """Explicit alternate SIM command law, retaining total motor saturation.

    The feedforward is a joint torque, never a root or ground external force.
    This helper has no hardware transport. Zero feedforward is bit-exact PD.
    """
    feedforward = np.asarray(feedforward, dtype=np.float64)
    effort = np.asarray(profile.effort)
    if (
        feedforward.shape != (23,)
        or not np.isfinite(feedforward).all()
        or np.any(np.abs(feedforward) > effort + 1e-6)
    ):
        raise ValueError("feedforward requires finite bounded native23 joint torque")
    original = native_model_pd_numpy(target, q, dq, profile)
    if not np.any(feedforward):
        return original
    requested = original[0] + feedforward
    invalid = ~np.isfinite(requested).all(axis=-1)
    applied = np.where(np.expand_dims(invalid, -1), 0.0, np.clip(requested, -effort, effort))
    excess = np.maximum(np.abs(requested) / effort - 1.0, 0.0)
    cost = np.square(np.nan_to_num(excess, nan=10, posinf=10, neginf=10).clip(0, 10)).mean(axis=-1)
    return requested, applied, invalid, cost


def solve_tracker_qp(diagonal, linear, matrix, lower, upper):
    """Isolated strict backend; leave the offline reference solver unchanged.

    Static KKT regularization stalled this strictly convex contact QP above
    the requested dual tolerance. Disable that numerical perturbation, while
    retaining dynamic pivot protection/refinement and all strict tolerances.
    https://clarabel.org/stable/api_settings/
    """
    diagonal, linear, matrix, lower, upper = (
        np.asarray(value, dtype=np.float64) for value in (diagonal, linear, matrix, lower, upper)
    )
    if (
        diagonal.ndim != 1
        or not len(diagonal)
        or linear.shape != diagonal.shape
        or lower.ndim != 1
        or upper.shape != lower.shape
        or matrix.shape != (len(lower), len(diagonal))
        or any(not np.isfinite(value).all() for value in (diagonal, linear, matrix))
        or np.any(diagonal <= 0)
        or np.isnan(lower).any()
        or np.isnan(upper).any()
        or np.isposinf(lower).any()
        or np.isneginf(upper).any()
        or np.any(lower > upper)
    ):
        raise ValueError("tracker QP requires finite strictly convex data and consistent bounds")
    equality = np.isfinite(lower) & (lower == upper)
    high, low = np.isfinite(upper) & ~equality, np.isfinite(lower) & ~equality
    cone_matrix = sparse.csc_matrix(np.vstack((matrix[equality], matrix[high], -matrix[low])))
    rhs = np.r_[upper[equality], upper[high], -lower[low]]
    cones = []
    if equality.any():
        cones.append(clarabel.ZeroConeT(int(equality.sum())))
    if high.any() or low.any():
        cones.append(clarabel.NonnegativeConeT(int(high.sum() + low.sum())))
    settings = clarabel.DefaultSettings()
    settings.verbose, settings.max_iter, settings.max_threads = False, 200, 1
    settings.direct_solve_method = "qdldl"
    settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = 1e-9
    settings.static_regularization_enable = False
    answer = clarabel.DefaultSolver(
        sparse.diags(diagonal, format="csc"), linear, cone_matrix, rhs, cones, settings
    ).solve()
    report = dict(
        backend="clarabel",
        backend_version=clarabel.__version__,
        status=str(answer.status),
        iterations=int(answer.iterations),
        solve_time_s=float(answer.solve_time),
        primal_residual=float(answer.r_prim),
        dual_residual=float(answer.r_dual),
        static_regularization=False,
        requested_feasibility_and_gap_tolerance=1e-9,
        original_constraint_audit_tolerance=1e-8,
        accepted=False,
    )
    for key, value in report.items():
        if isinstance(value, float) and not np.isfinite(value):
            report[key] = None
    if answer.status != clarabel.SolverStatus.Solved or answer.x is None:
        return None, report
    solution = np.asarray(answer.x, dtype=np.float64)
    if solution.shape != diagonal.shape or not np.isfinite(solution).all():
        report["status"] = "nonfinite_or_wrong_shape_solution"
        return None, report
    actual = matrix @ solution
    violation = max(np.maximum(lower - actual, 0).max(initial=0), np.maximum(actual - upper, 0).max(initial=0))
    if not np.isfinite(violation) or violation > 1e-8:
        report["status"] = "independent_original_constraints_failed"
        return None, report
    report.update(accepted=True, independent_original_constraint_violation=float(violation))
    return solution, report


def reachable_target_bounds():
    """Intersection of the unchanged source raw and native inverse-tanh bounds."""
    order = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
    native_scale = np.empty(23)
    source_scale = np.empty(23)
    native_scale[order], source_scale[order] = NATIVE_IL23_ACTION_SCALE, SOURCE_SCALE_NATIVE_IL23
    capacities = np.asarray([SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE, SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE])
    fractions = np.minimum(np.tanh((SAFE_TARGET_RAW_ACTION_CLIP - 0.001) * native_scale / capacities), 1 - 1e-6)
    extent = np.minimum(capacities * fractions, (SAFE_TARGET_RAW_ACTION_CLIP - 0.001) * source_scale)
    # Stay inside the existing envelope despite the subsequent float32 codec.
    extent = np.maximum(extent - 2e-6, 0)
    center = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    return center - extent[0], center + extent[1]


def reference_candidate_corners(gaps):
    """Nominate a stance foot; actual collision geometry selects its corners.

    A slightly tilted reference can have raised toes that physically settle on
    the floor. Rejecting those measured toe contacts creates heel-only support
    even when the real simulator has full-sole contact. No airborne reference
    foot is nominated, and actual gap checks still reject floating corners.
    """
    gaps = np.asarray(gaps, dtype=float)
    if gaps.shape != (2, 4) or not np.isfinite(gaps).all():
        raise ValueError("inverse-dynamics reference requires eight finite sole gaps")
    minimum = gaps.min(axis=1, keepdims=True)
    return np.repeat(minimum <= CANDIDATE_GAP_M, 4, axis=1)


def desired_generalized_acceleration(model, measured_qpos, measured_qvel, received_poses):
    positions = np.asarray(received_poses, dtype=float)
    measured, velocity = np.asarray(measured_qpos, dtype=float), np.asarray(measured_qvel, dtype=float)
    if (
        positions.shape != (3, 30)
        or measured.shape != (30,)
        or velocity.shape != (29,)
        or any(not np.isfinite(value).all() for value in (positions, measured, velocity))
        or not np.allclose(np.linalg.norm(positions[:, 3:7], axis=1), 1, atol=2e-6, rtol=0)
        or not np.isclose(np.linalg.norm(measured[3:7]), 1, atol=2e-6, rtol=0)
    ):
        raise ValueError(
            "tracker requires copied finite measured state and exactly three received unit-quaternion poses"
        )
    reference_v, reference_a = pose_path_derivatives(model, positions, 0.02)
    desired_v, desired_a = reference_v[1].copy(), reference_a[1].copy()
    actual_rotation, reference_rotation = np.zeros((3, 3)), np.zeros((3, 3))
    mujoco.mju_quat2Mat(actual_rotation.reshape(9), measured[3:7])
    mujoco.mju_quat2Mat(reference_rotation.reshape(9), positions[1, 3:7])
    transport = actual_rotation.T @ reference_rotation
    desired_v[3:6] = transport @ reference_v[1, 3:6]
    desired_a[3:6] = transport @ reference_a[1, 3:6] - np.cross(velocity[3:6], desired_v[3:6])
    error = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, error, 1.0, measured, positions[1])
    return desired_a + POSE_KP * error + POSE_KD * (desired_v - velocity), reference_v[1], reference_a[1]


def foot_acceleration_tasks(model, measured, reference, reference_acceleration, sole_geoms):
    """World-space foot goals compensate root error during swing and landing.

    Tracking joint angles alone reproduces base-position error at the foot.
    These tasks may instead adjust leg posture to place each foot correctly.
    They are soft acceleration goals; no reference or physical state is moved.
    """
    jacobians, targets, errors = [], [], []
    for geom in np.asarray(sole_geoms).ravel():
        body = int(model.geom_bodyid[geom])
        point, desired_point = measured.geom_xpos[geom], reference.geom_xpos[geom]
        jacobian, derivative, desired_jacobian, desired_derivative = (np.zeros((3, 29)) for _ in range(4))
        mujoco.mj_jac(model, measured, jacobian, None, point, body)
        mujoco.mj_jacDot(model, measured, derivative, None, point, body)
        mujoco.mj_jac(model, reference, desired_jacobian, None, desired_point, body)
        mujoco.mj_jacDot(model, reference, desired_derivative, None, desired_point, body)
        error = desired_point - point
        desired_velocity = desired_jacobian @ reference.qvel
        desired_acceleration = desired_jacobian @ reference_acceleration + desired_derivative @ reference.qvel
        jacobians.append(jacobian)
        targets.append(
            desired_acceleration
            + FOOT_KP * error
            + FOOT_KD * (desired_velocity - jacobian @ measured.qvel)
            - derivative @ measured.qvel
        )
        errors.append(error)
    return np.vstack(jacobians), np.concatenate(targets), np.asarray(errors)


class InverseDynamicsTracker:
    def __init__(self, model, actuation, *, actuation_mode=PD_ONLY):
        tracker_contract(1, actuation_mode=actuation_mode)
        self.actuation_mode = actuation_mode
        if (model.nq, model.nv, model.nu, model.neq, model.ntendon) != (30, 29, 23, 0, 0):
            raise ValueError("tracker requires unmodified free-root native23 articulation")
        if not np.allclose(model.opt.gravity, [0, 0, -9.81], atol=1e-8, rtol=0):
            raise ValueError("tracker requires nominal Earth gravity")
        self.original, self.original_sha = model, compiled_model_sha256(model)
        self.model = copy.copy(model)
        self.model.geom_margin[:] = np.maximum(self.model.geom_margin, CANDIDATE_GAP_M)
        self.model.pair_margin[:] = np.maximum(self.model.pair_margin, CANDIDATE_GAP_M)
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        self.data, self.reference = mujoco.MjData(self.model), mujoco.MjData(self.model)
        planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        if len(planes) != 1:
            raise ValueError("tracker requires a single horizontal floor")
        self.plane = int(planes[0])
        self.sole_geoms = []
        for side in ("left", "right"):
            body = model.body(side + "_ankle_roll_link").id
            ids = np.flatnonzero(
                (model.geom_bodyid == body) & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
            )
            if len(ids) != 4 or not np.all(model.geom_type[ids] == mujoco.mjtGeom.mjGEOM_SPHERE):
                raise ValueError("tracker requires four physical sole spheres per foot")
            self.sole_geoms.append(ids)
        self.sole_geoms = np.asarray(self.sole_geoms)
        self.kp, self.kd, self.effort = (np.asarray(getattr(actuation, name)) for name in ("kp", "kd", "effort"))
        self.target_lower, self.target_upper = reachable_target_bounds()
        self.bodyweight = float(model.body_mass.sum() * 9.81)
        self.joint_map = np.vstack([np.zeros((6, 23)), np.eye(23)])
        # Refuse actuator mappings that would invalidate the dynamics equality.
        if not np.array_equal(model.actuator_trnid[:, 0], np.arange(1, 24)) or not np.array_equal(
            model.actuator_gear[:, 0], np.ones(23)
        ):
            raise ValueError("tracker requires unit-gear hardware-order joint torque actuators")

    def assert_original_unchanged(self):
        if compiled_model_sha256(self.original) != self.original_sha:
            raise RuntimeError("inverse-dynamics diagnostic changed original physics")

    def infer(self, measured_qpos, measured_qvel, received_poses):
        desired, reference_v, reference_a = desired_generalized_acceleration(
            self.model, measured_qpos, measured_qvel, received_poses
        )
        data, reference, model = self.data, self.reference, self.model
        data.qpos[:], data.qvel[:] = measured_qpos, measured_qvel
        reference.qpos[:], reference.qvel[:] = received_poses[1], reference_v
        for state in (data, reference):
            mujoco.mj_fwdPosition(model, state)
            mujoco.mj_fwdVelocity(model, state)
        reference_points = reference.geom_xpos[self.sole_geoms].copy()
        reference_points[:, :, 2] -= model.geom_size[self.sole_geoms, 0]
        selected = reference_candidate_corners(reference_points[:, :, 2])
        allowed = {
            int(g): reference_points[side, corner]
            for side in range(2)
            for corner, g in enumerate(self.sole_geoms[side])
            if selected[side, corner]
        }
        full_map, candidates = floor_contact_map(model, data, self.plane, CANDIDATE_GAP_M)
        maps, jacobians, accelerations, geom_ids = [], [], [], []
        for contact in candidates:
            geom = contact["robot_geom_id"]
            if geom not in allowed:
                continue
            start, count = contact["cone_ray_start"], contact["cone_ray_count"]
            maps.append(full_map[:, start : start + count])
            geom_ids.append(geom)
            point = np.asarray(contact["position_w"])
            body = int(model.geom_bodyid[geom])
            jacobian, jacobian_dot = np.zeros((3, 29)), np.zeros((3, 29))
            mujoco.mj_jac(model, data, jacobian, None, point, body)
            mujoco.mj_jacDot(model, data, jacobian_dot, None, point, body)
            target = allowed[geom].copy()
            target[2] = 0
            jacobians.append(jacobian)
            accelerations.append(
                -jacobian_dot @ data.qvel - CONTACT_KD * (jacobian @ data.qvel) - CONTACT_KP * (point - target)
            )
        force_map = np.column_stack(maps) if maps else np.zeros((29, 0))
        contact_jacobian = np.vstack(jacobians) if jacobians else np.zeros((0, 29))
        contact_acceleration = np.concatenate(accelerations) if accelerations else np.zeros(0)
        foot_jacobian, foot_acceleration, foot_error = foot_acceleration_tasks(
            model, data, reference, reference_a, self.sole_geoms
        )
        jacobian = np.vstack((contact_jacobian, foot_jacobian))
        task_acceleration = np.r_[contact_acceleration, foot_acceleration]
        task_weights = np.r_[
            np.full(len(contact_acceleration), CONTACT_WEIGHT), np.full(len(foot_acceleration), FOOT_CORNER_WEIGHT)
        ]
        rays, tasks = force_map.shape[1], len(task_acceleration)
        if rays > 32 or len(geom_ids) > 8:
            raise ValueError("unexpected native sole cone topology")
        mass = np.zeros((29, 29))
        mujoco.mj_fullM(model, mass, data.qM)
        bias = data.qfrc_bias - data.qfrc_passive
        pd_lower = self.kp * (self.target_lower - data.qpos[7:]) - self.kd * data.qvel[6:]
        pd_upper = self.kp * (self.target_upper - data.qpos[7:]) - self.kd * data.qvel[6:]
        feedforward_capacity = self.effort if self.actuation_mode == PD_FEEDFORWARD else np.zeros(23)
        torque_lower = np.maximum(-self.effort, pd_lower - feedforward_capacity)
        torque_upper = np.minimum(self.effort, pd_upper + feedforward_capacity)
        if np.any(torque_lower > torque_upper):
            raise ValueError("no PD target inside unchanged torque and target envelopes")
        # Residual variables retain a diagonal strictly convex objective.
        # x = [generalized acceleration29, joint torque23, rays, task residuals].
        size = 52 + rays + tasks
        dynamics = np.column_stack((mass, -self.joint_map, -force_map, np.zeros((29, tasks))))
        contact_rows = np.column_stack((jacobian, np.zeros((tasks, 23 + rays)), -np.eye(tasks)))
        matrix = np.vstack((dynamics, contact_rows, np.eye(size)))
        lower = np.r_[
            -bias,
            task_acceleration,
            -ACCELERATION_BOUNDS,
            torque_lower,
            np.zeros(rays),
            np.full(tasks, -np.inf),
        ]
        upper = np.r_[
            -bias,
            task_acceleration,
            ACCELERATION_BOUNDS,
            torque_upper,
            np.full(rays, 5 * self.bodyweight),
            np.full(tasks, np.inf),
        ]
        diagonal = np.r_[
            ACCELERATION_WEIGHTS,
            0.001 / self.effort**2,
            np.full(rays, 0.000001 / self.bodyweight**2),
            task_weights,
        ]
        linear = np.r_[-ACCELERATION_WEIGHTS * desired, np.zeros(size - 29)]
        # Equivalent variable substitution x = scale*y, NOT a different loss,
        # weaker solver tolerance, altered constraint, or fallback controller.
        # Raw ray costs near 1e-11 unnecessarily ill-condition the Hessian.
        scale = np.r_[np.ones(29), self.effort, np.full(rays, self.bodyweight), np.ones(tasks)]
        scaled_solution, report = solve_tracker_qp(
            diagonal * scale**2, linear * scale, matrix * scale[None, :], lower, upper
        )
        if scaled_solution is None:
            raise ValueError("inverse-dynamics QP rejected: " + str(report))
        solution = scale * scaled_solution
        actual_rows = matrix @ solution
        violation = max(
            np.maximum(lower - actual_rows, 0).max(initial=0), np.maximum(actual_rows - upper, 0).max(initial=0)
        )
        if not np.isfinite(solution).all() or not np.isfinite(violation) or violation > 1e-8:
            raise ValueError("inverse-dynamics unscaled original constraints failed strict audit")
        acceleration, torque, weights = solution[:29], solution[29:52], solution[52 : 52 + rays]
        residual = mass @ acceleration + bias - self.joint_map @ torque - force_map @ weights
        if np.max(np.abs(residual)) > 1e-5:
            raise ValueError("inverse-dynamics original generalized-force residual failed")
        unbounded_target = data.qpos[7:] + (torque + self.kd * data.qvel[6:]) / self.kp
        target = (
            np.clip(unbounded_target, self.target_lower, self.target_upper)
            if self.actuation_mode == PD_FEEDFORWARD
            else unbounded_target
        )
        order = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
        raw = ((target - SAFE_TARGET_DEFAULT_Q_HARDWARE)[order] / SOURCE_SCALE_NATIVE_IL23).astype(np.float32)
        inverse, projection = source_scaled_precompensation(raw)
        if np.max(np.abs(projection)) > 1e-6:
            raise ValueError("inverse-dynamics target unexpectedly projected by existing codec")
        feedforward = np.zeros(23)
        if self.actuation_mode == PD_FEEDFORWARD:
            # Account for exactly the same float32 target decoded by the referee.
            _, emitted_target = safe_target_transform_numpy(inverse)
            feedforward = torque - self.kp * (emitted_target - data.qpos[7:]) + self.kd * data.qvel[6:]
            if np.any(np.abs(feedforward) > self.effort + 1e-6):
                raise ValueError("joint feedforward exceeded its independently bounded capacity")
        diagnostics = dict(
            tracker_raw23=raw.copy(),
            tracker_target23=target,
            tracker_unbounded_pd_target23=unbounded_target,
            tracker_feedforward_torque23=feedforward,
            tracker_torque23=torque,
            tracker_qdd29=acceleration,
            tracker_desired_qdd29=desired,
            tracker_force_residual29=residual,
            tracker_ray_weights32=np.pad(weights, (0, 32 - rays)),
            tracker_contact_geom8=np.pad(geom_ids, (0, 8 - len(geom_ids)), constant_values=-1),
            tracker_contact_count1=np.asarray([len(geom_ids)]),
            tracker_ray_count1=np.asarray([rays]),
            tracker_qp_solve_seconds1=np.asarray([report["solve_time_s"]]),
            tracker_original_constraint_violation1=np.asarray([violation]),
            tracker_foot_world_error8x3=foot_error,
            tracker_foot_task_acceleration_error24=foot_jacobian @ acceleration - foot_acceleration,
        )
        return raw, diagnostics
