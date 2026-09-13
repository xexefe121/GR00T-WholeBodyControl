"""Bounded whole-path contact conditioning on unchanged native23 physics.

This explicitly changes kinematic reference targets; it does not inherit raw
original29 fidelity or constitute a policy. Full source timing remains fixed.
Pelvis translation, twelve legs and ten arms are optimized together so ground,
self-clearance and temporal bounds cannot be independently contradicted.
"""

from dataclasses import asdict

import mujoco
import numpy as np
from scipy import sparse

from gear_sonic.scripts.audit_g1_true23_reference_bank_self_contacts import measure_self_contacts
from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_trajectory_projection import _constraint_system, audit_trajectory_constraints
from gear_sonic.utils.g1_true23_affine_task_soc import solve_affine_task_soc
from gear_sonic.utils.g1_true23_collision_sampling import interpolate_original_poses, interpolation_weights
from gear_sonic.utils.g1_true23_self_collision import SelfCollisionLinearizer
from gear_sonic.utils.g1_true23_stance_foot_cleanup import (
    bounded_stance_targets,
    flat_sole_height,
    foot_frames,
    foot_task_residual_jacobian,
)


def solve_elastic_task_step(
    residual, jacobian, matrix, lower, upper, groups, collision_jacobian, distances, *, objective_scale=1.0
):
    n, c = jacobian.shape[1], len(distances)
    constraint = sparse.vstack(
        (
            sparse.hstack((matrix, sparse.csc_matrix((matrix.shape[0], c)))),
            sparse.hstack((collision_jacobian, sparse.eye(c))),
            sparse.hstack((sparse.csc_matrix((c, n)), sparse.eye(c))),
        ),
        format="csc",
    )
    delta, report = solve_affine_task_soc(
        np.r_[np.full(n, 0.001), np.full(c, 1e5)],
        np.zeros(n + c),
        residual,
        sparse.hstack((jacobian, sparse.csc_matrix((len(residual), c))), format="csc"),
        constraint,
        np.r_[lower, 0.001 - distances, np.zeros(c)],
        np.r_[upper, np.full(2 * c, np.inf)],
        groups,
        objective_profile="task_lsq_v1",
        objective_scale=objective_scale,
    )
    return (None if delta is None else delta[:n]), report


class ContactConditionedPath:
    def __init__(self, model, poses, *, initial_joint_velocity=None):
        self.model, self.data = model, mujoco.MjData(model)
        self.original = np.array(poses, dtype=float, copy=True)
        if (
            self.original.ndim != 2
            or self.original.shape[1] != 30
            or len(poses) < 3
            or not np.isfinite(poses).all()
        ):
            raise ValueError("contact path requires full finite native23 poses")
        # Source-only clips may begin in motion. An explicit boundary is not
        # evidence that a robot can acquire that state from standing. Keep the
        # historical zero-velocity boundary unless the caller declares one.
        self.moving_source_boundary_declared = initial_joint_velocity is not None
        self.initial_joint_velocity = (
            np.zeros(23) if initial_joint_velocity is None else np.array(initial_joint_velocity, copy=True)
        )
        if (
            self.initial_joint_velocity.shape != (23,)
            or self.initial_joint_velocity.dtype.kind not in "fiu"
            or not np.isfinite(self.initial_joint_velocity).all()
            or np.any(np.abs(self.initial_joint_velocity) > 5)
        ):
            raise ValueError("initial joint velocity requires 23 finite values within existing +/-5 rad/s bounds")
        self.initial_joint_velocity = self.initial_joint_velocity.astype(float)
        self.count = len(poses)
        times = np.arange(self.count)
        self.weights = interpolation_weights(times, np.arange(2 * self.count - 1) / 2)
        self.template = interpolate_original_poses(poses, times, np.arange(2 * self.count - 1) / 2)
        self.mapping = sparse.kron(self.weights, sparse.eye(26), format="csc")
        self.columns = np.r_[0:3, 6:29]
        self.feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
        tasks = {task.name: task for task in ik.DEFAULT_TASKS}
        selected = [tasks[name] for name in ("left_hand", "right_hand", "head_proxy")]
        self.uppers = [model.body(task.target_body).id for task in selected]
        self.upper_offsets = [np.asarray(task.target_point) for task in selected]
        self.geoms = [
            int(g)
            for body in self.feet
            for g in np.flatnonzero(model.geom_bodyid == body)
            if model.geom_contype[g] or model.geom_conaffinity[g]
        ]
        if len(self.geoms) != 8:
            raise ValueError("contact path requires exactly eight unchanged sole spheres")
        positions, rotations = foot_frames(model, poses)
        self.contacts = ik.infer_foot_contacts(
            positions, fps=50, height_tolerance_m=0.035, speed_tolerance_m_s=0.45
        )
        # A half sample inherits stance only if both adjacent controls agree.
        self.query_contacts = self.weights @ self.contacts.astype(float) > 1 - 1e-10
        self.reference_positions, self.reference_rotations = foot_frames(model, self.template)
        self.target_pos, self.target_rot, self.target_proof = bounded_stance_targets(
            self.reference_positions,
            self.reference_rotations,
            self.query_contacts,
            [flat_sole_height(model, body) for body in self.feet],
        )
        self.upper_positions, self.selected_points = [], []
        for frame, pose in enumerate(self.template):
            self.data.qpos[:] = pose
            mujoco.mj_forward(model, self.data)
            self.upper_positions.append(
                np.asarray(
                    [
                        self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ offset
                        for body, offset in zip(self.uppers, self.upper_offsets, strict=True)
                    ]
                )
            )
            for side, body in enumerate(self.feet):
                geoms = self.geoms[4 * side : 4 * side + 4]
                # Preserve the actual nearest support feature. Flattened targets
                # have tied corner heights and can arbitrarily choose a raised
                # heel instead of the original toe, creating a false conflict.
                original_gaps = self.data.geom_xpos[geoms, 2] - model.geom_size[geoms, 0]
                if self.query_contacts[frame, side]:
                    self.selected_points.append(8 * frame + 4 * side + int(np.argmin(original_gaps)))
        self.upper_positions = np.asarray(self.upper_positions)
        self.selected_points = np.asarray(self.selected_points, dtype=int)
        self.query = SelfCollisionLinearizer(model)
        self.initial = np.column_stack((np.zeros((self.count, 3)), poses[:, 7:]))
        low, high = ik.safe_target_joint_bounds(model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
        capacity = np.r_[np.full(12, 0.2), 0.0, np.full(10, 0.3)]
        self.lower = np.column_stack((np.full((self.count, 3), -0.03), np.maximum(low, poses[:, 7:] - capacity)))
        self.upper = np.column_stack((np.full((self.count, 3), 0.03), np.minimum(high, poses[:, 7:] + capacity)))
        # Waist yaw is unchanged; missing waist axes are not invented.
        self.velocity = np.r_[np.full(3, 0.75), np.full(23, 5.0)]
        self.acceleration = np.r_[np.full(3, 0.5), np.full(23, 80.0)]
        operator, self.linear_low, self.linear_high = _constraint_system(
            self.count,
            self.lower,
            self.upper,
            self.velocity * 0.02 * 0.995,
            self.acceleration * 0.02**2 * 0.995,
            np.r_[np.zeros(3), self.initial_joint_velocity] * 0.02,
        )
        self.operator = sparse.kron(operator, sparse.eye(26), format="csc")
        self.linear_low, self.linear_high = self.linear_low.ravel(), self.linear_high.ravel()
        self.norm_groups = []
        width = 33  # two desired feet, two original feet, three upper positions
        for frame in range(len(self.template)):
            offset = width * frame
            for side in range(2):
                base = offset + 12 + 6 * side
                self.norm_groups.extend(
                    (
                        dict(indices=np.arange(base, base + 3), scales=np.full(3, 0.01), radius=0.0195),
                        dict(indices=np.arange(base + 3, base + 6), scales=np.full(3, 0.1), radius=0.1495),
                    )
                )
            for upper in range(3):
                self.norm_groups.append(
                    dict(
                        indices=np.arange(offset + 24 + 3 * upper, offset + 27 + 3 * upper),
                        scales=np.full(3, 0.05),
                        radius=0.0495,
                    )
                )

    def poses(self, variables):
        result = self.original.copy()
        result[:, :3] += variables[:, :3]
        result[:, 7:] = variables[:, 3:]
        return result

    def evaluate(self, variables):
        controls = self.poses(variables)
        samples = self.template.copy()
        samples[:, :3] += self.weights @ variables[:, :3]
        samples[:, 7:] = self.weights @ variables[:, 3:]
        residual, blocks, gaps, floor_blocks = [], [], [], []
        for frame, pose in enumerate(samples):
            self.data.qpos[:] = pose
            mujoco.mj_forward(self.model, self.data)
            r, j = [], []
            for positions, rotations in (
                (self.target_pos, self.target_rot),
                (self.reference_positions, self.reference_rotations),
            ):
                for side, body in enumerate(self.feet):
                    error, jac = foot_task_residual_jacobian(
                        self.model, self.data, body, positions[frame, side], rotations[frame, side], self.columns
                    )
                    r.extend(error)
                    j.extend(jac)
            for index, body in enumerate(self.uppers):
                jac = np.zeros((3, self.model.nv))
                point = self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ self.upper_offsets[index]
                mujoco.mj_jac(self.model, self.data, jac, None, point, body)
                r.extend(20 * (point - self.upper_positions[frame, index]))
                j.extend(20 * jac[:, self.columns])
            residual.extend(r)
            blocks.append(sparse.csc_matrix(j))
            floor = []
            for geom in self.geoms:
                jac = np.zeros((3, self.model.nv))
                mujoco.mj_jacGeom(self.model, self.data, jac, None, geom)
                gaps.append(self.data.geom_xpos[geom, 2] - self.model.geom_size[geom, 0])
                floor.append(jac[2, self.columns])
            floor_blocks.append(sparse.csc_matrix(floor))
        # A smooth offset cannot add high-frequency acceleration to an otherwise
        # slow reference. The hard0.5m/s2 bound is tighter than the old6m/s2 cap.
        residual = np.asarray(residual)
        jacobian = sparse.block_diag(blocks, format="csc") @ self.mapping
        gaps = np.asarray(gaps)
        floor_jacobian = sparse.block_diag(floor_blocks, format="csc") @ self.mapping
        collision_jacobian, distances, _ = self.query.path_rows(samples, self.columns)
        shortfall = np.maximum(0.001 - distances, 0)
        lower_defect = np.maximum(0.0001 - gaps, 0)
        upper_defect = np.maximum(gaps[self.selected_points] - 0.001, 0)
        merit = float(shortfall @ shortfall + lower_defect @ lower_defect + upper_defect @ upper_defect)
        return dict(
            controls=controls,
            samples=samples,
            residual=residual,
            jacobian=jacobian,
            gaps=gaps,
            floor_jacobian=floor_jacobian,
            collision_jacobian=collision_jacobian @ self.mapping,
            distances=distances,
            merit=merit,
        )

    def correction_audit(self, values, state):
        error = state["residual"].reshape(-1, 33)
        feet_position = np.stack(
            [np.linalg.norm(error[:, offset : offset + 3], axis=1) / 100 for offset in (12, 18)], axis=1
        )
        feet_rotation = np.stack(
            [np.linalg.norm(error[:, offset : offset + 3], axis=1) / 10 for offset in (15, 21)], axis=1
        )
        upper = np.stack(
            [np.linalg.norm(error[:, offset : offset + 3], axis=1) / 20 for offset in (24, 27, 30)], axis=1
        )
        linear = self.operator @ values.ravel()
        violation = float(
            max(np.maximum(self.linear_low - linear, 0).max(), np.maximum(linear - self.linear_high, 0).max())
        )
        return dict(
            passed=bool(
                violation <= 2e-7
                and feet_position.max() <= 0.02
                and feet_rotation.max() <= 0.15
                and upper.max() <= 0.05
            ),
            linear_violation=violation,
            maximum_ankle_translation_m=float(feet_position.max()),
            maximum_ankle_rotation_rad=float(feet_rotation.max()),
            maximum_upper_position_change_m=float(upper.max()),
        )


def _select_contact_step(problem, state, values, matrix, lower, upper, trust):
    """Retry a stalled nonlinear step in smaller boxes; final bounds never change."""
    trials, best, best_row = [], None, None
    for factor in (1.0, 0.1, 0.01):
        low, high = lower.copy(), upper.copy()
        start = problem.operator.shape[0]
        low[start : start + values.size] = -factor * trust
        high[start : start + values.size] = factor * trust
        step, solve = solve_elastic_task_step(
            state["residual"],
            state["jacobian"],
            matrix,
            low,
            high,
            problem.norm_groups,
            state["collision_jacobian"],
            state["distances"],
            # A common positive scalar preserves all task/collision weight
            # ratios. Independent original-row tolerances stay unchanged.
            objective_scale=1 / problem.count,
        )
        trial = dict(trust_region_scale=factor, solver=solve, candidate_found=False)
        trials.append(trial)
        if step is not None:
            if step.shape != (values.size,) or not np.isfinite(step).all():
                raise ValueError("contact step must be a finite complete path correction")
            if np.any(np.abs(step) > factor * trust + 1e-8):
                raise ValueError("contact step exceeds its independently checked intermediate trust box")
            for fraction in (1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
                candidate = values + fraction * step.reshape(values.shape)
                candidate[:, 15] = problem.initial[:, 15]
                after = problem.evaluate(candidate)
                audit = problem.correction_audit(candidate, after)
                if audit["passed"] and after["merit"] < state["merit"] - 1e-12:
                    trial.update(candidate_found=True, fraction=fraction, merit_after=after["merit"])
                    if best_row is None or after["merit"] < best_row["merit_after"]:
                        best, best_row = (
                            candidate,
                            dict(
                                accepted=True,
                                solver=solve,
                                fraction=fraction,
                                merit_after=after["merit"],
                                correction_audit=audit,
                                trust_region_scale=factor,
                                selected_trust_trial=len(trials) - 1,
                            ),
                        )
                    break
        if factor == 1 and (step is None or best is not None):
            # Preserve successful normal steps and infeasible/no-step behavior.
            # Smaller boxes cannot establish feasibility of an infeasible QP.
            break
    if best_row is None:
        best_row = dict(accepted=False, solver=trials[-1]["solver"], selected_trust_trial=None)
    return best, {**best_row, "trust_region_trials": trials}


def fit_contact_conditioned_path(model, poses, *, progress=None, initial_joint_velocity=None):
    problem = ContactConditionedPath(model, poses, initial_joint_velocity=initial_joint_velocity)
    values = problem.initial.copy()
    iterations = []
    for iteration in range(32):
        state = problem.evaluate(values)
        audit = problem.correction_audit(values, state)
        collisions = measure_self_contacts(model, state["samples"])
        if (
            audit["passed"]
            and not collisions["frames_with_robot_robot_penetration"]
            and state["gaps"].min() >= -2e-7
            and state["gaps"][problem.selected_points].max(initial=0) <= 0.002
        ):
            break
        linear = problem.operator @ values.ravel()
        # Existing absolute correction boxes still apply. Root translation is
        # exactly linear; nonlinear joint steps are checked by full line search.
        trust = np.tile(np.r_[np.full(3, 0.03), np.full(23, 0.3)], problem.count)
        matrix = sparse.vstack(
            (
                problem.operator,
                sparse.eye(values.size),
                state["floor_jacobian"],
                state["floor_jacobian"][problem.selected_points],
            ),
            format="csc",
        )
        low = np.r_[
            problem.linear_low - linear,
            -trust,
            0.0001 - state["gaps"],
            np.full(len(problem.selected_points), -np.inf),
        ]
        high = np.r_[
            problem.linear_high - linear,
            trust,
            np.full(len(state["gaps"]), np.inf),
            0.001 - state["gaps"][problem.selected_points],
        ]
        candidate, selected = _select_contact_step(problem, state, values, matrix, low, high, trust)
        row = dict(iteration=iteration, merit_before=state["merit"], **selected)
        if candidate is not None:
            values = candidate
        iterations.append(row)
        if progress:
            progress(
                {k: v for k, v in row.items() if k not in ("solver", "trust_region_trials")}
                | {"solver_status": row["solver"]["status"], "trust_trials": len(row["trust_region_trials"])}
            )
        if not row["accepted"]:
            break
    state = problem.evaluate(values)
    audit = problem.correction_audit(values, state)
    collisions = measure_self_contacts(model, state["samples"])
    low, high = ik.safe_target_joint_bounds(model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    bounds = audit_trajectory_constraints(
        state["controls"][:, 7:],
        lower_bounds=low,
        upper_bounds=high,
        dt=0.02,
        max_velocity=5,
        max_acceleration=80,
        initial_velocity=problem.initial_joint_velocity,
        tolerance=2e-7,
    )
    return (
        state["controls"],
        problem.contacts,
        dict(
            kind=(
                "g1_true23_joint_contact_clearance_moving_source_boundary_v2"
                if problem.moving_source_boundary_declared
                else "g1_true23_joint_contact_clearance_whole_path_hypothesis_v1"
            ),
            initial_joint_velocity_rad_s=problem.initial_joint_velocity.tolist(),
            numerical_profile="objective_mean_over_control_frames_v2",
            objective_normalization=1 / problem.count,
            step_selection_profile="stalled_nonlinear_step_trust_backoff_v1",
            fallback_trust_region_scales=[0.1, 0.01],
            final_path_or_collision_acceptance_relaxed=False,
            relative_task_weights_and_constraints_changed=False,
            initial_velocity_boundary=(
                "explicit_source_velocity_not_standing_acquisition"
                if problem.moving_source_boundary_declared
                else "stationary"
            ),
            standing_acquisition_or_velocity_matching_proven=False,
            iterations=iterations,
            target_proof=problem.target_proof,
            selected_contact_feature="nearest_original_native_sole_sphere_v2",
            correction_bounds_passed=audit["passed"],
            correction_audit=audit,
            maximum_ankle_translation_m=audit["maximum_ankle_translation_m"],
            maximum_ankle_rotation_rad=audit["maximum_ankle_rotation_rad"],
            bounds=asdict(bounds),
            root_translation_enabled=True,
            maximum_joint_change_rad=float(np.max(np.abs(state["controls"][:, 7:] - np.asarray(poses)[:, 7:]))),
            source_frames_removed=0,
            source_retimed=False,
            final_100hz_self_contacts=collisions,
            minimum_sole_gap_m=float(state["gaps"].min()),
            maximum_selected_contact_gap_m=float(state["gaps"][problem.selected_points].max(initial=0)),
            original_root_attitude_bit_exact=np.array_equal(state["controls"][:, 3:7], np.asarray(poses)[:, 3:7]),
            dynamic_feasibility_proven=False,
            training_reference_accepted=False,
            hardware_authorized=False,
            deployment_ready=False,
        ),
    )
