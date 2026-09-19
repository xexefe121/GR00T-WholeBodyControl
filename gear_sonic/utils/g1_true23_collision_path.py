"""Bounded offline contact-clearance restoration with existing task constraints.

Only numerical intermediate iterates may have contact/task residual. No iterate
is an accepted reference until the independent full serialized control-grid and
all-original-timestamp tests pass. This is not dynamics or live-control code.
"""

from itertools import product

import numpy as np
from scipy import sparse

from gear_sonic.utils.g1_23dof_trajectory_projection import _constraint_system
from gear_sonic.utils.g1_true23_collision_sampling import CollisionPathSampler
from gear_sonic.utils.g1_true23_generalist_protected_root import audit_norms, residual_groups, solve_box_soc
from gear_sonic.utils.g1_true23_self_collision import SelfCollisionLinearizer


def collision_line_search_fractions(profile="four_step_v1"):
    if profile == "four_step_v1":
        return (1.0, 0.5, 0.25, 0.125)
    if profile == "extended_bisection_v2":
        # Smaller numerical trials address linearization curvature near a hard
        # task boundary. They do not enlarge any physical or acceptance bound.
        return tuple(2.0**-i for i in range(8))
    raise ValueError("unknown collision line search profile")


def clearance_merit(distances, profile="maximum_clearance_v1"):
    values = np.asarray(distances, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("collision merit requires finite signed distances")
    violation = np.maximum(0.001 - values, 0)
    if profile == "maximum_clearance_v1":
        return float(violation.max(initial=0))
    if profile == "sum_squared_clearance_v2":
        return float(violation @ violation)
    raise ValueError("unknown collision merit profile")


def clearance_step_targets(
    distances, *, clearance_m=0.001, restoration_step_m=0.005, strategy="all_contacts_step_v1"
):
    values = np.asarray(distances, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("contact distances must be a finite vector")
    if not 0 < clearance_m <= 0.005 or not 0 < restoration_step_m <= 0.01:
        raise ValueError("clearance and numerical restoration step must be bounded and positive")
    if strategy == "worst_contact_first_v2":
        # Improve the deepest distance without forcing every smaller contact
        # to improve in this same numerical step. Final acceptance is unchanged.
        worst_floor = float(values.min(initial=clearance_m)) + restoration_step_m
        return np.minimum(clearance_m, np.maximum(values, worst_floor))
    if strategy != "all_contacts_step_v1":
        raise ValueError("unknown collision restoration strategy")
    # Negative intermediate distances are not accepted physical trajectories.
    return np.minimum(clearance_m, values + restoration_step_m)


def collision_step(
    problem,
    current,
    residual,
    jacobian,
    groups,
    contact_jacobian,
    distances,
    step_m,
    strategy="all_contacts_step_v1",
    solver_form="augmented_residual_v1",
    objective_profile="task_lsq_v1",
    objective_target=None,
):
    if objective_profile not in ("task_lsq_v1", "minimum_change_v2", "pose_escape_guided_v3"):
        raise ValueError("unknown collision objective profile")
    if objective_profile != "task_lsq_v1" and solver_form != "eliminated_residual_v2":
        raise ValueError("minimum-change collision objective requires eliminated residual formulation")
    if objective_profile == "pose_escape_guided_v3":
        if (
            objective_target is None
            or np.shape(objective_target) != current.shape
            or not np.isfinite(objective_target).all()
        ):
            raise ValueError("pose-guided objective requires an explicit finite full-path objective target")
    elif objective_target is not None:
        raise ValueError("objective guidance cannot be silently applied to another objective")
    cfg = problem.config
    trust = np.repeat([cfg.root_trust_m, cfg.rotation_trust_rad, cfg.joint_trust_rad], [3, 3, 23])
    lo, hi = np.maximum(problem.lower, current - trust), np.minimum(problem.upper, current + trust)
    operator, lower, upper = _constraint_system(
        len(current),
        lo,
        hi,
        problem.velocity * cfg.serialization_margin_fraction * 0.02,
        problem.acceleration * cfg.serialization_margin_fraction * 0.02**2,
        problem.initial_velocity * 0.02,
    )
    temporal = sparse.kron(operator, sparse.eye(29), format="csc")
    facets = np.zeros((8, 29))
    facets[:, 3:6] = list(product((-1.0, 1.0), repeat=3))
    rotation = sparse.kron(sparse.eye(len(current)), sparse.csc_matrix(facets), format="csc")
    n, m = current.size, len(residual)
    if strategy == "elastic_squared_clearance_v3":
        from gear_sonic.utils.g1_true23_elastic_collision_step import solve_elastic_collision_step

        if solver_form != "eliminated_residual_v2" or objective_profile == "task_lsq_v1":
            raise ValueError("elastic clearance requires eliminated residuals and a movement objective")
        matrix = sparse.vstack((temporal, rotation), format="csc")
        absolute_lower = np.r_[lower.ravel(), np.full(rotation.shape[0], -np.inf)]
        absolute_upper = np.r_[upper.ravel(), np.full(rotation.shape[0], cfg.maximum_root_rotation_l1_rad)]
        origin = matrix @ current.ravel()
        step, report = solve_elastic_collision_step(
            problem.posture_weight,
            (current - (objective_target if objective_target is not None else problem.initial)).ravel(),
            residual,
            jacobian,
            matrix,
            absolute_lower - origin,
            absolute_upper - origin,
            [{**group, "radius": group["radius"] * cfg.serialization_margin_fraction} for group in groups],
            contact_jacobian,
            distances,
            objective_scale=1 / len(current),
            objective_profile=objective_profile,
        )
        if step is None:
            return None, report
        candidate = current + step.reshape(current.shape)
        actual = matrix @ candidate.ravel()
        violation = float(
            max(
                np.maximum(absolute_lower - actual, 0).max(initial=0),
                np.maximum(actual - absolute_upper, 0).max(initial=0),
            )
        )
        report["independent_absolute_noncontact_row_violation"] = violation
        if not np.isfinite(violation) or violation > 1e-8:
            report.update(accepted=False, status="elastic_absolute_noncontact_row_audit_failed")
            return None, report
        return candidate, report
    floor = clearance_step_targets(distances, restoration_step_m=step_m, strategy=strategy)
    if solver_form == "eliminated_residual_v2":
        from gear_sonic.utils.g1_true23_affine_task_soc import solve_affine_task_soc

        matrix = sparse.vstack((temporal, rotation, contact_jacobian), format="csc")
        absolute_lower = np.r_[
            lower.ravel(),
            np.full(rotation.shape[0], -np.inf),
            floor - distances + contact_jacobian @ current.ravel(),
        ]
        absolute_upper = np.r_[
            upper.ravel(),
            np.full(rotation.shape[0], cfg.maximum_root_rotation_l1_rad),
            np.full(len(distances), np.inf),
        ]
        origin = matrix @ current.ravel()
        step, report = solve_affine_task_soc(
            problem.posture_weight,
            (current - (objective_target if objective_target is not None else problem.initial)).ravel(),
            residual,
            jacobian,
            matrix,
            absolute_lower - origin,
            absolute_upper - origin,
            [{**group, "radius": group["radius"] * cfg.serialization_margin_fraction} for group in groups],
            objective_scale=1 / len(current),
            objective_profile=objective_profile,
        )
        report.update(contact_constraint_count=len(distances), restoration_step_m=step_m)
        if step is None:
            return None, report
        candidate = current + step.reshape(current.shape)
        actual = matrix @ candidate.ravel()
        violation = float(
            max(
                np.maximum(absolute_lower - actual, 0).max(initial=0),
                np.maximum(actual - absolute_upper, 0).max(initial=0),
            )
        )
        report["independent_absolute_row_violation"] = violation
        if not np.isfinite(violation) or violation > 1e-8:
            report.update(accepted=False, status="collision_absolute_row_audit_failed")
            return None, report
        return candidate, report
    if solver_form != "augmented_residual_v1":
        raise ValueError("unknown collision solver formulation")
    matrix = sparse.vstack(
        (
            sparse.hstack((temporal, sparse.csc_matrix((temporal.shape[0], m)))),
            sparse.hstack((rotation, sparse.csc_matrix((rotation.shape[0], m)))),
            sparse.hstack((jacobian, -sparse.eye(m))),
            sparse.hstack((contact_jacobian, sparse.csc_matrix((len(distances), m)))),
        ),
        format="csc",
    )
    floor = clearance_step_targets(distances, restoration_step_m=step_m, strategy=strategy)
    absolute_lower = np.r_[
        lower.ravel(),
        np.full(rotation.shape[0], -np.inf),
        jacobian @ current.ravel() - residual,
        floor - distances + contact_jacobian @ current.ravel(),
    ]
    absolute_upper = np.r_[
        upper.ravel(),
        np.full(rotation.shape[0], cfg.maximum_root_rotation_l1_rad),
        jacobian @ current.ravel() - residual,
        np.full(len(distances), np.inf),
    ]
    origin = np.r_[
        temporal @ current.ravel(),
        rotation @ current.ravel(),
        jacobian @ current.ravel(),
        contact_jacobian @ current.ravel(),
    ]
    scale = 1 / len(current)
    answer, report = solve_box_soc(
        scale * np.r_[problem.posture_weight, np.ones(m)],
        scale * np.r_[problem.posture_weight * (current - problem.initial).ravel(), np.zeros(m)],
        matrix,
        absolute_lower - origin,
        absolute_upper - origin,
        [
            {
                **group,
                "indices": group["indices"] + n,
                "radius": group["radius"] * cfg.serialization_margin_fraction,
            }
            for group in groups
        ],
        solver_tolerance=1e-11,
        accept_independently_feasible_inaccurate=True,
    )
    report.update(contact_constraint_count=len(distances), restoration_step_m=step_m)
    if answer is None:
        return None, report
    candidate = current + answer[:n].reshape(current.shape)
    actual = matrix @ np.r_[candidate.ravel(), answer[n:]]
    violation = float(
        max(
            np.maximum(absolute_lower - actual, 0).max(initial=0),
            np.maximum(actual - absolute_upper, 0).max(initial=0),
        )
    )
    report["independent_absolute_row_violation"] = violation
    if not np.isfinite(violation) or violation > 1e-8:
        report.update(accepted=False, status="collision_absolute_row_audit_failed")
        return None, report
    return candidate, report


def fit_collision_clearance_path(
    problem,
    initial,
    baseline,
    physical_model,
    *,
    progress=None,
    strategy="all_contacts_step_v1",
    solver_form="augmented_residual_v1",
    objective_profile="task_lsq_v1",
    control_source_times=None,
    original_times=None,
    original_time_tasks=None,
    objective_target=None,
    collision_merit_profile="maximum_clearance_v1",
    upper_landmark_constraints=None,
    line_search_profile="four_step_v1",
):
    fractions = collision_line_search_fractions(line_search_profile)
    if strategy == "elastic_squared_clearance_v3" and (
        solver_form != "eliminated_residual_v2"
        or objective_profile not in ("minimum_change_v2", "pose_escape_guided_v3")
        or collision_merit_profile != "sum_squared_clearance_v2"
    ):
        raise ValueError("elastic clearance requires explicit eliminated movement objective and squared merit")
    current = np.array(initial, dtype=float, copy=True)
    if not problem.audit(current)["passed"]:
        raise ValueError("collision fit must start from the intact temporally feasible full path")
    query = SelfCollisionLinearizer(physical_model)
    groups, _ = residual_groups(problem, baseline)
    if upper_landmark_constraints is not None:
        groups.extend(upper_landmark_constraints)
    if original_time_tasks is not None:
        residual_count = len(problem.evaluate(current, derivatives=False)[0])
        groups.extend(
            {**group, "indices": group["indices"] + residual_count} for group in original_time_tasks.groups
        )

    def task_rows(variables, *, derivatives=True):
        residual, jacobian, _ = problem.evaluate(variables, derivatives=derivatives)
        if original_time_tasks is not None:
            extra_residual, extra_jacobian = original_time_tasks.evaluate(variables, derivatives=derivatives)
            residual = np.r_[residual, extra_residual]
            if derivatives:
                jacobian = sparse.vstack((jacobian, extra_jacobian), format="csc")
        return residual, jacobian

    # Rigid whole-robot motion cannot change self-contact distance. Map only
    # joint derivatives; the six root-reference variable columns remain zero.
    joint_selector = sparse.kron(
        sparse.eye(len(current)), sparse.hstack((sparse.csc_matrix((23, 6)), sparse.eye(23))), format="csc"
    )
    if (control_source_times is None) != (original_times is None):
        raise ValueError("collision sampling requires both control-source and original timestamps")
    sampler = (
        CollisionPathSampler(control_source_times, original_times) if control_source_times is not None else None
    )

    def contact_rows(variables):
        poses = problem.qpos(variables)
        if sampler is not None:
            return sampler.path_rows(query, poses, problem.layout.dof_addresses)
        jac, distances, identities = query.path_rows(poses, problem.layout.dof_addresses)
        return jac @ joint_selector, distances, identities

    history, failure = [], None
    for iteration in range(problem.config.maximum_iterations):
        residual, jacobian = task_rows(current)
        protected = audit_norms(residual, groups)
        contact_jac, distance, _ = contact_rows(current)
        penetration = float(np.maximum(0.001 - distance, 0).max(initial=0))
        before_merit = clearance_merit(distance, collision_merit_profile)
        if penetration == 0 and protected["passed"]:
            break
        row = {
            "iteration": iteration + 1,
            "before_clearance_violation_m": penetration,
            "before_collision_merit": before_merit,
            "before_protected_normalized_excess": protected["maximum_normalized_excess"],
            "accepted": False,
            "attempts": [],
            "line_search_trials": [],
        }
        # A bounded second attempt addresses an overly large numerical step,
        # not a weakened final clearance or tracking gate.
        for step_m in (0.005,) if strategy == "elastic_squared_clearance_v3" else (0.005, 0.001):
            candidate, solver = collision_step(
                problem,
                current,
                residual,
                jacobian,
                groups,
                contact_jac,
                distance,
                step_m,
                strategy,
                solver_form,
                objective_profile,
                objective_target,
            )
            row["attempts"].append(solver)
            if candidate is None:
                continue
            for fraction in fractions:
                trial = current + fraction * (candidate - current)
                path_passed = problem.audit(trial)["passed"]
                trial_report = {"attempt": len(row["attempts"]), "fraction": fraction, "path_passed": path_passed}
                row["line_search_trials"].append(trial_report)
                if not path_passed:
                    continue
                trial_residual, _ = task_rows(trial, derivatives=False)
                trial_protected = audit_norms(trial_residual, groups)
                _, trial_distance, trial_ids = contact_rows(trial)
                trial_penetration = float(np.maximum(0.001 - trial_distance, 0).max(initial=0))
                trial_report.update(
                    clearance_violation_m=trial_penetration,
                    sum_squared_clearance_violation_m2=float(
                        np.square(np.maximum(0.001 - trial_distance, 0)).sum()
                    ),
                    worst_query_frame_and_geom_ids=list(trial_ids[int(np.argmin(trial_distance))])
                    if len(trial_distance)
                    else None,
                    protected_normalized_excess=trial_protected["maximum_normalized_excess"],
                    failed_protected_categories={
                        name: {
                            "count": len(detail["failed_frames"]),
                            "first_frame_indices": detail["failed_frames"][:8],
                            "maximum_excess": detail["maximum_excess"],
                        }
                        for name, detail in trial_protected["categories"].items()
                        if detail["failed_frames"]
                    },
                )
                # Numerical intermediates may have <=0.1% normalized task
                # residual. This is NOT an acceptance tolerance: final FK
                # gates remain exact and are independently rerun after save.
                protected_progress = trial_protected["maximum_normalized_excess"] <= max(
                    0.001, protected["maximum_normalized_excess"]
                )
                trial_merit = clearance_merit(trial_distance, collision_merit_profile)
                collision_progress = trial_merit < before_merit - (
                    1e-8 if collision_merit_profile == "maximum_clearance_v1" else 1e-8 * max(before_merit, 1e-12)
                )
                final_restore = (
                    # A task-restoration numerical step may spend positive
                    # query clearance margin, but not introduce query overlap.
                    # Final exact-physical collision audits remain authoritative.
                    penetration <= 0.001
                    and trial_penetration <= 0.001
                    and trial_protected["maximum_normalized_excess"] < protected["maximum_normalized_excess"]
                )
                if protected_progress and (collision_progress or final_restore):
                    current = trial
                    row.update(
                        accepted=True,
                        fraction=fraction,
                        after_clearance_violation_m=trial_penetration,
                        after_collision_merit=trial_merit,
                        after_protected_normalized_excess=trial_protected["maximum_normalized_excess"],
                    )
                    break
            if row["accepted"]:
                break
        history.append(row)
        if progress:
            progress(row)
        if not row["accepted"]:
            failure = "bounded collision restoration stalled; not a physical infeasibility certificate"
            break
    return current, {
        "kind": "g1_true23_existing_task_and_self_collision_path_restoration_v1",
        "restoration_strategy": strategy,
        "solver_form": solver_form,
        "objective_profile": objective_profile,
        "collision_merit_profile": collision_merit_profile,
        "line_search_profile": line_search_profile,
        "line_search_fractions": list(fractions),
        "intermediate_contact_constraints_elastic": strategy == "elastic_squared_clearance_v3",
        "original_noncontact_and_final_acceptance_constraints_changed": False,
        "numerical_worst_contact_may_temporarily_increase": collision_merit_profile == "sum_squared_clearance_v2",
        "collision_query_grid": "control_and_all_original_timestamps_v2" if sampler else "control_only_v1",
        "collision_query_pose_count": len(sampler.query_times) if sampler else len(current),
        "all_original_time_task_norms_constrained_during_fit": original_time_tasks is not None,
        "fixed_reference_rank_upper_landmark_constraints_enabled": upper_landmark_constraints is not None,
        "original_time_task_frame_count": len(original_time_tasks.original_times)
        if original_time_tasks is not None
        else 0,
        "iterations": history,
        "failure": failure,
        "query_clearance_target_m": 0.001,
        "near_contact_query_copy_margin_m": 0.03,
        "physical_model_modified": False,
        "source_targets_replaced": False,
        "frames_dropped": 0,
        "intermediate_iterates_are_accepted_references": False,
        "minimum_norm_optimality_proven": False,
        "dynamic_feasibility_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
