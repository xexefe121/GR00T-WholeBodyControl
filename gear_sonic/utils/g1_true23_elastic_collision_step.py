"""Offline elastic clearance restoration; every non-contact bound stays hard.

The subproblem minimizes path movement plus squared nonnegative clearance
slacks. A contact may temporarily worsen to escape competing contact gradients.
These slacks are NOT accepted physical clearance: the caller must line-search
the actual nonlinear full-path merit and independently audit the final model.
No physics, robot commands, trajectory acceptance, or limit changes live here.
"""

import numpy as np
from scipy import sparse

from gear_sonic.utils.g1_true23_affine_task_soc import solve_affine_task_soc


def solve_elastic_collision_step(
    posture,
    displacement,
    residual,
    jacobian,
    matrix,
    lower,
    upper,
    groups,
    contact_jacobian,
    distances,
    *,
    objective_scale=1.0,
    objective_profile="minimum_change_v2",
    clearance_m=0.001,
    slack_weight=10000.0,
):
    """Solve for delta only; return slack diagnostics, never an acceptance pass.

    Variables are (delta, s). Constraints are the supplied unchanged linear
    bounds and task SOCs, J_contact delta + s >= clearance - distance, s >= 0.
    Only the intermediate contact rows are elastic. No task group receives a
    slack column. The returned numerical step may still be in self-collision.
    """
    posture = np.asarray(posture, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    distances = np.asarray(distances, dtype=float)
    jacobian, matrix, contact_jacobian = [
        sparse.csc_matrix(value) for value in (jacobian, matrix, contact_jacobian)
    ]
    n, c = posture.size, distances.size
    if (
        objective_profile not in ("minimum_change_v2", "pose_escape_guided_v3")
        or distances.ndim != 1
        or contact_jacobian.shape != (c, n)
        or not np.isfinite(distances).all()
        or not np.isfinite(contact_jacobian.data).all()
        or not np.isfinite(clearance_m)
        or not 0 < clearance_m <= 0.005
        or not np.isfinite(slack_weight)
        or not 1 <= slack_weight <= 1e6
    ):
        raise ValueError("invalid bounded elastic collision subproblem")
    constraints = sparse.vstack(
        (
            sparse.hstack((matrix, sparse.csc_matrix((matrix.shape[0], c)))),
            sparse.hstack((contact_jacobian, sparse.eye(c))),
            sparse.hstack((sparse.csc_matrix((c, n)), sparse.eye(c))),
        ),
        format="csc",
    )
    # All original group indices refer to residual rows, not decision columns.
    task_jacobian = sparse.hstack((jacobian, sparse.csc_matrix((jacobian.shape[0], c))), format="csc")
    answer, report = solve_affine_task_soc(
        np.r_[posture, np.full(c, slack_weight)],
        np.r_[displacement, np.zeros(c)],
        residual,
        task_jacobian,
        constraints,
        np.r_[lower, clearance_m - distances, np.zeros(c)],
        np.r_[upper, np.full(2 * c, np.inf)],
        groups,
        objective_scale=objective_scale,
        objective_profile=objective_profile,
    )
    report.update(
        contact_restoration_profile="elastic_squared_clearance_v3",
        contact_constraint_count=c,
        clearance_slack_weight=slack_weight,
        query_clearance_target_m=clearance_m,
        intermediate_contact_nonworsening_constraint_removed=True,
        original_noncontact_linear_and_task_constraints_changed=False,
        final_collision_acceptance_changed=False,
        numerical_step_is_accepted_reference=False,
        physical_collision_clearance_proven=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    if answer is None:
        return None, report
    delta, slack = answer[:n], answer[n:]
    predicted_distance = distances + contact_jacobian @ delta
    shortfall = np.maximum(clearance_m - predicted_distance, 0.0)
    report.update(
        clearance_slack_max_m=float(slack.max(initial=0)),
        predicted_squared_clearance_violation_m2=float(shortfall @ shortfall),
        predicted_clearance_violation_max_m=float(shortfall.max(initial=0)),
        predicted_contacts_worsened=int(np.count_nonzero(predicted_distance < distances - 1e-8)),
        slack_nonnegativity_violation=float(np.maximum(-slack, 0).max(initial=0)),
        contact_slack_row_violation=float(np.maximum(shortfall - slack, 0).max(initial=0)),
    )
    return delta, report
