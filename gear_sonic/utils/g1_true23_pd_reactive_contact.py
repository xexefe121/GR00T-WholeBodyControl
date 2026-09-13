"""State-reactive constraints in offline trajectory trials, not live teleop.

Quadratic value functions use the full future motion. Current simulated state
changes which constraints are solved; this module has no robot interface.
"""

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_pd_contact_step import solve_contact_quadratic
from gear_sonic.utils.g1_true23_self_collision import SelfCollisionLinearizer

TERMINAL_FIELDS = (
    "final_proof_standing_joint_error_max_rad",
    "final_proof_root_speed_max_m_s",
    "final_proof_root_position_error_max_m",
    "final_proof_root_orientation_error_max_rad",
)


def terminal_nonregression(initial, candidate):
    violations = []
    for name in TERMINAL_FIELDS:
        ceiling, value = initial["lifecycle"][name], candidate["lifecycle"][name]
        if ceiling is None or value is None or not np.isfinite([ceiling, value]).all() or value > ceiling + 1e-12:
            violations.append(dict(metric=name, candidate=value, fixed_initial_ceiling=ceiling))
    return dict(
        accepted=not violations, violations=violations, research_nonregression_not_standing_qualification=True
    )


def make_reactive_contact_law(plant, local, nominal, controls, stage_models, alpha, limits, *, clearance_m=0.001):
    """Re-solve QP with geometry at the state-shifted next pose.

    Geometry normals are local approximations only. The private query model
    never supplies forces or overwrites the integrated plant. Every actual
    physics-substep contact remains checked by the original protected plant.
    """
    if not np.isfinite(clearance_m) or not 0 <= clearance_m <= 0.005:
        raise ValueError("reactive planning clearance must be between zero and five millimetres")
    query = SelfCollisionLinearizer(plant.model, near_distance_m=0.03)
    columns, identity = np.arange(29), np.eye(23)
    stats = dict(
        controls_solved=0,
        controls_with_active_contact_constraints=0,
        maximum_original_linear_violation=0.0,
        maximum_near_contact_pairs=0,
        geometry_requeried_from_state_shifted_prediction=True,
        geometry_normal_derivative_is_approximate=True,
        full_future_value_function_used_offline=True,
        stricter_linear_planning_clearance_buffer_m=clearance_m,
    )

    def law(index, difference):
        predicted_pose = nominal["qpos"][index + 1].copy()
        predicted_shift = local["a"][index] @ difference
        mujoco.mj_integratePos(plant.model, predicted_pose, predicted_shift[:29], 1.0)
        contacts = [row for row in query.pose_rows(predicted_pose, columns) if row["distance_m"] < 0.025]
        jacobian = np.asarray([row["joint_jacobian"] for row in contacts]).reshape(-1, 29)
        jacobian[:, :6] = 0.0
        matrix = np.vstack((identity, -identity, -jacobian @ local["b"][index, :29]))
        bound = np.r_[
            plant.upper - controls[index],
            controls[index] - plant.lower,
            [row["distance_m"] - limits.get(row["geoms"], 0.0) - clearance_m for row in contacts],
        ]
        stage = stage_models[index]
        gradient = alpha * stage["gradient"] + stage["gradient_state"] @ difference
        try:
            delta, _, result = solve_contact_quadratic(
                stage["hessian"], gradient, matrix, bound, np.zeros((23, 0)), np.zeros((len(bound), 0))
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"reactive contact control {index}: {exc}") from exc
        stats["controls_solved"] += 1
        stats["controls_with_active_contact_constraints"] += int(any(row >= 46 for row in result["active_rows"]))
        stats["maximum_near_contact_pairs"] = max(stats["maximum_near_contact_pairs"], len(contacts))
        stats["maximum_original_linear_violation"] = max(
            stats["maximum_original_linear_violation"], result["maximum_original_row_violation"]
        )
        return delta

    return law, stats
