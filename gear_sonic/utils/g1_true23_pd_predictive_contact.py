"""Actual-state substep contact prediction for offline trajectory research only.

Every probe starts from a private copy of the complete current integration state.
The original plant, physical model, contact settings and codec are untouched.
Full-future cached value functions still make this unsuitable for live teleop.
"""

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_pd_contact_step import solve_contact_quadratic
from gear_sonic.utils.g1_true23_pd_shooting import PdShootingPlant
from gear_sonic.utils.g1_true23_pd_substep_limits import (
    JOINT_PLANNING_INSET_RAD,
    SPEED_PLANNING_INSET_RAD_S,
    substep_bound_jacobian,
    substep_bound_margins,
)
from gear_sonic.utils.g1_true23_pd_target_lattice import nearest_original_codec_target
from gear_sonic.utils.g1_true23_self_collision import SelfCollisionLinearizer


class SubstepContactPredictor:
    def __init__(self, plant):
        self.plant = PdShootingPlant(plant.model, plant.profile)
        self.query = SelfCollisionLinearizer(plant.model, near_distance_m=0.03)

    def poses(self, state, target):
        """Exactly the original held-target physics, on private data only."""
        plant = self.plant
        data = plant.scratch(state)
        target = plant.validate_target(target)
        mujoco.mj_kinematics(plant.model, data)
        mujoco.mj_comPos(plant.model, data)
        mujoco.mj_comVel(plant.model, data)
        poses, velocities = [], []
        for _ in range(10):
            plant._tick(data, target)
            poses.append(data.qpos.copy())
            velocities.append(data.qvel.copy())
        return np.asarray(poses), np.asarray(velocities)

    def rows(self, poses, limits, clearance_m):
        rows = []
        for substep, pose in enumerate(poses):
            for contact in self.query.pose_rows(pose, np.arange(29)):
                if contact["distance_m"] < 0.015:
                    rows.append(
                        dict(
                            substep=substep,
                            geoms=contact["geoms"],
                            distance=contact["distance_m"],
                            floor=limits.get(contact["geoms"], 0.0) + clearance_m,
                            jacobian=contact["joint_jacobian"],
                        )
                    )
        return rows

    def target_derivatives(self, state, target, *, epsilon=1e-5, include_velocity=False):
        """Differentiate all ten original PD substeps at the actual state.

        Local finite differences do not re-encode through the float32 codec.
        Accepted proposed commands do, and are independently re-predicted.
        """
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("substep derivative epsilon must be positive")
        result = np.zeros((10, 29, 23))
        velocity_result = np.zeros_like(result)
        for joint in range(23):
            lo, hi = np.asarray(target, dtype=float).copy(), np.asarray(target, dtype=float).copy()
            lo[joint] = max(self.plant.lower[joint], lo[joint] - epsilon)
            hi[joint] = min(self.plant.upper[joint], hi[joint] + epsilon)
            width = hi[joint] - lo[joint]
            if width <= 0:
                raise ValueError("no finite-difference interval inside original target bounds")
            minus, minus_velocity = self.poses(state, lo)
            plus, plus_velocity = self.poses(state, hi)
            velocity_result[:, :, joint] = (plus_velocity - minus_velocity) / width
            for step in range(10):
                derivative = np.empty(29)
                mujoco.mj_differentiatePos(self.plant.model, derivative, width, minus[step], plus[step])
                result[step, :, joint] = derivative
        return (result, velocity_result) if include_velocity else result


def make_predictive_contact_law(plant, controls, stages, alpha, limits, *, clearance_m=0.001, max_repairs=6):
    """Sequential local QPs, with actual-state nonlinear substep rechecks."""
    if not np.isfinite(clearance_m) or not 0 <= clearance_m <= 0.005 or max_repairs < 1:
        raise ValueError("invalid offline substep planning parameters")
    predictor = SubstepContactPredictor(plant)
    identity = np.eye(23)
    stats = dict(
        controls_solved=0,
        controls_requiring_repair=0,
        nonlinear_prediction_calls=0,
        actual_state_derivative_calls=0,
        maximum_repairs_in_one_control=0,
        maximum_linear_constraint_violation=0.0,
        predicted_substeps_per_control=10,
        uses_complete_actual_integration_state=True,
        full_future_value_function_used_offline=True,
        geometry_normal_derivative_is_approximate=True,
        stricter_planning_clearance_m=clearance_m,
        stricter_joint_planning_inset_rad=JOINT_PLANNING_INSET_RAD,
        stricter_speed_planning_inset_rad_s=SPEED_PLANNING_INSET_RAD_S,
        maximum_extra_contact_padding_m=0.0,
    )

    def solve(stage, gradient, matrix, bound):
        delta, _, result = solve_contact_quadratic(
            stage["hessian"], gradient, matrix, bound, np.zeros((23, 0)), np.zeros((len(bound), 0))
        )
        stats["maximum_linear_constraint_violation"] = max(
            stats["maximum_linear_constraint_violation"], result["maximum_original_row_violation"]
        )
        return delta

    def law(index, difference, state):
        stage, nominal = stages[index], controls[index]
        gradient = alpha * stage["gradient"] + stage["gradient_state"] @ difference
        box, bounds = np.vstack((identity, -identity)), np.r_[plant.upper - nominal, nominal - plant.lower]
        delta = solve(stage, gradient, box, bounds)
        for repair in range(max_repairs + 1):
            target = nearest_original_codec_target(np.clip(nominal + delta, plant.lower, plant.upper))[0]
            poses, velocities = predictor.poses(state, target)
            stats["nonlinear_prediction_calls"] += 1
            rows = predictor.rows(poses, limits, clearance_m)
            physical_margins = substep_bound_margins(plant, poses, velocities)
            violation = max((row["floor"] - row["distance"] for row in rows), default=0.0)
            physical_violation = max(0.0, float(-np.min(physical_margins)))
            if violation <= 1e-8 and physical_violation <= 1e-8:
                stats["controls_solved"] += 1
                stats["controls_requiring_repair"] += int(repair > 0)
                stats["maximum_repairs_in_one_control"] = max(stats["maximum_repairs_in_one_control"], repair)
                return target.astype(float) - nominal
            if repair == max_repairs:
                raise ValueError(
                    f"actual-state contact prediction control {index} unresolved: "
                    f"contact planning violation={violation:.9g} m, "
                    f"joint/speed planning violation={physical_violation:.9g} after {max_repairs} repairs"
                )
            derivatives, velocity_derivatives = predictor.target_derivatives(state, target, include_velocity=True)
            stats["actual_state_derivative_calls"] += 1
            jacobian = np.asarray([row["jacobian"] @ derivatives[row["substep"]] for row in rows]).reshape(-1, 23)
            physical_jacobian = substep_bound_jacobian(derivatives, velocity_derivatives)
            emitted_delta = target.astype(float) - nominal
            # Request outward clearance instead of repeatedly rounding back to
            # the same violating codec value. After a nonlinear mismatch, add
            # bounded extra planning padding. Acceptance tolerances never grow.
            padding = 2e-6 if repair == 0 else min(5e-4, max(2e-6, 2 * violation))
            stats["maximum_extra_contact_padding_m"] = max(stats["maximum_extra_contact_padding_m"], padding)
            matrix = np.vstack((box, -jacobian, -physical_jacobian))
            bound = np.r_[
                bounds,
                [row["distance"] - row["floor"] - padding for row in rows] - jacobian @ emitted_delta,
                physical_margins - 1e-6 - physical_jacobian @ emitted_delta,
            ]
            delta = solve(stage, gradient, matrix, bound)
        raise RuntimeError("unreachable substep repair branch")

    return law, stats
