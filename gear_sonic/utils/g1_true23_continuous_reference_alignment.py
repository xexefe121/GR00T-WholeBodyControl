"""SIM-only causal, foot-priority reference IK with hard temporal bounds.

Each accepted frame depends on the current original29 pose and preceding
accepted references only. Feet are fitted first; torso fitting may use only
their declared residual budget, and arm IK cannot move the feet. Position,
velocity, acceleration, braking viability, source correction and pelvis tilt
constraints are checked before accepting state. This is not a motor controller,
an automatically qualified reference, or permission to actuate a robot.
"""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_true23_original_task_trajectory import skew, so3_left_jacobian
from gear_sonic.utils.g1_true23_torso_reference_alignment import TorsoReferenceAlignment, _pose


@dataclass(frozen=True)
class ContinuousAlignmentConfig:
    root_position_limit_m: float = 0.08
    root_rotation_l1_limit_rad: float = 0.45
    base_tilt_limit_rad: float = 0.5
    joint_correction_limit_rad: float = 0.6
    root_position_velocity_m_s: float = 0.75
    root_position_acceleration_m_s2: float = 6.0
    root_rotation_coordinate_velocity_rad_s: float = 1.5
    root_rotation_coordinate_acceleration_rad_s2: float = 12.0
    joint_velocity_rad_s: float = 5.0
    joint_acceleration_rad_s2: float = 80.0
    foot_position_limit_m: float = 0.005
    foot_orientation_limit_rad: float = 0.15
    reserve_fraction: float = 0.995
    maximum_iterations: int = 24

    def __post_init__(self):
        limits = (0.08, 0.45, 0.5, 0.6, 0.75, 6.0, 1.5, 12.0, 5.0, 80.0, 0.005, 0.15)
        for (name, value), maximum in zip(list(asdict(self).items())[:12], limits, strict=True):
            if isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"invalid or expanded reference limit: {name}")
        if not 0 < self.reserve_fraction < 1:
            raise ValueError("continuous reference requires a strict internal reserve")
        if type(self.maximum_iterations) is not int or self.maximum_iterations < 1:
            raise ValueError("continuous reference iteration count must be positive integer")


def temporal_bounds(previous, velocity, lower, upper, maximum_velocity, maximum_acceleration, *, dt=0.02):
    """Same conservative braking envelope as existing trajectory retargeter.

    Generalized to mixed reference-coordinate units. Never relax acceleration
    to repair an empty box, and never silently repair a nonviable past state.
    """
    arrays = [
        np.asarray(value, dtype=float)
        for value in (previous, velocity, lower, upper, maximum_velocity, maximum_acceleration)
    ]
    previous, velocity, lower, upper, maximum_velocity, maximum_acceleration = arrays
    if (
        previous.ndim != 1
        or not len(previous)
        or any(value.shape != previous.shape or not np.isfinite(value).all() for value in arrays)
        or not np.isfinite(dt)
        or dt <= 0
        or np.any(lower >= upper)
        or np.any(maximum_velocity <= 0)
        or np.any(maximum_acceleration <= 0)
        or np.any(previous < lower - 1e-12)
        or np.any(previous > upper + 1e-12)
        or np.any(np.abs(velocity) > maximum_velocity + 1e-9)
    ):
        raise ValueError("invalid reference state or temporal bounds")
    acceleration_step = maximum_acceleration * dt * dt
    distance_low, distance_high = np.maximum(previous - lower, 0), np.maximum(upper - previous, 0)
    inward_low = -acceleration_step + np.sqrt(acceleration_step**2 + 2 * acceleration_step * distance_low)
    inward_high = -acceleration_step + np.sqrt(acceleration_step**2 + 2 * acceleration_step * distance_high)
    lo = np.maximum.reduce(
        (
            lower,
            previous - maximum_velocity * dt,
            previous - inward_low,
            previous + velocity * dt - acceleration_step,
        )
    )
    hi = np.minimum.reduce(
        (
            upper,
            previous + maximum_velocity * dt,
            previous + inward_high,
            previous + velocity * dt + acceleration_step,
        )
    )
    if np.any(lo > hi):
        raise ValueError("no acceleration-bounded viable next reference; constraints not relaxed")
    return lo, hi


class ContinuousReferenceAlignment(TorsoReferenceAlignment):
    """Stateful reference conversion, explicitly separate from frozen prototype."""

    def __init__(
        self, source_model, native_model, initial_source_pose, *, temporal_config=ContinuousAlignmentConfig()
    ):
        super().__init__(source_model, native_model)
        self.temporal_config = temporal_config
        c, r = temporal_config, temporal_config.reserve_fraction
        self.static_lower = np.r_[
            np.full(3, -c.root_position_limit_m * r), np.full(3, -c.root_rotation_l1_limit_rad * r), self.lower
        ]
        self.static_upper = np.r_[
            np.full(3, c.root_position_limit_m * r), np.full(3, c.root_rotation_l1_limit_rad * r), self.upper
        ]
        self.maximum_velocity = (
            np.repeat(
                [c.root_position_velocity_m_s, c.root_rotation_coordinate_velocity_rad_s, c.joint_velocity_rad_s],
                [3, 3, 23],
            )
            * r
        )
        self.maximum_acceleration = (
            np.repeat(
                [
                    c.root_position_acceleration_m_s2,
                    c.root_rotation_coordinate_acceleration_rad_s2,
                    c.joint_acceleration_rad_s2,
                ],
                [3, 3, 23],
            )
            * r
        )
        self.initial_source = _pose(initial_source_pose, 36)
        self.previous = np.r_[np.zeros(6), self.initial_source[7 + self.keep]]
        self.velocity = np.zeros(29)
        self.index = 0
        self.failed = False
        self.signs = np.asarray([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])
        if np.any(self.initial_source[7 + self.missing] != 0):
            raise ValueError("reference initialization requires declared zero-missing-axis standing pose")
        if np.any(self.previous < self.static_lower) or np.any(self.previous > self.static_upper):
            raise ValueError("initial reference outside unchanged bounds")
        self.current_source = self.initial_source.copy()
        self.source_rotation = Rotation.from_quat(self.current_source[[4, 5, 6, 3]])
        if np.min(self.root_constraints(self.previous[:19])[0]) < 0:
            raise ValueError("initial reference outside pelvis envelope")

    def qpos(self, variables):
        x = np.asarray(variables)
        rotation = Rotation.from_rotvec(x[3:6]) * self.source_rotation
        return np.r_[self.current_source[:3] + x[:3], rotation.as_quat()[[3, 0, 1, 2]], x[6:]]

    def root_constraints(self, lower_variables):
        c = self.temporal_config
        angular = so3_left_jacobian(lower_variables[3:6])
        axis = (Rotation.from_rotvec(lower_variables[3:6]) * self.source_rotation).apply([0, 0, 1])
        values = np.r_[
            c.root_rotation_l1_limit_rad * c.reserve_fraction - self.signs @ lower_variables[3:6],
            axis[2] - np.cos(c.base_tilt_limit_rad * c.reserve_fraction),
        ]
        jacobian = np.zeros((9, 19))
        jacobian[:8, 3:6] = -self.signs
        jacobian[8, 3:6] = (-skew(axis) @ angular)[2]
        return values, jacobian

    def features(self, lower_variables, tasks):
        """Exact world point/rotation residual Jacobians for root+12legs+yaw."""
        variables = self.previous.copy()
        variables[:19] = lower_variables
        pose = self.qpos(variables)
        self.native_data.qpos[:] = pose
        mujoco.mj_fwdPosition(self.native, self.native_data)
        angular_root = so3_left_jacobian(lower_variables[3:6])
        result = []
        for task in tasks:
            body = self.native.body(task.target_body).id
            rotation = self.native_data.xmat[body].reshape(3, 3)
            point = self.native_data.xpos[body] + rotation @ task.target_point
            source_body = self.source.body(task.source_body).id
            desired_rotation = self.source_data.xmat[source_body].reshape(3, 3)
            desired_point = self.source_data.xpos[source_body] + desired_rotation @ task.source_point
            jp, jr = np.zeros((3, self.native.nv)), np.zeros((3, self.native.nv))
            mujoco.mj_jac(self.native, self.native_data, jp, jr, point, body)
            point_jac = np.c_[np.eye(3), -skew(point - pose[:3]) @ angular_root, jp[:, 6:19]]
            angle_jac = np.c_[np.zeros((3, 3)), angular_root, jr[:, 6:19]]
            rotation_jac = np.concatenate(
                [np.cross(angle_jac.T, rotation[:, column]).T for column in range(3)]
            ) / np.sqrt(2)
            result.append(
                (
                    point - desired_point,
                    point_jac,
                    (rotation - desired_rotation).T.ravel() / np.sqrt(2),
                    rotation_jac,
                )
            )
        return result

    def _step(self, source_pose, frame_index):
        if type(frame_index) is not int or frame_index != self.index:
            raise ValueError("continuous reference requires exact sequential frame index")
        source_pose = _pose(source_pose, 36)
        self.current_source = source_pose
        self.source_rotation = Rotation.from_quat(source_pose[[4, 5, 6, 3]])
        self.source_data.qpos[:] = source_pose
        mujoco.mj_fwdPosition(self.source, self.source_data)
        lo, hi = temporal_bounds(
            self.previous,
            self.velocity,
            self.static_lower,
            self.static_upper,
            self.maximum_velocity,
            self.maximum_acceleration,
        )
        correction = self.temporal_config.joint_correction_limit_rad * self.temporal_config.reserve_fraction
        lo[6:] = np.maximum(lo[6:], source_pose[7 + self.keep] - correction)
        hi[6:] = np.minimum(hi[6:], source_pose[7 + self.keep] + correction)
        if np.any(lo > hi):
            raise ValueError("source correction and temporal envelopes do not intersect")
        # Preserve declared initial reference exactly; no hidden one-centimetre
        # standing reset to optimize a proxy that belongs to a different model.
        if (
            np.array_equal(source_pose, self.initial_source)
            and np.all(self.velocity == 0)
            and np.array_equal(self.previous, np.r_[np.zeros(6), source_pose[7 + self.keep]])
        ):
            self.index += 1
            return np.r_[source_pose[:7], source_pose[7 + self.keep]], dict(
                frame=frame_index, initial_reference_noop=True, foot_position_error_m=[0.0, 0.0], qualified=False
            )
        initial = np.clip(self.previous + self.velocity * 0.02, lo, hi)
        feet = [row[0] for row in self.limbs[:2]]
        head = next(task for task in self.tasks if task.name == "head_proxy")
        regularizer = 1e-8

        def foot_objective(value):
            cost, gradient = 0.0, np.zeros(19)
            for p, jp, rotation, jr in self.features(value, feet):
                cost += np.dot(p, p) + 0.05**2 * np.dot(rotation, rotation)
                gradient += 2 * (jp.T @ p + 0.05**2 * jr.T @ rotation)
            difference = value - initial[:19]
            return cost + regularizer * np.dot(difference, difference), gradient + 2 * regularizer * difference

        root_constraint = dict(
            type="ineq", fun=lambda x: self.root_constraints(x)[0], jac=lambda x: self.root_constraints(x)[1]
        )
        foot_fit = minimize(
            foot_objective,
            initial[:19],
            jac=True,
            method="SLSQP",
            bounds=list(zip(lo[:19], hi[:19], strict=True)),
            constraints=root_constraint,
            options=dict(maxiter=self.temporal_config.maximum_iterations, ftol=1e-13),
        )

        def root_feasible(value):
            return (
                value.shape == (19,)
                and np.isfinite(value).all()
                and np.all(value >= lo[:19] - 1e-10)
                and np.all(value <= hi[:19] + 1e-10)
                and np.min(self.root_constraints(value)[0]) >= -1e-10
            )

        candidates = [value for value in (initial[:19], foot_fit.x) if root_feasible(value)]
        if not candidates:
            raise ValueError("no root/temporal-feasible reference found; state not advanced")
        foot_best = min(candidates, key=lambda value: foot_objective(value)[0]).copy()
        foot_features = self.features(foot_best, feet)
        c = self.temporal_config
        position_budgets = [
            max(c.foot_position_limit_m * c.reserve_fraction, np.linalg.norm(row[0])) for row in foot_features
        ]
        # Matrix residual norm = 2*sin(angle/2). Retain this monotonic metric.
        orientation_budgets = [
            max(2 * np.sin(c.foot_orientation_limit_rad * c.reserve_fraction / 2), np.linalg.norm(row[2]))
            for row in foot_features
        ]

        def protected_feet(value):
            values, jac = [], []
            for row, position_budget, orientation_budget in zip(
                self.features(value, feet), position_budgets, orientation_budgets, strict=True
            ):
                p, jp, rotation, jr = row
                # Normalize, so SLSQP cannot accept a centimetre error because
                # its squared metre residual looked numerically small.
                values.extend(
                    (1 - np.dot(p, p) / position_budget**2, 1 - np.dot(rotation, rotation) / orientation_budget**2)
                )
                jac.extend((-2 * p @ jp / position_budget**2, -2 * rotation @ jr / orientation_budget**2))
            return np.asarray(values), np.asarray(jac)

        def upper_objective(value):
            p, jp, rotation, jr = self.features(value, [head])[0]
            difference = value - foot_best
            return np.dot(p, p) + 0.2**2 * np.dot(rotation, rotation) + 1e-6 * np.dot(
                difference, difference
            ), 2 * (jp.T @ p + 0.2**2 * jr.T @ rotation + 1e-6 * difference)

        upper = minimize(
            upper_objective,
            foot_best,
            jac=True,
            method="SLSQP",
            bounds=list(zip(lo[:19], hi[:19], strict=True)),
            constraints=[
                root_constraint,
                dict(type="ineq", fun=lambda x: protected_feet(x)[0], jac=lambda x: protected_feet(x)[1]),
            ],
            options=dict(maxiter=c.maximum_iterations, ftol=1e-12),
        )
        use_upper = (
            root_feasible(upper.x)
            and np.min(protected_feet(upper.x)[0]) >= -1e-8
            and upper_objective(upper.x)[0] < upper_objective(foot_best)[0]
        )
        chosen = initial.copy()
        chosen[:19] = upper.x if use_upper else foot_best
        pose = self.qpos(chosen)
        arm_nfev = []
        for task, indices in self.limbs[2:]:
            source_body = self.source.body(task.source_body).id
            rotation = self.source_data.xmat[source_body].reshape(3, 3).copy()
            desired = (self.source_data.xpos[source_body] + rotation @ task.source_point, rotation)
            kwargs = dict(
                pose=pose, task=task, indices=indices, desired=desired, posture=source_pose[7 + self.keep[indices]]
            )
            coordinates = 6 + indices
            arm = chosen[coordinates].copy()
            active = hi[coordinates] > lo[coordinates]
            arm[~active] = lo[coordinates][~active]

            def assemble(value):
                q = arm.copy()
                q[active] = value
                return q

            if np.any(active):
                fit = least_squares(
                    lambda q: self.limb_residual_jacobian(assemble(q), **kwargs)[0],
                    arm[active],
                    jac=lambda q: self.limb_residual_jacobian(assemble(q), **kwargs)[1][:, active],
                    bounds=(lo[coordinates][active], hi[coordinates][active]),
                    max_nfev=self.config.max_function_evaluations,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                )
                arm = assemble(fit.x)
                arm_nfev.append(int(fit.nfev))
            else:
                arm_nfev.append(0)
            chosen[coordinates] = arm
            pose[7 + indices] = arm
        if (
            not np.isfinite(chosen).all()
            or np.any(chosen < lo - 1e-10)
            or np.any(chosen > hi + 1e-10)
            or not root_feasible(chosen[:19])
        ):
            raise ValueError("reference solve violated original current-frame constraints")
        # Upper/lower stages may fail their fidelity objective. Such outcomes
        # remain explicit, not a license to expand the final foot acceptance.
        residuals = self.features(chosen[:19], feet)
        errors = [float(np.linalg.norm(row[0])) for row in residuals]
        self.velocity = (chosen - self.previous) / 0.02
        self.previous = chosen.copy()
        self.index += 1
        return pose, dict(
            frame=frame_index,
            initial_reference_noop=False,
            foot_solver_success=bool(foot_fit.success),
            foot_solver_status=int(foot_fit.status),
            foot_solver_iterations=int(foot_fit.nit),
            upper_solver_success=bool(upper.success),
            upper_solver_status=int(upper.status),
            upper_solver_iterations=int(upper.nit),
            upper_feasible_improvement_accepted=bool(use_upper),
            protected_position_budgets_m=position_budgets,
            foot_position_error_m=errors,
            foot_position_gate_passed=bool(max(errors) <= c.foot_position_limit_m),
            arm_nfev=arm_nfev,
            qualified=False,
        )

    def push(self, source_pose, *, frame_index):
        if self.failed:
            raise RuntimeError("continuous reference failure latched; no automatic frame skip or reset")
        try:
            return self._step(source_pose, frame_index)
        except Exception:
            self.failed = True
            raise

    def contract(self):
        return dict(
            kind="native23_causal_continuous_foot_priority_reference_v1",
            temporal_config=asdict(self.temporal_config),
            limb_config=asdict(self.config),
            source_joint_names=[self.source.joint(i).name for i in range(1, 30)],
            native_joint_names=[self.native.joint(i).name for i in range(1, 24)],
            lower_joint_bounds_rad=self.lower.tolist(),
            upper_joint_bounds_rad=self.upper.tolist(),
            hand_convention=self.hand_convention,
            uses_current_source_and_preceding_accepted_reference_only=True,
            future_source_frames_consumed=False,
            source_speed_factor=1.0,
            original_source_truth_replaced=False,
            foot_best_excess_retained_as_failure_not_accepted=True,
            failure_latched=True,
            physical_motor_output=False,
            controller_or_contact_feasibility_proven=False,
            deployment_ready=False,
            hardware_authorized=False,
        )
