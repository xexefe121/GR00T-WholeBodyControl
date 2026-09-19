"""Offline native23 trajectory repair using actual bounded-PD contact dynamics.

This is an optimizer for research/reference preparation, NOT a SONIC policy,
teleop controller, admissible training parent, or hardware execution path.
"""

import mujoco
import numpy as np

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_generalist_benchmark import LANDMARKS, task_points
from gear_sonic.utils.g1_true23_pd_shooting import state_difference
from gear_sonic.utils.g1_true23_pd_target_lattice import nearest_original_codec_target
from gear_sonic.utils.g1_true23_reference_floor import motion_qpos
from gear_sonic.utils.g1_true23_reference_support import pose_path_derivatives

CONTROL_WEIGHT, SLEW_WEIGHT = 0.02, 0.2


def canonical_target(target):
    return nearest_original_codec_target(target)[0]


class MotionObjective:
    def __init__(self, plant, motion, timeline):
        self.plant, self.model = plant, plant.model
        poses = motion_qpos(self.model, motion)
        velocity, _ = pose_path_derivatives(self.model, poses, 0.02)
        self.poses, self.velocity = poses[10:], velocity[10:]
        self.points = np.asarray(
            [task_points(p, q) for p, q in zip(motion["body_pos_w"][10:], motion["body_quat_w"][10:], strict=True)]
        )
        self.data = mujoco.MjData(self.model)
        self.count = len(self.poses) - 1
        if self.count != timeline["total_requested_controls"]:
            raise ValueError("trajectory objective must preserve complete source and lifecycle")
        self.standing = np.zeros(self.count + 1, bool)
        for phase in timeline["phases"]:
            if phase["name"] in ("initial_standing", "returned_standing", "standing_proof_margin"):
                self.standing[phase["control_start"] : phase["control_stop"] + 1] = True
        self.point_weights = np.repeat([400, 400, 100, 100, 100], 3)
        self.velocity_weights = np.r_[np.full(3, 1.0), np.full(3, 0.4), np.full(23, 0.05)]

    def state_cost(self, qpos, qvel, index, *, derivatives=False, terminal=False):
        data, model = self.data, self.model
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        errors, jacobians = [], []
        for (_, body, offset), goal in zip(LANDMARKS, self.points[index], strict=True):
            point = data.xpos[body + 1] + data.xmat[body + 1].reshape(3, 3) @ offset
            errors.extend(point - goal)
            if derivatives:
                jacobian = np.empty((3, 29))
                mujoco.mj_jac(model, data, jacobian, None, point, body + 1)
                jacobians.append(jacobian)
        pose_error = state_difference(model, self.poses[index], self.velocity[index], qpos, qvel)[:29]
        pose_weights = np.r_[
            np.full(3, 400.0), np.full(3, 25.0), np.full(23, 100.0 if self.standing[index] else 4.0)
        ]
        velocity_error = np.asarray(qvel) - self.velocity[index]
        error = np.asarray(errors)
        factor = 10.0 if terminal else 1.0
        cost = (
            0.5
            * factor
            * (
                np.sum(self.point_weights * error**2)
                + np.sum(pose_weights * pose_error**2)
                + np.sum(self.velocity_weights * velocity_error**2)
            )
        )
        if not derivatives:
            return float(cost)
        jpose = np.eye(29)
        rotation_jacobian = np.empty((3, 3))
        mujoco.mjd_subQuat(qpos[3:7], self.poses[index, 3:7], rotation_jacobian, None)
        jpose[3:6, 3:6] = rotation_jacobian
        jpoints = np.vstack(jacobians)
        gradient, hessian = np.zeros(81), np.zeros((81, 81))
        gradient[:29] = jpoints.T @ (self.point_weights * error) + jpose.T @ (pose_weights * pose_error)
        gradient[29:58] = self.velocity_weights * velocity_error
        hessian[:29, :29] = jpoints.T @ (self.point_weights[:, None] * jpoints) + jpose.T @ (
            pose_weights[:, None] * jpose
        )
        hessian[29:58, 29:58] = np.diag(self.velocity_weights)
        return float(cost), factor * gradient, factor * hessian

    def cost(self, trajectory, controls, seed_controls):
        result = 0.0
        previous = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
        for i, control in enumerate(controls):
            result += self.state_cost(trajectory["qpos"][i], trajectory["qvel"][i], i)
            result += 0.5 * CONTROL_WEIGHT * np.sum((control - seed_controls[i]) ** 2)
            result += 0.5 * SLEW_WEIGHT * np.sum((control - previous) ** 2)
            previous = control
        result += self.state_cost(trajectory["qpos"][-1], trajectory["qvel"][-1], self.count, terminal=True)
        return float(result)


def linearize_trajectory(plant, objective, trajectory, controls, progress=None):
    count = len(controls)
    a, b = np.zeros((count, 81, 81)), np.zeros((count, 81, 23))
    gradient, hessian = np.zeros((count, 81)), np.zeros((count, 81, 81))
    for i in range(count):
        physical_a, physical_b = plant.linearize_control(trajectory["integration_state"][i], controls[i])
        a[i, :58, :58], b[i, :58] = physical_a, physical_b
        b[i, 58:] = np.eye(23)
        _, gradient[i], hessian[i] = objective.state_cost(
            trajectory["qpos"][i], trajectory["qvel"][i], i, derivatives=True
        )
        if progress is not None and ((i + 1) % 100 == 0 or i + 1 == count):
            progress(dict(stage="linearize_actual_pd_contact_dynamics", completed=i + 1, total=count))
    _, terminal_g, terminal_h = objective.state_cost(
        trajectory["qpos"][-1], trajectory["qvel"][-1], count, derivatives=True, terminal=True
    )
    return dict(a=a, b=b, gradient=gradient, hessian=hessian, terminal_g=terminal_g, terminal_h=terminal_h)


def backward_pass(plant, local, controls, seed_controls, regularization):
    """Damped local quadratic step; true nonlinear rollouts decide acceptance."""
    count = len(controls)
    increments, feedback = np.zeros((count, 23)), np.zeros((count, 23, 81))
    vg, vh = local["terminal_g"].copy(), local["terminal_h"].copy()
    identity = np.eye(23)
    for i in reversed(range(count)):
        a, b = local["a"][i], local["b"][i]
        previous = controls[i - 1] if i else np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
        delta, slew = controls[i] - seed_controls[i], controls[i] - previous
        lx, lxx = local["gradient"][i].copy(), local["hessian"][i].copy()
        lx[58:] -= SLEW_WEIGHT * slew
        lxx[58:, 58:] += SLEW_WEIGHT * identity
        lu = CONTROL_WEIGHT * delta + SLEW_WEIGHT * slew
        lux = np.zeros((23, 81))
        lux[:, 58:] = -SLEW_WEIGHT * identity
        qx, qu = lx + a.T @ vg, lu + b.T @ vg
        qxx = lxx + a.T @ vh @ a
        quu = (CONTROL_WEIGHT + SLEW_WEIGHT) * identity + b.T @ vh @ b
        qux = lux + b.T @ vh @ a
        h = 0.5 * (quu + quu.T) + regularization * identity
        factor = np.linalg.cholesky(h)
        answer = -np.linalg.solve(factor.T, np.linalg.solve(factor, np.column_stack((qu, qux))))
        k, gain = answer[:, 0], answer[:, 1:]
        bounded = np.clip(controls[i] + k, plant.lower, plant.upper) - controls[i]
        active = np.abs(bounded - k) > 1e-10
        k, gain[active] = bounded, 0
        increments[i], feedback[i] = k, gain
        vg = qx + gain.T @ qu + qux.T @ k + gain.T @ quu @ k
        vh = qxx + gain.T @ quu @ gain + gain.T @ qux + qux.T @ gain
        vh = 0.5 * (vh + vh.T)
        if not np.isfinite(vg).all() or not np.isfinite(vh).all():
            raise RuntimeError("trajectory quadratic approximation diverged")
    return increments, feedback


def feedback_trial(
    plant, initial, nominal, controls, increments, feedback, alpha, *, control_law=None, state_control_law=None
):
    if control_law is not None and state_control_law is not None:
        raise ValueError("choose only one offline control law")
    data = plant.scratch(initial)
    states, qpos, qvel, actual_controls = [plant.state(data)], [data.qpos.copy()], [data.qvel.copy()], []
    previous = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    records = {
        key: []
        for key in (
            "physics_pre_qpos",
            "physics_pre_qvel",
            "physics_post_qpos",
            "physics_post_qvel",
            "physics_time",
            "requested_torque23",
            "applied_torque23",
            "engine_actuator_force23",
            "physics_contact_count",
            "torque_saturated23",
        )
    }
    for i in range(len(controls)):
        nominal_previous = controls[i - 1] if i else np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
        difference = np.r_[
            state_difference(plant.model, nominal["qpos"][i], nominal["qvel"][i], data.qpos, data.qvel),
            previous - nominal_previous,
        ]
        raw_target = (
            controls[i] + alpha * increments[i] + feedback[i] @ difference
            if control_law is None
            else controls[i] + control_law(i, difference)
        )
        if state_control_law is not None:
            raw_target = controls[i] + state_control_law(i, difference, plant.state(data))
        target = canonical_target(np.clip(raw_target, plant.lower, plant.upper))
        plant.integrate_control(data, target, records)
        # Reject incomplete or unsafe trials; never assign a low prefix cost.
        recent_q, recent_v = (
            np.asarray(records["physics_post_qpos"][-10:]),
            np.asarray(records["physics_post_qvel"][-10:]),
        )
        if (
            np.any(recent_q[:, 7:] < plant.model.jnt_range[1:, 0] - 1e-6)
            or np.any(recent_q[:, 7:] > plant.model.jnt_range[1:, 1] + 1e-6)
            or np.any(np.abs(recent_v[:, 6:]) > np.asarray(plant.profile.velocity))
        ):
            raise ValueError("trajectory trial violated unchanged physical joint bounds")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
        if data.qpos[2] < 0.12 or np.arccos(np.clip(rotation[8], -1, 1)) > 2.2:
            raise ValueError("trajectory trial hit original absolute height/tilt stop")
        states.append(plant.state(data))
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        actual_controls.append(target.copy())
        previous = target
    plant.assert_unchanged()
    trajectory = {
        "integration_state": np.asarray(states),
        "qpos": np.asarray(qpos),
        "qvel": np.asarray(qvel),
        **{key: np.asarray(value) for key, value in records.items()},
    }
    return trajectory, np.asarray(actual_controls)
