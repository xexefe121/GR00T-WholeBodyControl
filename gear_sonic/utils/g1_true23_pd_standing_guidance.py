"""Original standing-reference velocity guidance for offline search only."""

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_reference_floor import motion_qpos


class StandingVelocityGuidance:
    def __init__(
        self,
        model,
        motion,
        timeline,
        *,
        root_weight=3000.0,
        angular_weight=100.0,
        joint_weight=1.0,
        orientation_weight=0.0,
    ):
        weights = np.asarray([root_weight, angular_weight, joint_weight, orientation_weight], dtype=float)
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("standing velocity guide weights must be finite and nonnegative")
        poses = motion_qpos(model, motion)[10:]
        count = timeline["total_requested_controls"]
        if len(poses) != count + 1 or (model.nq, model.nv) != (30, 29):
            raise ValueError("standing guide requires the unchanged complete native23 lifecycle")
        self.model, self.poses = model, poses.copy()
        self.orientation_weight = float(orientation_weight)
        self.weights = np.repeat(weights[:3], [3, 3, 23])
        self.active = np.zeros(count + 1, bool)
        self.velocity = np.zeros((count + 1, 29))
        self.phases = []
        for phase in timeline["phases"]:
            if phase["name"] not in ("returned_standing", "standing_proof_margin"):
                continue
            start, stop = phase["control_start"], phase["control_stop"]
            if not 0 <= start < stop <= count:
                raise ValueError("standing guide phase lies outside original lifecycle")
            self.phases.append(dict(name=phase["name"], control_start=start, control_stop=stop))
            self.active[start + 1 : stop + 1] = True
            for index in range(start + 1, stop + 1):
                mujoco.mj_differentiatePos(model, self.velocity[index], 0.02, poses[index - 1], poses[index])
        if not self.phases:
            raise ValueError("standing velocity guidance requires an original returned-standing phase")

    def state_terms(self, velocity, index, pose=None):
        gradient, hessian = np.zeros(81), np.zeros((81, 81))
        if not self.active[index]:
            return 0.0, gradient, hessian
        velocity = np.asarray(velocity)
        if velocity.shape != (29,) or not np.isfinite(velocity).all():
            raise ValueError("standing guide needs finite native29 generalized velocity")
        error = velocity - self.velocity[index]
        gradient[29:58] = self.weights * error
        hessian[29:58, 29:58] = np.diag(self.weights)
        cost = float(0.5 * error @ (self.weights * error))
        if self.orientation_weight:
            pose = np.asarray(pose)
            if (
                pose.shape != (30,)
                or not np.isfinite(pose).all()
                or not np.isclose(np.linalg.norm(pose[3:7]), 1.0, atol=2e-6, rtol=0)
            ):
                raise ValueError("standing orientation guide needs a finite native23 pose with unit quaternion")
            difference, jacobian = np.empty(29), np.empty((3, 3))
            mujoco.mj_differentiatePos(self.model, difference, 1.0, self.poses[index], pose)
            mujoco.mjd_subQuat(pose[3:7], self.poses[index, 3:7], jacobian, None)
            rotation_error = difference[3:6]
            gradient[3:6] = self.orientation_weight * jacobian.T @ rotation_error
            hessian[3:6, 3:6] = self.orientation_weight * jacobian.T @ jacobian
            cost += float(0.5 * self.orientation_weight * (rotation_error @ rotation_error))
        return cost, gradient, hessian

    def augment(self, local, trajectory):
        result = {
            **local,
            **{name: local[name].copy() for name in ("gradient", "hessian", "terminal_g", "terminal_h")},
        }
        count = len(local["gradient"])
        if np.shape(trajectory["qvel"]) != (count + 1, 29) or len(self.active) != count + 1:
            raise ValueError("standing guide cannot crop the physical trajectory")
        if self.orientation_weight and np.shape(trajectory.get("qpos")) != (count + 1, 30):
            raise ValueError("standing orientation guide cannot crop or omit physical poses")
        indices = np.flatnonzero(self.active)
        extra_g, extra_h, cost = [], [], 0.0
        for index in indices:
            value, gradient, hessian = self.state_terms(
                trajectory["qvel"][index],
                index,
                trajectory["qpos"][index] if self.orientation_weight else None,
            )
            if index == count:
                result["terminal_g"] += gradient
                result["terminal_h"] += hessian
            else:
                result["gradient"][index] += gradient
                result["hessian"][index] += hessian
            cost += value
            extra_g.append(gradient)
            extra_h.append(hessian)
        return (
            result,
            dict(indices=indices, gradient=np.asarray(extra_g), hessian=np.asarray(extra_h)),
            dict(
                kind=(
                    "g1_true23_original_standing_velocity_orientation_local_guidance_v2"
                    if self.orientation_weight
                    else "g1_true23_original_standing_velocity_local_guidance_v1"
                ),
                **(
                    dict(
                        root_orientation_weight=self.orientation_weight,
                        orientation_target="unchanged_original_returned_standing_quaternion",
                        orientation_error="shortest_arc_native_rotation_vector_not_quaternion_components",
                        orientation_hessian="positive_semidefinite_gauss_newton",
                    )
                    if self.orientation_weight
                    else {}
                ),
                root_linear_weight=float(self.weights[0]),
                root_angular_weight=float(self.weights[3]),
                joint_velocity_weight=float(self.weights[6]),
                target="original_reference_backward_native_velocity_at_post_control_state",
                phases=self.phases,
                guided_state_nodes=len(indices),
                terminal_node_included=bool(self.active[-1]),
                local_guide_cost_at_nominal=cost,
                original_acceptance_objective_and_thresholds_unchanged=True,
                cached_physical_dynamics_unchanged=True,
                no_extra_physical_forces=True,
                not_policy_training_or_hardware_qualification=True,
            ),
        )
