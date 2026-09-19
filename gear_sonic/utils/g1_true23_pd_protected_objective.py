"""SIM-only protected motion repair, never a SONIC or hardware controller.

Near-contact derivatives use the existing private geometry-query model. Actual
rollouts keep original physics, and inspect contacts at every 500 Hz step.
Existing self-contact allowances are only a nonregression research envelope,
NOT permission for contact in a deployed robot or collision qualification.
"""

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_generalist_benchmark import LANDMARKS
from gear_sonic.utils.g1_true23_pd_shooting import PdShootingPlant
from gear_sonic.utils.g1_true23_pd_trajectory_optimizer import MotionObjective
from gear_sonic.utils.g1_true23_self_collision import SelfCollisionLinearizer, self_contact_rows

LANDMARK_THRESHOLDS = np.asarray([0.05, 0.05, 0.10, 0.10, 0.10])
ENVELOPE_WEIGHT = 100.0
SELF_CLEARANCE_M = 0.001
SELF_DISTANCE_SCALE_M = 0.001
CONTACT_COMPARISON_TOLERANCE_M = 1e-6
COLLISION_DERIVATIVE_EPSILON_RAD = 1e-6


class ProtectedMotionObjective(MotionObjective):
    def __init__(self, plant, motion, timeline):
        super().__init__(plant, motion, timeline)
        self.collision_query = SelfCollisionLinearizer(plant.model, near_distance_m=0.03)
        self.columns = np.arange(plant.model.nv)
        self.source = np.zeros(self.count + 1, dtype=bool)
        phase = next(item for item in timeline["phases"] if item["name"] == "source_motion")
        self.source[phase["control_start"] + 1 : phase["control_stop"] + 1] = True

    def collision_terms(self, qpos, derivatives):
        active = [
            row
            for row in self.collision_query.pose_rows(qpos, self.columns)
            if row["distance_m"] < SELF_CLEARANCE_M
        ]
        residuals = np.asarray([(SELF_CLEARANCE_M - row["distance_m"]) / SELF_DISTANCE_SCALE_M for row in active])
        jacobian = np.zeros((len(active), 29))
        if derivatives and active:
            # Mesh penetration normals are approximate. Differentiate the same
            # signed geometry values that define this cost, without modifying
            # physical collision tolerances. Self-distances are rigid-root invariant.
            epsilon = COLLISION_DERIVATIVE_EPSILON_RAD
            for joint in range(23):
                distances = []
                for sign in (-1, 1):
                    pose = np.asarray(qpos).copy()
                    pose[7 + joint] += sign * epsilon
                    rows = {
                        row["geoms"]: row["distance_m"]
                        for row in self.collision_query.pose_rows(pose, self.columns)
                    }
                    if any(row["geoms"] not in rows for row in active):
                        raise ValueError("active self-contact vanished inside the geometry derivative probe")
                    distances.append(np.asarray([rows[row["geoms"]] for row in active]))
                jacobian[:, 6 + joint] = -(distances[1] - distances[0]) / (2 * epsilon * SELF_DISTANCE_SCALE_M)
        return residuals, jacobian

    def state_cost(self, qpos, qvel, index, *, derivatives=False, terminal=False):
        base = super().state_cost(qpos, qvel, index, derivatives=derivatives, terminal=terminal)
        if derivatives:
            cost, gradient, hessian = base
        else:
            cost = base
        residuals, jacobians = [], []
        if self.source[index]:
            for (_, body, offset), desired, limit in zip(
                LANDMARKS, self.points[index], LANDMARK_THRESHOLDS, strict=True
            ):
                point = self.data.xpos[body + 1] + self.data.xmat[body + 1].reshape(3, 3) @ offset
                error = point - desired
                distance = np.linalg.norm(error)
                if distance > limit:
                    residuals.append(np.sqrt(ENVELOPE_WEIGHT) * (distance / limit - 1))
                    if derivatives:
                        jacobian = np.empty((3, 29))
                        mujoco.mj_jac(self.model, self.data, jacobian, None, point, body + 1)
                        jacobians.append(np.sqrt(ENVELOPE_WEIGHT) / limit * (error / distance) @ jacobian)
        collision_residuals, collision_jacobian = self.collision_terms(qpos, derivatives)
        residuals.extend(collision_residuals)
        if derivatives:
            jacobians.extend(collision_jacobian)
        factor = 10.0 if terminal else 1.0
        residuals = np.asarray(residuals)
        cost += factor * 0.5 * float(residuals @ residuals)
        if not derivatives:
            return cost
        if len(residuals):
            jacobian = np.asarray(jacobians)
            gradient[:29] += factor * jacobian.T @ residuals
            hessian[:29, :29] += factor * jacobian.T @ jacobian
        return cost, gradient, hessian

    def contract(self):
        return dict(
            kind="g1_true23_original_motion_per_landmark_and_self_clearance_v1",
            source_landmark_thresholds_m=LANDMARK_THRESHOLDS.tolist(),
            source_envelope_penalty_weight=ENVELOPE_WEIGHT,
            self_clearance_target_m=SELF_CLEARANCE_M,
            self_distance_penalty_scale_m=SELF_DISTANCE_SCALE_M,
            self_distance_derivative="centered_native_joint_geometry_difference",
            self_distance_derivative_epsilon_rad=COLLISION_DERIVATIVE_EPSILON_RAD,
            source_reference_and_timing_unchanged=True,
            private_collision_query_not_used_for_physics=True,
            penalties_do_not_constitute_constraint_or_hardware_qualification=True,
        )


class ProtectedPdPlant(PdShootingPlant):
    def __init__(self, model, profile):
        super().__init__(model, profile)
        self.self_contact_limits = None
        self.self_contact_depths = {}
        self.physics_steps = 0
        self.foot_bodies = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        }
        if min(self.foot_bodies) < 0:
            raise ValueError("protected plant requires original native23 foot bodies")

    def scratch(self, state):
        self.self_contact_depths, self.physics_steps = {}, 0
        return super().scratch(state)

    def set_research_contact_envelope(self, distances):
        limits = {tuple(pair): float(distance) for pair, distance in distances.items()}
        if any(len(pair) != 2 or not np.isfinite(distance) or distance >= 0 for pair, distance in limits.items()):
            raise ValueError("research contact envelope requires finite actually measured penetrations")
        self.self_contact_limits = limits

    def _tick(self, data, target):
        result = super()._tick(data, target)
        self.physics_steps += 1
        self._check_contact_geometry(data)
        return result

    def check_terminal_geometry(self, qpos, qvel):
        """Cover the final post-step pose without perturbing integrated state."""
        probe = mujoco.MjData(self.model)
        probe.qpos[:], probe.qvel[:] = qpos, qvel
        mujoco.mj_forward(self.model, probe)
        self._check_contact_geometry(probe)

    def _check_contact_geometry(self, data):
        for contact in data.contact[: data.ncon]:
            bodies = [int(self.model.geom_bodyid[int(geom)]) for geom in (contact.geom1, contact.geom2)]
            if 0 in bodies and contact.dist < 0 and bodies[1 - bodies.index(0)] not in self.foot_bodies:
                raise ValueError(f"non-foot ground penetration at physics step {self.physics_steps}")
        for row in self_contact_rows(self.model, data):
            distance, pair = row["distance_m"], row["geoms"]
            if distance >= 0:
                continue
            self.self_contact_depths[pair] = min(self.self_contact_depths.get(pair, 0.0), distance)
            if self.self_contact_limits is not None:
                floor = self.self_contact_limits.get(pair, 0.0)
                if distance < floor - CONTACT_COMPARISON_TOLERANCE_M:
                    raise ValueError(
                        f"self-contact nonregression rejected pair {pair} at physics step {self.physics_steps}: "
                        f"distance={distance:.9g}, initial_envelope={floor:.9g}"
                    )

    def contact_observations(self):
        return dict(
            physics_steps=self.physics_steps,
            minimum_self_distance_by_pair=[
                dict(geoms=list(pair), distance_m=distance)
                for pair, distance in sorted(self.self_contact_depths.items())
            ],
            research_envelope_not_collision_or_hardware_qualification=True,
        )


def protected_tracking_acceptance(initial_metrics, candidate_metrics):
    """Do not sacrifice a previously passing source landmark to lower mean cost."""
    initial = initial_metrics["lifecycle"]["source_motion_tracking"]["landmark_position_p95_m"]
    candidate = candidate_metrics["lifecycle"]["source_motion_tracking"]["landmark_position_p95_m"]
    violations = []
    for (name, _, _), threshold in zip(LANDMARKS, LANDMARK_THRESHOLDS, strict=True):
        ceiling = max(initial[name], float(threshold))
        if not np.isfinite(candidate[name]) or candidate[name] > ceiling + 1e-12:
            violations.append(dict(landmark=name, candidate_p95_m=candidate[name], fixed_ceiling_m=ceiling))
    return dict(
        accepted=not violations,
        violations=violations,
        original_qualification_thresholds_unchanged=True,
        simulator_qualified=False,
    )
