"""Offline whole-path inverse-force derivatives for native23 retargeting.

Contact forces are external optimization variables, so inverse dynamics is
evaluated on a separate constraint-disabled model, not a changed simulator.
No reference, policy, controller, hardware or motion acceptance is implied.

MuJoCo returns transposed, continuous-time finite-difference Jacobians:
https://mujoco.readthedocs.io/en/3.5.0/APIreference/APIfunctions.html#mjd-inversefd
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear

from gear_sonic.utils.g1_true23_contact_trajectory import (
    VARIABLE_DOFS,
    ContactLinearization,
    ContactTrajectoryConfig,
)
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_reference_support import (
    contact_cone_rays,
    floor_contact_map,
    pose_path_derivatives,
)


def pose_derivative_operator(frames, dt=0.02):
    """Sparse central differences with the same second-order one-sided ends."""
    if type(frames) is not int or frames < 3 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("force trajectory needs at least three frames and positive finite timestep")
    operator = sparse.diags(
        [-np.ones(frames - 1), np.ones(frames - 1)], [-1, 1], shape=(frames, frames), format="lil"
    )
    operator[0, :3] = [-3, 4, -1]
    operator[-1, -3:] = [1, -4, 3]
    return (operator / (2 * dt)).tocsc()


class InverseForcePath:
    """M(q) a + bias(q,v) - passive(q,v), including all six unactuated rows.

    Path variables are original-root XYZ offsets and all 23 joint positions.
    Root orientation and timing remain fixed. Velocity and acceleration come
    from the complete path, not from stale archived channels or independent
    frame resets. Scalar derivatives couple neighboring frames and endpoints.
    """

    def __init__(self, model, source_qpos, *, derivative_epsilon=1e-6):
        source = np.asarray(source_qpos, dtype=float)
        if (
            source.ndim != 2
            or source.shape[1] != 30
            or len(source) < 3
            or not np.isfinite(source).all()
            or np.any(np.linalg.norm(source[:, 3:7], axis=1) < 1e-12)
        ):
            raise ValueError("force path requires at least three finite native23 poses with nonzero rotations")
        if not np.isfinite(derivative_epsilon) or not 0 < derivative_epsilon <= 1e-3:
            raise ValueError("inverse-force derivative epsilon must be finite and in (0,1e-3]")
        if model.neq or model.ntendon or model.opt.noslip_iterations:
            raise ValueError("inverse-force path requires unconstrained native23 articulation without noslip")
        if model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError("inverse-force derivatives do not support RK4")
        # Validate the exact joint/state layout and fixed horizontal floor.
        ContactLinearization({"source": model}, source, [()] * len(source), ContactTrajectoryConfig())
        self.source = source.copy()
        self.original_model = model
        self.original_model_sha256 = compiled_model_sha256(model)
        self.model = copy.copy(model)
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        self.model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
        self.data = mujoco.MjData(self.model)
        self.epsilon = derivative_epsilon
        self.first = pose_derivative_operator(len(source))
        self.second = self.first @ self.first
        self.velocity_jacobian = sparse.kron(self.first, sparse.eye(26), format="csc")
        self.acceleration_jacobian = sparse.kron(self.second, sparse.eye(26), format="csc")

    def state(self, path):
        path = np.asarray(path, dtype=float)
        if path.shape != (len(self.source), 26) or not np.isfinite(path).all():
            raise ValueError("force path must be finite [frames,26] root-offset/joint positions")
        qpos = self.source.copy()
        qpos[:, :3] += path[:, :3]
        qpos[:, 7:] = path[:, 3:]
        qvel, qacc = pose_path_derivatives(self.original_model, qpos, 0.02)
        return qpos, qvel, qacc

    def evaluate(self, path, *, jacobian=True):
        """Return all required generalized forces and their full-path Jacobian.

        Df/Dq and Df/Dv use MuJoCo's forward finite differences. Df/Da is
        exactly the mass matrix because constraint forces are explicit external
        unknowns and invdiscrete is disabled on this private derivative model.
        No support or joint friction-loss force is silently supplied here.
        """
        qpos, qvel, qacc = self.state(path)
        model, data = self.model, self.data
        forces, position_blocks, velocity_blocks, acceleration_blocks = [], [], [], []
        for position, velocity, acceleration in zip(qpos, qvel, qacc):
            data.qpos[:] = position
            data.qvel[:] = velocity
            data.qacc[:] = acceleration
            mujoco.mj_inverse(model, data)
            forces.append(data.qfrc_inverse.copy())
            if jacobian:
                dq, dv, mass = (np.zeros((model.nv, model.nv)) for _ in range(3))
                mujoco.mj_fullM(model, mass, data.qM)
                mujoco.mjd_inverseFD(model, data, self.epsilon, 0, dq, dv, None, None, None, None, None)
                position_blocks.append(sparse.csr_matrix(dq.T[:, VARIABLE_DOFS]))
                velocity_blocks.append(sparse.csr_matrix(dv.T[:, VARIABLE_DOFS]))
                acceleration_blocks.append(sparse.csr_matrix(mass[:, VARIABLE_DOFS]))
        forces = np.asarray(forces)
        derivative = None
        if jacobian:
            derivative = (
                sparse.block_diag(position_blocks, format="csc")
                + sparse.block_diag(velocity_blocks, format="csc") @ self.velocity_jacobian
                + sparse.block_diag(acceleration_blocks, format="csc") @ self.acceleration_jacobian
            )
        if not np.isfinite(forces).all() or (jacobian and not np.isfinite(derivative.data).all()):
            raise ValueError("inverse-force path produced nonfinite force/derivative values")
        if compiled_model_sha256(self.original_model) != self.original_model_sha256:
            raise RuntimeError("inverse-force path source model changed")
        return forces, derivative


def fit_contact_forces(required, contact_map, torque_limits, friction_limits, *, body_weight_n):
    """Bounded least-squares warm start, not a force-feasibility certificate.

    Forces are scaled by body weight and joint torques by their unchanged
    supplied limits. Friction-loss assistance remains explicitly optimistic,
    as in the independent support audit. A residual never counts as support.
    """
    required, contact_map, torque_limits, friction_limits = (
        np.asarray(value, dtype=float) for value in (required, contact_map, torque_limits, friction_limits)
    )
    if (
        required.shape != (29,)
        or contact_map.ndim != 2
        or contact_map.shape[0] != 29
        or torque_limits.shape != (23,)
        or friction_limits.shape != (23,)
        or any(not np.isfinite(value).all() for value in (required, contact_map, torque_limits, friction_limits))
        or not np.isfinite(body_weight_n)
        or body_weight_n <= 0
        or np.any(torque_limits <= 0)
        or np.any(friction_limits < 0)
    ):
        raise ValueError("contact force fit requires finite native23 forces and positive torque/weight scales")
    scale = np.r_[np.full(3, body_weight_n), np.full(3, body_weight_n * 0.5), torque_limits]
    joint_map = np.vstack((np.zeros((6, 23)), np.diag(torque_limits)))
    friction_indices = np.flatnonzero(friction_limits > 0)
    friction_map = joint_map[:, friction_indices]
    mapping = np.column_stack((body_weight_n * contact_map, joint_map, friction_map))
    rays = contact_map.shape[1]
    lower = np.r_[
        np.zeros(rays), -np.ones(23), -friction_limits[friction_indices] / torque_limits[friction_indices]
    ]
    upper = np.r_[
        np.full(rays, np.inf), np.ones(23), friction_limits[friction_indices] / torque_limits[friction_indices]
    ]
    # Strictly convex regularized seed; final acceptance uses an independent
    # unregularized force LP and all nonlinear contacts, never this optimum.
    matrix = np.vstack((mapping / scale[:, None], 1e-6 * np.eye(mapping.shape[1])))
    target = np.r_[required / scale, np.zeros(mapping.shape[1])]
    result = lsq_linear(matrix, target, bounds=(lower, upper), method="bvls", tol=1e-9, max_iter=500)
    if not np.isfinite(result.x).all() or np.any(result.x < lower - 1e-8) or np.any(result.x > upper + 1e-8):
        raise ValueError("contact force seed failed independent finite/bound checks")
    physical_friction = np.zeros(23)
    physical_friction[friction_indices] = result.x[rays + 23 :] * torque_limits[friction_indices]
    residual = required - mapping @ result.x
    return {
        "ray_weights": result.x[:rays] * body_weight_n,
        "joint_torque": result.x[rays : rays + 23] * torque_limits,
        "friction_assistance": physical_friction,
        "residual": residual,
        "normalized_residual": residual / scale,
        "force_scale": scale,
        "seed_solver_success": bool(result.success),
        "force_feasibility_proven": False,
    }


class FrozenContactLoad:
    """Local linearization hypothesis: body-attached points, world cone rays.

    These loads match the supplied current contact map exactly at the base
    pose. Nearby-pose derivatives hold the force weights, attachment points
    and world cone directions fixed. They do NOT predict changing closest
    features or establish contact persistence. Every candidate must recompute
    actual contacts and independent force feasibility before acceptance.
    """

    def __init__(self, model, data, contacts, ray_weights):
        weights = np.asarray(ray_weights, dtype=float)
        if weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < -1e-8):
            raise ValueError("contact load requires finite nonnegative ray weights")
        self.model, self.data = model, mujoco.MjData(model)
        self.base_qpos = data.qpos.copy()
        self.loads = []
        count = 0
        for record in contacts:
            body = model.body(record["body"]).id
            start, length = record["cone_ray_start"], record["cone_ray_count"]
            if start != count or start + length > len(weights):
                raise ValueError("contact ray layout must cover each force coefficient exactly once")
            # Flat floor uses force coordinates [normal, tangent1, tangent2].
            # Recover MuJoCo's contact frame from the actual matching record;
            # do not infer an arbitrary tangent orientation from the normal.
            matches = [
                item
                for item in data.contact[: data.ncon]
                if record["robot_geom_id"] in item.geom
                and np.allclose(item.pos, record["position_w"], atol=1e-12, rtol=0)
            ]
            if len(matches) != 1:
                raise ValueError("frozen load requires one exact actual contact frame")
            contact = matches[0]
            plane_first = model.geom_type[int(contact.geom[0])] == mujoco.mjtGeom.mjGEOM_PLANE
            rotation = (1 if plane_first else -1) * np.asarray(contact.frame).reshape(3, 3).T
            if not np.allclose(rotation[:, 0], [0, 0, 1], atol=1e-7, rtol=0):
                raise ValueError("contact load requires upward flat-floor cone normals")
            local_point = data.xmat[body].reshape(3, 3).T @ (np.asarray(record["position_w"]) - data.xpos[body])
            rays = contact_cone_rays(record["dimension"], record["friction"])
            if rays.shape[1] != length:
                raise ValueError("contact cone dimension disagrees with supplied force coefficients")
            local_wrench = rays @ weights[start : start + length]
            self.loads.append((body, local_point, rotation @ local_wrench[:3], rotation @ local_wrench[3:]))
            count += length
        if count != len(weights):
            raise ValueError("unused contact force coefficients")

    def evaluate(self, qpos):
        model, data = self.model, self.data
        qpos = np.asarray(qpos, dtype=float)
        if qpos.shape != (30,) or not np.isfinite(qpos).all():
            raise ValueError("frozen contact load needs a finite native23 pose")
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        load = np.zeros(model.nv)
        for body, local_point, force, torque in self.loads:
            point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local_point
            jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, jacr, point, body)
            load += jacp.T @ force + jacr.T @ torque
        return load

    def derivative(self, epsilon=1e-6):
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("contact-load derivative epsilon must be finite and positive")
        baseline = self.evaluate(self.base_qpos)
        result = np.empty((self.model.nv, 26))
        for column, index in enumerate(np.r_[0:3, 7:30]):
            position = self.base_qpos.copy()
            position[index] += epsilon
            result[:, column] = (self.evaluate(position) - baseline) / epsilon
        return result


class ForceLinearization:
    """Actual candidate cones plus whole-path required-force linearizations.

    Every model has its own forces but shares one native23 reference path.
    Frame-local force variables are normalized unilateral cone coefficients,
    bounded joint torques and explicitly optimistic friction-loss assistance.
    No generalized-force actuator is attached to the floating base.
    """

    def __init__(self, models, source_qpos, torque_limits, *, gap_tolerance_m=0.002, progress=None):
        limits = np.asarray(torque_limits, dtype=float)
        if limits.shape != (23,) or not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("force linearization requires positive finite native23 torque limits")
        if not models or not np.isfinite(gap_tolerance_m) or not 0 < gap_tolerance_m <= 0.01:
            raise ValueError("force linearization needs models and a bounded explicit floor gap")
        self.limits = limits.copy()
        self.gap = gap_tolerance_m
        self.progress = progress
        self.models = {}
        for name, model in models.items():
            if not np.allclose(model.opt.gravity, [0, 0, -9.81], atol=1e-8, rtol=0):
                raise ValueError("force linearization requires explicit Earth gravity")
            inverse = InverseForcePath(model, source_qpos)
            candidate = copy.copy(model)
            candidate.geom_margin[:] = np.maximum(candidate.geom_margin, self.gap)
            candidate.pair_margin[:] = np.maximum(candidate.pair_margin, self.gap)
            candidate.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
            plane = int(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)[0])
            self.models[name] = (inverse, candidate, mujoco.MjData(candidate), plane)

    def evaluate(self, path, *, jacobian=True):
        forces, derivatives, force_maps, lower, upper, seeds, scales, model_reports = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        joint_map = np.vstack((np.zeros((6, 23)), np.diag(self.limits)))
        for name, (inverse, model, data, plane) in self.models.items():
            if self.progress:
                self.progress({"force_model": name, "linearizing": jacobian, "starting_frames": len(path)})
            required, derivative = inverse.evaluate(path, jacobian=jacobian)
            qpos, qvel, _qacc = inverse.state(path)
            weight = float(model.body_mass.sum() * 9.81)
            friction = model.dof_frictionloss[6:]
            loads, frame_maps, frame_seeds, frame_scales = [], [], [], []
            fit_norms, no_contacts, seed_failures = [], 0, 0
            for frame, (position, velocity) in enumerate(zip(qpos, qvel)):
                data.qpos[:], data.qvel[:] = position, velocity
                mujoco.mj_fwdPosition(model, data)
                mujoco.mj_fwdVelocity(model, data)
                force_map, records = floor_contact_map(model, data, plane, self.gap)
                fit = fit_contact_forces(required[frame], force_map, self.limits, friction, body_weight_n=weight)
                rays = force_map.shape[1]
                frame_maps.append(sparse.csc_matrix(np.column_stack((weight * force_map, joint_map, joint_map))))
                frame_seeds.append(
                    np.r_[
                        fit["ray_weights"] / weight,
                        fit["joint_torque"] / self.limits,
                        fit["friction_assistance"] / self.limits,
                    ]
                )
                frame_scales.append(fit["force_scale"])
                lower.append(np.r_[np.zeros(rays), -np.ones(23), -friction / self.limits])
                upper.append(np.r_[np.full(rays, np.inf), np.ones(23), friction / self.limits])
                fit_norms.append(float(np.sum(fit["normalized_residual"] ** 2)))
                no_contacts += not records
                seed_failures += not fit["seed_solver_success"]
                if jacobian:
                    load = FrozenContactLoad(model, data, records, fit["ray_weights"])
                    if not np.allclose(
                        load.evaluate(position), force_map @ fit["ray_weights"], atol=1e-8, rtol=1e-10
                    ):
                        raise ValueError("current attached load disagrees with actual contact cone map")
                    loads.append(sparse.csr_matrix(load.derivative(inverse.epsilon)))
                if self.progress and ((frame + 1) % 200 == 0 or frame == len(qpos) - 1):
                    self.progress({"force_model": name, "linearizing": jacobian, "frames_processed": frame + 1})
            forces.append(required.ravel())
            if jacobian:
                derivatives.append(derivative - sparse.block_diag(loads, format="csc"))
            force_maps.append(sparse.block_diag(frame_maps, format="csc"))
            seeds.extend(frame_seeds)
            scales.extend(frame_scales)
            model_reports.append(
                {
                    "model": name,
                    "frames": len(qpos),
                    "frames_without_candidate_contact": no_contacts,
                    "seed_solver_failures": seed_failures,
                    "summed_normalized_force_residual_squared": sum(fit_norms),
                    "maximum_normalized_frame_force_residual_squared": max(fit_norms),
                    "candidate_cones_recomputed_from_actual_geometry": True,
                    "force_feasibility_proven": False,
                    "source_model_sha256": inverse.original_model_sha256,
                    "private_inverse_model_sha256": compiled_model_sha256(inverse.model),
                    "candidate_contact_model_sha256": compiled_model_sha256(model),
                }
            )
        required = np.concatenate(forces)
        force_map = sparse.block_diag(force_maps, format="csc")
        seed, scale = np.concatenate(seeds), np.concatenate(scales)
        residual = (required - force_map @ seed) / scale
        return {
            "required": required,
            "jacobian": sparse.vstack(derivatives, format="csc") if jacobian else None,
            "force_map": force_map,
            "lower": np.concatenate(lower),
            "upper": np.concatenate(upper),
            "seed": seed,
            "scale": scale,
            "normalized_residual": residual,
            "summed_normalized_force_residual_squared": float(np.sum(residual**2)),
            "maximum_absolute_generalized_force_residual": float(np.max(np.abs(residual * scale))),
            "models": model_reports,
            "force_feasibility_proven": False,
        }
