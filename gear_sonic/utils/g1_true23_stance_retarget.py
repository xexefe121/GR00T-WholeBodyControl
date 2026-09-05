"""Offline native23 contact/COM retargeting; no control or authorization.

Explicit stance hypotheses may change all 23 joint references and root XYZ,
never root orientation or sample timing. Position IK is followed by whole-path
derivative projection and independent contact/force audits. Those audits, not
optimizer success, determine which hypotheses failed. No clip is discarded.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
)
from gear_sonic.utils.g1_23dof_task_space_retarget import build_mjlab_motion_arrays, safe_target_joint_bounds
from gear_sonic.utils.g1_23dof_trajectory_projection import (
    audit_trajectory_constraints,
    project_nearest_trajectory,
)
from gear_sonic.utils.g1_true23_contact_geometry import lift_reset_floor_overlap
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos, reference_geometry
from gear_sonic.utils.g1_true23_sim_acquisition import audit_reference_kinematics

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
HANDS = ("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand")
ELBOWS = ("left_elbow_link", "right_elbow_link")
FAMILIES = ("biped_stance", "biped_motion", "hand_crawl", "elbow_crawl")
VARIABLE_DOFS = np.r_[0:3, 6:29]


@dataclass(frozen=True)
class StanceRetargetConfig:
    maximum_root_offset_m: float = 0.08
    maximum_joint_change_rad: float = 0.6
    maximum_joint_velocity_rad_s: float = 5.0
    maximum_joint_acceleration_rad_s2: float = 80.0
    maximum_offset_velocity_m_s: float = 0.75
    maximum_offset_acceleration_m_s2: float = 6.0
    candidate_gap_m: float = 0.002
    contact_clearance_m: float = 0.0002
    com_margin_m: float = 0.015
    center_biped_stance_com: bool = True
    floor_residual_weight: float = 5000.0
    maximum_frame_evaluations: int = 80

    def __post_init__(self):
        if any(
            not np.isfinite(value) or value <= 0
            for name, value in asdict(self).items()
            if name != "center_biped_stance_com"
        ):
            raise ValueError("stance retarget bounds must be finite and positive")
        if type(self.center_biped_stance_com) is not bool:
            raise ValueError("stance COM centering must be explicit boolean")
        if self.contact_clearance_m >= self.candidate_gap_m or self.candidate_gap_m > 0.01:
            raise ValueError("stance clearance must be smaller than explicit near-contact gap")
        if type(self.maximum_frame_evaluations) is not int:
            raise ValueError("frame evaluation budget must be an integer")


def skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


class SupportGeometry:
    """Pose a single body's colliders for contact-target construction only."""

    def __init__(self, model, gap):
        lift_reset_floor_overlap(model, model.qpos0[None, :])
        self.model = copy.copy(model)
        self.model.geom_margin[:] = np.maximum(self.model.geom_margin, gap)
        self.model.pair_margin[:] = np.maximum(self.model.pair_margin, gap)
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        self.data = mujoco.MjData(self.model)
        self.plane = int(np.flatnonzero(self.model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)[0])
        self.gap = gap
        self.cache = {}

    def _pose(self, name, rotation, height):
        model, data = self.model, self.data
        body = model.body(name).id
        data.qpos[:] = model.qpos0
        mujoco.mj_kinematics(model, data)
        delta = rotation @ data.xmat[body].reshape(3, 3).T
        root_rotation = Rotation.from_quat(data.qpos[[4, 5, 6, 3]]).as_matrix()
        data.qpos[3:7] = Rotation.from_matrix(delta @ root_rotation).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_kinematics(model, data)
        data.qpos[:3] += np.array([0, 0, height]) - data.xpos[body]
        mujoco.mj_fwdPosition(model, data)
        return body

    def depth(self, name, rotation):
        key = (name, np.asarray(rotation).tobytes())
        if key not in self.cache:
            body = self._pose(name, rotation, 1.5)
            geoms = [
                i
                for i in range(self.model.ngeom)
                if self.model.geom_bodyid[i] == body
                and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])
            ]
            if not geoms:
                raise ValueError(f"no collision geometry on support body {name}")
            distance = min(mujoco.mj_geomDistance(self.model, self.data, self.plane, i, 3.0, None) for i in geoms)
            self.cache[key] = 1.5 - distance
        return self.cache[key]

    def points(self, name, rotation, height):
        body = self._pose(name, rotation, height)
        points = []
        for contact in self.data.contact[: self.data.ncon]:
            if self.plane not in contact.geom or contact.dist > self.gap:
                continue
            other = int(contact.geom[1] if contact.geom[0] == self.plane else contact.geom[0])
            if self.model.geom_bodyid[other] == body:
                points.append(contact.pos[:2].copy())
        return np.asarray(points).reshape(-1, 2)


def stance_schedule(model, motion, family):
    """Explicit family plus conservative geometric/velocity contact inference.

    Biped stance is an explicit two-foot hypothesis. Motion families infer
    contacts from each candidate body's lowest collider relative to the clip's
    2% floor quantile and body speed <=0.45 m/s. This is not observed contact.
    """
    if family not in FAMILIES:
        raise ValueError("stance family must be explicit and supported")
    bodies = FEET + (HANDS if family == "hand_crawl" else ELBOWS if family == "elbow_crawl" else ())
    positions = motion_qpos(model, motion)
    if family == "biped_stance":
        return bodies, np.ones((len(positions), len(bodies)), dtype=bool)
    data = mujoco.MjData(model)
    plane = model.geom("floor").id
    ids = [model.body(name).id for name in bodies]
    geom_ids = [
        [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == body and (model.geom_contype[i] or model.geom_conaffinity[i])
        ]
        for body in ids
    ]
    distances, body_positions = [], []
    for position in positions:
        data.qpos[:] = position
        mujoco.mj_fwdPosition(model, data)
        distances.append(
            [min(mujoco.mj_geomDistance(model, data, plane, i, 3.0, None) for i in group) for group in geom_ids]
        )
        body_positions.append(data.xpos[ids].copy())
    distances = np.asarray(distances)
    speed = np.linalg.norm(np.gradient(np.asarray(body_positions), 0.02, axis=0), axis=2)
    return bodies, (distances <= np.quantile(distances, 0.02) + 0.035) & (speed <= 0.45)


def closest_supported_com(desired, polygons, margin):
    """Nearest XY in the intersection of full-dimensional support polygons.

    Degenerate/missing polygons do not acquire an invented support area. Such
    frames keep the source COM objective and are explicitly marked unsupported.
    """
    equations = []
    for points in polygons:
        if len(points) < 3:
            return np.array(desired, copy=True), False
        try:
            hull = ConvexHull(points)
        except QhullError:
            return np.array(desired, copy=True), False
        equations.append(hull.equations)
    if not equations:
        return np.array(desired, copy=True), False
    equations = np.vstack(equations)
    a, b = equations[:, :2], equations[:, 2] + margin
    result = minimize(
        lambda value: 0.5 * np.sum((value - desired) ** 2),
        np.array(desired),
        jac=lambda value: value - desired,
        method="SLSQP",
        constraints={"type": "ineq", "fun": lambda value: -(a @ value + b), "jac": lambda value: -a},
        options={"maxiter": 100, "ftol": 1e-12},
    )
    if not result.success or np.max(a @ result.x + b) > 1e-7:
        return np.array(desired, copy=True), False
    return result.x, True


class FloorResidual:
    """Soft clearance objective using actual collider distance/Jacobian.

    The plane-distance derivative is the world-z point Jacobian at the closest
    point on the robot shape. Feature changes are nonsmooth. This objective is
    NOT a hard floor constraint, especially after temporal projection.
    """

    def __init__(self, model, clearance, weight):
        self.model, self.data = model, mujoco.MjData(model)
        self.clearance, self.weight = clearance, weight
        self.plane = int(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)[0])
        self.geoms = [
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] > 0 and (model.geom_contype[i] or model.geom_conaffinity[i])
        ]

    def evaluate(self, qpos):
        model, data = self.model, self.data
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        residual = np.zeros(len(self.geoms))
        jacobian = np.zeros((len(self.geoms), 26))
        for row, geom in enumerate(self.geoms):
            segment = np.empty(6)
            gap = mujoco.mj_geomDistance(model, data, self.plane, geom, 0.05, segment)
            if gap >= self.clearance:
                continue
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, None, segment[3:], int(model.geom_bodyid[geom]))
            residual[row] = self.weight * (gap - self.clearance)
            jacobian[row] = self.weight * jacp[2, VARIABLE_DOFS]
        return residual, jacobian


class FrameProblem:
    """Bounded whole-body IK with analytic position/axis/COM Jacobians."""

    def __init__(
        self,
        model,
        source,
        support_targets,
        com_target,
        previous_delta,
        *,
        collision_models=None,
        config=StanceRetargetConfig(),
    ):
        self.model, self.source = model, source.copy()
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = source
        mujoco.mj_kinematics(model, self.data)
        mujoco.mj_comPos(model, self.data)
        self.targets = []
        for name in FEET + HANDS + ELBOWS + ("torso_link",):
            if name in support_targets:
                continue
            body = model.body(name).id
            self.targets.append(
                (
                    body,
                    self.data.xpos[body].copy(),
                    self.data.xmat[body].reshape(3, 3).copy(),
                    20.0 if name in HANDS + FEET else 5.0,
                    2.0,
                )
            )
        for name, (position, rotation) in support_targets.items():
            self.targets.append((model.body(name).id, position, rotation, 500.0, 50.0))
        self.com_target = com_target
        self.floor_objectives = [
            FloorResidual(value, config.contact_clearance_m, config.floor_residual_weight)
            for value in (collision_models or {}).values()
        ]
        self.previous_delta = previous_delta.copy()
        self.root_body = model.body("pelvis").id
        self.last_x = None

    def evaluate(self, x):
        if self.last_x is not None and np.array_equal(self.last_x, x):
            return self.last_result
        model, data = self.model, self.data
        data.qpos[:] = self.source
        data.qpos[:3] += x[:3]
        data.qpos[7:] = x[3:]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        residuals, jacobians = [], []
        for body, target, desired_rotation, position_weight, rotation_weight in self.targets:
            jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body], body)
            residuals.append(position_weight * (data.xpos[body] - target))
            jacobians.append(position_weight * jacp[:, VARIABLE_DOFS])
            rotation = data.xmat[body].reshape(3, 3)
            for axis in range(3):
                residuals.append(rotation_weight * (rotation[:, axis] - desired_rotation[:, axis]))
                jacobians.append(-rotation_weight * skew(rotation[:, axis]) @ jacr[:, VARIABLE_DOFS])
        if self.com_target is not None:
            jaccom = np.zeros((3, model.nv))
            mujoco.mj_jacSubtreeCom(model, data, jaccom, self.root_body)
            residuals.append(200.0 * (data.subtree_com[self.root_body, :2] - self.com_target))
            jacobians.append(200.0 * jaccom[:2, VARIABLE_DOFS])
        source_x = np.r_[np.zeros(3), self.source[7:]]
        weights = np.r_[np.full(3, 10.0), np.full(23, 0.5)]
        residuals.extend((weights * (x - source_x), 0.1 * weights * (x - source_x - self.previous_delta)))
        jacobians.extend((np.diag(weights), np.diag(0.1 * weights)))
        for objective in self.floor_objectives:
            residual, jacobian = objective.evaluate(data.qpos)
            residuals.append(residual)
            jacobians.append(jacobian)
        self.last_x = x.copy()
        self.last_result = (np.concatenate(residuals), np.vstack(jacobians))
        return self.last_result


def retarget_stance_motion(
    model, collision_models, motion, family, *, config=StanceRetargetConfig(), progress=None
):
    """Whole-clip candidate, never automatic teacher/model promotion."""
    source = motion_qpos(model, motion)
    fk = audit_reference_kinematics(SimpleNamespace(module=mujoco, model=model), motion)
    if not fk["position_fk_consistent"] or not fk["orientation_fk_consistent"]:
        raise ValueError("stance retarget input body channels disagree with native23 FK")
    if not collision_models:
        raise ValueError("stance retarget requires explicit collision models")
    model_hash = compiled_model_sha256(model)
    model_hashes = {name: compiled_model_sha256(value) for name, value in collision_models.items()}
    geometries = {name: SupportGeometry(value, config.candidate_gap_m) for name, value in collision_models.items()}
    bodies, active = stance_schedule(model, motion, family)
    source_data = mujoco.MjData(model)
    anchors = {}
    solved = []
    records = []
    previous_delta = np.zeros(26)
    safe_low, safe_high = safe_target_joint_bounds(model, native_action_clip=9.5)
    safe_low = np.maximum(safe_low, np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + 0.05)
    safe_high = np.minimum(safe_high, np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) - 0.05)
    lower = np.column_stack(
        (
            np.full((len(source), 3), -config.maximum_root_offset_m),
            np.maximum(source[:, 7:] - config.maximum_joint_change_rad, safe_low),
        )
    )
    upper = np.column_stack(
        (
            np.full((len(source), 3), config.maximum_root_offset_m),
            np.minimum(source[:, 7:] + config.maximum_joint_change_rad, safe_high),
        )
    )
    if np.any(lower >= upper):
        raise ValueError("source needs more than the explicit joint-correction bound; no frame may be removed")
    for index, pose in enumerate(source):
        source_data.qpos[:] = pose
        mujoco.mj_kinematics(model, source_data)
        mujoco.mj_comPos(model, source_data)
        targets = {}
        polygons = {name: [] for name in geometries}
        for column, body_name in enumerate(bodies):
            if not active[index, column]:
                anchors.pop(body_name, None)
                continue
            body = model.body(body_name).id
            rotation = source_data.xmat[body].reshape(3, 3).copy()
            if body_name not in anchors:
                yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                anchors[body_name] = (source_data.xpos[body, :2].copy(), Rotation.from_euler("z", yaw).as_matrix())
            xy, flat_rotation = anchors[body_name]
            if body_name in FEET:
                rotation = flat_rotation
            height = (
                max(value.depth(body_name, rotation) for value in geometries.values()) + config.contact_clearance_m
            )
            targets[body_name] = (np.r_[xy, height], rotation)
            for name, geometry in geometries.items():
                polygons[name].extend(geometry.points(body_name, rotation, height) + xy)
        desired_com = source_data.subtree_com[model.body("pelvis").id, :2]
        if family == "biped_stance" and config.center_biped_stance_com:
            desired_com = np.mean([targets[name][0][:2] for name in FEET], axis=0)
        com, polygon_pass = closest_supported_com(
            desired_com,
            [np.asarray(value) for value in polygons.values()],
            config.com_margin_m,
        )
        problem = FrameProblem(
            model,
            pose,
            targets,
            com if targets else None,
            previous_delta,
            collision_models=collision_models,
            config=config,
        )
        start = np.clip(np.r_[np.zeros(3), pose[7:]] + previous_delta, lower[index] + 1e-9, upper[index] - 1e-9)
        result = least_squares(
            lambda x: problem.evaluate(x)[0],
            start,
            jac=lambda x: problem.evaluate(x)[1],
            bounds=(lower[index], upper[index]),
            max_nfev=config.maximum_frame_evaluations,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        solved.append(result.x.copy())
        previous_delta = result.x - np.r_[np.zeros(3), pose[7:]]
        records.append(
            {
                "frame": index,
                "support_bodies": list(targets),
                "common_support_polygon_found": polygon_pass,
                "com_target_xy": com.tolist(),
                "support_targets": {
                    name: {"position_w": position.tolist(), "rotation_matrix_w": rotation.tolist()}
                    for name, (position, rotation) in targets.items()
                },
                "solver_success": bool(result.success),
                "nfev": result.nfev,
                "cost": float(result.cost),
            }
        )
        if progress is not None and (index % 100 == 0 or index == len(source) - 1):
            progress({"frames_solved": index + 1, "total_frames": len(source)})
    solved = np.asarray(solved)
    velocity = np.r_[
        np.full(3, config.maximum_offset_velocity_m_s), np.full(23, config.maximum_joint_velocity_rad_s)
    ]
    acceleration = np.r_[
        np.full(3, config.maximum_offset_acceleration_m_s2), np.full(23, config.maximum_joint_acceleration_rad_s2)
    ]
    initial_velocity = np.clip((solved[1] - solved[0]) / 0.02, -velocity * 0.995, velocity * 0.995)
    projected = project_nearest_trajectory(
        solved,
        lower_bounds=lower,
        upper_bounds=upper,
        dt=0.02,
        max_velocity=velocity * 0.995,
        max_acceleration=acceleration * 0.995,
        initial_velocity=initial_velocity,
    )
    root_pos = (source[:, :3] + projected.projected_path[:, :3]).astype(np.float32)
    joints = projected.projected_path[:, 3:].astype(np.float32)
    serialized = np.column_stack((root_pos.astype(float) - source[:, :3], joints))
    trajectory_audit = audit_trajectory_constraints(
        serialized,
        lower_bounds=lower,
        upper_bounds=upper,
        dt=0.02,
        max_velocity=velocity,
        max_acceleration=acceleration,
        initial_velocity=initial_velocity,
        tolerance=2e-7,
    )
    if not trajectory_audit.passed:
        raise ValueError("serialized stance candidate failed whole-path derivative bounds")
    arrays = build_mjlab_motion_arrays(
        model,
        SimpleNamespace(root_pos_w=root_pos, root_quat_wxyz=source[:, 3:7], joint_pos_hardware=joints, fps=50.0),
    )
    output_fk = audit_reference_kinematics(SimpleNamespace(module=mujoco, model=model), arrays)
    if not output_fk["position_fk_consistent"] or not output_fk["orientation_fk_consistent"]:
        raise RuntimeError("serialized stance candidate failed native23 FK consistency")
    after = {name: reference_geometry(value, arrays) for name, value in collision_models.items()}
    if model_hash != compiled_model_sha256(model) or model_hashes != {
        name: compiled_model_sha256(value) for name, value in collision_models.items()
    }:
        raise RuntimeError("stance retarget mutated a supplied collision model")
    return arrays, {
        "kind": "g1_true23_whole_path_stance_retarget_candidate_v1",
        "family": family,
        "config": asdict(config),
        "frames_in": len(source),
        "frames_out": len(joints),
        "frames_dropped": 0,
        "controlled_joint_count": 23,
        "optimized_joint_names": list(HARDWARE_23_JOINT_NAMES),
        "root_orientation_changed": False,
        "time_scale": 1.0,
        "maximum_joint_change_rad": float(np.abs(joints - source[:, 7:]).max()),
        "maximum_root_offset_m_per_axis": np.abs(root_pos - source[:, :3]).max(axis=0).tolist(),
        "maximum_body_position_change_m": float(
            np.linalg.norm(arrays["body_pos_w"] - motion["body_pos_w"], axis=2).max()
        ),
        "initial_projection_velocity": initial_velocity.tolist(),
        "serialized_trajectory_audit": asdict(trajectory_audit),
        "projection_iterations": projected.iterations,
        "input_fk": fk,
        "output_fk": output_fk,
        "kinematic_model_sha256": model_hash,
        "collision_model_sha256": model_hashes,
        "geometry_after": after,
        "stance_hypothesis_observed_or_verified": False,
        "floor_objective_is_hard_constraint": False,
        "solver_rows": records,
        "old_packet_joint_proofs_reusable": False,
        "causal_terms_must_be_rebuilt_from_new_positions": True,
        "torque_feasibility_optimized_or_proven": False,
        "dynamic_feasibility_proven": False,
        "full_clip_tracking_qualified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
