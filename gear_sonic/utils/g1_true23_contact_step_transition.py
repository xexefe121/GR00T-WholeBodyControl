"""Offline native23 stepping entry/exit references, never robot commands.

Unlike joint-space ramps, the explicit contact schedule keeps support feet at
fixed world XY while transferring COM, then lifts and places one foot at a time.
This is a reference hypothesis, not dynamically certified policy behavior.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

from gear_sonic.utils.g1_23dof_task_space_retarget import safe_target_joint_bounds
from gear_sonic.utils.g1_true23_affine_task_soc import solve_affine_task_soc
from gear_sonic.utils.g1_true23_generalist_lifecycle import _blend_poses
from gear_sonic.utils.g1_true23_stance_foot_cleanup import foot_frames, foot_task_residual_jacobian

PROFILE = "fixed_world_lipm_single_support_step_transitions_v2"
STAGES = (
    ("ground", 50),
    ("shift_first", 75),
    ("swing_first", 35),
    ("shift_second", 75),
    ("swing_second", 35),
    ("settle", 100),
)


def smooth_fraction(value):
    value = np.asarray(value, dtype=float)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def swing_lift(value, height=0.04):
    """Zero value/velocity/acceleration at lift/land; requested apex 4 cm."""
    value = np.asarray(value, dtype=float)
    return height * 64 * value**3 * (1 - value) ** 3


def _rotation_mix(start, end, fraction):
    return Slerp([0, 1], Rotation.from_matrix(np.stack((start, end))))(float(fraction)).as_matrix()


def _sole_height(model, body, rotation, gap):
    geoms = [
        int(g)
        for g in np.flatnonzero(model.geom_bodyid == body)
        if model.geom_contype[g] or model.geom_conaffinity[g]
    ]
    if len(geoms) != 4 or any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE for g in geoms):
        raise ValueError("step transitions require four physical sole spheres per foot")
    return float(gap + np.max(model.geom_size[geoms, 0] - (model.geom_pos[geoms] @ rotation.T)[:, 2]))


def step_targets(model, start, finish):
    """Produce a fixed offline contact/COM schedule; never read rollout state."""
    endpoints = np.asarray([start, finish], dtype=float)
    if endpoints.shape != (2, 30) or not np.isfinite(endpoints).all():
        raise ValueError("step transition needs two finite exact native23 endpoint poses")
    if (model.nq, model.nv, model.nu, model.nbody) != (30, 29, 23, 25):
        raise ValueError("step transition requires native23 physical topology")
    feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    positions, rotations = foot_frames(model, endpoints)
    if np.max(np.linalg.norm(positions[1, :, :2] - positions[0, :, :2], axis=1)) > 0.4:
        raise ValueError("one-step entry cannot bridge a foot displacement above 40 cm")
    data = mujoco.MjData(model)
    com = []
    for pose in endpoints:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        com.append(data.subtree_com[model.body("pelvis").id, :2].copy())
    com = np.asarray(com)
    yaw = np.arctan2(rotations[:, :, 1, 0], rotations[:, :, 0, 0])
    flat = Rotation.from_euler("z", yaw.ravel()).as_matrix().reshape(2, 2, 3, 3)
    grounded = positions.copy()
    for end in range(2):
        for side in range(2):
            grounded[end, side, 2] = _sole_height(model, feet[side], flat[end, side], 0.0005)
    # Stay inside each sole, nearer the other foot, so weight transfer does not
    # demand more ankle roll than the unchanged native action bounds permit.
    centers = (
        grounded[:, :, :2]
        + np.einsum("esij,sj->esi", flat, np.array([[0.035, -0.012, 0], [0.035, 0.012, 0]]))[:, :, :2]
    )
    costs = [
        np.linalg.norm(com[0] - centers[0, 1 - first])
        + np.linalg.norm(centers[0, 1 - first] - centers[1, first])
        + np.linalg.norm(centers[1, first] - com[1])
        for first in (0, 1)
    ]
    first = int(np.argmin(costs))
    second = 1 - first
    total = sum(count for _, count in STAGES)
    # Endpoints remain outside the generated interior, exactly as the source.
    templates = np.vstack((start, _blend_poses(start, finish, total - 1), finish))
    targets = [
        dict(
            positions=positions[0].copy(),
            rotations=rotations[0].copy(),
            com=com[0].copy(),
            root_z=float(start[2]),
            support=(True, True),
            stage="initial_endpoint",
        )
    ]
    current_pos, current_rot = positions[0].copy(), rotations[0].copy()
    current_com, current_z = com[0].copy(), float(start[2])
    low_z = min(float(start[2]), float(finish[2])) - 0.04
    for stage, count in STAGES:
        goal_pos, goal_rot = current_pos.copy(), current_rot.copy()
        goal_com, goal_z = current_com.copy(), current_z
        swing = None
        if stage == "ground":
            goal_pos, goal_rot, goal_z = grounded[0].copy(), flat[0].copy(), low_z
        elif stage == "shift_first":
            goal_com = centers[0, second].copy()
            goal_rot[first] = (
                flat[0, first] @ Rotation.from_euler("x", (1 if first == 0 else -1) * 0.1).as_matrix()
            )
            goal_pos[first, 2] = _sole_height(model, feet[first], goal_rot[first], 0.0005)
        elif stage == "swing_first":
            swing = first
        elif stage == "shift_second":
            goal_com = centers[1, first].copy()
            goal_pos[first], goal_rot[first] = grounded[1, first], flat[1, first]
            goal_rot[second] = (
                flat[0, second] @ Rotation.from_euler("x", (1 if second == 0 else -1) * 0.1).as_matrix()
            )
            goal_pos[second, 2] = _sole_height(model, feet[second], goal_rot[second], 0.0005)
        elif stage == "swing_second":
            swing = second
        else:
            goal_pos, goal_rot = positions[1].copy(), rotations[1].copy()
            goal_com, goal_z = com[1].copy(), float(finish[2])
        if swing is not None:
            goal_pos[swing], goal_rot[swing] = grounded[1, swing], flat[1, swing]
            # Land on the inner sole edge, then flatten in double support.
            goal_rot[swing] = (
                goal_rot[swing] @ Rotation.from_euler("x", (1 if swing == 0 else -1) * 0.25).as_matrix()
            )
            goal_pos[swing, 2] = _sole_height(model, feet[swing], goal_rot[swing], 0.0005)
        for k in range(1, count + 1):
            t = k / count
            blend = float(smooth_fraction(t))
            pos = current_pos + blend * (goal_pos - current_pos)
            rot = np.stack([_rotation_mix(current_rot[side], goal_rot[side], blend) for side in range(2)])
            if swing is not None:
                pos[swing, 2] += float(swing_lift(t))
                # The unloaded foot may roll while clear of the floor. Keeping
                # it artificially horizontal at the apex overconstrains the
                # native ankle; this extra roll vanishes at lift and landing.
                angle = 0.0  # Toe-off/landing interpolation supplies the needed unloaded-foot roll.
                rot[swing] = rot[swing] @ Rotation.from_euler("x", angle).as_matrix()
            # During grounding/settling keep the original endpoint gap evolution
            # but remove orientation-induced floor penetration analytically.
            if swing is None:
                for side, body in enumerate(feet):
                    gap_a = current_pos[side, 2] - _sole_height(model, body, current_rot[side], 0)
                    gap_b = goal_pos[side, 2] - _sole_height(model, body, goal_rot[side], 0)
                    pos[side, 2] = _sole_height(model, body, rot[side], (1 - blend) * gap_a + blend * gap_b)
            targets.append(
                dict(
                    positions=pos,
                    rotations=rot,
                    com=current_com + blend * (goal_com - current_com),
                    root_z=current_z + blend * (goal_z - current_z),
                    support=tuple(side != swing for side in range(2)),
                    stage=stage,
                )
            )
        current_pos, current_rot = goal_pos, goal_rot
        current_com, current_z = goal_com, goal_z
    if len(targets) != len(templates):
        raise ValueError("transition target and pose timelines differ")
    return (
        templates,
        targets,
        dict(
            profile=PROFILE,
            first_swing_foot=("left", "right")[first],
            generated_interior_controls=total - 1,
            duration_s=total * 0.02,
            stages=[dict(name=name, controls=count) for name, count in STAGES],
            swing_clearance_m=0.04,
            landing_foot_roll_rad=0.25,
            additional_swing_foot_roll_rad=0.0,
            root_crouch_m=0.04,
            measured_robot_state_used=False,
            contact_schedule_is_hypothesis=True,
        ),
    )


def plan_dynamic_com(model, targets):
    """Fixed-height LIPM necessary-condition plan, not a full rigid-body proof."""
    count, dt, height = len(targets), 0.02, 0.65
    identity = sparse.eye(count, format="csc")
    d1 = sparse.diags((-np.ones(count - 1), np.ones(count - 1)), (0, 1), shape=(count - 1, count)) / dt
    d2 = (
        sparse.diags(
            (np.ones(count - 2), -2 * np.ones(count - 2), np.ones(count - 2)), (0, 1, 2), shape=(count - 2, count)
        )
        / dt**2
    )
    d3 = (
        sparse.diags(
            (-np.ones(count - 3), 3 * np.ones(count - 3), -3 * np.ones(count - 3), np.ones(count - 3)),
            (0, 1, 2, 3),
            shape=(count - 3, count),
        )
        / dt**3
    )
    zmp = sparse.kron(identity[1:-1] - height / 9.81 * d2, sparse.eye(2), format="csr")
    nominal = np.asarray(targets[0]["com"])[None] + smooth_fraction(np.linspace(0, 1, count))[:, None] * (
        np.asarray(targets[-1]["com"]) - targets[0]["com"]
    )
    lower, upper = nominal - 0.25, nominal + 0.25
    lower[:3] = upper[:3] = nominal[0]
    lower[-3:] = upper[-3:] = nominal[-1]
    operator = sparse.kron(sparse.vstack((identity, d1, d2, d3)), sparse.eye(2), format="csc")
    limits = np.r_[np.full(2 * (count - 1), 0.7), np.full(2 * (count - 2), 3.0), np.full(2 * (count - 3), 30.0)]
    lows, highs = [lower.ravel(), -limits], [upper.ravel(), limits]
    rows, facets = [operator], []
    feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    for frame in range(1, count - 1):
        target = targets[frame]
        points = []
        for side, body in enumerate(feet):
            if not target["support"][side]:
                continue
            geoms = [
                int(g)
                for g in np.flatnonzero(model.geom_bodyid == body)
                if model.geom_contype[g] or model.geom_conaffinity[g]
            ]
            corners = target["positions"][side] + model.geom_pos[geoms] @ target["rotations"][side].T
            gaps = corners[:, 2] - model.geom_size[geoms, 0]
            points.extend(corners[gaps <= gaps.min() + 0.002, :2])
        equations = ConvexHull(np.asarray(points)).equations
        for nx, ny, offset in equations:
            rows.append(nx * zmp[2 * (frame - 1)] + ny * zmp[2 * (frame - 1) + 1])
            lows.append(np.array([-np.inf]))
            highs.append(np.array([-offset - 0.004]))
        facets.append(equations)
    objective = sparse.kron(sparse.vstack((0.1 * d1, 0.1 * d2, 0.03 * d3)), sparse.eye(2), format="csc")
    values, report = solve_affine_task_soc(
        np.ones(2 * count),
        -nominal.ravel(),
        np.zeros(objective.shape[0]),
        objective,
        sparse.vstack(rows, format="csc"),
        np.concatenate(lows),
        np.concatenate(highs),
        [],
    )
    if values is None:
        raise ValueError(f"LIPM contact schedule infeasible: {report['status']}")
    com, support_point = values.reshape(count, 2), (zmp @ values).reshape(-1, 2)
    violation = max(
        float(np.max(eq[:, :2] @ point + eq[:, 2] + 0.004))
        for eq, point in zip(facets, support_point, strict=True)
    )
    if violation > 1e-8:
        raise ValueError("independent LIPM support-polygon check failed")
    for target, position in zip(targets, com, strict=True):
        target["com"] = position.copy()
    report.update(
        assumed_constant_com_height_m=height,
        inward_support_polygon_margin_m=0.004,
        maximum_support_polygon_violation_m=max(0.0, violation),
        full_rigid_body_dynamics_qualified=False,
    )
    return report


def solve_step_transition(model, start, finish, *, progress=None):
    templates, targets, report = step_targets(model, start, finish)
    report["lipm_plan"] = plan_dynamic_com(model, targets)
    data = mujoco.MjData(model)
    feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    pelvis = model.body("pelvis").id
    columns = np.arange(18)
    low, high = safe_target_joint_bounds(model, native_action_clip=9.5, safe_limit_guard_rad=0.05)
    bounds = (
        np.r_[np.minimum(start[:3], finish[:3]) - [0.4, 0.4, 0.08], [-0.25, -0.25], low[:12]],
        np.r_[np.maximum(start[:3], finish[:3]) + [0.4, 0.4, 0.02], [0.25, 0.25], high[:12]],
    )
    result = templates.copy()
    previous = np.r_[start[:3], [0.0, 0.0], start[7:19]]
    predecessor = previous.copy()
    maxima = np.zeros(4)
    solver_limit_frames = []
    for frame in range(1, len(templates) - 1):
        target, template = targets[frame], templates[frame]
        velocity_step = np.r_[np.full(3, 0.7), np.full(2, 1.5), np.full(12, 5.0)] * 0.02
        acceleration_step = np.r_[np.full(3, 3.0), np.full(2, 20.0), np.full(12, 80.0)] * 0.02**2
        step_low = np.maximum.reduce(
            (bounds[0], previous - velocity_step, 2 * previous - predecessor - acceleration_step)
        )
        step_high = np.minimum.reduce(
            (bounds[1], previous + velocity_step, 2 * previous - predecessor + acceleration_step)
        )
        # Force a smooth return of the extra base tilt to the exact source/end
        # posture instead of leaving a one-frame quaternion/IK branch jump.
        fade = min(
            float(smooth_fraction(min(1.0, frame / 50))),
            float(smooth_fraction(min(1.0, (len(templates) - 1 - frame) / 100))),
        )
        step_low[3:5] = np.maximum(step_low[3:5], -0.25 * fade)
        step_high[3:5] = np.minimum(step_high[3:5], 0.25 * fade)
        if np.any(step_low >= step_high):
            bad = np.flatnonzero(step_low >= step_high)
            raise ValueError(
                f"step IK rate/endpoint bounds infeasible at frame {frame}: "
                f"variables {bad.tolist()}, lower {step_low[bad].tolist()}, upper {step_high[bad].tolist()}"
            )

        def evaluate(values, derivative=False):
            data.qpos[:] = template
            data.qpos[:3], data.qpos[7:19] = values[:3], values[5:]
            rotation = Rotation.from_rotvec([*values[3:5], 0.0])
            data.qpos[3:7] = (Rotation.from_quat(template[[4, 5, 6, 3]]) * rotation).as_quat()[[3, 0, 1, 2]]
            vector = np.r_[values[3:5], 0.0]
            angle = np.linalg.norm(vector)
            x, y, z = vector
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            right_jac = (
                np.eye(3) - 0.5 * skew + skew @ skew / 6
                if angle < 1e-6
                else np.eye(3)
                - (1 - np.cos(angle)) / angle**2 * skew
                + (angle - np.sin(angle)) / angle**3 * skew @ skew
            )
            mapping = np.zeros((18, 17))
            mapping[:3, :3] = np.eye(3)
            mapping[3:6, 3:5] = right_jac[:, :2]
            mapping[6:, 5:] = np.eye(12)
            mujoco.mj_forward(model, data)
            residual, jacobians = [], []
            for side, body in enumerate(feet):
                r, j = foot_task_residual_jacobian(
                    model, data, body, target["positions"][side], target["rotations"][side], columns
                )
                residual.extend(10 * r)
                jacobians.extend(10 * j @ mapping)
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacSubtreeCom(model, data, jac, pelvis)
            residual.extend(1000 * (data.subtree_com[pelvis, :2] - target["com"]))
            jacobians.extend(1000 * jac[:2, columns] @ mapping)
            residual.append(100 * (values[2] - target["root_z"]))
            row = np.zeros(17)
            row[2] = 100
            jacobians.append(row)
            # Weak, disclosed branch-selection regularization, not action targets.
            residual.extend(0.1 * (values - previous))
            jacobians.extend(0.1 * np.eye(17))
            residual.extend(0.005 * values[3:5])
            jacobians.extend(0.005 * np.eye(17)[3:5])
            return np.asarray(jacobians) if derivative else np.asarray(residual)

        solve = least_squares(
            evaluate,
            np.clip(previous, step_low, step_high),
            jac=lambda v: evaluate(v, True),
            bounds=(step_low, step_high),
            max_nfev=200,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-9,
        )
        residual = evaluate(solve.x)
        result[frame] = data.qpos.copy()
        errors = np.array(
            [
                max(np.linalg.norm(residual[:3]), np.linalg.norm(residual[6:9])) / 1000,
                max(np.linalg.norm(residual[3:6]), np.linalg.norm(residual[9:12])) / 100,
                np.linalg.norm(residual[12:14]) / 1000,
                abs(residual[14]) / 100,
            ]
        )
        maxima = np.maximum(maxima, errors)
        # The physical task residual and the subsequent full-path audit decide
        # feasibility. Exhausting iterations on weak posture regularization is
        # recorded, not confused with a failed foot/COM constraint.
        if not solve.success:
            solver_limit_frames.append(frame)
        if not np.isfinite(solve.x).all() or np.any(errors > [0.0005, 0.005, 0.0005, 0.003]):
            active = np.flatnonzero(np.minimum(solve.x - bounds[0], bounds[1] - solve.x) < 1e-6)
            raise ValueError(
                f"step IK failed at frame {frame} ({target['stage']}): {errors.tolist()}; "
                f"active variables {active.tolist()}, root {solve.x[:3].tolist()}, "
                f"tilt {solve.x[3:5].tolist()}, legs {solve.x[5:].tolist()}, target COM {target['com'].tolist()}"
            )
        predecessor, previous = previous, solve.x.copy()
        if progress and frame % 100 == 0:
            progress(dict(frame=frame, stage=target["stage"], maximum_foot_error_m=float(maxima[0])))
    np.testing.assert_array_equal(result[[0, -1]], np.asarray([start, finish]))
    report.update(
        maximum_foot_position_error_m=float(maxima[0]),
        maximum_foot_rotation_error_rad=float(maxima[1]),
        maximum_com_xy_error_m=float(maxima[2]),
        maximum_root_z_error_m=float(maxima[3]),
        endpoint_poses_exact=True,
        ik_iteration_limit_frames=solver_limit_frames,
        all_ik_optimizers_converged=not solver_limit_frames,
        generated_root_roll_pitch_offset_bound_rad=0.25,
        source_root_attitudes_modified=False,
        physical_integration_performed=False,
        dynamic_feasibility_qualified=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    return result, targets, report


def solve_return_step_transition(model, source_endpoint, standing, *, progress=None):
    """Reverse a bounded standing-to-endpoint step plan, then re-audit dynamics.

    Time reversal preserves position/rate/acceleration magnitudes and the LIPM
    relation, not friction/actuation feasibility. The caller reruns full native
    collision, floor, rates and inverse-dynamics checks in execution order.
    """
    poses, targets, report = solve_step_transition(model, standing, source_endpoint, progress=progress)
    result = poses[::-1].copy()
    reversed_targets = [{**row, "stage": "reverse_" + row["stage"]} for row in reversed(targets)]
    np.testing.assert_array_equal(result[[0, -1]], [source_endpoint, standing])
    report.update(
        return_built_by_explicit_time_reversal=True,
        first_swing_foot="right" if report["first_swing_foot"] == "left" else "left",
        stages=[
            dict(name="reverse_" + stage["name"], controls=stage["controls"])
            for stage in reversed(report["stages"])
        ],
        reversed_rigid_body_force_feasibility_inherited=False,
    )
    return result, reversed_targets, report
