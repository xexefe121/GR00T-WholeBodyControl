from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_23dof_contract import SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks
from gear_sonic.utils.g1_true23_intent_arm_ik import HAND_POINTS_LOCAL, Native23ArmIK


@pytest.fixture(scope="module")
def models():
    root = Path(__file__).resolve().parents[2] / "gear_sonic/data/robots/g1"
    return tuple(mujoco.MjModel.from_xml_path(str(root / name)) for name in
                 ("g1_29dof.xml", "g1_23dof_rev_1_0.xml"))


def native_pose():
    value = np.r_[np.array((.2, -.3, .8, 1., 0., 0., 0.)), SAFE_TARGET_DEFAULT_Q_HARDWARE]
    value[3:7] = Rotation.from_euler("xyz", (.14, -.17, .6)).as_quat()[[3, 0, 1, 2]]
    return value


def test_hand_frames_exactly_match_original_convention(models):
    source, native = models
    tasks, _ = neutral_wrist_hand_tasks(source, native)
    solver = Native23ArmIK(native)
    hands = [next(t for t in tasks if t.name == side + "_hand") for side in ("left", "right")]
    np.testing.assert_allclose(HAND_POINTS_LOCAL, [t.target_point for t in hands], atol=1e-12, rtol=0)
    assert list(solver.body_ids) == [native.body(t.target_body).id for t in hands]


@pytest.mark.parametrize("side", (0, 1))
def test_analytic_jacobian_matches_central_difference(models, side):
    solver = Native23ArmIK(models[1])
    pose = native_pose()
    points, rotations = solver.hand_poses(pose)
    joints = pose[solver.qpos_indices[side]] + np.array((.12, -.09, .07, -.04, .11))
    kwargs = dict(side=side, qpos=pose, target_position_w=points[side] + .01,
                  target_rotation_w=rotations[side], posture=pose[solver.qpos_indices[side]])
    _, jacobian = solver.residual_jacobian(joints, **kwargs)
    numerical = np.column_stack([(solver.residual_jacobian(joints + d, **kwargs)[0] -
                                  solver.residual_jacobian(joints - d, **kwargs)[0]) / 2e-6
                                 for d in np.eye(5) * 1e-6])
    np.testing.assert_allclose(jacobian, numerical, atol=2e-8, rtol=2e-6)


def test_recoverable_arm_targets_preserve_other_joints(models):
    solver = Native23ArmIK(models[1])
    seed = native_pose()
    desired = seed.copy()
    desired[solver.qpos_indices] += np.array(((.15, .08, -.12, .15, .25), (-.17, -.11, .08, -.15, -.2)))
    points, rotations = solver.hand_poses(desired)
    saved = seed.copy()
    fitted = solver.solve(seed, points, target_rotations_w=rotations)
    np.testing.assert_array_equal(seed, saved)
    fixed = np.setdiff1d(np.arange(30), solver.qpos_indices.ravel())
    np.testing.assert_array_equal(fitted.qpos[fixed], seed[fixed])
    assert fitted.position_errors_m.max() < 2e-4
    assert fitted.orientation_errors_rad.max() < .003
    assert fitted.success.all()
    assert np.all(fitted.qpos[solver.qpos_indices] >= solver.lower)
    assert np.all(fitted.qpos[solver.qpos_indices] <= solver.upper)
    quats = Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
    other = solver.solve(seed, points, quats)
    np.testing.assert_allclose(other.qpos, fitted.qpos, atol=1e-10, rtol=0)


def test_original29_missing_wrist_axes_position_improves(models):
    source, native = models
    solver = Native23ArmIK(native)
    native_q = native_pose()
    source_q = np.zeros(36)
    source_q[:7] = native_q[:7]
    source_q[7 + np.asarray(SOURCE_MJ29_KEEP_INDICES)] = native_q[7:]
    for name, value in (("left_wrist_pitch_joint", .4), ("left_wrist_yaw_joint", -.3),
                        ("right_wrist_pitch_joint", -.35), ("right_wrist_yaw_joint", .3)):
        source_q[int(source.joint(name).qposadr[0])] = value
    data = mujoco.MjData(source)
    data.qpos[:] = source_q
    mujoco.mj_fwdPosition(source, data)
    tasks, _ = neutral_wrist_hand_tasks(source, native)
    points, rotations = [], []
    for side in ("left", "right"):
        task = next(t for t in tasks if t.name == side + "_hand")
        body = source.body(task.source_body).id
        rotation = data.xmat[body].reshape(3, 3).copy()
        points.append(data.xpos[body] + rotation @ task.source_point)
        rotations.append(rotation)
    baseline = np.linalg.norm(solver.hand_poses(native_q)[0] - points, axis=1)
    fitted = solver.solve(native_q, points, target_rotations_w=rotations)
    assert np.all(baseline > .05)
    assert np.all(fitted.position_errors_m < baseline * .15)
    assert np.all(fitted.qpos[solver.qpos_indices] >= solver.lower)
    assert np.all(fitted.qpos[solver.qpos_indices] <= solver.upper)


def test_previous_arm_step_box_intersects_physical_bounds_and_keeps_fixed_pose(models):
    solver = Native23ArmIK(models[1], posture_weight=.02)
    previous = native_pose()
    previous[solver.qpos_indices] = solver.upper - .03
    current = native_pose()
    wanted = current.copy()
    wanted[solver.qpos_indices] = solver.lower + .2
    points, rotations = solver.hand_poses(wanted)
    result = solver.solve(current, points, target_rotations_w=rotations,
                          posture_qpos=previous, previous_qpos=previous, max_step_rad=.15)
    fitted = result.qpos[solver.qpos_indices]
    assert np.all(np.abs(fitted - previous[solver.qpos_indices]) <= .15 + 1e-12)
    assert np.all(fitted >= solver.lower) and np.all(fitted <= solver.upper)
    fixed = np.setdiff1d(np.arange(30), solver.qpos_indices.ravel())
    np.testing.assert_array_equal(result.qpos[fixed], current[fixed])
    assert np.all(result.position_errors_m > .01)  # Unreachable step goals stay visible.
    result2 = solver.solve(current, points, target_rotations_w=rotations,
                           previous_qpos=result.qpos, max_step_rad=np.full((2, 5), .03))
    assert np.max(np.abs(result2.qpos[solver.qpos_indices] - fitted)) <= .03 + 1e-12


@pytest.mark.parametrize("step", (0., -.1, np.nan, np.inf, [1., 2.]))
def test_invalid_step_box_rejected(models, step):
    solver = Native23ArmIK(models[1])
    pose = native_pose()
    points, rotations = solver.hand_poses(pose)
    with pytest.raises(ValueError, match="max_step_rad"):
        solver.solve(pose, points, previous_qpos=pose, max_step_rad=step)


def test_step_box_requires_previous_and_nonempty_physical_intersection(models):
    solver = Native23ArmIK(models[1])
    pose = native_pose()
    points, _ = solver.hand_poses(pose)
    with pytest.raises(ValueError, match="supplied together"):
        solver.solve(pose, points, max_step_rad=.15)
    with pytest.raises(ValueError, match="supplied together"):
        solver.solve(pose, points, previous_qpos=pose)
    previous = pose.copy()
    previous[solver.qpos_indices] = solver.upper + 1.
    with pytest.raises(ValueError, match="no interior"):
        solver.solve(pose, points, previous_qpos=previous, max_step_rad=.15)


def test_margin_keeps_unreachable_boundary_goal_inside_native_range(models):
    model = models[1]
    original = Native23ArmIK(model)
    solver = Native23ArmIK(model, joint_margin=.03)
    boundary = native_pose()
    boundary[original.qpos_indices] = original.upper
    points, rotations = original.hand_poses(boundary)
    result = solver.solve(boundary, points, target_rotations_w=rotations)
    q = result.qpos[solver.qpos_indices]
    assert np.all(q <= original.upper - .03 + 1e-12)
    assert np.all(q >= original.lower + .03 - 1e-12)
    np.testing.assert_array_equal(model.jnt_range[model.joint("left_shoulder_pitch_joint").id],
                                  [-3.0892, 2.6704])


@pytest.mark.parametrize("margin", (-.01, np.nan, np.inf, 10.))
def test_invalid_interior_margin_rejected(models, margin):
    with pytest.raises(ValueError, match="joint margin"):
        Native23ArmIK(models[1], joint_margin=margin)
