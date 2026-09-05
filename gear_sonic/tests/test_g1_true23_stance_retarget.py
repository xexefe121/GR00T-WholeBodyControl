import copy
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_23dof_task_space_retarget import build_mjlab_motion_arrays
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos
from gear_sonic.utils.g1_true23_stance_retarget import (
    FEET,
    FloorResidual,
    FrameProblem,
    StanceRetargetConfig,
    SupportGeometry,
    closest_supported_com,
    retarget_stance_motion,
    stance_schedule,
)


@pytest.fixture
def materials():
    root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"))
    motion = build_mjlab_motion_arrays(
        model,
        SimpleNamespace(
            root_pos_w=np.tile([0, 0, 0.76], (16, 1)),
            root_quat_wxyz=np.tile([1, 0, 0, 0], (16, 1)),
            joint_pos_hardware=np.tile(SAFE_TARGET_DEFAULT_Q_HARDWARE, (16, 1)),
            fps=50.0,
        ),
    )
    return model, motion


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maximum_root_offset_m": 0},
        {"maximum_joint_change_rad": float("nan")},
        {"candidate_gap_m": 0.02},
        {"contact_clearance_m": 0.002},
        {"maximum_frame_evaluations": 4.5},
    ],
)
def test_invalid_bounds_fail_closed(kwargs):
    with pytest.raises(ValueError):
        StanceRetargetConfig(**kwargs)


def test_support_polygon_intersection_and_degeneracy():
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    position, supported = closest_supported_com(np.array([3.0, 0]), [square, square + [1, 0]], 0.1)
    assert supported
    np.testing.assert_allclose(position, [0.9, 0], atol=1e-8)
    for polygons in ([], [square[:2]], [np.zeros((4, 2))], [square, square + [5, 0]]):
        position, supported = closest_supported_com(np.array([3.0, 0]), polygons, 0.1)
        assert not supported
        np.testing.assert_array_equal(position, [3.0, 0])


def test_geometry_uses_actual_shapes_without_mutating_model(materials):
    model, _ = materials
    initial_hash = compiled_model_sha256(model)
    geometry = SupportGeometry(model, 0.002)
    rotation = Rotation.from_euler("z", 0.4).as_matrix()
    for foot in FEET:
        depth = geometry.depth(foot, rotation)
        assert 0.01 < depth < 0.1
        points = geometry.points(foot, rotation, depth + 0.0002)
        assert points.shape == (4, 2)
    assert compiled_model_sha256(model) == initial_hash


def test_analytic_ik_jacobian_matches_pose_differentiation(materials):
    model, motion = materials
    pose = motion_qpos(model, motion)[0]
    pose[3:7] = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_quat()[[3, 0, 1, 2]]
    targets = {FEET[0]: (np.array([0.1, 0.1, 0.03]), np.eye(3))}
    problem = FrameProblem(model, pose, targets, np.array([0.02, 0]), np.zeros(26))
    x = np.r_[[0.01, -0.01, 0.02], pose[7:] + 0.01]
    _, jacobian = problem.evaluate(x)
    numerical = []
    for column in range(26):
        step = np.zeros(26)
        step[column] = 1e-6
        numerical.append((problem.evaluate(x + step)[0] - problem.evaluate(x - step)[0]) / 2e-6)
    np.testing.assert_allclose(jacobian, np.array(numerical).T, atol=2e-6, rtol=1e-6)


def test_penetrating_collision_distance_jacobian_matches_finite_difference(materials):
    model, motion = materials
    pose = motion_qpos(model, motion)[0]
    pose[2] -= 0.08
    pose[3:7] = Rotation.from_euler("xyz", [0.03, 0.04, 0.1]).as_quat()[[3, 0, 1, 2]]
    objective = FloorResidual(model, 0.0002, 5000)
    residual, jacobian = objective.evaluate(pose)
    assert np.any(residual < 0)
    numerical = []
    for coordinate in list(range(3)) + list(range(7, 30)):
        step = np.zeros(30)
        step[coordinate] = 1e-6
        numerical.append((objective.evaluate(pose + step)[0] - objective.evaluate(pose - step)[0]) / 2e-6)
    np.testing.assert_allclose(jacobian, np.array(numerical).T, atol=2e-5, rtol=1e-5)


def test_contact_family_is_explicit_and_does_not_invent_other_supports(materials):
    model, motion = materials
    bodies, active = stance_schedule(model, motion, "biped_stance")
    assert bodies == FEET and active.shape == (16, 2) and active.all()
    with pytest.raises(ValueError, match="explicit"):
        stance_schedule(model, motion, "automatic")


def test_retarget_full_clip_preserves_inputs_and_false_readiness(materials):
    model, motion = materials
    before = copy.deepcopy(motion)
    model_hash = compiled_model_sha256(model)
    output, report = retarget_stance_motion(model, {"mesh": model}, motion, "biped_stance")
    assert report["frames_in"] == report["frames_out"] == 16
    assert report["frames_dropped"] == 0 and report["controlled_joint_count"] == 23
    assert report["serialized_trajectory_audit"]["passed"]
    assert report["output_fk"]["position_fk_consistent"]
    assert report["output_fk"]["orientation_fk_consistent"]
    assert report["maximum_joint_change_rad"] <= 0.600001
    assert max(report["maximum_root_offset_m_per_axis"]) <= 0.080001
    for key in (
        "root_orientation_changed",
        "dynamic_feasibility_proven",
        "hardware_authorized",
        "deployment_ready",
        "old_packet_joint_proofs_reusable",
    ):
        assert not report[key]
    np.testing.assert_allclose(output["body_quat_w"][:, 0], motion["body_quat_w"][:, 0], atol=1e-7)
    assert compiled_model_sha256(model) == model_hash
    for key, value in before.items():
        np.testing.assert_array_equal(motion[key], value)


def test_bad_fk_and_missing_collision_models_rejected(materials):
    model, motion = materials
    altered = copy.deepcopy(motion)
    altered["body_pos_w"][:, 4, 2] += 0.1
    with pytest.raises(ValueError, match="disagree"):
        retarget_stance_motion(model, {"mesh": model}, altered, "biped_stance")
    with pytest.raises(ValueError, match="collision models"):
        retarget_stance_motion(model, {}, motion, "biped_stance")


def test_manifest_rejects_missing_extra_or_implicit_contact_families():
    from gear_sonic.scripts.retarget_g1_true23_stance import validate_manifest

    manifest = {"motions": [{"name": "dance", "path": "dance.npz"}]}
    hypotheses = {"families": {"dance": "biped_motion"}, "observed_or_verified_contacts": False}
    assert validate_manifest(manifest, hypotheses)[1] == {"dance": "biped_motion"}
    for families in ({}, {"dance": "biped_motion", "extra": "biped_stance"}, {"dance": "automatic"}):
        with pytest.raises(ValueError, match="exactly"):
            validate_manifest(manifest, {**hypotheses, "families": families})
    with pytest.raises(ValueError, match="must not claim"):
        validate_manifest(manifest, {**hypotheses, "observed_or_verified_contacts": True})
    with pytest.raises(ValueError, match="safe output"):
        validate_manifest({"motions": [{"name": "../dance"}]}, hypotheses)


def test_rebuilt_causal_proofs_cover_every_complete_history(materials):
    from gear_sonic.scripts.retarget_g1_true23_stance import audit_rebuilt_causal_terms

    _, motion = materials
    report = audit_rebuilt_causal_terms(motion)
    assert report["packets_rebuilt_and_validated"] == 6
    assert report["first_q9"] == 9 and report["last_q9"] == 14
    assert len(report["canonical_jsonl_sha256"]) == 64
    assert not report["stored_or_sent_to_robot"] and not report["live_pico_input_qualified"]


def test_standing_com_centering_repairs_conditional_effort_screen(materials):
    from gear_sonic.utils.g1_true23_actuation_profile import SIM_CONFIG, NativeSupportActuationProfile
    from gear_sonic.utils.g1_true23_reference_support import audit_reference_support

    model, _ = materials
    root = Path(__file__).resolve().parents[2]
    # Rounded frame-0 posture from saved upright PICO anchor. The legs put the
    # ankles behind COM; merely projecting COM inside the toe edge overloads
    # the unchanged quarter-effort ankle screen even with flat planted feet.
    joints = [
        0.131859,
        -0.016596,
        0.001326,
        0.169331,
        -0.206666,
        0.004629,
        0.146187,
        0.00545,
        -0.00197,
        0.163704,
        -0.215431,
        -0.018187,
        -0.024077,
        0.324795,
        0.16048,
        -0.099881,
        -0.105467,
        -0.111005,
        0.484963,
        -0.220321,
        0.131076,
        -0.281753,
        0.127813,
    ]
    motion = build_mjlab_motion_arrays(
        model,
        SimpleNamespace(
            root_pos_w=np.tile([0, 0, 0.800305], (16, 1)),
            root_quat_wxyz=np.tile([1, 0, 0, 0], (16, 1)),
            joint_pos_hardware=np.tile(joints, (16, 1)),
            fps=50.0,
        ),
    )
    limits = np.array(NativeSupportActuationProfile.from_sim_config(root / SIM_CONFIG).effort) * 0.95 * 0.25
    for center in (False, True):
        output, report = retarget_stance_motion(
            model,
            {"mesh": model},
            motion,
            "biped_stance",
            config=StanceRetargetConfig(center_biped_stance_com=center),
        )
        assert report["geometry_after"]["mesh"]["frames_with_floor_overlap"] == 0
        screen = audit_reference_support(model, output, limits)
        assert screen["frames_with_no_support_solution"] == 0
        assert screen["frames_with_conditional_solution_within_effort_limits"] == (16 if center else 0)
        assert not screen["dynamic_feasibility_proven"] and not screen["deployment_ready"]


def test_joint_correction_box_cannot_silently_expand_or_drop_frames(materials):
    model, motion = materials
    joints = motion["joint_pos"].copy()
    joints[:, 0] = 4.0
    invalid = build_mjlab_motion_arrays(
        model,
        SimpleNamespace(
            root_pos_w=motion["body_pos_w"][:, 0],
            root_quat_wxyz=motion["body_quat_w"][:, 0],
            joint_pos_hardware=joints,
            fps=50.0,
        ),
    )
    with pytest.raises(ValueError, match="no frame may be removed"):
        retarget_stance_motion(model, {"mesh": model}, invalid, "biped_stance")
