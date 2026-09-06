"""Whole-source phase accounting for actual one-policy lifecycle probes."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_lifecycle import (
    PREHISTORY_FRAMES,
    build_lifecycle_timeline,
    assess_lifecycle_diagnostic,
    _planned_endpoint_standing,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


@pytest.fixture
def timeline():
    root = Path(__file__).resolve().parents[2]
    assets = root.parent / "GR00T-WholeBodyControl"
    path = (
        root
        / "artifacts/g1_true23_frozen_lora/original29_neutral_hand_frame_fit_20260906_v1/happy_dance.native23.npz"
    )
    if not path.is_file() or not (assets / MODEL).is_file():
        pytest.skip("local native motion/model assets unavailable")
    with np.load(path, allow_pickle=False) as archive:
        source = {key: archive[key].copy() for key in archive.files}
    _, model, _ = prepare_true23_model(assets / MODEL, root / PHYSICS)
    motion, timeline = build_lifecycle_timeline(source, model=model, simulation_config=root / PHYSICS)
    return motion, timeline, source


def test_every_source_frame_and_channel_retained_exactly(timeline):
    motion, info, source = timeline
    start, stop = info["source_start_frame"], info["source_stop_frame_exclusive"]
    assert stop - start == 546
    assert info["total_requested_controls"] == 546 + 750
    assert len(motion["joint_pos"]) == PREHISTORY_FRAMES + 546 + 750
    for name, original in source.items():
        if name != "fps":
            np.testing.assert_array_equal(motion[name][start:stop], original)
    assert info["source_frame_indices"] == list(range(546))
    assert not info["source_history_frames_trimmed"]


def test_phase_controls_partition_complete_timeline(timeline):
    _, info, _ = timeline
    previous = 0
    for row in info["phases"]:
        assert row["control_start"] == previous
        assert row["control_stop"] - row["control_start"] == row["requested_controls"]
        assert row["frame_start"] == row["control_start"] + PREHISTORY_FRAMES
        previous = row["control_stop"]
    assert previous == info["total_requested_controls"]
    assert not info["generated_ramp_contact_or_force_feasibility_qualified"]


@pytest.mark.parametrize("completed", [0, 249, 250, 300, 350, 351, 896, 1296])
def test_lifecycle_phase_accounting_never_credits_unexecuted_frames(timeline, completed):
    _, info, _ = timeline
    report = dict(
        available_controls=1296,
        completed_controls=completed,
        failure=None if completed == 1296 else {"type": "Stop"},
    )
    arrays = dict(
        landmark_error_m=np.zeros((completed, 5)),
        qpos=np.tile(info["configured_standing_qpos"], (completed + 1, 1)),
        qvel=np.zeros((completed + 1, 29)),
    )
    result = assess_lifecycle_diagnostic(info, report, arrays)
    assert sum(row["completed_controls"] for row in result["phases"]) == completed
    assert result["single_policy_full_lifecycle_integrated"] == (completed == 1296)
    assert result["every_original_source_frame_evaluated"] == (completed >= 896)
    assert not result["standing_or_contact_qualification"]
    assert not result["simulator_qualified"]


def test_reject_shortened_lifecycle_accounting(timeline):
    _, info, _ = timeline
    with pytest.raises(ValueError, match="shortened"):
        assess_lifecycle_diagnostic(info, dict(available_controls=500), {})


def test_endpoint_standing_uses_planned_xy_heading_not_origin():
    standing = np.zeros(30)
    standing[2:4] = [0.75, 1.0]
    standing[7:] = np.linspace(-0.2, 0.3, 23)
    endpoint = standing.copy()
    endpoint[:3] = [8.0, -3.0, 0.68]
    endpoint[3:7] = Rotation.from_euler("ZYX", [1.4, -0.1, 0.2]).as_quat()[[3, 0, 1, 2]]
    result = _planned_endpoint_standing(standing, endpoint)
    np.testing.assert_array_equal(result[:3], [8.0, -3.0, 0.75])
    np.testing.assert_array_equal(result[7:], standing[7:])
    np.testing.assert_allclose(Rotation.from_quat(result[[4, 5, 6, 3]]).as_euler("ZYX"), [1.4, 0, 0], atol=1e-14)
    np.testing.assert_array_equal(standing[:3], [0, 0, 0.75])


def test_endpoint_rejects_undefined_upright_heading():
    standing = np.zeros(30)
    standing[3] = 1
    endpoint = standing.copy()
    endpoint[3:7] = Rotation.from_euler("y", np.pi / 2).as_quat()[[3, 0, 1, 2]]
    with pytest.raises(ValueError, match="heading"):
        _planned_endpoint_standing(standing, endpoint)


def test_full_planned_endpoint_lifecycle_preserves_source_without_eight_metre_return():
    root = Path(__file__).resolve().parents[2]
    assets = root.parent / "GR00T-WholeBodyControl"
    path = root / "artifacts/g1_true23_generalist/planned_dance_retarget_20260907_v5/adapted.true23.npz"
    if not path.is_file() or not (assets / MODEL).is_file():
        pytest.skip("accepted planned reference/model assets unavailable")
    keys = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    with np.load(path, allow_pickle=False) as archive:
        source = {key: archive[key].copy() for key in keys}
    _, model, _ = prepare_true23_model(assets / MODEL, root / PHYSICS)
    legacy, old_info = build_lifecycle_timeline(source, model=model, simulation_config=root / PHYSICS)
    explicit, explicit_info = build_lifecycle_timeline(
        source, model=model, simulation_config=root / PHYSICS, return_target="configured_origin"
    )
    for key in legacy:
        np.testing.assert_array_equal(legacy[key], explicit[key])
    assert old_info == explicit_info
    motion, info = build_lifecycle_timeline(
        source, model=model, simulation_config=root / PHYSICS, return_target="planned_endpoint"
    )
    assert info["kind"].endswith("_v2")
    assert info["total_requested_controls"] == 1841
    assert info["return_root_horizontal_displacement_m"] == 0
    assert info["return_root_horizontal_speed_max_m_s"] == 0
    assert not info["measured_robot_state_used_for_return_target"]
    assert not info["generated_ramp_contact_or_force_feasibility_qualified"]
    start, stop = info["source_start_frame"], info["source_stop_frame_exclusive"]
    for key in keys[1:]:
        np.testing.assert_array_equal(motion[key][start:stop], source[key])
    np.testing.assert_array_equal(motion["body_pos_w"][:stop], legacy["body_pos_w"][:stop])
    np.testing.assert_allclose(
        motion["body_pos_w"][stop:, 0, :2],
        np.tile(source["body_pos_w"][-1, 0, :2], (len(motion["joint_pos"]) - stop, 1)),
        atol=0,
    )
    assert np.max(np.linalg.norm(np.diff(legacy["body_pos_w"][stop - 1 :, 0, :2], axis=0) / 0.02, axis=1)) > 7
    count = info["total_requested_controls"]
    report = dict(available_controls=count, completed_controls=count, failure=None)
    arrays = dict(
        landmark_error_m=np.zeros((count, 5)),
        qpos=np.tile(info["returned_standing_qpos"], (count + 1, 1)),
        qvel=np.zeros((count + 1, 29)),
    )
    assessment = assess_lifecycle_diagnostic(info, report, arrays)
    assert assessment["final_proof_root_position_error_max_m"] == 0
    assert assessment["final_proof_standing_joint_error_max_rad"] == 0
    assert assessment["final_proof_root_orientation_error_max_rad"] < 1e-14
