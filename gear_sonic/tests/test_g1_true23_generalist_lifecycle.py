"""Whole-source phase accounting for actual one-policy lifecycle probes."""

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_generalist_lifecycle import (
    PREHISTORY_FRAMES,
    build_lifecycle_timeline,
    assess_lifecycle_diagnostic,
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
