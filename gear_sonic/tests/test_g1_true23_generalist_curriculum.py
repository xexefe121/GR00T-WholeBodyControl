"""Nominal stage mechanics are not held-out dance or teleop qualification."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_generalist_curriculum import (
    GeneralistCurriculumMotionCommand,
    advance_lifecycle_command,
    configure_curriculum_environment,
)
from gear_sonic.utils.g1_true23_generalist_curriculum import (
    MOTION_KEYS,
    array_digest,
    derive_curriculum,
    write_curriculum_bundle,
)


@pytest.fixture(scope="module")
def source():
    import mujoco
    from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

    root = Path(__file__).resolve().parents[2]
    config = root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
    model_path = root.parent / "GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
    _, model, _ = prepare_true23_model(model_path, config)
    data = mujoco.MjData(model)
    initial = json.loads(config.read_text())["initial_state"]
    data.qpos[:] = [
        *initial["base_position_m"],
        *initial["base_quaternion_wxyz"],
        *initial["joint_position_hardware_rad"],
    ]
    mujoco.mj_fwdPosition(model, data)
    # A synthetic test fixture, never described as a licensed recording.
    motion = {
        "joint_pos": np.tile(data.qpos[7:], (16, 1)),
        "joint_vel": np.zeros((16, 23)),
        "body_pos_w": np.tile(data.xpos[1:], (16, 1, 1)),
        "body_quat_w": np.tile(data.xquat[1:], (16, 1, 1)),
        "body_lin_vel_w": np.zeros((16, 24, 3)),
        "body_ang_vel_w": np.zeros((16, 24, 3)),
    }
    spans = {"spans": [dict(start=0, length=16, asset_id="synthetic", name="test")]}
    return motion, spans, model, config


def derive(source, stage="lifecycle", inputs=None):
    motion, spans, model, config = source
    return derive_curriculum(
        motion,
        spans,
        inputs or {"smoke_only": True, "corpus_audit": None},
        stage=stage,
        model=model,
        simulation_config=config,
    )


def test_full_source_channels_preserved_and_not_qualified(source):
    motion, spans, contract = derive(source)
    row = spans["spans"][0]
    timeline = row["timeline"]
    for key in MOTION_KEYS:
        np.testing.assert_array_equal(
            motion[key][timeline["source_start_frame"] : timeline["source_stop_frame_exclusive"]], source[0][key]
        )
    assert row["original_source_indices_requested"] == list(range(16))
    assert row["every_original_source_frame_requested"]
    assert timeline["total_requested_controls"] == 766
    assert not contract["full_lifecycle_training_completed"]
    assert not contract["deployment_ready"]


def test_acquisition_has_pose_hold_not_shortened_dance(source):
    _, spans, contract = derive(source, "acquisition")
    row = spans["spans"][0]
    assert row["original_source_indices_requested"] == [0]
    assert not row["every_original_source_frame_requested"]
    assert row["original_source_frames"] == 16
    assert "acquisition_pose_hold" in [phase["name"] for phase in row["timeline"]["phases"]]
    assert not contract["full_source_lifecycle_reference_enabled"]


@pytest.mark.parametrize("stage", ["", "dance_only", None])
def test_unknown_stage_rejected(source, stage):
    with pytest.raises(ValueError, match="stage"):
        derive(source, stage)


def test_unaudited_production_rejected(source):
    with pytest.raises(ValueError, match="audited"):
        derive(source, inputs={"smoke_only": False, "corpus_audit": None})


def audit(split="train", frames=16):
    return {
        "asset_splits": {"synthetic": split},
        "asset_metadata": {
            "synthetic": {"recording_id": "original", "timing": {"fps": 50, "frame_count": frames}}
        },
        "asset_bindings": {"synthetic": {"sha256": "a" * 64}},
    }


def test_derivative_keeps_original_train_ownership(source):
    _, spans, _ = derive(source, inputs={"smoke_only": False, "corpus_audit": audit()})
    assert spans["spans"][0]["ownership"] == {
        "asset_id": "synthetic",
        "recording_id": "original",
        "split": "train",
        "source_asset_sha256": "a" * 64,
    }


@pytest.mark.parametrize("split", ["validation", "test", None])
def test_held_out_derivatives_rejected(source, split):
    with pytest.raises(ValueError, match="train split"):
        derive(source, inputs={"smoke_only": False, "corpus_audit": audit(split)})


def test_asset_declared_timing_must_match(source):
    with pytest.raises(ValueError, match="timing"):
        derive(source, inputs={"smoke_only": False, "corpus_audit": audit(frames=17)})


def test_bundle_hashes_and_no_overwrite(source, tmp_path):
    motion, spans, contract = derive(source)
    paths = write_curriculum_bundle(tmp_path / "derived", motion, spans, contract)
    from gear_sonic.scripts.train_g1_23dof_mjlab_low_latency_recovery import _read_metadata

    assert _read_metadata(paths[1], paths[0])["curriculum"] == contract
    with np.load(paths[0], allow_pickle=False) as archive:
        assert array_digest(archive) == contract["derived_arrays_sha256"]
    with pytest.raises(FileExistsError):
        write_curriculum_bundle(tmp_path / "derived", motion, spans, contract)


def fake_command():
    zero = torch.zeros(2, 3)
    quat = torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1)
    return SimpleNamespace(
        robot_anchor_pos_w=zero + 2,
        robot_anchor_quat_w=quat,
        _causal_resampled=torch.tensor([False, True]),
        time_steps=torch.tensor([18, 10]),
        _lifecycle_last_anchor=torch.tensor([18, 30]),
        _causal_robot_anchor_pos_w=zero.clone(),
        _causal_robot_anchor_quat_w=quat.clone(),
        _causal_last_current_anchor_pos_w=zero + 1,
        _causal_last_current_anchor_quat_w=quat.clone(),
        _virtual_anchor_at_q9=lambda ids: (zero[ids], quat[ids]),
        _refresh_targets_from_causal_anchor=lambda: None,
        _resample_command=lambda ids: pytest.fail("update must never resample or write robot state"),
    )


def test_boundary_clamps_without_resampling_or_robot_state_writes():
    command = fake_command()
    for _ in range(50):
        advance_lifecycle_command(command)
    assert command.time_steps.tolist() == [18, 30]
    assert not command._causal_resampled.any()


def test_regular_update_advances_exactly_one_anchor():
    command = fake_command()
    command.time_steps[0] = 11
    advance_lifecycle_command(command)
    assert command.time_steps.tolist() == [12, 10]
    torch.testing.assert_close(command._causal_robot_anchor_pos_w[0], torch.ones(3))


@pytest.mark.parametrize("requested_controls", [16, 766, 1296])
def test_prephysics_reference_to_postphysics_timeout_counts_every_control(requested_controls):
    command = fake_command()
    command.time_steps[:] = 10  # prehistory 11; q10 proof is first requested frame.
    command._causal_resampled.zero_()  # initial environment prime already refreshed anchors.
    total_frames = 11 + requested_controls
    command._lifecycle_last_anchor[:] = total_frames - 2
    anchors = []
    for _ in range(requested_controls + 2):
        anchors.append(int(command.time_steps[0]))
        # MJLab performs one physical action before evaluating terminations,
        # and only then resets and updates commands.
        if int(command.time_steps[0]) >= total_frames - 2:
            break
        advance_lifecycle_command(command)
    assert len(anchors) == requested_controls
    assert [anchor + 1 for anchor in anchors] == list(range(11, total_frames))


def test_timed_compute_never_calls_resample():
    calls = []
    command = SimpleNamespace(
        _update_metrics=lambda: calls.append("metrics"),
        _update_command=lambda: calls.append("update"),
        command_update_count=0,
        time_left=torch.zeros(2),
        _resample=lambda ids: pytest.fail("timed resampling forbidden"),
    )
    GeneralistCurriculumMotionCommand.compute(command, 1000.0)
    assert calls == ["metrics", "update"]
    assert command.command_update_count == 1


def test_resample_outside_reset_rejected_before_state_access():
    with pytest.raises(RuntimeError, match="environment reset"):
        GeneralistCurriculumMotionCommand._resample_command(
            SimpleNamespace(_inside_environment_reset=False), torch.tensor([0])
        )


def test_config_timeout_exceeds_whole_source_and_zero_reset_noise(source):
    _, spans, _ = derive(source)
    cfg = SimpleNamespace(commands={"motion": SimpleNamespace()})
    configure_curriculum_environment(cfg, spans["spans"])
    assert cfg.episode_length_s == 768 * 0.02
    command = cfg.commands["motion"]
    assert command.pose_range == command.velocity_range == {}
    assert command.joint_position_range == (0.0, 0.0)
    assert command.sampling_mode == "uniform"


def test_array_hash_changes_when_one_frame_changes(source):
    altered = deepcopy(source[0])
    altered["joint_pos"][3, 2] += 0.001
    assert array_digest(altered) != array_digest(source[0])
