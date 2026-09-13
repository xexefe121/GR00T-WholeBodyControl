"""Whole-clip preservation, duplicate rejection and source-boundary isolation."""

import copy
import os
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_buffered_source import buffered_source_lower_body
from gear_sonic.scripts.prepare_g1_true23_contact_step_lifecycle import motion_from_poses
from gear_sonic.scripts.record_g1_sonic_public29_baseline import lifecycle29
from gear_sonic.utils.g1_23dof_contract import SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_true23_generalist_curriculum import MOTION_KEYS
from gear_sonic.utils.g1_true23_generalist_lifecycle import build_lifecycle_timeline
from gear_sonic.utils.g1_true23_original29_reference import build_original29_reference
from gear_sonic.utils.g1_true23_original_intent_mix import assemble_mix
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


@pytest.fixture(scope="module")
def fixture():
    root = Path(__file__).resolve().parents[2]
    assets = Path(os.environ.get("G1_TRUE23_TEST_ASSET_ROOT", root))
    config = root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
    _, native, _ = prepare_true23_model(assets / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml", config)
    geometry = mujoco.MjModel.from_xml_path(str(assets / "gear_sonic/data/robots/g1/g1_29dof.xml"))
    members = []
    for i in range(2):
        poses = np.tile(np.r_[[0, 0, 0.76, 1, 0, 0, 0], np.zeros(29)], (20 + i, 1))
        poses[:, 0] = np.linspace(0, 1 + i, len(poses))
        poses[:, 20] = 0.1 + i * 0.05  # Source-only waist roll is never erased.
        poses[:, 27] = 0.2 + i * 0.03  # Source-only wrist pitch is never erased.
        poses[:, 7] = i * 0.15
        projected = np.c_[poses[:, :7], poses[:, 7:][:, SOURCE_MJ29_KEEP_INDICES]]
        motion = motion_from_poses(native, projected)
        motion, timeline = build_lifecycle_timeline(
            motion, model=native, simulation_config=config, return_target="planned_endpoint"
        )
        standing = np.zeros(36)
        q0 = np.asarray(timeline["configured_standing_qpos"])
        standing[:7] = q0[:7]
        standing[7 + np.asarray(SOURCE_MJ29_KEEP_INDICES)] = q0[7:]
        original, _ = lifecycle29(poses, standing)
        members.append(
            dict(
                name=f"clip{i}",
                recording_id=f"recording{i}",
                motion=motion,
                timeline=timeline,
                original_reference=build_original29_reference(geometry, original).arrays(),
            )
        )
    return members, dict(source_model=geometry, native_model=native, simulation_config=config)


def test_all_source_and_lifecycle_arrays_preserved(fixture):
    members, kwargs = fixture
    raw, spans, original, derived, report = assemble_mix(members, ["evaluation"], **kwargs)
    assert spans["total_frames"] == 41
    assert report["lifecycle_frames"] == sum(len(m["motion"]["joint_pos"]) for m in members)
    for key in MOTION_KEYS:
        np.testing.assert_array_equal(derived[key], np.concatenate([m["motion"][key] for m in members]))
        np.testing.assert_array_equal(
            raw[key],
            np.concatenate(
                [
                    m["motion"][key][
                        m["timeline"]["source_start_frame"] : m["timeline"]["source_stop_frame_exclusive"]
                    ]
                    for m in members
                ]
            ),
        )
    for key in original:
        np.testing.assert_array_equal(
            original[key], np.concatenate([m["original_reference"][key] for m in members])
        )
    assert report["untouched_test_generalization_claimed"] is False
    assert report["deployment_ready"] is False


def test_horizon_at_each_span_tail_does_not_read_next_recording(fixture):
    members, kwargs = fixture
    _, _, _, derived, report = assemble_mix(members, [], **kwargs)
    rows = report["lifecycle_spans"]["spans"]
    anchors = [r["start"] + r["length"] - 2 for r in rows]
    motion = SimpleNamespace(**{key: torch.tensor(derived[key], dtype=torch.float32) for key in MOTION_KEYS})
    motion.time_step_total = len(derived["joint_pos"])
    command = SimpleNamespace(motion=motion, _curriculum_spans=rows, time_steps=torch.tensor(anchors))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    lower = buffered_source_lower_body(env).numpy()
    for i, anchor in enumerate(anchors):
        np.testing.assert_array_equal(
            lower[i, :120], np.tile(derived["joint_pos"][anchor, :12].astype(np.float32), 10)
        )
        np.testing.assert_array_equal(lower[i, 120:], np.zeros(120, dtype=np.float32))


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda m: m[1].update(name=m[0]["name"]), "member names"),
        (lambda m: m[1].update(recording_id=m[0]["recording_id"]), "recording IDs"),
        (lambda m: m[0]["timeline"].update(source_timing_scale=0.7), "original-speed"),
        (lambda m: m[0]["timeline"].update(source_frame_indices=[0]), "original-speed"),
        (lambda m: m[0]["timeline"].update(return_target="configured_origin"), "original-speed"),
        (lambda m: m[0]["timeline"].update(source_start_frame=True), "source range"),
        (lambda m: m[0]["timeline"].update(source_stop_frame_exclusive=0), "source range"),
        (lambda m: m[0]["motion"]["joint_vel"].__setitem__((0, 0), 1.0), "changes saved channel"),
        (lambda m: m[0]["original_reference"]["virtual_vr21"].__setitem__((0, 0), 1.0), "task reconstruction"),
        (lambda m: m[0]["motion"]["joint_pos"].__setitem__((0, 0), 1.0), "changed"),
    ],
)
def test_rejects_changed_or_partial_members(fixture, mutation, message):
    original, kwargs = fixture
    members = copy.deepcopy(original)
    mutation(members)
    with pytest.raises(ValueError, match=message):
        assemble_mix(members, [], **kwargs)


def test_duplicate_content_under_new_names_rejected(fixture):
    members, kwargs = fixture
    duplicate = copy.deepcopy(members[0])
    duplicate.update(name="another", recording_id="another_recording")
    with pytest.raises(ValueError, match="duplicate source content"):
        assemble_mix([members[0], duplicate], [], **kwargs)


def test_evaluation_recording_cannot_enter_optimizer(fixture):
    members, kwargs = fixture
    with pytest.raises(ValueError, match="evaluation recording"):
        assemble_mix(members, ["recording1"], **kwargs)


@pytest.mark.parametrize("ids", [["x", "x"], [1], [""], "x"])
def test_invalid_evaluation_ids_rejected(fixture, ids):
    members, kwargs = fixture
    with pytest.raises(ValueError, match="evaluation recording IDs"):
        assemble_mix(members, ids, **kwargs)


def test_one_recording_cannot_be_labelled_multimotion(fixture):
    members, kwargs = fixture
    with pytest.raises(ValueError, match="at least two"):
        assemble_mix(members[:1], [], **kwargs)
