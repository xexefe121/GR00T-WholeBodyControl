"""Synthetic, redistributable original-time audit tests; no BONES data fixture."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils import g1_23dof_task_space_retarget as ik
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_bones_seed_source_audit import audit_original_timestamps
from gear_sonic.utils.g1_true23_generalist_retarget import AdaptationLimits, _resample, validate_named_motion


@pytest.fixture(scope="module")
def models():
    root = Path(__file__).resolve().parents[2]
    return ik.load_models(root / ik.DEFAULT_SOURCE_MODEL, root / ik.DEFAULT_TARGET_MODEL)


def sample(models, pulse=False):
    names = ik._model_layout(models[0]).joint_names
    defaults = dict(zip(HARDWARE_23_JOINT_NAMES, SAFE_TARGET_DEFAULT_Q_HARDWARE, strict=True))
    source = {
        "joint_names": np.asarray(names),
        "joint_pos": np.tile([defaults.get(name, 0) for name in names], (9, 1)),
        "root_pos_w": np.tile([1.0, -2.0, 0.8], (9, 1)),
        "root_quat_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (9, 1)),
        "timestamps_s": 3 + np.arange(9) / 120,
        "fps": np.asarray([120.0]),
        "contact_flags": np.ones((9, 2), dtype=bool),
    }
    if pulse:
        source["root_pos_w"][2, 2] += 0.3
    limits = AdaptationLimits(duration_scales=(2.0,), excursion_scales=(1.0,), maximum_output_frames=2500)
    source = validate_named_motion(source, models[0])
    control, time_map, scale = _resample(source, 2.0, limits)
    target_names = ik._model_layout(models[1]).joint_names
    motion = SimpleNamespace(
        fps=50.0,
        joint_pos_hardware=control["joint_pos"][:, [names.index(name) for name in target_names]],
        root_pos_w=control["root_pos_w"],
        root_quat_wxyz=control["root_quat_wxyz"],
    )
    target = {
        **ik.build_mjlab_motion_arrays(models[1], motion),
        "source_time_map_s": time_map,
        "timestamps_s": np.arange(len(time_map)) / 50,
        "joint_names": np.asarray(target_names),
        "source_joint_names": source["joint_names"],
        **{f"source_{key}_resampled": control[key] for key in ("joint_pos", "root_pos_w", "root_quat_wxyz")},
        "achieved_task_pos_w": np.zeros((len(time_map), len(ik.DEFAULT_TASKS), 3)),
    }
    report = {
        "kind": "g1_true23_generalist_bounded_offline_retarget",
        "accepted": True,
        "source_role": "requested_choreography",
        "limits": asdict(limits),
        "ik_config": asdict(ik.RetargetConfig(optimize_lower_body=True)),
        "source_frame_count": 9,
        "source_fps": 120.0,
        "selected_attempt": 0,
        "attempts": [
            {
                "accepted": True,
                "failures": [],
                "requested_duration_scale": 2.0,
                "requested_excursion_scale": 1.0,
                "actual_duration_scale": scale,
            }
        ],
    }
    return source, target, report


def test_all_original_times_including_endpoints_are_evaluated(models):
    source, target, report = sample(models)
    before = {key: value.copy() for key, value in source.items()}
    result = audit_original_timestamps(*models, source, target, report)
    assert result["original_timestamp_fidelity_passed"]
    assert result["original_frames_evaluated"] == 9
    assert result["control_frames"] == 7
    assert result["original_time_sample_indices"] == list(range(9))
    assert not result["cached_achieved_task_positions_used"]
    assert not result["source_root_reanchored_or_joint_targets_clipped"]
    for key, value in before.items():
        np.testing.assert_array_equal(source[key], value)
    for key in (
        "dynamic_feasibility_verified",
        "deployment_ready",
        "hardware_authorized",
        "training_corpus_ready",
        "continuous_time_fidelity_proven",
        "control_grid_trajectory_audit_replaced",
    ):
        assert result[key] is False


def test_missing_between_grid_source_pose_is_not_hidden(models):
    source, target, report = sample(models, pulse=True)
    result = audit_original_timestamps(*models, source, target, report)
    assert not result["original_timestamp_fidelity_passed"]
    assert 2 in result["categories"]["left_foot_position"]["frame_indices"]
    assert result["fidelity"]["left_foot"]["achieved_to_original_m"]["max"] > 0.1


def test_cached_success_positions_do_not_change_audit(models):
    source, target, report = sample(models)
    first = audit_original_timestamps(*models, source, target, report)
    target["achieved_task_pos_w"][:] = np.nan
    second = audit_original_timestamps(*models, source, target, report)
    assert first == second


@pytest.mark.parametrize(
    "key,mutation,match",
    [
        ("source_time_map_s", lambda a: a[:-1], "shape"),
        ("source_time_map_s", lambda a: a + 0.01, "endpoint"),
        ("source_time_map_s", lambda a: a[::-1], "endpoint"),
        ("timestamps_s", lambda a: a + 0.01, "control grid"),
        ("fps", lambda a: np.asarray([60.0]), "fps"),
        ("joint_names", lambda a: a[::-1], "names/order"),
        ("source_joint_pos_resampled", lambda a: a + 0.01, "complete original"),
        ("body_pos_w", lambda a: a * np.nan, "finite shape"),
        ("body_quat_w", lambda a: a * 2, "normalized"),
        ("joint_pos", lambda a: a[:, :-1], "shape"),
    ],
)
def test_changed_or_ambiguous_payload_rejected(models, key, mutation, match):
    source, target, report = sample(models)
    target[key] = mutation(target[key])
    with pytest.raises(ValueError, match=match):
        audit_original_timestamps(*models, source, target, report)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r.update(accepted=False), "accepted requested"),
        (lambda r: r.update(source_role="robot_state"), "accepted requested"),
        (lambda r: r.update(source_frame_count=8), "frame count"),
        (lambda r: r.update(selected_attempt=True), "selected retarget"),
        (lambda r: r["attempts"][0].update(failures=["bad"]), "not accepted"),
        (lambda r: r["limits"].update(hand_head_p95_m=0.2), "cannot weaken"),
        (lambda r: r["ik_config"].update(valid_max_com_regression_m=0.1), "unchanged protected"),
    ],
)
def test_rejected_or_relaxed_report_cannot_pass(models, mutation, match):
    source, target, report = sample(models)
    changed = deepcopy(report)
    mutation(changed)
    with pytest.raises(ValueError, match=match):
        audit_original_timestamps(*models, source, target, changed)
