import copy

import numpy as np
import pytest

from gear_sonic.scripts.diagnose_g1_true23_source_initialization import (
    EXPERIMENT_KIND,
    NOMINAL_BASELINE_EXPERIMENT_KIND,
    NORMAL_KIND,
    source_initialized_tail,
    validate_baseline_cases,
)
from gear_sonic.utils.g1_true23_root_feedback_campaign import validate_campaign_evaluation


def fixture():
    phases, control = [], 0
    for name, count in (
        ("initial_standing", 20),
        ("acquisition_ramp", 10),
        ("source_motion", 12),
        ("return_ramp", 10),
        ("returned_standing", 20),
        ("standing_proof_margin", 5),
    ):
        phases.append(
            dict(
                name=name,
                control_start=control,
                control_stop=control + count,
                frame_start=11 + control,
                frame_stop=11 + control + count,
                requested_controls=count,
            )
        )
        control += count
    frames = 11 + control
    quaternions = np.zeros((frames, 24, 4))
    quaternions[..., 0] = 1
    motion = dict(
        fps=np.array([50.0]),
        joint_pos=np.tile(np.arange(frames)[:, None] * 0.001, (1, 23)),
        joint_vel=np.zeros((frames, 23)),
        body_pos_w=np.zeros((frames, 24, 3)),
        body_quat_w=quaternions,
        body_lin_vel_w=np.zeros((frames, 24, 3)),
        body_ang_vel_w=np.zeros((frames, 24, 3)),
    )
    timeline = dict(
        prehistory_frames=11,
        phases=phases,
        total_requested_controls=control,
        source_frames=12,
        source_frame_indices=list(range(12)),
    )
    return motion, timeline


def test_every_source_and_return_sample_exact_with_disclosed_single_initialization():
    motion, timeline = fixture()
    saved_motion, saved_timeline = copy.deepcopy(motion), copy.deepcopy(timeline)
    actual, changed = source_initialized_tail(motion, timeline)
    for key in motion:
        np.testing.assert_array_equal(motion[key], saved_motion[key])
        if key != "fps":
            np.testing.assert_array_equal(actual[key][11:], motion[key][41:])
            np.testing.assert_array_equal(actual[key][:11], np.repeat(motion[key][41:42], 11, axis=0))
    assert timeline == saved_timeline
    assert changed["kind"] == EXPERIMENT_KIND
    assert changed["phases"][0]["control_start"] == 0
    assert changed["total_requested_controls"] == 47
    assert changed["source_frame_indices"] == list(range(12))
    assert not changed["standing_acquisition_tested"]
    assert not changed["eligible_parent_for_training_continuation"]
    assert not changed["deployment_ready"]


@pytest.mark.parametrize("mutation", ["prehistory", "count", "crop", "missing_tail"])
def test_incomplete_or_changed_lifecycle_refused(mutation):
    motion, timeline = fixture()
    if mutation == "prehistory":
        timeline["prehistory_frames"] = 10
    elif mutation == "count":
        timeline["total_requested_controls"] -= 1
    elif mutation == "crop":
        timeline["source_frame_indices"] = list(range(1, 13))
    else:
        timeline["phases"].pop()
    with pytest.raises(ValueError):
        source_initialized_tail(motion, timeline)


@pytest.mark.parametrize("kind", [EXPERIMENT_KIND, NOMINAL_BASELINE_EXPERIMENT_KIND])
def test_source_initialized_report_cannot_be_normal_continuation_evidence(kind):
    with pytest.raises(ValueError, match="CPU lifecycle campaign"):
        validate_campaign_evaluation(
            {"kind": kind}, checkpoint_sha256="x", actor_sha256="y", lineage_sha256="z", updates=1200
        )


def baseline(cases):
    return dict(
        kind=NORMAL_KIND,
        records=[
            dict(
                case=case,
                result=dict(
                    available_controls=100,
                    requested_controls=100,
                    diagnostic_prefix_requested=False,
                    completed_controls=60,
                    failure={"message": "retained failure"},
                ),
            )
            for case in cases
        ],
    )


def test_nominal_only_baseline_requires_explicit_distinct_scope():
    report = baseline(["nominal"])
    with pytest.raises(ValueError, match="scope"):
        validate_baseline_cases(report)
    assert validate_baseline_cases(report, nominal_only_diagnostic=True) == ["nominal"]
    full = baseline(["nominal", "standing_push_x", "standing_push_y"])
    assert validate_baseline_cases(full) == ["nominal", "standing_push_x", "standing_push_y"]
    with pytest.raises(ValueError, match="scope"):
        validate_baseline_cases(full, nominal_only_diagnostic=True)
    assert report["records"][0]["result"]["failure"] == {"message": "retained failure"}


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "prefix", "short", "wrong_kind", "no_records"])
def test_baseline_scope_cannot_hide_missing_cases_prefixes_or_relabelled_results(mutation):
    report = baseline(["nominal", "standing_push_x", "standing_push_y"])
    if mutation == "duplicate":
        report["records"].append(copy.deepcopy(report["records"][0]))
    elif mutation == "missing":
        report["records"].pop(0)
    elif mutation == "prefix":
        report["records"][0]["result"]["diagnostic_prefix_requested"] = True
    elif mutation == "short":
        report["records"][0]["result"]["requested_controls"] -= 1
    elif mutation == "wrong_kind":
        report["kind"] = EXPERIMENT_KIND
    else:
        report.pop("records")
    with pytest.raises(ValueError):
        validate_baseline_cases(report)
