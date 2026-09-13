from copy import deepcopy

import pytest

from gear_sonic.scripts.compare_g1_true23_root_feedback_campaigns import CASES, compare_verified


def fixture():
    return dict(
        reference_arrays_sha256="reference",
        phases=["full"],
        release_compatibility="runtime",
        policy="parent",
        inputs={},
        cases=[
            dict(
                case=case,
                complete=True,
                physics={"model": "same"},
                thresholds_m={"left": 0.05, "right": 0.05},
                root_p95_m=0.1,
                landmarks_p95_m={"left": 0.07, "right": 0.09},
                final_standing={"joint": 0.2, "root": 0.08},
                source_landmark_screen_passed=False,
            )
            for case in CASES
        ],
    )


def test_identity_comparison_no_false_promotion():
    before = fixture()
    report = compare_verified(before, deepcopy(before))
    assert report["all_six_full_lifecycles_completed"]
    assert report["after_source_landmark_screen_pass_count"] == 0
    assert all(r["root_p95_change_m"] == 0 for r in report["cases"])
    assert not report["deployment_ready"]


def test_partial_failure_cannot_become_improvement():
    before, after = fixture(), fixture()
    after["cases"][0]["complete"] = False
    after["cases"][0]["landmarks_p95_m"] = {"left": 0, "right": 0}
    report = compare_verified(before, after)
    assert not report["all_six_full_lifecycles_completed"]
    assert report["cases"][0]["landmark_p95_change_m"] is None
    assert not report["cases"][0]["all_five_landmarks_nondegrading"]


def test_better_average_cannot_hide_one_worse_foot_or_return():
    before, after = fixture(), fixture()
    after["cases"][0]["landmarks_p95_m"] = {"left": 0.02, "right": 0.10}
    after["cases"][0]["final_standing"]["root"] = 0.1
    row = compare_verified(before, after)["cases"][0]
    assert not row["all_five_landmarks_nondegrading"]
    assert row["final_standing_changes"]["root"] > 0


@pytest.mark.parametrize(
    "field", ["reference_arrays_sha256", "phases", "release_compatibility", "physics", "thresholds_m", "cases"]
)
def test_changed_benchmark_rejected(field):
    before, after = fixture(), fixture()
    if field in ("physics", "thresholds_m"):
        after["cases"][0][field] = "changed"
    elif field == "cases":
        after["cases"].pop()
    else:
        after[field] = "changed"
    with pytest.raises(ValueError):
        compare_verified(before, after)


@pytest.mark.parametrize("mutation", [None, "kind", "profile", "geometry", "counterfactual"])
def test_step_comparison_is_explicit_and_cannot_adopt_counterfactuals(mutation):
    from gear_sonic.scripts.compare_g1_true23_root_feedback_campaigns import (
        STEP_PROFILE,
        validate_reference_profile,
    )

    report = dict(
        kind="g1_true23_contact_step_lifecycle_policy_diagnostic_v1",
        contact_step_reference_diagnostic=True,
        timeline=dict(
            generated_transition_profile=STEP_PROFILE,
            contact_step_independent_geometry_audit=dict(provisional_geometry_screen_passed=True),
        ),
    )
    with pytest.raises(ValueError):
        validate_reference_profile(report, "none")
    if mutation == "kind":
        report["kind"] = "g1_true23_root_feedback_single_policy_lifecycle_campaign_v1"
    elif mutation == "profile":
        report["timeline"]["generated_transition_profile"] = "none"
    elif mutation == "geometry":
        report["timeline"]["contact_step_independent_geometry_audit"] = {}
    elif mutation == "counterfactual":
        report["root_input_counterfactual"] = {"scale": 4}
    if mutation is None:
        validate_reference_profile(report, STEP_PROFILE)
    else:
        with pytest.raises(ValueError):
            validate_reference_profile(report, STEP_PROFILE)


def test_different_reference_profiles_cannot_be_compared_as_same_reference():
    before, after = fixture(), fixture()
    after["generated_transition_profile"] = "different"
    with pytest.raises(ValueError):
        compare_verified(before, after)


def test_phase_errors_do_not_hide_bad_entry_behind_good_source_or_omit_unrun_return():
    import numpy as np

    from gear_sonic.scripts.compare_g1_true23_root_feedback_campaigns import phase_tracking

    phases = [
        dict(name=name, control_start=i * 2, control_stop=(i + 1) * 2, requested_controls=2)
        for i, name in enumerate(("acquisition_ramp", "source_motion", "return_ramp"))
    ]
    arrays = dict(
        landmark_error_m=np.array([[0.2] * 5, [0.1] * 5, [0.01] * 5, [0.02] * 5]),
        desired_root_position_w=np.zeros((4, 3)),
        measured_root_position_w=np.zeros((4, 3)),
    )
    result = phase_tracking(dict(phases=phases), arrays, 4)
    assert result[0]["landmark_position_p95_m"]["left_ankle_origin"] == pytest.approx(0.195)
    assert result[1]["landmark_position_p95_m"]["left_ankle_origin"] == pytest.approx(0.0195)
    assert result[2]["completed_controls"] == 0
    assert not result[2]["complete"]
    assert result[2]["landmark_position_p95_m"] is None
    assert not any(row["contact_or_lifecycle_qualification"] for row in result)
    with pytest.raises(ValueError):
        phase_tracking(dict(phases=phases), arrays, 3)
