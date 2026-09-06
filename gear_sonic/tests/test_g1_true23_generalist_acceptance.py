import copy

import pytest

from gear_sonic.utils.g1_true23_generalist_corpus import canonical_digest

from gear_sonic.utils.g1_true23_generalist_acceptance import (
    ERROR_LIMITS,
    PHASES,
    SEEDS,
    ZERO_COUNTS,
    assess_generalist_suite,
    lifecycle_case_failures,
    progress_decision,
)


def good_case(recording="dance0", seed=1729):
    return {
        "kind": "g1_true23_generalist_measured_lifecycle_v1",
        "recording_id": recording,
        "seed": seed,
        "policy_sha256": "a" * 64,
        "split_sha256": "b" * 64,
        "status": "executed",
        "actuation_profile": "g1_true23_nominal_native_model_actuation_v1",
        "metric_frame": "world_source_aligned_no_per_frame_registration",
        "foot_metric": "sole_contact_points",
        **dict.fromkeys(ZERO_COUNTS, 0),
        "phases": {name: {"expected_controls": 500, "completed_controls": 500} for name in PHASES},
        "adapted_reference_p95_m": dict.fromkeys(ERROR_LIMITS, 0.01),
        "source_reference_p95_m": dict.fromkeys(ERROR_LIMITS, 0.03),
        "duration_scale": 1.2,
        "maximum_task_space_excursion_reduction_fraction": 0.1,
        "source_time_map_verified": True,
        "return_to_standing_stable": True,
        "perturbation_state_verified": True,
    }


def campaign(count=100):
    ids = [f"dance{i}" for i in range(count)]
    result = {
        "corpus_audit": {
            "manifest_valid": True,
            "manifest_sha256": "c" * 64,
            "recording_splits": dict.fromkeys(ids, "test"),
            "recording_families": dict.fromkeys(ids, "dance"),
            "split_sha256": "b" * 64,
            "corpus_quantity_and_coverage_sufficient": count >= 100,
        },
        "recording_ids": ids,
        "cases": [good_case(i, s) for i in ids for s in SEEDS],
        "policy_sha256": "a" * 64,
        "teleop_receipt": {
            "policy_sha256": "a" * 64,
            **dict.fromkeys(
                (
                    "saved_full_body_passed",
                    "paced_stream_passed",
                    "unexpected_changes_passed",
                    "stale_input_passed",
                    "standing_return_passed",
                ),
                True,
            ),
        },
        "export_receipt": {
            "policy_sha256": "a" * 64,
            **dict.fromkeys(
                ("encoder_parity_passed", "decoder_parity_passed", "causal_runtime_profile_verified"), True
            ),
        },
        "regression_receipt": {
            "policy_sha256": "a" * 64,
            "all_full_requested_motions_passed": True,
            "full_duration_videos_bound": True,
        },
    }

    result["corpus_audit"]["asset_metadata"] = {
        identifier: {
            "recording_id": identifier,
            "transform": "original",
            "timing": {"frame_count": 500, "fps": 50},
        }
        for identifier in ids
    }
    result["corpus_audit"]["asset_bindings"] = {identifier: {"sha256": "d" * 64} for identifier in ids}
    plan = {
        "kind": "g1_true23_generalist_full_source_plan_v1",
        "split_sha256": "b" * 64,
        "corpus_manifest_sha256": "c" * 64,
        "seeds": list(SEEDS),
        "recordings": {
            identifier: {
                "source_asset_id": identifier,
                "source_sha256": "d" * 64,
                "source_frame_count": 500,
                "source_fps": 50,
                "adapted_reference_sha256": "e" * 64,
                "time_map_sha256": "f" * 64,
                "phase_controls": dict.fromkeys(PHASES, 500),
            }
            for identifier in ids
        },
    }
    result["evaluation_plan"] = plan
    for case in result["cases"]:
        case["duration_scale"] = 1.0
        case["source_coverage_frames"] = [0, 499]
        case.update(
            evaluation_plan_sha256=canonical_digest(plan),
            adapted_reference_sha256="e" * 64,
            time_map_sha256="f" * 64,
        )
    return result


def test_full_complete_suite_can_only_qualify_simulation():
    result = assess_generalist_suite(**campaign())
    assert result["simulator_qualification_complete"]
    assert not result["hardware_authorized"] and not result["deployment_ready"]


def test_rejection_and_missing_seeds_never_leave_denominator():
    values = campaign()
    values["cases"] = values["cases"][15:]
    result = assess_generalist_suite(**values)
    assert result["planned_original_recordings"] == 100
    assert result["success_fraction"] == 0.95
    values["cases"][0]["status"] = "rejected"
    assert not assess_generalist_suite(**values)["simulator_qualification_complete"]


def test_one_dance_cannot_qualify():
    assert not assess_generalist_suite(**campaign(1))["simulator_qualification_complete"]


def test_self_reported_one_frame_full_dance_cannot_qualify():
    values = campaign()
    for case in values["cases"]:
        for phase in case["phases"].values():
            phase.update(expected_controls=1, completed_controls=1)
    assert not assess_generalist_suite(**values)["simulator_qualification_complete"]


def test_plan_cannot_replace_full_original_with_crop():
    values = campaign()
    values["corpus_audit"]["asset_metadata"]["dance0"]["transform"] = "crop"
    with pytest.raises(ValueError, match="complete original"):
        assess_generalist_suite(**values)
    values = campaign()
    values["evaluation_plan"]["recordings"]["dance0"]["phase_controls"]["motion"] = 103
    with pytest.raises(ValueError, match="truncates source"):
        assess_generalist_suite(**values)


@pytest.mark.parametrize("key", ["evaluation_plan_sha256", "adapted_reference_sha256", "time_map_sha256"])
def test_receipt_cannot_change_plan_source_or_time_map(key):
    values = campaign()
    values["cases"][0][key] = "0" * 64
    with pytest.raises(ValueError, match="immutable evaluation plan"):
        assess_generalist_suite(**values)


@pytest.mark.parametrize("key", ZERO_COUNTS)
def test_no_fallback_posewrites_or_violations(key):
    case = good_case()
    case[key] = 1
    assert lifecycle_case_failures(case)


@pytest.mark.parametrize(
    "key,value",
    [
        ("duration_scale", 2.01),
        ("maximum_task_space_excursion_reduction_fraction", 0.21),
        ("source_time_map_verified", False),
        ("return_to_standing_stable", False),
        ("perturbation_state_verified", False),
        ("foot_metric", "ankle_origin"),
    ],
)
def test_incomplete_adaptation_or_measurement_rejected(key, value):
    case = good_case()
    case[key] = value
    assert lifecycle_case_failures(case)


def test_full_prefix_and_nonfinite_never_pass():
    case = good_case()
    case["phases"]["motion"]["completed_controls"] = 103
    assert lifecycle_case_failures(case)
    case = good_case()
    case["adapted_reference_p95_m"]["head"] = float("nan")
    assert lifecycle_case_failures(case)


def test_mixed_policies_or_derivative_leakage_rejects_campaign():
    values = campaign()
    values["cases"][0]["policy_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="mixed checkpoints"):
        assess_generalist_suite(**values)
    values = campaign()
    values["corpus_audit"]["recording_splits"]["dance0"] = "train"
    with pytest.raises(ValueError, match="held-out"):
        assess_generalist_suite(**values)
    values = campaign()
    values["corpus_audit"]["recording_families"]["dance0"] = "locomotion"
    with pytest.raises(ValueError, match="dance recordings"):
        assess_generalist_suite(**values)


def test_three_unchanged_evaluations_require_diagnosis():
    rows = [
        {
            "update_count": i * 100,
            "evaluation_plan_sha256": "c" * 64,
            "full_lifecycle_success_fraction": 0.0,
            "mean_task_space_p95_m": 0.5,
        }
        for i in range(4)
    ]
    assert progress_decision(rows)["diagnosis_required"]
    changed = copy.deepcopy(rows)
    changed[-1]["mean_task_space_p95_m"] = 0.4
    assert not progress_decision(changed)["diagnosis_required"]
    changed[-1]["evaluation_plan_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="identical evaluation plan"):
        progress_decision(changed)
