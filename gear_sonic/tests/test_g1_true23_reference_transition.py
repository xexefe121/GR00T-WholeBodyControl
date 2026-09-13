from copy import deepcopy

import pytest

from gear_sonic.utils.g1_true23_reference_transition import validate_full_excursion_transition


def material():
    def fit(motion, excursion):
        return {
            "accepted": True,
            "source_field": "planned_qpos50",
            "recorded_policy_pose_used_as_choreography": False,
            "adapted_motion_sha256": motion,
            "named_source_sha256": "original29",
            "source_frame_count": 546,
            "source_fps": 50.0,
            "compiled_models": {"source": "source", "target": "target"},
            "ik_config": {"unchanged": True},
            "limits": {"excursion_scales": [excursion], "duration_scales": [2.0], "foot_p95_m": 0.05},
            "selected_attempt": 0,
            "attempts": [
                {
                    "accepted": True,
                    "failures": [],
                    "requested_duration_scale": 2.0,
                    "actual_duration_scale": 2.0,
                    "requested_excursion_scale": excursion,
                    "output_frames": 1091,
                }
            ],
        }

    def curriculum(motion):
        return {
            "stage": "lifecycle",
            "derived_arrays_sha256": "derived_" + motion,
            "native_reference_model_sha256": "native_model",
            "native_reference_sim_config_sha256": "physics",
            "original_training_inputs": {"motion_sha256": motion},
            "derived_spans": {"clip_count": 1, "total_frames": 1852, "spans": [{"original_source_frames": 1091}]},
        }

    old_fit, new_fit = fit("old", 0.9), fit("new", 1.0)
    audit = {
        "kind": "g1_true23_planned29_all_original_timestamps_fk_audit_v1",
        "original_timestamp_fidelity_passed": True,
        "failures": [],
        "all_original_timestamps_and_endpoints_evaluated": True,
        "original_frames_evaluated": 546,
        "control_frames": 1091,
        "compiled_models": deepcopy(new_fit["compiled_models"]),
    }
    return curriculum("old"), curriculum("new"), old_fit, new_fit, audit


def test_full_excursion_is_changed_reference_not_resumed_or_qualified():
    result = validate_full_excursion_transition(*material())
    assert result["reference_arrays_changed"]
    assert result["parent_evaluation_must_use_new_complete_reference"]
    assert result["actor_critic_optimizer_and_counters_preserved"]
    assert not result["same_reference_resume_claimed"]
    assert not result["old_tracking_results_qualify_new_reference"]
    assert not result["deployment_ready"]


@pytest.mark.parametrize("key", ["stage", "native_reference_model_sha256", "native_reference_sim_config_sha256"])
def test_fixed_training_material_cannot_change(key):
    values = material()
    values[1][key] = "changed"
    with pytest.raises(ValueError, match="fixed curriculum"):
        validate_full_excursion_transition(*values)


@pytest.mark.parametrize(
    "key", ["named_source_sha256", "source_frame_count", "source_fps", "compiled_models", "ik_config"]
)
def test_original_source_and_model_cannot_change(key):
    values = material()
    values[3][key] = "changed"
    with pytest.raises(ValueError, match="source/model/configuration"):
        validate_full_excursion_transition(*values)


@pytest.mark.parametrize(
    "defect",
    [
        "rejected",
        "failed_attempt",
        "cropped",
        "new_clip",
        "changed_motion",
        "relaxed_limit",
        "time",
        "still_shrunken",
        "same_reference",
        "failed_audit",
        "partial_audit",
    ],
)
def test_unqualified_or_unrelated_transition_rejected(defect):
    previous, current, old_fit, new_fit, audit = values = material()
    if defect == "rejected":
        new_fit["accepted"] = False
    elif defect == "failed_attempt":
        new_fit["attempts"][0]["failures"] = ["foot"]
    elif defect == "cropped":
        current["derived_spans"]["spans"][0]["original_source_frames"] -= 1
    elif defect == "new_clip":
        current["derived_spans"]["clip_count"] = 2
    elif defect == "changed_motion":
        current["original_training_inputs"]["motion_sha256"] = "other"
    elif defect == "relaxed_limit":
        new_fit["limits"]["foot_p95_m"] = 0.1
    elif defect == "time":
        new_fit["attempts"][0]["actual_duration_scale"] = 3.0
    elif defect == "still_shrunken":
        new_fit["attempts"][0]["requested_excursion_scale"] = 0.95
    elif defect == "same_reference":
        current["derived_arrays_sha256"] = previous["derived_arrays_sha256"]
    elif defect == "failed_audit":
        audit["original_timestamp_fidelity_passed"] = False
    else:
        audit["original_frames_evaluated"] -= 1
    with pytest.raises(ValueError):
        validate_full_excursion_transition(*values)
