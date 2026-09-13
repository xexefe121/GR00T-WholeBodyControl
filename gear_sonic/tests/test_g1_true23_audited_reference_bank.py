"""No licensed samples: synthetic accepted evidence and exact concatenation."""

from copy import deepcopy

import numpy as np
import pytest

from gear_sonic.scripts.build_g1_true23_audited_reference_bank import (
    MOTION_KEYS,
    concatenate_exact_members,
    validate_collision_evidence,
    validate_member_evidence,
)
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES


def _evidence():
    models = {"source": "source29", "target": "native23"}
    fit = {
        "accepted": True,
        "selected_attempt": 0,
        "source_frame_count": 9,
        "adapted_motion_sha256": "motion",
        "compiled_models": models,
        "attempts": [
            {
                "accepted": True,
                "failures": [],
                "requested_excursion_scale": 1.0,
                "output_frames": 17,
                "actual_duration_scale": 2.0,
            }
        ],
    }
    audit = {
        "kind": "g1_true23_planned29_all_original_timestamps_fk_audit_v1",
        "original_timestamp_fidelity_passed": True,
        "failures": [],
        "all_original_timestamps_and_endpoints_evaluated": True,
        "original_frames_evaluated": 9,
        "control_frames": 17,
        "control_fps": 50,
        "original_time_sample_indices": list(range(9)),
        "compiled_models": models.copy(),
    }
    return fit, audit


def _validate(fit, audit):
    return validate_member_evidence(fit, audit, motion_sha256="motion", original_frames=9, control_frames=17)


def test_accepted_complete_evidence_preserves_model_and_timing():
    fit, audit = _evidence()
    assert _validate(fit, audit) == {"target_model_sha256": "native23", "actual_duration_scale": 2.0}


@pytest.mark.parametrize(
    "mutation",
    [
        "rejected_fit",
        "bad_selection",
        "bool_selection",
        "failed_attempt",
        "scaled",
        "cropped",
        "wrong_hash",
        "failed_audit",
        "omitted_sample",
        "wrong_model",
        "bad_audit_kind",
        "wrong_fps",
    ],
)
def test_rejected_or_unbound_evidence_cannot_enter_bank(mutation):
    fit, audit = _evidence()
    if mutation == "rejected_fit":
        fit["accepted"] = False
    elif mutation == "bad_selection":
        fit["selected_attempt"] = -1
    elif mutation == "bool_selection":
        fit["selected_attempt"] = False
    elif mutation == "failed_attempt":
        fit["attempts"][0]["failures"] = ["com"]
    elif mutation == "scaled":
        fit["attempts"][0]["requested_excursion_scale"] = 0.9
    elif mutation == "cropped":
        fit["attempts"][0]["output_frames"] = 16
    elif mutation == "wrong_hash":
        fit["adapted_motion_sha256"] = "another"
    elif mutation == "failed_audit":
        audit["failures"] = ["com"]
    elif mutation == "omitted_sample":
        audit["original_time_sample_indices"] = list(range(8))
    elif mutation == "wrong_model":
        audit["compiled_models"]["target"] = "other"
    elif mutation == "bad_audit_kind":
        audit["kind"] = "control_grid_only"
    else:
        audit["control_fps"] = 49
    with pytest.raises(ValueError):
        _validate(fit, audit)


def _member(count, offset):
    result = {key: np.full((count, 23), offset, dtype=np.float32) for key in ("joint_pos", "joint_vel")}
    for key in MOTION_KEYS[2:]:
        result[key] = np.full((count, 24, 4 if key == "body_quat_w" else 3), offset, dtype=np.float32)
    result["fps"] = np.array([50.0])
    result["joint_names"] = np.array(HARDWARE_23_JOINT_NAMES)
    return result


def test_every_frame_and_boundary_remains_exact_without_reanchoring_or_blending():
    members = [_member(17, 0.2), _member(19, -1.2)]
    before = deepcopy(members)
    result = concatenate_exact_members(members)
    for key in MOTION_KEYS:
        np.testing.assert_array_equal(result[key][:17], before[0][key])
        np.testing.assert_array_equal(result[key][17:], before[1][key])
        for left, right in zip(members, before, strict=True):
            np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.parametrize("mutation", ["one", "seventeen", "joint_order", "fps", "nonfinite", "dtype", "shape"])
def test_concatenation_rejects_unbounded_or_incompatible_members(mutation):
    members = [_member(17, 0.2), _member(19, -1.2)]
    if mutation == "one":
        members = members[:1]
    elif mutation == "seventeen":
        members = members[:1] * 17
    elif mutation == "joint_order":
        members[1]["joint_names"] = members[1]["joint_names"][::-1]
    elif mutation == "fps":
        members[1]["fps"][0] = 120
    elif mutation == "nonfinite":
        members[1]["joint_pos"][2, 3] = np.nan
    elif mutation == "dtype":
        members[1]["joint_pos"] = members[1]["joint_pos"].astype(float)
    else:
        members[1]["body_pos_w"] = members[1]["body_pos_w"][:, :-1]
    with pytest.raises(ValueError):
        concatenate_exact_members(members)


def _collision_evidence():
    gates = (
        "complete_control_grid_fidelity_passed",
        "all_original_timestamp_fidelity_passed",
        "serialized_declared_path_bounds_passed",
        "no_control_grid_robot_robot_penetration",
        "no_original_grid_robot_robot_penetration",
    )
    result = {
        "kind": "g1_true23_full_collision_repair_audit_v1",
        "acceptance": {"passed": True, "checks": dict.fromkeys(gates, True)},
        "accepted_candidate": {"sha256": "motion"},
        "compiled_models": {"target": "native23"},
        "compiled_physics_model_sha256": "a" * 64,
        "control_frames": 17,
        "original_frames": 9,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    for key, count in (("after_control_self_contacts", 17), ("after_original_self_contacts", 9)):
        result[key] = {
            "frames": count,
            "frames_with_robot_robot_penetration": 0,
            "penetration_frame_indices": [],
            "maximum_penetration_m": 0.0,
            "physics_integration_steps": 0,
        }
    return result


def _validate_collision(audit):
    return validate_collision_evidence(
        audit, motion_sha256="motion", original_frames=9, control_frames=17, target_model_sha256="native23"
    )


def test_collision_gate_binds_both_full_grids_to_motion_and_physical_model():
    assert _validate_collision(_collision_evidence()) == "a" * 64


@pytest.mark.parametrize(
    "failure", ["missing", "rejected", "hash", "model", "physics", "control", "original", "omitted", "relabelled"]
)
def test_old_fk_acceptance_or_partial_collision_clearance_cannot_enter_new_bank(failure):
    audit = _collision_evidence()
    if failure == "missing":
        audit = {}
    elif failure == "rejected":
        audit["acceptance"]["passed"] = False
    elif failure == "hash":
        audit["accepted_candidate"]["sha256"] = "other"
    elif failure == "model":
        audit["compiled_models"]["target"] = "other"
    elif failure == "physics":
        audit.pop("compiled_physics_model_sha256")
    elif failure == "omitted":
        audit["after_original_self_contacts"]["frames"] -= 1
    elif failure == "relabelled":
        audit["acceptance"]["checks"]["no_original_grid_robot_robot_penetration"] = False
    else:
        audit["after_" + failure + "_self_contacts"]["frames_with_robot_robot_penetration"] = 1
    with pytest.raises(ValueError):
        _validate_collision(audit)
