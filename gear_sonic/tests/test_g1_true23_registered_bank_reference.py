from copy import deepcopy
import json

import numpy as np
import pytest

from gear_sonic.scripts.g1_true23_reference_bank_campaign import validate_bank_layout
from gear_sonic.tests.test_g1_true23_reference_bank_campaign import _materials
from gear_sonic.tests.test_g1_true23_start_registration import fixture
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_curriculum import array_digest
from gear_sonic.utils.g1_true23_registered_bank_reference import (
    START_REGISTRATION_PROFILE,
    load_registered_repair_member,
)
from gear_sonic.utils.g1_true23_start_registration import register_motion_start


def registered_materials():
    _, previous, bank = _materials()
    current = deepcopy(previous)
    current["derived_arrays_sha256"] = "registered-bank"
    current["source_start_registration"] = START_REGISTRATION_PROFILE
    for row in current["derived_spans"]["spans"]:
        _, proof = register_motion_start(fixture())
        proof["frame_count"] = row["original_source_frames"]
        row["source_start_registration"] = proof
        row["registered_source_arrays_sha256"] = row["name"] + "-registered"
        row["derived_arrays_sha256"] = row["name"] + "-new-lifecycle"
        row["timeline"]["generated_reference_repair"] = {"report_sha256": "repair"}
    return previous, current, bank


def test_registration_is_explicit_changed_world_reference_not_bank_expansion():
    previous, current, bank = registered_materials()
    proof = validate_bank_layout(previous, current, bank, allow_expansion=False, allow_start_registration=True)
    assert proof["original_audited_source_arrays_preserved"]
    assert not proof["same_reference_resume_claimed"]
    assert not proof["same_unregistered_world_benchmark_claimed"]
    assert proof["actor_critic_optimizer_and_counters_preserved"]
    assert not proof["deployment_ready"]
    with pytest.raises(ValueError, match="explicit"):
        validate_bank_layout(previous, current, bank, allow_expansion=False)
    with pytest.raises(ValueError, match="also expand"):
        validate_bank_layout(previous, current, bank, allow_expansion=True, allow_start_registration=True)
    # Once registered, exact same-bank continuation does not recalibrate again.
    same = validate_bank_layout(deepcopy(current), current, bank, allow_expansion=False)
    assert same["same_reference_resume_claimed"]
    with pytest.raises(ValueError, match="once-only"):
        validate_bank_layout(
            deepcopy(current), current, bank, allow_expansion=False, allow_start_registration=True
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "bank",
        "member",
        "frames",
        "ownership",
        "timing",
        "initial_state",
        "crop",
        "heading",
        "translation",
        "joints",
        "height",
        "travel",
        "measured",
        "reset",
        "hardware",
        "repair",
        "digest",
    ],
)
def test_registration_transition_cannot_change_choreography_or_hide_reference_change(mutation):
    previous, current, bank = registered_materials()
    row = current["derived_spans"]["spans"][0]
    proof = row["source_start_registration"]
    if mutation == "source":
        row["source_arrays_sha256"] = "changed"
    elif mutation == "bank":
        previous["original_training_inputs"]["motion_sha256"] = "changed"
    elif mutation == "member":
        previous["derived_spans"]["spans"].pop()
    elif mutation == "frames":
        row["length"] += 1
    elif mutation == "ownership":
        row["ownership"] = {"split": "test"}
    elif mutation == "timing":
        row["timeline"]["total_requested_controls"] -= 1
    elif mutation == "initial_state":
        row["timeline"]["configured_standing_qpos"] = [1]
    elif mutation == "crop":
        row["original_source_indices_requested"].pop()
    elif mutation == "heading":
        proof["target_heading_rad"] = 0.1
    elif mutation == "translation":
        proof["target_root_xy"] = [0.1, 0.0]
    elif mutation in {"joints", "height"}:
        proof[
            {"joints": "joint_positions_and_velocities_bit_exact", "height": "world_height_channels_bit_exact"}[
                mutation
            ]
        ] = False
    elif mutation in {"travel", "measured", "reset", "hardware"}:
        proof[
            {
                "travel": "relative_motion_excursion_scaled",
                "measured": "measured_robot_state_used",
                "reset": "midrun_reregistration_or_pose_reset",
                "hardware": "hardware_authorized",
            }[mutation]
        ] = True
    elif mutation == "repair":
        row["timeline"].pop("generated_reference_repair")
    else:
        row.pop("registered_source_arrays_sha256")
    with pytest.raises(ValueError):
        validate_bank_layout(previous, current, bank, allow_expansion=False, allow_start_registration=True)


def receipts(tmp_path, mutation=None):
    original = fixture()
    registered, proof = register_motion_start(original)
    if mutation is not None:
        registered[mutation].flat[-1] += 0.001
    original_path, registered_path, report_path = [
        tmp_path / name for name in ("original.npz", "registered.npz", "report.json")
    ]
    np.savez_compressed(original_path, **original)
    np.savez_compressed(registered_path, **registered)
    report = dict(
        kind="g1_true23_registered_source_lifecycle_geometry_diagnostic_v1",
        registration=proof,
        policy_evaluation_performed=False,
        training_reference_accepted=False,
        input_bindings={str(original_path): sha256_file(original_path)},
        timeline={"source_motion_sha256": sha256_file(registered_path)},
        hardware_authorized=False,
        deployment_ready=False,
    )
    report_path.write_text(json.dumps(report))
    clip = dict(
        source_path=str(original_path), source_sha256=sha256_file(original_path), length=len(original["joint_pos"])
    )
    row = dict(
        original_source_motion_sha256=sha256_file(original_path),
        registered_source_path=str(registered_path),
        source_motion_sha256=sha256_file(registered_path),
        registration_report_path=str(report_path),
        registration_report_sha256=sha256_file(report_path),
    )
    return row, clip, original, registered


def test_registration_loader_recomputes_all_arrays_not_report_claims(tmp_path):
    row, clip, original, registered = receipts(tmp_path)
    result, inputs = load_registered_repair_member(row, clip)
    assert result["original_source_arrays_sha256"] == array_digest(original)
    assert result["registered_source_arrays_sha256"] == array_digest(registered)
    assert len(inputs) == 3


@pytest.mark.parametrize(
    "key", ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "fps"]
)
def test_registration_loader_rejects_rehashed_motion_tampering(tmp_path, key):
    row, clip, _, _ = receipts(tmp_path, key)
    with pytest.raises(ValueError, match="independently recomputed"):
        load_registered_repair_member(row, clip)
