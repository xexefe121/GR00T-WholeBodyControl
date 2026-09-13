from copy import deepcopy

import pytest

from gear_sonic.scripts.g1_true23_reference_bank_campaign import validate_bank_layout
from gear_sonic.tests.test_g1_true23_contact_bank_reference import contact_materials
from gear_sonic.utils.g1_true23_contact_step_bank_reference import PHASES
from gear_sonic.utils.g1_true23_contact_step_transition import PROFILE


def step_materials(combined=True):
    unconditioned, conditioned, bank = contact_materials()
    old, new = (unconditioned if combined else conditioned), deepcopy(conditioned)
    new.update(derived_arrays_sha256="stepping-bank", generated_transition_profile=PROFILE)
    for contract, ramp_count in ((old, 100), (new, 369)):
        start = 0
        for row in contract["derived_spans"]["spans"]:
            timeline = row["timeline"]
            timeline.update(
                configured_standing_qpos=[0],
                returned_standing_qpos=[0],
                source_frames=row["original_source_frames"],
                source_timing_scale=1.0,
                source_frame_indices=list(range(row["original_source_frames"])),
                prehistory_frames=11,
                phases=[],
            )
            cursor = 0
            for name, count in zip(
                PHASES, (250, ramp_count, row["original_source_frames"], ramp_count, 250, 50), strict=True
            ):
                timeline["phases"].append(
                    dict(
                        name=name,
                        requested_controls=count,
                        control_start=cursor,
                        control_stop=cursor + count,
                        frame_start=cursor + 11,
                        frame_stop=cursor + count + 11,
                    )
                )
                cursor += count
            timeline.update(total_requested_controls=cursor, total_frames=cursor + 11)
            row.update(start=start, length=cursor + 11)
            start += row["length"]
            if contract is new:
                timeline.update(
                    contact_step_reference=dict(profile=PROFILE, input_bindings={"report": "sha"}),
                    contact_step_independent_geometry_audit=dict(provisional_geometry_screen_passed=True),
                    contact_step_conditional_support_failures=dict(acquisition_ramp=41, return_ramp=42),
                    contact_step_dynamic_feasibility_qualified=False,
                )
    return old, new, bank


def validate(old, new, bank, combined=True):
    return validate_bank_layout(
        old, new, bank, allow_expansion=False, allow_contact_steps=True, allow_contact_conditioning=combined
    )


@pytest.mark.parametrize("combined", [True, False])
def test_explicit_step_branch_preserves_recordings_and_discloses_failures(combined):
    old, new, bank = step_materials(combined)
    with pytest.raises(ValueError, match="explicit changed-reference"):
        validate_bank_layout(old, new, bank, allow_expansion=False)
    result = validate(old, new, bank, combined)
    assert result["source_contact_conditioning_also_changed"] is combined
    assert result["generated_transition_timing_changed"]
    assert not result["same_reference_resume_claimed"]
    assert not result["dynamic_feasibility_proven"]
    assert result["step_support_failures"]["dance"]["return_ramp"] == 42
    assert not result["deployment_ready"]


@pytest.mark.parametrize(
    "mutation",
    [
        "raw",
        "indices",
        "ownership",
        "start",
        "length",
        "phase_count",
        "phase_cursor",
        "phase_order",
        "source_speed",
        "total",
        "prehistory",
        "source_count",
        "standing",
        "source_support",
        "step_support",
        "negative_support",
        "unknown_support",
        "source_geometry",
        "step_geometry",
        "measured",
        "hardware",
        "fidelity",
        "calibration",
        "bindings",
    ],
)
def test_step_branch_rejects_hidden_scope_timing_or_qualification_changes(mutation):
    old, new, bank = step_materials()
    row = new["derived_spans"]["spans"][0]
    timeline, proof = row["timeline"], row["source_reference_conditioning"]
    if mutation == "raw":
        row["source_arrays_sha256"] = "different"
    elif mutation == "indices":
        timeline["source_frame_indices"].pop()
    elif mutation == "ownership":
        row["ownership"] = {"split": "test"}
    elif mutation == "start":
        row["start"] += 1
    elif mutation == "length":
        row["length"] -= 1
    elif mutation == "phase_count":
        timeline["phases"][0]["requested_controls"] -= 1
    elif mutation == "phase_cursor":
        timeline["phases"][1]["frame_start"] += 1
    elif mutation == "phase_order":
        timeline["phases"].reverse()
    elif mutation == "source_speed":
        timeline["source_timing_scale"] = 2.0
    elif mutation == "total":
        timeline["total_requested_controls"] -= 1
    elif mutation == "prehistory":
        timeline["prehistory_frames"] = 10
    elif mutation == "source_count":
        timeline["phases"][2]["requested_controls"] -= 1
    elif mutation == "standing":
        timeline["configured_standing_qpos"] = [1]
    elif mutation == "source_support":
        proof.pop("remaining_conditional_force_support_failures")
    elif mutation == "step_support":
        timeline.pop("contact_step_conditional_support_failures")
    elif mutation == "negative_support":
        timeline["contact_step_conditional_support_failures"]["return_ramp"] = -1
    elif mutation == "unknown_support":
        timeline["contact_step_conditional_support_failures"]["return_ramp"] = None
    elif mutation == "source_geometry":
        proof["geometry"]["independent_serialized_geometry_passed"] = False
    elif mutation == "step_geometry":
        timeline["contact_step_independent_geometry_audit"]["provisional_geometry_screen_passed"] = False
    elif mutation == "measured":
        proof["measured_robot_state_used"] = True
    elif mutation == "hardware":
        proof["hardware_authorized"] = True
    elif mutation == "fidelity":
        proof["raw_original_fidelity_acceptance_inherited"] = True
    elif mutation == "calibration":
        proof["final_source_start_registration"] = {"changed": True}
    else:
        timeline["contact_step_reference"]["input_bindings"] = {}
    with pytest.raises(ValueError):
        validate(old, new, bank)


@pytest.mark.parametrize("mutation", ["conditioned_source", "registration", "double_conditioning"])
def test_step_only_branch_cannot_change_conditioned_source(mutation):
    old, new, bank = step_materials(False)
    row = new["derived_spans"]["spans"][0]
    if mutation == "conditioned_source":
        row["registered_source_arrays_sha256"] += "changed"
    elif mutation == "registration":
        row["source_start_registration"] = {"changed": True}
    with pytest.raises(ValueError):
        validate(old, new, bank, combined=mutation == "double_conditioning")
