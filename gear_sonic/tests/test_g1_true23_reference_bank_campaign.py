from copy import deepcopy

import pytest

from gear_sonic.scripts.g1_true23_reference_bank_campaign import validate_bank_layout


def _materials():
    rows = [
        dict(
            name=name,
            length=781,
            original_source_frames=20,
            source_arrays_sha256=name + "-source",
            derived_arrays_sha256=name + "-lifecycle",
            timeline={"total_requested_controls": 770},
            every_original_source_frame_requested=True,
            original_source_indices_requested=list(range(20)),
        )
        for name in ("dance", "walk")
    ]
    previous = dict(
        stage="lifecycle",
        native_reference_model_sha256="model",
        native_reference_sim_config_sha256="config",
        original_training_inputs={"motion_sha256": "old"},
        derived_arrays_sha256="old-derived",
        derived_spans={"clip_count": 1, "spans": deepcopy(rows[:1])},
    )
    current = deepcopy(previous)
    current.update(
        original_training_inputs={"motion_sha256": "bank"},
        derived_arrays_sha256="bank-derived",
        derived_spans={"clip_count": 2, "spans": rows},
    )
    bank = dict(
        kind="g1_true23_small_collision_audited_local_reference_bank_v2",
        accepted_reference_bank=True,
        all_member_control_and_original_timestamp_gates_passed=True,
        all_member_control_and_original_timestamp_self_collision_gates_passed=True,
        all_six_serialized_training_arrays_exactly_equal_original_members=True,
        bones_members_train_only=True,
        source_resampling_root_alignment_joint_deletion_or_transition_blending=False,
        hardware_authorized=False,
        deployment_ready=False,
        clip_count=2,
        output={"sha256": "bank"},
        clips=[
            dict(name="dance", length=20, split="local_regression"),
            dict(name="walk", length=20, split="train"),
        ],
    )
    return previous, current, bank


def test_expansion_preserves_old_reference_and_does_not_claim_same_reference_resume():
    before, after, bank = _materials()
    proof = validate_bank_layout(before, after, bank, allow_expansion=True)
    assert proof["previous_clip_count"] == 1 and proof["new_clip_count"] == 2
    assert proof["previous_source_and_lifecycle_arrays_preserved"]
    assert proof["reference_arrays_changed"]
    assert not proof["same_reference_resume_claimed"]
    assert not proof["old_single_clip_scores_qualify_bank"]
    assert not proof["deployment_ready"]


def test_unchanged_bank_can_continue_without_relabelling_an_expansion():
    _, after, bank = _materials()
    proof = validate_bank_layout(deepcopy(after), after, bank, allow_expansion=False)
    assert proof["same_reference_resume_claimed"]
    with pytest.raises(ValueError, match="unchanged"):
        validate_bank_layout(deepcopy(after), after, bank, allow_expansion=True)


def test_expansion_requires_explicit_flag():
    before, after, bank = _materials()
    with pytest.raises(ValueError, match="explicit"):
        validate_bank_layout(before, after, bank, allow_expansion=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "model",
        "physics",
        "stage",
        "membership",
        "dropped_old",
        "source_modified",
        "lifecycle_modified",
        "old_timing",
        "cropped",
        "omitted_index",
        "validation",
        "test",
        "bank_hash",
        "hardware",
        "gate_failed",
        "collision_missing",
        "legacy_bank",
    ],
)
def test_bank_expansion_cannot_waive_identity_coverage_or_previous_reference(mutation):
    before, after, bank = _materials()
    if mutation == "model":
        after["native_reference_model_sha256"] = "changed"
    elif mutation == "physics":
        after["native_reference_sim_config_sha256"] = "changed"
    elif mutation == "stage":
        before["stage"] = after["stage"] = "acquisition"
    elif mutation == "membership":
        bank["clips"].reverse()
    elif mutation == "dropped_old":
        before["derived_spans"]["spans"][0]["name"] = "omitted-old-reference"
    elif mutation == "source_modified":
        after["derived_spans"]["spans"][0]["source_arrays_sha256"] = "changed"
    elif mutation == "lifecycle_modified":
        after["derived_spans"]["spans"][0]["derived_arrays_sha256"] = "changed"
    elif mutation == "old_timing":
        after["derived_spans"]["spans"][0]["timeline"]["total_requested_controls"] -= 1
    elif mutation == "cropped":
        after["derived_spans"]["spans"][1]["original_source_frames"] -= 1
    elif mutation == "omitted_index":
        after["derived_spans"]["spans"][1]["original_source_indices_requested"].pop()
    elif mutation in {"validation", "test"}:
        bank["clips"][1]["split"] = mutation
    elif mutation == "bank_hash":
        bank["output"]["sha256"] = "changed"
    elif mutation == "hardware":
        bank["hardware_authorized"] = True
    elif mutation == "collision_missing":
        bank.pop("all_member_control_and_original_timestamp_self_collision_gates_passed")
    elif mutation == "legacy_bank":
        bank["kind"] = "g1_true23_small_audited_local_reference_bank_v1"
    else:
        bank["all_member_control_and_original_timestamp_gates_passed"] = False
    with pytest.raises(ValueError):
        validate_bank_layout(before, after, bank, allow_expansion=True)


def _repair_materials():
    _, current, bank = _materials()
    for clip in bank["clips"]:
        clip.update(
            recording_id=clip["name"] + "-recording",
            original_frames=21,
            actual_duration_scale=2.0,
            source_sha256=clip["name"] + "-new",
        )
    previous = deepcopy(current)
    previous["original_training_inputs"]["motion_sha256"] = "old-bank"
    previous["derived_arrays_sha256"] = "old-derived"
    old_clips = deepcopy(bank["clips"])
    for clip in old_clips:
        clip["source_sha256"] = clip["name"] + "-old"
    old_clips.append({**deepcopy(old_clips[0]), "name": "bad", "source_sha256": "bad-old"})
    previous["derived_spans"]["spans"].append({**deepcopy(previous["derived_spans"]["spans"][0]), "name": "bad"})
    previous["derived_spans"]["clip_count"] = 3
    old_bank = dict(accepted_reference_bank=True, output={"sha256": "old-bank"}, clips=old_clips)
    plan = dict(
        kind="g1_true23_collision_repaired_reference_bank_transition_v1",
        replacements=[
            dict(name=c["name"], previous_source_sha256=c["name"] + "-old", new_source_sha256=c["source_sha256"])
            for c in bank["clips"]
        ],
        quarantined=[
            dict(name="bad", previous_source_sha256="bad-old", reason="Unresolved reference self-penetration")
        ],
        all_omitted_references_remain_unqualified=True,
        hardware_authorized=False,
        deployment_ready=False,
    )
    return previous, current, bank, plan, old_bank


def test_explicit_collision_repair_preserves_recordings_and_discloses_quarantined_scope():
    before, after, bank, plan, old_bank = _repair_materials()
    proof = validate_bank_layout(
        before, after, bank, allow_expansion=False, repair_plan=plan, previous_bank=old_bank
    )
    assert proof["reference_arrays_changed"] and not proof["previous_source_and_lifecycle_arrays_preserved"]
    assert not proof["same_reference_resume_claimed"] and not proof["held_out_generalization_verified"]
    assert [r["name"] for r in proof["quarantined"]] == ["bad"]
    assert proof["actor_critic_optimizer_and_counters_preserved"] and not proof["deployment_ready"]
    with pytest.raises(ValueError, match="retain every previous"):
        validate_bank_layout(before, after, bank, allow_expansion=False)


@pytest.mark.parametrize(
    "change",
    [
        "source_hash",
        "recording",
        "frames",
        "duration",
        "split",
        "missing_quarantine",
        "missing_replacement",
        "reason",
        "previous_bank",
        "hidden_failure",
        "expansion",
        "unchanged",
        "hardware",
    ],
)
def test_collision_repair_cannot_hide_identity_scope_or_failures(change):
    before, after, bank, plan, old_bank = _repair_materials()
    expansion = False
    if change in {"recording", "frames", "duration", "split"}:
        key = {
            "recording": "recording_id",
            "frames": "original_frames",
            "duration": "actual_duration_scale",
            "split": "split",
        }[change]
        bank["clips"][0][key] = "changed"
    elif change == "source_hash":
        plan["replacements"][0]["previous_source_sha256"] = "changed"
    elif change == "missing_quarantine":
        plan["quarantined"].clear()
    elif change == "missing_replacement":
        plan["replacements"].pop()
    elif change == "reason":
        plan["quarantined"][0]["reason"] = ""
    elif change == "previous_bank":
        old_bank["output"]["sha256"] = "changed"
    elif change == "hidden_failure":
        plan["all_omitted_references_remain_unqualified"] = False
    elif change == "expansion":
        expansion = True
    elif change == "unchanged":
        after["derived_arrays_sha256"] = before["derived_arrays_sha256"]
    else:
        plan["hardware_authorized"] = True
    with pytest.raises(ValueError):
        validate_bank_layout(
            before, after, bank, allow_expansion=expansion, repair_plan=plan, previous_bank=old_bank
        )
