from copy import deepcopy

import pytest

from gear_sonic.utils.g1_true23_root_feedback_campaign import validate_campaign_evaluation


def receipt():
    source = dict(checkpoint_sha256="a", actor_state_sha256="b", lineage_sha256="c", completed_update_count=100)
    result = dict(
        requested_controls=1841,
        available_controls=1841,
        completed_controls=1841,
        diagnostic_prefix_requested=False,
        state_pose_writes_after_reset=0,
        history_resets_during_motion=0,
        initial_state_and_history_sha256="initial",
        compiled_model_sha256="model",
        physics_config_sha256="physics",
        motion_sha256="motion",
        kp_hardware=[1],
        kd_hardware=[1],
        effort_limit_hardware_nm=[1],
        failure=None,
    )
    return dict(
        kind="g1_true23_root_feedback_single_policy_lifecycle_campaign_v1",
        only_one_hash_bound_policy_per_case=True,
        no_postinitial_robot_pose_rewrites=True,
        no_fallback_controller=True,
        deployment_ready=False,
        hardware_authorized=False,
        timeline={"total_requested_controls": 1841},
        records=[
            dict(
                case=case,
                policy_identity={"source": deepcopy(source)},
                result=deepcopy(result),
                lifecycle={"source_motion_tracking": {"lifecycle_qualified": False}},
            )
            for case in ("nominal", "standing_push_x", "standing_push_y")
        ],
    )


def validate(report):
    return validate_campaign_evaluation(
        report, checkpoint_sha256="a", actor_sha256="b", lineage_sha256="c", updates=100
    )


def test_failed_evaluation_retains_failure_not_a_promotion():
    report = receipt()
    report["records"][1]["result"].update(completed_controls=600, failure="absolute height/tilt stop")
    outcomes = validate(report)
    assert len(outcomes) == 3
    assert outcomes[1]["completed_controls"] == 600
    assert outcomes[1]["requested_controls"] == 1841
    assert outcomes[1]["failure"] == "absolute height/tilt stop"
    assert not outcomes[1]["source_motion_tracking"]["lifecycle_qualified"]


@pytest.mark.parametrize(
    "tamper", ["checkpoint", "actor", "count", "missing_case", "prefix", "reset", "physics", "hardware"]
)
def test_invalid_continuation_evidence_rejected(tamper):
    report = receipt()
    row = report["records"][1]
    if tamper == "checkpoint":
        row["policy_identity"]["source"]["checkpoint_sha256"] = "other"
    elif tamper == "actor":
        row["policy_identity"]["source"]["actor_state_sha256"] = "other"
    elif tamper == "count":
        row["policy_identity"]["source"]["completed_update_count"] = 99
    elif tamper == "missing_case":
        report["records"].pop()
    elif tamper == "prefix":
        row["result"]["diagnostic_prefix_requested"] = True
    elif tamper == "reset":
        row["result"]["state_pose_writes_after_reset"] = 1
    elif tamper == "physics":
        row["result"]["compiled_model_sha256"] = "other"
    elif tamper == "hardware":
        report["hardware_authorized"] = True
    with pytest.raises(ValueError):
        validate(report)


def test_new_profile_requires_matching_evaluation_runtime():
    from gear_sonic.utils.g1_true23_release_compatibility import release_compatibility_contract

    report = receipt()
    contract = release_compatibility_contract("a" * 64)
    report["release_compatibility"] = contract
    with pytest.raises(ValueError, match="compatibility"):
        validate(report)
    for row in report["records"]:
        row["policy_identity"]["release_compatibility"] = contract
        row["result"]["release_compatibility"] = contract
        row["result"]["runtime_adapter"] = dict(
            mode="causal_past_virtual_source",
            action_convention="released_bounded_linear",
            future_reference_consumed=False,
            case=row["case"],
        )
    arguments = dict(
        checkpoint_sha256="a",
        actor_sha256="b",
        lineage_sha256="c",
        updates=100,
        expected_release_compatibility=contract,
    )
    assert len(validate_campaign_evaluation(report, **arguments)) == 3
    report["records"][1]["result"]["runtime_adapter"]["action_convention"] = "native_tanh"
    with pytest.raises(ValueError, match="runtime"):
        validate_campaign_evaluation(report, **arguments)


def test_buffered_campaign_requires_exact_timing_and_force_contracts():
    import json

    from gear_sonic.teleop.buffered_source_simulation import BufferedSourceSimulationAdapter
    from gear_sonic.tests.test_g1_true23_buffered_reference import material
    from gear_sonic.utils.g1_true23_buffered_reference import BUFFERED_TIMING
    from gear_sonic.utils.g1_true23_release_compatibility import release_compatibility_contract
    from gear_sonic.utils.g1_true23_root_feedback_benchmark import ForcePulse

    report = receipt()
    contract = release_compatibility_contract("a" * 64, "released29_scale_bounded_linear_v2", BUFFERED_TIMING)
    report["release_compatibility"] = contract
    motion, vr = material()
    for row in report["records"]:
        pulses = (
            ()
            if row["case"] == "nominal"
            else (ForcePulse(500, 50, (40.0, 0.0, 0.0) if row["case"] == "standing_push_x" else (0.0, 40.0, 0.0)),)
        )
        row["policy_identity"]["release_compatibility"] = contract
        row["result"]["release_compatibility"] = contract
        row["result"]["runtime_adapter"] = json.loads(
            json.dumps(BufferedSourceSimulationAdapter(motion, vr, pulses).contract())
        )
    arguments = dict(
        checkpoint_sha256="a",
        actor_sha256="b",
        lineage_sha256="c",
        updates=100,
        expected_release_compatibility=contract,
    )
    assert len(validate_campaign_evaluation(report, **arguments)) == 3
    for field, value in (
        ("source_clock_latency_s", 0.0),
        ("future_reference_consumed", True),
        ("scored_reference_timing_or_physics_changed", True),
        ("reference_timing", "causal_history"),
    ):
        changed = deepcopy(report)
        changed["records"][0]["result"]["runtime_adapter"][field] = value
        with pytest.raises(ValueError, match="runtime"):
            validate_campaign_evaluation(changed, **arguments)


def test_exact_training_model_counterfactual_cannot_replace_primary_cpu_evaluation():
    report = receipt()
    for row in report["records"]:
        row["result"]["training_model_counterfactual"] = {
            "kind": "exact_compiled_training_model_counterfactual_v1",
            "original_replay_model_qualification": False,
        }
    with pytest.raises(ValueError, match="counterfactual physics"):
        validate(report)


def test_training_scene_change_cannot_silently_continue_old_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from gear_sonic.scripts import export_g1_true23_root_feedback as exporter
    from gear_sonic.utils.g1_true23_root_feedback_campaign import campaign_parent_contract

    checkpoint, evaluation = tmp_path / "parent.pt", tmp_path / "evaluation.json"
    torch.save({}, checkpoint)
    evaluation.write_text("{}")
    monkeypatch.setattr(
        exporter, "validate_export_semantics", lambda _: {"root_feedback_training_configuration": {}}
    )
    args = SimpleNamespace(training_physics_contract={"name": "pinned_cpu_referee_scene_v1"})
    with pytest.raises(ValueError, match="physics change requires fresh"):
        campaign_parent_contract(checkpoint, evaluation, args=args, curriculum={})


@pytest.mark.parametrize("tamper", [None, "kind", "profile", "geometry", "counterfactual", "missing_case"])
def test_contact_step_replay_requires_explicit_separate_reference_profile(tamper):
    from gear_sonic.utils.g1_true23_contact_step_transition import PROFILE

    report = receipt()
    report.update(
        kind="g1_true23_contact_step_lifecycle_policy_diagnostic_v1", contact_step_reference_diagnostic=True
    )
    report["timeline"].update(
        generated_transition_profile=PROFILE,
        contact_step_independent_geometry_audit=dict(provisional_geometry_screen_passed=True),
    )
    with pytest.raises(ValueError):
        validate(report)
    if tamper == "kind":
        report["kind"] = "g1_true23_root_feedback_single_policy_lifecycle_campaign_v1"
    elif tamper == "profile":
        report["timeline"]["generated_transition_profile"] = "none"
    elif tamper == "geometry":
        report["timeline"]["contact_step_independent_geometry_audit"] = {}
    elif tamper == "counterfactual":
        report["root_input_counterfactual"] = {"scale": 4}
    elif tamper == "missing_case":
        report["records"].pop()
    args = dict(
        checkpoint_sha256="a",
        actor_sha256="b",
        lineage_sha256="c",
        updates=100,
        expected_generated_transition_profile=PROFILE,
    )
    if tamper is None:
        result = validate_campaign_evaluation(report, **args)
        assert len(result) == 3
        assert not result[0]["source_motion_tracking"]["lifecycle_qualified"]
    else:
        with pytest.raises(ValueError):
            validate_campaign_evaluation(report, **args)
