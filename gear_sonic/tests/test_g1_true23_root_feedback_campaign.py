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
