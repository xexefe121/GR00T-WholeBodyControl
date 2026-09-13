import copy

import pytest

from gear_sonic.envs.mjlab.sonic_true23_world_tracking_termination import termination_contract
from gear_sonic.scripts.record_g1_true23_world_quality_lifecycle import verified_update_checkpoint
from gear_sonic.utils.g1_true23_world_quality import quality_contract


def fixture_update(run):
    checkpoint = (run / "train/checkpoints/world_quality_model_100.pt").resolve()
    return dict(
        kind="native23_world_quality_actual_update_audit_v1",
        passed=True,
        completed_updates=100,
        actual_transitions=204800,
        initial_actor_and_critic_equal_matched_world_predecessor=True,
        additional_training_failure=termination_contract(),
        world_quality_bonus=quality_contract(),
        hardware_authorized=False,
        deployment_ready=False,
        inputs={str(checkpoint): "a" * 64},
    )


def test_only_new_audited_full_update_snapshot_is_bound(tmp_path):
    audit = fixture_update(tmp_path)
    original = copy.deepcopy(audit)
    path, digest = verified_update_checkpoint(tmp_path, audit)
    assert audit == original
    assert path.name == "world_quality_model_100.pt"
    assert digest == "a" * 64


@pytest.mark.parametrize(
    "key,value",
    [
        ("kind", "native23_world_tracking_actual_update_audit_v1"),
        ("passed", False),
        ("completed_updates", 2),
        ("actual_transitions", 64),
        ("inputs", {}),
        ("world_quality_bonus", {}),
        ("additional_training_failure", {}),
        ("initial_actor_and_critic_equal_matched_world_predecessor", False),
        ("hardware_authorized", True),
        ("deployment_ready", True),
    ],
)
def test_old_partial_or_mislabeled_evidence_rejected(tmp_path, key, value):
    audit = fixture_update(tmp_path)
    audit[key] = value
    with pytest.raises(ValueError):
        verified_update_checkpoint(tmp_path, audit)
