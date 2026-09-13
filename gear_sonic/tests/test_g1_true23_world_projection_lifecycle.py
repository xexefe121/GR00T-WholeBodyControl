"""The unchanged SIM plant cannot accept a previous, partial, or live checkpoint."""

import copy

import pytest

from gear_sonic.scripts.record_g1_true23_world_projection_lifecycle import verified_update_checkpoint
from gear_sonic.trl.mjlab.native23_world_projection_runner import training_contract


def valid_update(run):
    path = (run / "train/checkpoints/world_projection_model_100.pt").resolve()
    return dict(
        kind="native23_world_projection_actual_update_audit_v1",
        passed=True,
        update_verification_passed=True,
        completed_updates=100,
        actual_transitions=204800,
        initial_actor_and_critic_equal_matched_quality_predecessor=True,
        world_projection=training_contract(),
        projection_loss=dict(actual_ppo_minibatches=800),
        hardware_authorized=False,
        deployment_ready=False,
        inputs={str(path): "a" * 64},
    )


def test_new_complete_audit_selects_exact_snapshot(tmp_path):
    path, digest = verified_update_checkpoint(tmp_path, valid_update(tmp_path))
    assert path == (tmp_path / "train/checkpoints/world_projection_model_100.pt").resolve()
    assert digest == "a" * 64


@pytest.mark.parametrize(
    "key,value",
    [
        ("kind", "native23_world_quality_actual_update_audit_v1"),
        ("passed", False),
        ("update_verification_passed", False),
        ("completed_updates", 2),
        ("actual_transitions", 64),
        ("initial_actor_and_critic_equal_matched_quality_predecessor", False),
        ("world_projection", {}),
        ("projection_loss", {"actual_ppo_minibatches": 799}),
        ("hardware_authorized", True),
        ("deployment_ready", True),
        ("inputs", {}),
    ],
)
def test_incomplete_or_previous_audit_rejected(tmp_path, key, value):
    update = copy.deepcopy(valid_update(tmp_path))
    update[key] = value
    with pytest.raises(ValueError):
        verified_update_checkpoint(tmp_path, update)
