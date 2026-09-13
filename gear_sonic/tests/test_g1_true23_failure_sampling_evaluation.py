"""Evaluation admission cannot mistake a smoke or previous recipe for qualification."""

import pytest

from gear_sonic.scripts.record_g1_true23_failure_sampling_lifecycle import verified_update_checkpoint


def valid_update(run):
    checkpoint = (run / "train/checkpoints/failure_sampling_model_100.pt").resolve()
    return dict(
        kind="native23_failure_sampling_actual_update_audit_v1",
        passed=True,
        update_verification_passed=True,
        completed_updates=100,
        actual_transitions=204800,
        actual_sampling_controls=6400,
        initial_actor_and_critic_equal_Q0=True,
        actual_sampling_source_failures=20,
        actual_source_anchors_different_from_uniform=10,
        standing_start_allocation_unchanged=True,
        all_sampled_anchors_within_original_source=True,
        weights_uses_only_already_observed_failures=True,
        hardware_authorized=False,
        deployment_ready=False,
        inputs={str(checkpoint): "a" * 64},
    )


def test_exact_bound_checkpoint_is_returned(tmp_path):
    update = valid_update(tmp_path)
    path, digest = verified_update_checkpoint(tmp_path, update)
    assert path == (tmp_path / "train/checkpoints/failure_sampling_model_100.pt").resolve()
    assert digest == "a" * 64


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("kind", "native23_world_quality_actual_update_audit_v1"),
        ("kind", "native23_world_projection_actual_update_audit_v1"),
        ("passed", False),
        ("update_verification_passed", False),
        ("completed_updates", 2),
        ("actual_transitions", 4096),
        ("actual_sampling_controls", 128),
        ("initial_actor_and_critic_equal_Q0", False),
        ("actual_sampling_source_failures", 0),
        ("actual_source_anchors_different_from_uniform", 0),
        ("standing_start_allocation_unchanged", False),
        ("all_sampled_anchors_within_original_source", False),
        ("weights_uses_only_already_observed_failures", False),
        ("hardware_authorized", True),
        ("deployment_ready", True),
        ("inputs", {}),
    ],
)
def test_incomplete_or_wrong_experiment_rejected(tmp_path, key, value):
    update = valid_update(tmp_path)
    update[key] = value
    with pytest.raises(ValueError):
        verified_update_checkpoint(tmp_path, update)


@pytest.mark.parametrize("digest", ["a" * 63, "A" * 64, "z" * 64, None, 123])
def test_malformed_checkpoint_identity_rejected(tmp_path, digest):
    update = valid_update(tmp_path)
    update["inputs"][next(iter(update["inputs"]))] = digest
    with pytest.raises(ValueError):
        verified_update_checkpoint(tmp_path, update)
