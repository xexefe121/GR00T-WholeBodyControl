import numpy as np
import pytest
import torch

from gear_sonic.utils.g1_true23_failure_sampling import FailureBinSampler, sampling_contract


def test_initial_mapping_bit_matches_uniform_and_does_not_draw_rng():
    sampler = FailureBinSampler([10, 1000], [137, 102], device="cpu")
    choices = torch.arange(10000).remainder(2)
    fractions = torch.rand(10000)
    rng = torch.random.get_rng_state().clone()
    expected = sampler.first[choices] + torch.floor(fractions * sampler.lengths[choices]).long()
    torch.testing.assert_close(sampler.sample(choices, fractions), expected, atol=0, rtol=0)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_failure_counts_ema_and_ignores_other_clips_and_nonfailed_controls():
    sampler = FailureBinSampler([10, 1000], [137, 102], device="cpu")
    counts = sampler.observe(
        torch.tensor([9, 10, 61, 62, 146, 147, 1101]), torch.tensor([True, False, True, True, True, True, True])
    )
    assert counts.tolist() == [[0, 2, 1], [0, 0, 1]]
    np.testing.assert_allclose(sampler.ema.numpy(), np.array([[0, 2, 1], [0, 0, 1]]) * 0.01, atol=0, rtol=0)
    sampler.observe(torch.tensor([15]), torch.tensor([False]))
    np.testing.assert_allclose(sampler.ema.numpy(), np.array([[0, 2, 1], [0, 0, 1]]) * 0.01 * 0.99, atol=0, rtol=0)
    assert sampler.controls == 2


def test_failure_weight_spreads_backward_without_crossing_clip_boundaries():
    sampler = FailureBinSampler([10, 1000], [137, 102], device="cpu")
    sampler.observe(torch.tensor([146]), torch.tensor([True]))
    p = sampler.probabilities().numpy()
    expected = 0.2 * np.array([50, 50, 37]) / 137 + 0.8 * np.array([0.25, 0.5, 1.0]) / 1.75
    np.testing.assert_allclose(p[0], expected, atol=1e-15, rtol=0)
    np.testing.assert_allclose(p[1], np.array([50, 50, 2]) / 102, atol=1e-15, rtol=0)


def test_inverse_cdf_preserves_all_frames_and_expected_mass():
    sampler = FailureBinSampler([10], [137], device="cpu")
    sampler.observe(torch.tensor([80]), torch.tensor([True]))
    fractions = (torch.arange(137000, dtype=torch.float64) + 0.5) / 137000
    anchors = sampler.sample(torch.zeros(len(fractions), dtype=torch.long), fractions)
    counts = torch.bincount(anchors - 10, minlength=137).double()
    assert (counts > 0).all()
    actual = torch.tensor([counts[:50].sum(), counts[50:100].sum(), counts[100:].sum()]) / len(fractions)
    torch.testing.assert_close(actual, sampler.probabilities()[0], atol=2 / len(fractions), rtol=0)


def test_short_last_bin_and_endpoint_cannot_sample_padding():
    sampler = FailureBinSampler([10, 1000], [101, 1], device="cpu")
    sampler.observe(torch.tensor([110, 1000]), torch.tensor([True, True]))
    choices = torch.tensor([0, 0, 1, 1])
    fractions = torch.tensor([0.0, 1.0 - 1e-15, 0.0, 1.0 - 1e-15], dtype=torch.float64)
    assert sampler.sample(choices, fractions).tolist() == [10, 110, 1000, 1000]


@pytest.mark.parametrize("fraction", [float("nan"), -0.1, 1.0])
def test_invalid_fraction_fails_closed(fraction):
    sampler = FailureBinSampler([10], [100], device="cpu")
    with pytest.raises(ValueError):
        sampler.sample(torch.tensor([0]), torch.tensor([fraction]))


def test_contract_is_a_training_sampler_not_a_motion_or_acceptance_change():
    contract = sampling_contract()
    assert not contract["mid_episode_pose_writes"]
    assert contract["uniform_support_on_every_original_source_frame"]
    assert not contract["sampled_suffix_completion_is_full_lifecycle_completion"]
    assert not contract["reference_reward_actor_physics_and_limits_changed"]


def command_fixture(cls):
    from types import SimpleNamespace

    command = cls.__new__(cls)
    command._curriculum_spans = [{}, {}]
    command._lifecycle_starts = torch.tensor([10, 1000])
    command._lifecycle_ends = torch.tensor([500, 1500])
    command._lifecycle_choice = torch.zeros(8, dtype=torch.long)
    command._lifecycle_last_anchor = torch.zeros(8, dtype=torch.long)
    command._env_clip_stop = torch.zeros(8, dtype=torch.long)
    command.time_steps = torch.zeros(8, dtype=torch.long)
    command._source_first_anchors = torch.tensor([150, 1200])
    command._source_lengths = torch.tensor([137, 102])
    command._episode_initial_anchor = torch.zeros(8, dtype=torch.long)
    command._episode_began_standing = torch.zeros(8, dtype=torch.bool)
    command.standing_reset_samples = command.source_reset_samples = 0
    command._source_reset_bin_counts = torch.zeros(10, dtype=torch.long)
    command.failure_sampler = FailureBinSampler([150, 1200], [137, 102], device="cpu")
    command.failure_reset_samples, command.failure_observations = [], []
    command._failure_monitor_installed = False
    command._env = SimpleNamespace(common_step_counter=0, device="cpu", num_envs=8)
    return command


def test_actual_command_uniform_fallback_keeps_old_reset_anchors_and_rng():
    from gear_sonic.envs.mjlab.sonic_true23_failure_sampling import FailureAdaptiveRootFeedbackCommand
    from gear_sonic.envs.mjlab.sonic_true23_mixed_reference_starts import MixedReferenceRootFeedbackCommand

    old, new = (
        command_fixture(MixedReferenceRootFeedbackCommand),
        command_fixture(FailureAdaptiveRootFeedbackCommand),
    )
    torch.manual_seed(809)
    old._uniform_sampling(torch.arange(8))
    rng_after = torch.random.get_rng_state().clone()
    torch.manual_seed(809)
    new._uniform_sampling(torch.arange(8))
    assert torch.equal(old.time_steps, new.time_steps)
    assert torch.equal(rng_after, torch.random.get_rng_state())
    assert new.standing_reset_samples == 2 and new.source_reset_samples == 6
    assert new.failure_reset_samples[0]["standing"].tolist() == [True, False, False, False] * 2


def test_monitor_consumes_original_done_once_and_preserves_return():
    from types import SimpleNamespace

    from gear_sonic.envs.mjlab.sonic_true23_failure_sampling import FailureAdaptiveRootFeedbackCommand

    command = command_fixture(FailureAdaptiveRootFeedbackCommand)
    original_calls = []
    token = torch.tensor([True, False, False, False, False, False, False, False])

    def compute():
        original_calls.append(1)
        return token

    command.time_steps[:] = 155
    command._env.termination_manager = SimpleNamespace(compute=compute, terminated=token)
    command.install_failure_monitor()
    command._env.common_step_counter = 1
    assert command._env.termination_manager.compute() is token
    assert len(original_calls) == 1 and command.failure_sampler.controls == 1
    assert command.failure_sampler.total_failures.tolist() == [[1, 0, 0], [0, 0, 0]]
    with pytest.raises(ValueError, match="already installed"):
        command.install_failure_monitor()
    with pytest.raises(ValueError, match="one observation"):
        command._env.termination_manager.compute()


def test_state_rejects_impossible_statistics_and_source_boundary_tampering():
    import copy

    from gear_sonic.trl.mjlab.native23_failure_sampling_runner import validate_sampler_state

    state = FailureBinSampler([150], [137], device="cpu").state()
    checkpoint = dict(
        failure_sampling_state=state,
        trainer_state=dict(env_common_step_counter=0),
        lineage=dict(
            materials=dict(
                resolved_config=dict(
                    payload=dict(
                        num_envs=32,
                        native23_root_feedback=dict(
                            curriculum=dict(
                                derived_spans=dict(
                                    spans=[
                                        dict(
                                            start=0,
                                            timeline=dict(source_start_frame=151, source_frames=137),
                                        )
                                    ]
                                )
                            )
                        ),
                    )
                )
            )
        ),
    )
    assert validate_sampler_state(checkpoint) is state
    invalid = copy.deepcopy(checkpoint)
    invalid["failure_sampling_state"]["ema"][0, 0] = 0.1
    with pytest.raises(ValueError, match="exceed observed"):
        validate_sampler_state(invalid)
    invalid = copy.deepcopy(checkpoint)
    invalid["failure_sampling_state"]["first"][0] = 149
    with pytest.raises(ValueError, match="source boundaries"):
        validate_sampler_state(invalid)
