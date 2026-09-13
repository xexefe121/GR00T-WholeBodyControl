from types import SimpleNamespace

import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_phase_reference_starts import (
    PROFILE,
    PhaseReferenceRootFeedbackCommand,
    phase_reference_anchors,
)
from gear_sonic.utils.g1_true23_start_schedule import start_schedule_contract, start_schedule_transition_contract


def inputs():
    return (
        torch.arange(16),
        torch.full((16,), 10),
        torch.tensor([[260, 629, 1370]]).repeat(16, 1),
        torch.tensor([[369, 741, 369]]).repeat(16, 1),
        torch.arange(16) / 16,
    )


def test_exact_quarters_sample_only_requested_phase_and_keep_history():
    values = inputs()
    anchors, roles = phase_reference_anchors(*values)
    assert torch.bincount(roles).tolist() == [4, 4, 4, 4]
    assert anchors[roles == 0].tolist() == [10] * 4
    assert torch.all(anchors - 9 >= 0)
    for role, first, length in ((1, 260, 369), (2, 629, 741), (3, 1370, 369)):
        assert torch.all(anchors[roles == role] >= first)
        assert torch.all(anchors[roles == role] < first + length)


@pytest.mark.parametrize("fraction", [-0.1, 1.0, float("nan"), float("inf")])
def test_invalid_sampling_rejected(fraction):
    values = list(inputs())
    values[-1][1] = fraction
    with pytest.raises(ValueError):
        phase_reference_anchors(*values)


def test_per_clip_offsets_and_full_first_last_phase_samples():
    values = list(inputs())
    values[1][8:] += 2500
    values[2][8:] += 2500
    for fraction in (0.0, 1 - 1e-7):
        values[-1][:] = fraction
        anchors, roles = phase_reference_anchors(*values)
        for i in range(16):
            if roles[i] == 0:
                assert anchors[i] == values[1][i]
            else:
                expected = values[2][i, roles[i] - 1].clone()
                if fraction > 0:
                    expected += values[3][i, roles[i] - 1] - 1
                assert anchors[i] == expected


def test_sampling_overlap_and_mid_episode_physical_rewrites_rejected():
    values = list(inputs())
    values[2][2, 1] = 300
    with pytest.raises(ValueError, match="overlap"):
        phase_reference_anchors(*values)
    with pytest.raises(RuntimeError, match="environment reset"):
        PhaseReferenceRootFeedbackCommand._resample_command(
            SimpleNamespace(_inside_environment_reset=False), torch.tensor([0])
        )


def test_schedule_change_requires_explicit_flag_and_preserves_benchmark():
    previous = dict(start_schedule=start_schedule_contract("mixed_reference_reset_v1"))
    args = SimpleNamespace(start_schedule=PROFILE, allow_start_schedule_transition=False)
    with pytest.raises(ValueError, match="explicit"):
        start_schedule_transition_contract(previous, args)
    args.allow_start_schedule_transition = True
    transition = start_schedule_transition_contract(previous, args)
    assert transition["rollout_schedule_changed"]
    assert transition["reference_arrays_and_evaluation_unchanged"]
    assert not transition["fresh_standing_environment_reset"]
    assert not transition["new"]["sampled_suffix_completion_is_full_lifecycle_completion"]
    assert not transition["new"]["sampled_reference_state_dynamically_qualified"]


def test_launcher_rejects_incomplete_lifecycle_or_nonquarter_batch():
    from gear_sonic.scripts.train_g1_true23_root_feedback import validate_bounds
    from gear_sonic.tests.test_train_g1_true23_root_feedback import arguments

    args = arguments("regression", "--start-schedule", PROFILE)
    with pytest.raises(ValueError, match="lifecycle curriculum"):
        validate_bounds(args)
    args.curriculum_stage = "lifecycle"
    validate_bounds(args)
    args.num_envs = 6
    with pytest.raises(ValueError, match="multiple of four"):
        validate_bounds(args)
