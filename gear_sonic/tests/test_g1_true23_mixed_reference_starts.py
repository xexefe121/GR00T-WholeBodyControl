from types import SimpleNamespace

import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_mixed_reference_starts import (
    MixedReferenceRootFeedbackCommand,
    mixed_reference_anchors,
)
from gear_sonic.utils.g1_true23_start_schedule import start_schedule_contract, start_schedule_transition_contract


def inputs():
    return (
        torch.arange(16),
        torch.full((16,), 10),
        torch.full((16,), 360),
        torch.full((16,), 1091),
        torch.arange(16) / 16,
    )


def test_quarter_standing_other_anchors_have_valid_q9_q10_source_samples():
    ids, start, first, length, fractions = inputs()
    anchors, standing = mixed_reference_anchors(ids, start, first, length, fractions)
    assert standing.nonzero().flatten().tolist() == [0, 4, 8, 12]
    torch.testing.assert_close(anchors[standing], start[standing])
    assert (anchors[~standing] + 1 >= 361).all()
    assert (anchors[~standing] + 1 < 1452).all()
    assert (anchors - 9 >= 0).all()
    assert anchors[~standing].unique().numel() == 12


@pytest.mark.parametrize("fraction", [-0.1, 1.0, float("nan"), float("inf")])
def test_invalid_sampling_rejected(fraction):
    values = list(inputs())
    values[-1][2] = fraction
    with pytest.raises(ValueError, match="invalid"):
        mixed_reference_anchors(*values)


def test_per_clip_offsets_are_respected():
    ids, start, first, length, fractions = inputs()
    start[1::2] += 2000
    first[1::2] += 2000
    length[1::2] = 50
    anchors, standing = mixed_reference_anchors(ids, start, first, length, fractions)
    assert torch.all(anchors[1::2] >= 2360)
    assert torch.all(anchors[1::2] < 2410)
    assert standing.sum() == 4


def test_sampling_cannot_write_physical_state_outside_environment_reset():
    with pytest.raises(RuntimeError, match="environment reset"):
        MixedReferenceRootFeedbackCommand._resample_command(
            SimpleNamespace(_inside_environment_reset=False), torch.tensor([0])
        )


def test_suffix_completion_is_not_relabelled_as_full_lifecycle(monkeypatch):
    from gear_sonic.envs.mjlab.sonic_true23_root_feedback import RootFeedbackCurriculumCommand

    command = object.__new__(MixedReferenceRootFeedbackCommand)
    command._inside_environment_reset = True
    command.time_steps = torch.tensor([100, 100, 60])
    command._lifecycle_last_anchor = torch.tensor([100, 100, 100])
    command._episode_began_standing = torch.tensor([True, False, False])
    command.completed_reference_timelines = 0
    command.completed_reference_suffixes = 0

    def base_resample(self, ids):
        self.completed_reference_timelines += int((self.time_steps[ids] >= self._lifecycle_last_anchor[ids]).sum())

    monkeypatch.setattr(RootFeedbackCurriculumCommand, "_resample_command", base_resample)
    command._resample_command(torch.arange(3))
    assert command.completed_reference_timelines == 1
    assert command.completed_reference_suffixes == 1


def test_mixed_reset_transition_does_not_claim_all_standing_or_dynamics_qualification():
    args = SimpleNamespace(start_schedule="mixed_reference_reset_v1", allow_start_schedule_transition=True)
    transition = start_schedule_transition_contract({}, args)
    assert not transition["fresh_standing_environment_reset"]
    assert transition["reference_arrays_and_evaluation_unchanged"]
    contract = start_schedule_contract("mixed_reference_reset_v1")
    assert contract["evaluation_requires_full_standing_start_lifecycle"]
    assert not contract["sampled_suffix_completion_is_full_lifecycle_completion"]
    assert not contract["sampled_reference_state_dynamically_qualified"]


def test_mixed_launcher_requires_complete_source_and_quarter_environment_allocation():
    from gear_sonic.scripts.train_g1_true23_root_feedback import validate_bounds
    from gear_sonic.tests.test_train_g1_true23_root_feedback import arguments

    args = arguments("regression", "--start-schedule", "mixed_reference_reset_v1")
    with pytest.raises(ValueError, match="lifecycle curriculum"):
        validate_bounds(args)
    args.curriculum_stage = "lifecycle"
    validate_bounds(args)
    args.num_envs = 6
    with pytest.raises(ValueError, match="multiple of four"):
        validate_bounds(args)
