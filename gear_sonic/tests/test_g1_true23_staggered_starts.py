"""Staggered training remains standing-start, complete, and simulation-only."""

from types import SimpleNamespace

import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_staggered_starts import (
    advance_staggered_command,
    assign_standing_start_delays,
    configure_staggered_start_timeout,
    standing_start_delays,
)
from gear_sonic.tests.test_g1_true23_generalist_curriculum import fake_command
from gear_sonic.utils.g1_true23_start_schedule import (
    start_schedule_contract,
    start_schedule_transition_contract,
)


def command_with_delays(delays):
    command = fake_command()
    command.time_steps[:] = 10
    command._lifecycle_last_anchor[:] = 30
    command._causal_resampled.zero_()
    command._start_hold_remaining = torch.tensor(delays)
    command._start_hold_executed = torch.zeros(2, dtype=torch.long)
    return command


def test_delays_span_lifecycle_without_skipping_frames():
    delays = standing_start_delays(torch.arange(16), torch.full((16,), 1841), 16)
    assert delays.tolist() == [(i * 1841) // 16 for i in range(16)]
    assert delays.unique().numel() == 16
    assert delays.max() < 1841


def test_subsequent_resets_do_not_repeat_initial_wait():
    command = command_with_delays([0, 0])
    command.num_envs = 2
    command._lifecycle_ends = torch.tensor([30])
    command._lifecycle_choice = torch.zeros(2, dtype=torch.long)
    command._start_hold_assigned = torch.zeros(2, dtype=torch.long)
    command._start_schedule_initialized = torch.zeros(2, dtype=torch.bool)
    assign_standing_start_delays(command, torch.tensor([0, 1]))
    assert command._start_hold_remaining.tolist() == [0, 10]
    assign_standing_start_delays(command, torch.tensor([1]))
    assert command._start_hold_remaining.tolist() == [0, 0]
    assert command._start_hold_assigned.tolist() == [0, 10]


@pytest.mark.parametrize("ids,count", [(torch.tensor([-1]), 2), (torch.tensor([2]), 2), (torch.tensor([0]), 0)])
def test_invalid_environment_assignment_rejected(ids, count):
    with pytest.raises(ValueError, match="environment indices"):
        standing_start_delays(ids, torch.ones_like(ids), count)


def test_hold_preserves_reference_but_updates_measured_anchor_history():
    command = command_with_delays([0, 3])
    before_position = command.robot_anchor_pos_w.clone()
    before_quaternion = command.robot_anchor_quat_w.clone()
    advance_staggered_command(command)
    assert command.time_steps.tolist() == [11, 10]
    assert command._start_hold_remaining.tolist() == [0, 2]
    torch.testing.assert_close(command._causal_robot_anchor_pos_w, torch.ones(2, 3))
    torch.testing.assert_close(command._causal_last_current_anchor_pos_w, before_position)
    torch.testing.assert_close(command.robot_anchor_pos_w, before_position)
    torch.testing.assert_close(command.robot_anchor_quat_w, before_quaternion)


def test_reset_prime_does_not_consume_physical_hold_interval():
    command = command_with_delays([2, 3])
    command._causal_resampled[:] = True
    advance_staggered_command(command)
    assert command.time_steps.tolist() == [10, 10]
    assert command._start_hold_remaining.tolist() == [2, 3]
    assert command._start_hold_executed.tolist() == [0, 0]


def test_every_control_occurs_after_exact_delay_without_mid_episode_resampling():
    command = command_with_delays([0, 5])
    anchors = [[], []]
    for _ in range(26):
        for i in range(2):
            anchors[i].append(int(command.time_steps[i]))
        advance_staggered_command(command)
    assert anchors[0][:21] == list(range(10, 31))
    assert anchors[1] == [10] * 5 + list(range(10, 31))
    assert command._start_hold_executed.tolist() == [0, 5]


def test_episode_timeout_does_not_cut_off_delayed_full_lifecycle():
    cfg = SimpleNamespace(commands={"motion": SimpleNamespace()})
    configure_staggered_start_timeout(cfg, [{"timeline": {"total_requested_controls": 1841}}])
    assert cfg.episode_length_s == (3682 + 2) * 0.02
    assert cfg.commands["motion"].resampling_time_range == (cfg.episode_length_s + 1,) * 2


def test_schedule_transition_is_explicit_not_same_configuration_resume():
    args = SimpleNamespace(start_schedule="staggered_standing_start_v1")
    with pytest.raises(ValueError, match="explicit"):
        start_schedule_transition_contract({}, args)
    args.allow_start_schedule_transition = True
    contract = start_schedule_transition_contract({}, args)
    assert contract["rollout_schedule_changed"]
    assert not contract["same_schedule_resume_claimed"]
    assert contract["actor_critic_optimizer_and_counters_preserved"]
    assert contract["reference_arrays_and_evaluation_unchanged"]
    assert not contract["new"]["deployment_ready"]
    same = start_schedule_transition_contract({"start_schedule": contract["new"]}, args)
    assert not same["rollout_schedule_changed"]


def test_schedule_contract_tampering_rejected():
    contract = start_schedule_contract()
    contract["source_frames_skipped"] = 1
    with pytest.raises(ValueError, match="invalid"):
        start_schedule_transition_contract({"start_schedule": contract}, SimpleNamespace())
