from types import SimpleNamespace

import pytest
import torch

from gear_sonic.envs.mjlab import sonic_true23_world_tracking_termination as task


def test_horizontal_drift_fails_even_when_height_matches():
    desired = torch.zeros(3, 3)
    measured = torch.tensor([[0.0, 0.0, 0.0], [0.31, 0.0, 0.0], [0.0, -0.31, 0.0]])
    error, failure = task.root_error_and_failure(desired, measured)
    assert failure.tolist() == [False, True, True]
    torch.testing.assert_close(error, torch.tensor([0.0, 0.31, 0.31]))


def test_boundary_is_strict_and_translation_invariant():
    desired = torch.zeros(3, 3, dtype=torch.float64)
    measured = torch.tensor([[0.3, 0.0, 0.0], [0.18, 0.24, 0.0], [0.18, 0.24, 0.001]], dtype=torch.float64)
    _, failure = task.root_error_and_failure(desired, measured)
    assert failure.tolist() == [False, False, True]
    origin = torch.tensor([4.0, 8.0, 0.0])
    # Well away from the strict boundary; cancellation cannot change classification.
    _, moved = task.root_error_and_failure(desired + origin, measured * 2 + origin)
    assert moved.tolist() == [True, True, True]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_error_fails(bad):
    _, failure = task.root_error_and_failure(torch.zeros(1, 3), torch.tensor([[bad, 0.0, 0.0]]))
    assert failure.item()


@pytest.mark.parametrize("desired,measured", [(torch.zeros(3), torch.zeros(3)),
                                            (torch.zeros(2, 3), torch.zeros(2, 4))])
def test_wrong_shapes_rejected(desired, measured):
    with pytest.raises(ValueError):
        task.root_error_and_failure(desired, measured)


def test_threshold_cannot_be_silently_relaxed():
    with pytest.raises(ValueError):
        task.root_error_and_failure(torch.zeros(1, 3), torch.zeros(1, 3), threshold_m=0.31)


def test_config_adds_a_failure_without_touching_old_terms_or_rewards():
    pytest.importorskip("mjlab")
    from mjlab.managers.termination_manager import TerminationTermCfg

    old = TerminationTermCfg(func=lambda env: torch.zeros(1, dtype=torch.bool), time_out=True)
    cfg = SimpleNamespace(terminations={"original_timeout": old}, rewards={"original": 7}, decimation=10)
    result = task.configure_environment(cfg)
    assert set(cfg.terminations) == {"original_timeout"}
    assert result.terminations["original_timeout"].func is old.func
    assert result.terminations["original_timeout"].time_out is True
    assert result.rewards == cfg.rewards and result.decimation == 10
    assert result.terminations[task.TERM_NAME].time_out is False
    assert result.terminations[task.TERM_NAME].params == {"threshold_m": 0.3}
    with pytest.raises(ValueError):
        task.configure_environment(result)


def test_capture_records_actual_world_pair_without_writing_state(monkeypatch):
    desired = torch.tensor([[1.0, 2.0, 0.76]])
    measured = torch.tensor([[0.5, 2.0, 0.76]])
    command = SimpleNamespace(robot_anchor_pos_w=measured, time_steps=torch.tensor([18]))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command),
                          common_step_counter=1, _world_root_failure_capture=[])
    monkeypatch.setattr(task, "current_root_reference", lambda cmd: (desired, torch.zeros(1, 3)))
    assert task.world_root_tracking_failure(env).item()
    row = env._world_root_failure_capture[0]
    assert row["common_step_counter"] == 1 and row["reference_q0"].item() == 18
    torch.testing.assert_close(row["measured_position_w"], measured)
    torch.testing.assert_close(row["desired_position_w"], desired)
    assert command.robot_anchor_pos_w is measured
    with pytest.raises(ValueError):
        task.world_root_tracking_failure(env)
