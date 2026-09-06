import copy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from gear_sonic.utils.g1_true23_reset_curriculum import (
    reset_curriculum_contract,
    reset_scale,
    resample_with_curriculum,
    scaled_reset_ranges,
    summarize_reset_events,
)


@pytest.mark.parametrize("control,expected", [(0, 0), (8, 0), (9, 0.0625), (16, 0.5), (24, 1), (99, 1)])
def test_schedule_uses_actual_control_counter(control, expected):
    assert reset_scale(control, reset_curriculum_contract(8, 24)) == expected


@pytest.mark.parametrize("warm,end", [(True, 24), (-1, 24), (24, 24), (25, 24), (0, 100000001), (0, 2.0)])
def test_invalid_schedule_rejected(warm, end):
    with pytest.raises(ValueError):
        reset_curriculum_contract(warm, end)


def test_changed_contract_and_invalid_counter_rejected():
    contract = reset_curriculum_contract(8, 24)
    with pytest.raises(ValueError):
        reset_scale(-1, contract)
    contract["preserve_motor_limits_and_gains"] = False
    with pytest.raises(ValueError):
        reset_scale(10, contract)


def command(control):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="2 2 .1"/><body pos="0 0 .05">'
        '<freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    cfg = SimpleNamespace(
        pose_range={"z": (-0.01, 0.01)}, velocity_range={"x": (-0.5, 0.5)}, joint_position_range=(-0.1, 0.1)
    )
    data = SimpleNamespace(
        qpos=torch.tensor([[0, 0, 0.05, 1, 0, 0, 0]], dtype=torch.float64),
        qvel=torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=torch.float64),
    )
    env = SimpleNamespace(
        common_step_counter=control, device="cpu", sim=SimpleNamespace(mj_model=model, data=data)
    )

    def write(pose, env_ids):
        data.qpos[env_ids] = pose

    robot = SimpleNamespace(write_root_link_pose_to_sim=write, clear_state=lambda env_ids: None)
    return SimpleNamespace(cfg=cfg, _env=env, robot=robot, time_steps=torch.tensor([50]))


def test_reset_scales_only_requested_ranges_then_restores_them():
    cmd = command(16)
    initial_cfg = copy.deepcopy(cmd.cfg)
    before = cmd._env.sim.data.qpos.clone()
    velocity = cmd._env.sim.data.qvel.clone()
    calls = []

    def original(actual, ids):
        calls.append(copy.deepcopy(vars(actual.cfg)))

    contract = reset_curriculum_contract(8, 24)
    resample_with_curriculum(cmd, torch.tensor([0]), original, contract)
    assert calls == [scaled_reset_ranges(initial_cfg, 0.5)]
    assert cmd.cfg == initial_cfg
    actual = cmd._env.sim.data.qpos
    assert actual[0, 2] > 0.1
    np.testing.assert_array_equal(actual[:, [0, 1, 3, 4, 5, 6]], before[:, [0, 1, 3, 4, 5, 6]])
    assert torch.equal(cmd._env.sim.data.qvel, velocity)
    summary = summarize_reset_events(cmd._reset_curriculum_events, contract)
    assert summary["reset_rows"] == summary["lifted_rows"] == 1
    assert summary["maximum_scale"] == 0.5
    assert summary["remaining_detected_floor_penetrations"] == 0
    assert summary["deployment_ready"] is False


def test_original_failure_restores_reset_configuration():
    cmd = command(16)
    initial_cfg = copy.deepcopy(cmd.cfg)

    def fail(actual, ids):
        raise RuntimeError("original resample failed")

    with pytest.raises(RuntimeError, match="original resample"):
        resample_with_curriculum(cmd, torch.tensor([0]), fail, reset_curriculum_contract(8, 24))
    assert cmd.cfg == initial_cfg


def test_empty_reset_does_not_create_fake_events():
    cmd = command(0)
    resample_with_curriculum(
        cmd, torch.tensor([], dtype=torch.long), lambda actual, ids: None, reset_curriculum_contract(8, 24)
    )
    assert not hasattr(cmd, "_reset_curriculum_events")


@pytest.mark.parametrize("change", ["scale", "penetration", "velocity_check", "lift", "rows"])
def test_runtime_summary_rejects_invalid_evidence(change):
    cmd, contract = command(24), reset_curriculum_contract(8, 24)
    resample_with_curriculum(cmd, torch.tensor([0]), lambda actual, ids: None, contract)
    events = copy.deepcopy(cmd._reset_curriculum_events)
    if change == "scale":
        events[0]["scale"] = 0
    elif change == "penetration":
        events[0]["minimum_floor_distances_after_m"] = [-0.01]
    elif change == "velocity_check":
        events[0]["expected_positions_and_unchanged_velocities_verified"] = False
    elif change == "lift":
        events[0]["root_lifts_m"] = [1.0]
    else:
        events[0]["source_phases"] = []
    with pytest.raises(ValueError):
        summarize_reset_events(events, contract)


def test_launcher_rejects_resume_before_loading_any_checkpoint():
    from gear_sonic.scripts.train_g1_true23_reset_curriculum import main

    with pytest.raises(ValueError, match="fresh-only"):
        main(["train", "--resume", "unused.pt"])
