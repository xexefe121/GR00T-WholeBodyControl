from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_measured_range_failure import TERM_NAME
from gear_sonic.envs.mjlab.sonic_true23_passive_range_capture import PassiveRangeCapture
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE as LOW,
    SAFE_TARGET_HARD_UPPER_HARDWARE as HIGH,
)


def fixture_env():
    events, token = [], object()
    q = torch.zeros((2, 30), dtype=torch.float64)
    q[:, 3] = 1
    q[:, 7:] = (torch.tensor(LOW) + torch.tensor(HIGH)) / 2
    q0 = q.clone()
    model = SimpleNamespace(
        nq=30,
        nv=29,
        njnt=24,
        joint=lambda i: SimpleNamespace(name=HARDWARE_23_JOINT_NAMES[i - 1]),
        jnt_range=np.vstack(([0, 0], np.column_stack((LOW, HIGH)))),
    )
    manager = SimpleNamespace(
        active_terms=["existing_failure", "time_out"],
        get_term_cfg=lambda name: SimpleNamespace(time_out=name == "time_out"),
        terminated=torch.tensor([False, True]),
        time_outs=torch.zeros(2, dtype=torch.bool),
    )
    manager.get_term = lambda n: manager.time_outs if n == "time_out" else manager.terminated

    def compute():
        events.append("termination")
        return manager.terminated | manager.time_outs

    def physics():
        index = events.count("physics") % 10
        events.append("physics")
        q[:] = q0
        if index == 3:
            q[0, 7] = HIGH[0] + 0.01
        if index == 9:
            q[1, 7] += 0.2
        return token

    sim = SimpleNamespace(mj_model=model, data=SimpleNamespace(qpos=q, qvel=torch.zeros((2, 29))), step=physics)
    manager.compute = compute
    env = SimpleNamespace(
        sim=sim, termination_manager=manager, physics_dt=0.002, step_dt=0.02, num_envs=2, device="cpu"
    )

    def step(action):
        for _ in range(10):
            assert sim.step() is token
        done = manager.compute()
        events.append("reset")
        q[done] = q0[done]
        return action

    env.step = step
    return env, events, token


def test_observes_real_steps_and_preserves_result_order_original_step_and_failures():
    env, events, token = fixture_env()
    old_step = env.step
    observer = PassiveRangeCapture(env)
    assert env.step is old_step
    assert observer.run(env.step, token) is token
    assert events == ["physics"] * 10 + ["termination", "reset"]
    row = observer.rows[0]
    assert row["range_failure"].tolist() == [True, False]
    assert row["terminated"].tolist() == [False, True]
    assert row["term_flags"].tolist() == [[False, False], [True, False]]
    assert torch.equal(row["terminal_qpos"], row["physics_qpos"][-1])
    assert not torch.equal(row["terminal_qpos"][1], row["post_reset_qpos"][1])
    assert observer.capture()["physics_qpos"].shape == (1, 10, 2, 30)
    assert env.termination_manager.active_terms == ["existing_failure", "time_out"]
    with pytest.raises(ValueError, match="already installed"):
        PassiveRangeCapture(env)


def test_requires_original_inventory_not_new_range_termination():
    env, _, _ = fixture_env()
    env.termination_manager.active_terms.append(TERM_NAME)
    with pytest.raises(ValueError, match="unchanged training failures"):
        PassiveRangeCapture(env)


def test_missing_substeps_rejected_and_observer_deactivated():
    env, _, token = fixture_env()
    observer = PassiveRangeCapture(env)
    with pytest.raises(ValueError, match="missed actual"):
        observer.run(lambda _: token, token)
    assert observer.pending is None and not observer.latch.active and not observer.rows


def test_off_control_physics_not_mislabeled_as_rollout():
    env, _, token = fixture_env()
    observer = PassiveRangeCapture(env)
    assert env.sim.step() is token
    assert observer.latch.calls == 0 and observer.rows == []


def test_failure_inventory_cannot_change_mid_step():
    env, _, token = fixture_env()
    observer = PassiveRangeCapture(env)
    env.termination_manager.active_terms = ["existing_failure"]
    with pytest.raises(ValueError, match="inventory changed"):
        observer.run(env.step, token)


def test_no_observer_control_and_passive_control_match_bit_exactly():
    plain, plain_events, _ = fixture_env()
    observed, observed_events, _ = fixture_env()
    observer = PassiveRangeCapture(observed)
    for index in range(2):
        assert plain.step(index) == observer.run(observed.step, index)
        assert torch.equal(plain.sim.data.qpos, observed.sim.data.qpos)
        assert torch.equal(plain.sim.data.qvel, observed.sim.data.qvel)
        assert plain_events == observed_events


def test_exact_mjlab_robot_namespace_is_accepted_but_arbitrary_prefix_is_not():
    env, _, token = fixture_env()
    env.sim.mj_model.joint = lambda i: SimpleNamespace(name="robot/" + HARDWARE_23_JOINT_NAMES[i - 1])
    observer = PassiveRangeCapture(env)
    assert observer.run(env.step, token) is token
    invalid, _, _ = fixture_env()
    invalid.sim.mj_model.joint = lambda i: SimpleNamespace(name="other/" + HARDWARE_23_JOINT_NAMES[i - 1])
    with pytest.raises(ValueError, match="physical joint order"):
        PassiveRangeCapture(invalid)
