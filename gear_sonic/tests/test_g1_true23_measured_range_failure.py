from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_measured_range_failure import (
    TERM_NAME,
    MeasuredRangeLatch,
    configure_environment,
    install_observer,
    measured_range_failure,
)
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
)


def poses():
    q = torch.zeros((2, 30), dtype=torch.float64)
    q[:, 3] = 1
    q[:, 7:] = (torch.tensor(SAFE_TARGET_HARD_LOWER_HARDWARE) + torch.tensor(SAFE_TARGET_HARD_UPPER_HARDWARE)) / 2
    return q


def test_brief_excursion_survives_return_inside_before_control_end():
    latch = MeasuredRangeLatch(2, "cpu")
    q = poses()
    latch.begin()
    for step in range(10):
        value = q.clone()
        if step == 3:
            value[0, 7] = SAFE_TARGET_HARD_UPPER_HARDWARE[0] + 0.001
        latch.observe(value)
    env = SimpleNamespace(_native_measured_range_latch=latch)
    assert measured_range_failure(env).tolist() == [True, False]
    latch.end()
    latch.begin()
    assert not latch.failure.any()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_base_or_joint_is_failure(nonfinite):
    latch = MeasuredRangeLatch(2, "cpu")
    latch.begin()
    q = poses()
    q[0, 0], q[1, 8] = nonfinite, nonfinite
    latch.observe(q)
    assert latch.failure.all() and torch.isinf(latch.max_excess).all()


def test_exact_bounds_accepted_but_no_tolerance_added():
    latch = MeasuredRangeLatch(2, "cpu")
    latch.begin()
    q = poses()
    q[0, 7:], q[1, 7:] = latch.low, latch.high
    latch.observe(q)
    assert not latch.failure.any()
    q[0, 7] = torch.nextafter(latch.low[0], torch.tensor(-torch.inf))
    latch.observe(q)
    assert latch.failure.tolist() == [True, False]


def test_termination_cannot_read_missing_or_incomplete_capture():
    with pytest.raises(ValueError):
        measured_range_failure(SimpleNamespace())
    latch = MeasuredRangeLatch(2, "cpu")
    latch.begin()
    latch.observe(poses())
    with pytest.raises(ValueError):
        latch.end()


def test_configure_preserves_old_terms_and_original_config():
    old = SimpleNamespace(terminations={"old": object()}, rewards={"joint_limit": -20})
    new = configure_environment(old)
    assert set(new.terminations) == {"old", TERM_NAME}
    assert set(old.terminations) == {"old"} and new.rewards == old.rewards
    with pytest.raises(ValueError):
        configure_environment(new)


def test_observer_preserves_actual_call_order_and_return_objects():
    names = HARDWARE_23_JOINT_NAMES
    model = SimpleNamespace(
        nq=30,
        njnt=24,
        joint=lambda i: SimpleNamespace(name=names[i - 1]),
        jnt_range=np.vstack(
            ([0, 0], np.column_stack((SAFE_TARGET_HARD_LOWER_HARDWARE, SAFE_TARGET_HARD_UPPER_HARDWARE)))
        ),
    )
    events, token = [], object()
    sim = SimpleNamespace(mj_model=model, data=SimpleNamespace(qpos=poses()))
    env = SimpleNamespace(
        sim=sim,
        physics_dt=0.002,
        step_dt=0.02,
        num_envs=2,
        device="cpu",
        termination_manager=SimpleNamespace(
            get_term_cfg=lambda name: SimpleNamespace(func=measured_range_failure, time_out=False)
        ),
    )

    def physics():
        events.append("physics")
        return token

    def control(action):
        for _ in range(10):
            assert sim.step() is token
        events.append("termination")
        assert not measured_range_failure(env).any()
        return action

    sim.step, env.step = physics, control
    install_observer(env)
    assert env.step(token) is token
    assert events == ["physics"] * 10 + ["termination"]
    with pytest.raises(ValueError):
        install_observer(env)
