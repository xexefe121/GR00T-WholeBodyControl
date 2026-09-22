from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_bringup import (
    ABORT_DAMPING_RAMP_S, BRAKE_STEP_RAD, BringupLadder, DT, LiveState,
    Stage, crc32_unitree_words, validate_real_arm_request,
)


LOWER = np.full(23, -2.0)
UPPER = np.full(23, 2.0)
DEFAULT = np.linspace(-0.2, 0.2, 23)


def state(now=100.0, *, q=None, dq=None, quat=None, mode=4, slots=35):
    return LiveState(now, np.zeros(23) if q is None else q, np.zeros(23) if dq is None else dq,
                     np.array([1.0, 0.0, 0.0, 0.0]) if quat is None else quat, mode, slots)


def armed(now=100.0):
    ladder = BringupLadder(LOWER, UPPER, DEFAULT)
    ladder.arm(state(now), now)
    return ladder


def to_policy(ladder, now=100.0):
    for _ in range(5):
        ladder.advance(state(now), now)
        now += 0.001
    assert ladder.stage is Stage.POLICY
    return now


def test_ladder_requires_fresh_explicit_actions_and_emits_promised_commands():
    ladder = armed()
    observe = ladder.command(state(), 100.0, operator_liveness_s=100.0)
    assert np.all(observe.kp == 0) and np.all(observe.kd == 0) and np.all(observe.tau == 0)
    ladder.advance(state(), 100.001)
    zero = ladder.command(state(100.002), 100.002, operator_liveness_s=100.002)
    assert np.all(zero.kp == 0) and np.all(zero.kd == 0) and np.all(zero.tau == 0)
    ladder.advance(state(), 100.003)
    damp = ladder.command(state(100.004), 100.004, operator_liveness_s=100.004)
    assert np.all(damp.kp == 0) and np.all(damp.kd > 0) and np.all(damp.tau == 0)
    ladder.advance(state(q=np.arange(23) * .001), 100.005)
    frames = [ladder.command(state(100.006 + n * .1, q=np.arange(23) * .001), 100.006 + n * .1,
                             operator_liveness_s=100.006 + n * .1) for n in range(31)]
    assert all(np.array_equal(frame.q, np.arange(23) * .001) for frame in frames)
    assert np.all(np.diff([frame.kp[0] for frame in frames]) >= 0)
    assert max(frame.kp[0] for frame in frames) <= ladder.operating_kp[0]


def test_pose_and_policy_are_bounded_and_abort_latches_reentry():
    ladder = armed(); now = 100.001
    ladder.advance(state(), now); now += .001
    ladder.advance(state(), now); now += .001
    ladder.advance(state(), now); now += .001
    ladder.advance(state(), now); now += .001
    previous = ladder.command(state(now), now, operator_liveness_s=now).q
    for index in range(200):
        now += DT
        command = ladder.command(state(now, q=previous), now, operator_liveness_s=now)
        assert np.max(np.abs(command.q - previous)) <= .20 * DT + 1e-12
        previous = command.q
        if np.allclose(previous, DEFAULT): break
    ladder.advance(state(now, q=previous), now); now += DT
    target = previous + .05
    command = ladder.command(state(now, q=previous), now, policy_q=target, policy_received_s=now,
                             operator_liveness_s=now)
    assert np.max(np.abs(command.q - previous)) <= BRAKE_STEP_RAD
    ladder.abort("manual", now)
    assert ladder.command(state(now + ABORT_DAMPING_RAMP_S / 2), now + ABORT_DAMPING_RAMP_S / 2,
                          operator_liveness_s=now).stage is Stage.ABORT_DAMPING
    with pytest.raises(RuntimeError, match="latched"):
        ladder.advance(state(now + 1), now + 1)


@pytest.mark.parametrize("mutator", [
    lambda kw: kw.update(arm_flag=False),
    lambda kw: kw.update(domain_was_explicit=False),
    lambda kw: kw.update(interface_was_explicit=False),
    lambda kw: kw.update(token_file=None),
    lambda kw: kw.update(supplied_token=None),
    lambda kw: kw.update(state=None),
])
def test_each_arming_requirement_refuses_independently(tmp_path, mutator):
    token = tmp_path / "token"; token.write_text("fresh-token\n")
    now = 1_000.0; os.utime(token, (now, now))
    kwargs = dict(arm_flag=True, domain_was_explicit=True, interface_was_explicit=True,
                  dds_domain=0, dds_interface="eth0", token_file=token, supplied_token="fresh-token",
                  state=state(now), lower=LOWER, upper=UPPER, now_s=now)
    mutator(kwargs)
    assert validate_real_arm_request(**kwargs)


@pytest.mark.parametrize("fault", ["stale_state", "stale_target", "deadline", "out_of_limit", "step",
                                    "position_error", "velocity", "tilt", "nonfinite", "liveness"])
def test_every_abort_condition_falls_back_to_damping_then_refuses_resume(fault):
    ladder = armed(); now = to_policy(ladder)
    kwargs = dict(policy_q=np.zeros(23), policy_received_s=now, operator_liveness_s=now)
    observed = state(now)
    if fault == "stale_state": observed = state(now - .021)
    elif fault == "stale_target": kwargs["policy_received_s"] = now - .101
    elif fault == "deadline":
        ladder.command(observed, now, **kwargs); now += DT; kwargs["policy_received_s"] = now; kwargs["operator_liveness_s"] = now; kwargs["deadline_missed"] = True
        ladder.command(state(now), now, **kwargs); now += DT; kwargs["policy_received_s"] = now; kwargs["operator_liveness_s"] = now
    elif fault == "out_of_limit": kwargs["policy_q"] = np.full(23, 3.0)
    elif fault == "step": kwargs["policy_q"] = np.full(23, BRAKE_STEP_RAD + .001)
    elif fault == "position_error": observed = state(now, q=np.full(23, .36))
    elif fault == "velocity": observed = state(now, dq=np.full(23, 6.1))
    elif fault == "tilt": observed = state(now, quat=np.array([.9, .4358899, 0., 0.]))
    elif fault == "nonfinite": observed = state(now, q=np.full(23, np.nan))
    elif fault == "liveness": kwargs["operator_liveness_s"] = now - 1.001
    command = ladder.command(observed, now, **kwargs)
    assert command.stage is Stage.ABORT_DAMPING
    with pytest.raises(RuntimeError, match="latched"):
        ladder.advance(state(now + 1), now + 1)


def test_unitree_crc_core_has_known_nonzero_vector_and_rejects_partial_words():
    assert crc32_unitree_words(bytes(range(16))) == 0x081B46CA
    with pytest.raises(ValueError):
        crc32_unitree_words(b"bad")
