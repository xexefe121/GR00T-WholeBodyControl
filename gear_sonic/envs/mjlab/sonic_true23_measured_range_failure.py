"""Opt-in training failure for actual substep joint-range excursions.

This is not the predictive deployment guard and cannot replace it. It observes
each completed simulation step without changing torques, poses, rewards, limits,
or simulator call order. A separate, explicitly versioned training recipe must
install it before claiming the additional termination criterion is trained.
"""

import copy

import numpy as np
import torch

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
)

TERM_NAME = "measured_native_joint_range_failure"


def range_failure_contract():
    return dict(
        kind="native23_measured_substep_range_failure_v1",
        term_name=TERM_NAME,
        physics_substeps_per_control=10,
        actual_position_bounds="unchanged_native_hardware_order_hard_joint_ranges",
        comparison="strict_outside_float64_bounds_no_tolerance_or_inset",
        nonfinite_position_fails=True,
        brief_excursions_latched_until_control_end=True,
        existing_terminations_preserved=True,
        timeout=False,
        joint_limit_penalty_unchanged=True,
        physical_state_or_torque_writes=False,
        predictive_deployment_guard_replacement=False,
        hardware_authorized=False,
        deployment_ready=False,
    )


class MeasuredRangeLatch:
    def __init__(self, num_envs, device):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("requires a positive environment count")
        self.low = torch.tensor(SAFE_TARGET_HARD_LOWER_HARDWARE, device=device, dtype=torch.float64)
        self.high = torch.tensor(SAFE_TARGET_HARD_UPPER_HARDWARE, device=device, dtype=torch.float64)
        self.failure = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.max_excess = torch.zeros(num_envs, device=device, dtype=torch.float64)
        self.calls, self.active = 0, False

    def begin(self):
        if self.active:
            raise ValueError("range observation control nested")
        self.failure.zero_()
        self.max_excess.zero_()
        self.calls, self.active = 0, True

    def observe(self, qpos):
        if not self.active or self.calls >= 10:
            raise ValueError("range observation outside ten-step control")
        if qpos.shape != (len(self.failure), 30) or not qpos.is_floating_point():
            raise ValueError("requires actual native23 [env,30] qpos")
        if qpos.device != self.failure.device:
            raise ValueError("range observation must stay on simulator device")
        finite = torch.isfinite(qpos).all(-1)
        joints = qpos[:, 7:].double()
        excess = torch.maximum(self.low - joints, joints - self.high).clamp_min(0).amax(-1)
        self.failure |= ~finite | (excess > 0)
        # Nonfinite states are failures, not zero-valued physical evidence.
        excess = torch.where(finite, excess, torch.full_like(excess, torch.inf))
        self.max_excess = torch.maximum(self.max_excess, excess)
        self.calls += 1

    def end(self):
        self.active = False
        if self.calls != 10:
            raise ValueError("range observation missed actual simulation substeps")


def measured_range_failure(env):
    monitor = getattr(env, "_native_measured_range_latch", None)
    if monitor is None or not monitor.active or monitor.calls != 10:
        raise ValueError("termination requires ten actual post-step observations")
    return monitor.failure.clone()


def configure_environment(cfg):
    from mjlab.managers.termination_manager import TerminationTermCfg

    if TERM_NAME in cfg.terminations:
        raise ValueError("measured range failure already configured")
    result = copy.deepcopy(cfg)
    result.terminations[TERM_NAME] = TerminationTermCfg(func=measured_range_failure, time_out=False)
    return result


def install_observer(env):
    if hasattr(env, "_native_measured_range_latch"):
        raise ValueError("measured range observer already installed")
    if abs(env.physics_dt - 0.002) > 1e-12 or abs(env.step_dt - 0.02) > 1e-12:
        raise ValueError("range observer requires unchanged500/50Hz simulation")
    model = env.sim.mj_model
    names = tuple(model.joint(i).name for i in range(1, model.njnt))
    if model.nq != 30 or names != tuple(HARDWARE_23_JOINT_NAMES):
        raise ValueError("range observer requires exact native23 joint order")
    np.testing.assert_array_equal(model.jnt_range[1:, 0], SAFE_TARGET_HARD_LOWER_HARDWARE)
    np.testing.assert_array_equal(model.jnt_range[1:, 1], SAFE_TARGET_HARD_UPPER_HARDWARE)
    term = env.termination_manager.get_term_cfg(TERM_NAME)
    if term.func is not measured_range_failure or term.time_out is not False:
        raise ValueError("range observation requires its explicit training termination")
    monitor = MeasuredRangeLatch(env.num_envs, env.device)
    physics_step, control_step = env.sim.step, env.step

    def observe_physics(*args, **kwargs):
        result = physics_step(*args, **kwargs)
        if monitor.active:
            monitor.observe(env.sim.data.qpos)
        return result

    def observe_control(*args, **kwargs):
        monitor.begin()
        try:
            result = control_step(*args, **kwargs)
        except BaseException:
            monitor.active = False
            raise
        monitor.end()
        return result

    env._native_measured_range_latch = monitor
    env.sim.step, env.step = observe_physics, observe_control
    return monitor
