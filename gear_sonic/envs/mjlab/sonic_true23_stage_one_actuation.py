"""Optional explicit-PD stage-one mechanics for SONIC LoRA training only.

The profile closes the gains/scaling/slew gap; it does not emulate DDS timing,
ownership, e-stop, measured-motor faults, or return-to-Unitree recovery. Guard
crossings terminate at the next 50 Hz training boundary. No robot API exists.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import re
from typing import Any

import torch

from gear_sonic.envs.mjlab.sonic_true23 import _MJLAB_IMPORT_ERROR, _require_last_dim
from gear_sonic.envs.mjlab.sonic_true23_causal_history_safe_target_v11 import (
    SafeTargetNativeIl23JointPositionAction,
    SafeTargetNativeIl23JointPositionActionCfg,
)
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_ACTION_SCALE, HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    SAFE_TARGET_HARD_LOWER_HARDWARE,
    SAFE_TARGET_HARD_UPPER_HARDWARE,
    safe_target_transform_torch,
)
from gear_sonic.utils.g1_true23_actuation_profile import NativeSupportActuationProfile, StageOneActuationProfile
from gear_sonic.utils.g1_true23_projected_controller_state import (
    applied_target_native_torch,
    synthetic_reset_target_torch,
)


def requested_projection_cost(requested: torch.Tensor, projected: torch.Tensor) -> torch.Tensor:
    """Reward-only normalized request residual, not an action/effort override.

    The per-joint squared cost is bounded at 100. Nonfinite requests receive
    that maximum cost while the existing action guard still terminates them.
    """
    if requested.shape != projected.shape or requested.ndim != 2 or requested.shape[1] != 23:
        raise ValueError("projection cost requires equal [env,23] target arrays")
    delta = (requested - projected) / requested.new_tensor(HARDWARE_23_ACTION_SCALE)
    delta = torch.nan_to_num(delta, nan=10.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
    return delta.square().mean(dim=1)


def projection_cost_contract(weight: float) -> dict:
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or not 0 <= weight <= 100
    ):
        raise ValueError("projection penalty weight must be finite within 0..100")
    return {
        "kind": "g1_true23_requested_projection_cost_v1",
        "enabled": weight > 0,
        "reward_weight": -float(weight),
        "cost": "mean_over_23_joints_and_10_physics_substeps_of_normalized_request_minus_projection_squared",
        "normalization_hardware_rad": list(HARDWARE_23_ACTION_SCALE),
        "maximum_per_joint_squared_cost": 100.0,
        "nonfinite_component_squared_cost": 100.0,
        "invalid_substeps_included": True,
        "original_tracking_rewards_retained": True,
        "controller_and_limits_changed": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def requested_projection_reward_term(env: Any, action_name: str = "joint_pos") -> torch.Tensor:
    action = env.action_manager.get_term(action_name)
    if not action.cfg.record_requested_projection:
        raise ValueError("requested projection reward requires substep recording")
    return action._projection_cost_sum / action._projection_cost_count.clamp_min(1)


def requested_stage_one_target(full_target: torch.Tensor, profile: StageOneActuationProfile) -> torch.Tensor:
    default = full_target.new_tensor(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    scale = full_target.new_tensor(profile.joint_scale)
    return default + (full_target - default) * profile.fraction * scale


def stage_one_pd_step(
    requested: torch.Tensor,
    previous: torch.Tensor,
    q: torch.Tensor,
    dq: torch.Tensor,
    profile: StageOneActuationProfile,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One explicit 2 ms PD substep; guard flags do not silently clip targets."""
    delta = profile.slew_rad_s * profile.timestep_s
    target = torch.maximum(torch.minimum(requested, previous + delta), previous - delta)
    effort = q.new_tensor(profile.effort)
    predicted = q.new_tensor(profile.kp) * (target - q) - q.new_tensor(profile.kd) * dq
    lower = q.new_tensor(SAFE_TARGET_HARD_LOWER_HARDWARE) + profile.target_margin_rad
    upper = q.new_tensor(SAFE_TARGET_HARD_UPPER_HARDWARE) - profile.target_margin_rad
    invalid = (
        (~torch.isfinite(requested)).any(dim=-1)
        | (~torch.isfinite(predicted)).any(dim=-1)
        | ((requested < lower) | (requested > upper)).any(dim=-1)
        | (predicted.abs() > 0.25 * effort).any(dim=-1)
    )
    torque = torch.maximum(torch.minimum(predicted, effort), -effort)
    return target, torque, invalid


def actuation_guard_violated(env: Any, action_name: str = "joint_pos") -> torch.Tensor:
    return env.action_manager.get_term(action_name).envelope_violation


def native_support_pd_step(requested, previous, q, dq, profile: NativeSupportActuationProfile):
    """Vectorized equivalent of the simulator's effort-feasible target projection.

    Infeasible rows signal episode termination and produce zero effort. This
    training-only response is NOT a hardware recovery behavior.
    """
    kp, kd, effort = (q.new_tensor(getattr(profile, name)) for name in ("kp", "kd", "effort"))
    cap = 0.95 * 0.25 * effort
    lower = torch.maximum(q.new_tensor(SAFE_TARGET_HARD_LOWER_HARDWARE) + 0.05, q + (kd * dq - cap) / kp)
    upper = torch.minimum(q.new_tensor(SAFE_TARGET_HARD_UPPER_HARDWARE) - 0.05, q + (kd * dq + cap) / kp)
    delta = profile.slew_rad_s * profile.timestep_s
    lower = torch.maximum(lower, previous - delta)
    upper = torch.minimum(upper, previous + delta)
    invalid = (lower > upper).any(dim=-1)
    for value in (requested, previous, q, dq):
        invalid |= (~torch.isfinite(value)).any(dim=-1)
    target = torch.maximum(torch.minimum(requested, upper), lower)
    target = torch.where(invalid[:, None], torch.nan_to_num(previous), target)
    torque = kp * (target - q) - kd * dq
    torque = torch.where(invalid[:, None], torch.zeros_like(torque), torque)
    return target, torque, invalid


if _MJLAB_IMPORT_ERROR is None:

    @dataclass(kw_only=True)
    class StageOneActuationActionCfg(SafeTargetNativeIl23JointPositionActionCfg):
        profile: StageOneActuationProfile
        record_requested_projection: bool = False

        def build(self, env: Any) -> StageOneActuationAction:
            return StageOneActuationAction(self, env)

    class StageOneActuationAction(SafeTargetNativeIl23JointPositionAction):
        cfg: StageOneActuationActionCfg

        def __init__(self, cfg: StageOneActuationActionCfg, env: Any) -> None:
            super().__init__(cfg, env)
            if abs(env.physics_dt - cfg.profile.timestep_s) > 1e-12 or abs(env.step_dt - 0.02) > 1e-12:
                raise ValueError("stage-one action requires 500 Hz PD and 50 Hz policy")
            self._requested_targets = self._processed_actions.clone()
            self._previous_targets = self._processed_actions.clone()
            self._needs_seed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self.envelope_violation = torch.zeros_like(self._needs_seed)
            self._projection_cost_sum = torch.zeros(self.num_envs, device=self.device)
            self._projection_cost_count = torch.zeros_like(self._projection_cost_sum)

        @property
        def _stateful_profile(self) -> bool:
            return (
                isinstance(self.cfg.profile, NativeSupportActuationProfile)
                and self.cfg.profile.consistent_controller_state
            )

        def _seed_controller_state(self) -> None:
            # Called by the first previous-action observation after all reset
            # managers, not by ActionManager.reset (which precedes motion reset).
            if not self._stateful_profile or not torch.any(self._needs_seed):
                return
            q = self._entity.data.joint_pos[:, self._target_ids]
            dq = self._entity.data.joint_vel[:, self._target_ids]
            target, invalid = synthetic_reset_target_torch(q, dq, self.cfg.profile)
            mask = self._needs_seed[:, None]
            self._previous_targets[:] = torch.where(mask, target, self._previous_targets)
            self._processed_actions[:] = torch.where(mask, target, self._processed_actions)
            self._safe_native_actions[:] = torch.where(
                mask, applied_target_native_torch(target), self._safe_native_actions
            )
            self.envelope_violation |= self._needs_seed & invalid
            self._needs_seed.zero_()

        @property
        def safe_native_action(self) -> torch.Tensor:
            self._seed_controller_state()
            return self._safe_native_actions

        def process_actions(self, actions: torch.Tensor) -> None:
            _require_last_dim(actions, 23, "stage-one raw native23 action")
            self._raw_actions[:] = actions
            safe, full_target = safe_target_transform_torch(actions)
            if not self._stateful_profile:
                self._safe_native_actions[:] = safe
            self._requested_targets[:] = requested_stage_one_target(full_target, self.cfg.profile)
            # The outer RL wrapper clips at 10; equality therefore also means
            # an unqualified clipped action. Runtime rejects abs(raw) >= 10.
            self.envelope_violation |= (actions.abs() >= 10.0).any(dim=-1)
            if self.cfg.record_requested_projection:
                self._projection_cost_sum.zero_()
                self._projection_cost_count.zero_()

        def apply_actions(self) -> None:
            self._seed_controller_state()
            q = self._entity.data.joint_pos[:, self._target_ids]
            dq = self._entity.data.joint_vel[:, self._target_ids]
            bias = self._entity.data.encoder_bias[:, self._target_ids]
            # Command resets run after ActionManager.reset. Seed lazily from
            # the actual final reset pose, never an earlier episode's target.
            self._previous_targets[:] = torch.where(self._needs_seed[:, None], q, self._previous_targets)
            self._needs_seed.zero_()
            step = (
                native_support_pd_step
                if isinstance(self.cfg.profile, NativeSupportActuationProfile)
                else stage_one_pd_step
            )
            target, torque, invalid = step(
                self._requested_targets,
                self._previous_targets,
                q,
                dq,
                self.cfg.profile,
            )
            if self.cfg.record_requested_projection:
                self._projection_cost_sum += requested_projection_cost(self._requested_targets, target)
                self._projection_cost_count += 1
            if torch.any(bias != 0):
                raise ValueError("stage-one profile requires disabled encoder-bias randomization")
            self._processed_actions[:] = target
            self._previous_targets[:] = target
            if self._stateful_profile:
                self._safe_native_actions[:] = applied_target_native_torch(target)
            self.envelope_violation |= invalid
            if isinstance(self.cfg.profile, NativeSupportActuationProfile):
                torque = torch.where(self.envelope_violation[:, None], torch.zeros_like(torque), torque)
            self._entity.set_joint_effort_target(torque, joint_ids=self._target_ids)

        def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
            super().reset(env_ids)
            ids = slice(None) if env_ids is None else env_ids
            self._requested_targets[ids] = self._processed_actions[ids]
            self._needs_seed[ids] = True
            self.envelope_violation[ids] = False
            self._projection_cost_sum[ids] = 0
            self._projection_cost_count[ids] = 0


def apply_stage_one_actuation_profile(
    cfg: Any, profile: StageOneActuationProfile, *, projection_penalty_weight: float = 0.0
) -> Any:
    """Replace only local config mechanics; original asset/config is untouched."""
    if _MJLAB_IMPORT_ERROR is not None:
        raise RuntimeError("MJLab runtime required") from _MJLAB_IMPORT_ERROR
    from mjlab.actuator import BuiltinMotorActuatorCfg, BuiltinPositionActuatorCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg

    cost = projection_cost_contract(projection_penalty_weight)
    if cost["enabled"] and not (
        isinstance(profile, NativeSupportActuationProfile) and profile.consistent_controller_state
    ):
        raise ValueError("projection penalty requires native-support stateful V2")

    cfg = copy.deepcopy(cfg)
    if "encoder_bias" in cfg.events:
        raise ValueError("stage-one exact mechanics cannot include encoder-bias randomization")
    robot = cfg.scene.entities["robot"]
    existing = robot.articulation.actuators
    motors = []
    for index, joint in enumerate(HARDWARE_23_JOINT_NAMES):
        matches = [
            actuator
            for actuator in existing
            if any(re.fullmatch(pattern, joint) for pattern in actuator.target_names_expr)
        ]
        if len(matches) != 1 or not isinstance(matches[0], BuiltinPositionActuatorCfg):
            raise ValueError(f"ambiguous original actuator binding for {joint}")
        original = matches[0]
        motors.append(
            BuiltinMotorActuatorCfg(
                target_names_expr=(joint,),
                effort_limit=profile.effort[index],
                armature=original.armature,
                frictionloss=original.frictionloss,
            )
        )
    original_spec_fn = robot.spec_fn

    def profiled_spec():
        spec = original_spec_fn()
        for joint_name, effort in zip(HARDWARE_23_JOINT_NAMES, profile.effort, strict=True):
            joint = spec.joint(joint_name)
            joint.actfrclimited = True
            joint.actfrcrange[:] = (-effort, effort)
        return spec

    robot.spec_fn = profiled_spec
    robot.articulation.actuators = tuple(motors)
    cfg.sim.mujoco.timestep = profile.timestep_s
    cfg.sim.mujoco.integrator = "euler"
    cfg.decimation = 10
    cfg.actions["joint_pos"] = StageOneActuationActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        profile=profile,
        record_requested_projection=cost["enabled"],
    )
    cfg.terminations["stage_one_actuation_guard"] = TerminationTermCfg(func=actuation_guard_violated)
    if cost["enabled"]:
        cfg.rewards["requested_projection_l2"] = RewardTermCfg(
            func=requested_projection_reward_term, weight=cost["reward_weight"]
        )
    return cfg
