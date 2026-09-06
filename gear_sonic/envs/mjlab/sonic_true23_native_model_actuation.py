"""Nominal native23 motor simulation, separate from every gantry profile."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch

from gear_sonic.envs.mjlab.sonic_true23 import _MJLAB_IMPORT_ERROR
from gear_sonic.envs.mjlab.sonic_true23_causal_history_safe_target_v11 import (
    SafeTargetNativeIl23JointPositionAction,
    SafeTargetNativeIl23JointPositionActionCfg,
)
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_native_model_actuation import (
    NativeModelActuationProfile,
    native_model_pd_torch,
)


def invalid_native_model_actuation(env, action_name="joint_pos"):
    return env.action_manager.get_term(action_name).invalid_actuation


def requested_effort_excess(env, action_name="joint_pos"):
    action = env.action_manager.get_term(action_name)
    return action.excess_sum / action.substeps.clamp_min(1)


if _MJLAB_IMPORT_ERROR is None:

    @dataclass(kw_only=True)
    class NativeModelActuationActionCfg(SafeTargetNativeIl23JointPositionActionCfg):
        profile: NativeModelActuationProfile

        def build(self, env):
            return NativeModelActuationAction(self, env)

    class NativeModelActuationAction(SafeTargetNativeIl23JointPositionAction):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            if abs(env.physics_dt - 0.002) > 1e-12 or abs(env.step_dt - 0.02) > 1e-12:
                raise ValueError("native-model action requires 500/50 Hz")
            self.invalid_actuation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.excess_sum = torch.zeros(self.num_envs, device=self.device)
            self.substeps = torch.zeros_like(self.excess_sum)
            self.saturation_sum = torch.zeros_like(self.excess_sum)
            self.requested_torque = torch.zeros_like(self._processed_actions)
            self.applied_torque = torch.zeros_like(self._processed_actions)

        def process_actions(self, actions):
            super().process_actions(actions)
            self.invalid_actuation |= ~torch.isfinite(actions).all(dim=-1) | (actions.abs() >= 10).any(dim=-1)
            self.excess_sum.zero_()
            self.substeps.zero_()
            self.saturation_sum.zero_()

        def apply_actions(self):
            q = self._entity.data.joint_pos[:, self._target_ids]
            dq = self._entity.data.joint_vel[:, self._target_ids]
            # This nominal profile has no encoder bias event. Fail closed if
            # another environment silently introduces it.
            if torch.any(self._entity.data.encoder_bias != 0):
                raise ValueError("nominal native-model requires unbiased simulation encoders")
            requested, applied, invalid, cost = native_model_pd_torch(
                self._processed_actions,
                q,
                dq,
                self.cfg.profile,
            )
            self.invalid_actuation |= invalid
            applied = torch.where(self.invalid_actuation[:, None], 0.0, applied)
            self.requested_torque[:] = requested
            self.applied_torque[:] = applied
            self.excess_sum += cost
            self.substeps += 1
            self.saturation_sum += (requested.abs() > q.new_tensor(self.cfg.profile.effort)).float().mean(dim=-1)
            self._entity.set_joint_effort_target(applied, joint_ids=self._target_ids)

        def reset(self, env_ids=None):
            super().reset(env_ids)
            if env_ids is None:
                env_ids = slice(None)
            for name in (
                "invalid_actuation",
                "excess_sum",
                "substeps",
                "saturation_sum",
                "requested_torque",
                "applied_torque",
            ):
                getattr(self, name)[env_ids] = 0


def apply_native_model_actuation_profile(cfg, profile: NativeModelActuationProfile, *, effort_penalty_weight=0.1):
    from mjlab.actuator import BuiltinMotorActuatorCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg

    if (
        isinstance(effort_penalty_weight, bool)
        or not math.isfinite(effort_penalty_weight)
        or not 0 <= effort_penalty_weight <= 100
    ):
        raise ValueError("effort penalty weight must be finite in 0..100")
    cfg = copy.deepcopy(cfg)
    if "encoder_bias" in cfg.events:
        raise ValueError("remove encoder-bias randomization explicitly before nominal profile")
    robot = cfg.scene.entities["robot"]
    original_spec_fn = robot.spec_fn

    def native_spec():
        spec = original_spec_fn()
        for i, name in enumerate(HARDWARE_23_JOINT_NAMES):
            joint = spec.joint(name)
            joint.actfrclimited = True
            joint.actfrcrange[:] = (-profile.effort[i], profile.effort[i])
            joint.damping = profile.damping[i]
        return spec

    robot.spec_fn = native_spec
    robot.articulation.actuators = tuple(
        BuiltinMotorActuatorCfg(
            target_names_expr=(name,),
            effort_limit=profile.effort[i],
            armature=profile.armature[i],
            frictionloss=profile.frictionloss[i],
        )
        for i, name in enumerate(HARDWARE_23_JOINT_NAMES)
    )
    cfg.sim.mujoco.timestep = profile.timestep_s
    cfg.sim.mujoco.integrator = "euler"
    cfg.decimation = profile.decimation
    cfg.actions["joint_pos"] = NativeModelActuationActionCfg(
        entity_name="robot", actuator_names=(".*",), profile=profile
    )
    cfg.terminations["invalid_native_model_actuation"] = TerminationTermCfg(func=invalid_native_model_actuation)
    cfg.rewards["requested_effort_excess"] = RewardTermCfg(
        func=requested_effort_excess, weight=-float(effort_penalty_weight)
    )
    return cfg
