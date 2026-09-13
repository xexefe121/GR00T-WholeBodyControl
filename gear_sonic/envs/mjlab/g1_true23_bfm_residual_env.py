"""BFM-residual simulator action term and frozen-policy observation driver."""

from dataclasses import dataclass, replace

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23 import hardware_to_native_il23, pad_native_il23_to_canonical_il29
from gear_sonic.envs.mjlab.sonic_true23_native_model_actuation import NativeModelActuationAction, NativeModelActuationActionCfg
from gear_sonic.utils.g1_true23_compact_features import pack_observation
from gear_sonic.utils.g1_true23_bfm_residual import FrozenBFMController


def combined_bfm_previous_action(env):
    action = env.action_manager.get_term("joint_pos")
    return pad_native_il23_to_canonical_il29(hardware_to_native_il23(action.combined_action))


def bfm_physical_limit_failure(env):
    action = env.action_manager.get_term("joint_pos")
    q, dq = env.sim.data.qpos[:, 7:], env.sim.data.qvel[:, 6:]
    failed = ((q < action.lower - .01) | (q > action.upper + .01)).any(-1)
    failed |= (dq.abs() > dq.new_tensor(action.cfg.profile.velocity)).any(-1)
    return failed | action.physical_limit_violation


@dataclass(kw_only=True)
class BFMResidualActionCfg(NativeModelActuationActionCfg):
    bfm_default: tuple
    bfm_action_scale: tuple

    def build(self, env):
        return BFMResidualAction(self, env)


class BFMResidualAction(NativeModelActuationAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.env = env
        self.combined_action = torch.zeros_like(self._raw_actions)
        self.residual_delta = torch.zeros_like(self._raw_actions)
        self.physical_limit_violation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lower = torch.as_tensor(env.sim.mj_model.jnt_range[1:, 0].copy(), device=self.device, dtype=torch.float32)
        self.upper = torch.as_tensor(env.sim.mj_model.jnt_range[1:, 1].copy(), device=self.device, dtype=torch.float32)

    def process_actions(self, actions):
        if actions.shape != (self.num_envs, 23) or not torch.isfinite(actions).all():
            raise ValueError("BFM residual requires finite pre-tanh actions [env,23]")
        target, delta, combined = self.env.bfm_controller.targets(actions, self.lower, self.upper)
        self._raw_actions.copy_(actions)
        self._processed_actions.copy_(target)
        self.residual_delta.copy_(delta)
        self.combined_action.copy_(combined)
        self._safe_native_actions.copy_(hardware_to_native_il23(combined))
        self.excess_sum.zero_()
        self.substeps.zero_()
        self.saturation_sum.zero_()

    def apply_actions(self):
        q, dq = self.env.sim.data.qpos[:, 7:], self.env.sim.data.qvel[:, 6:]
        self.physical_limit_violation |= ((q < self.lower - .01) | (q > self.upper + .01)).any(-1)
        self.physical_limit_violation |= (dq.abs() > dq.new_tensor(self.cfg.profile.velocity)).any(-1)
        super().apply_actions()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._processed_actions[ids] = self._processed_actions.new_tensor(self.cfg.bfm_default)
        if hasattr(self, "combined_action"):
            self.combined_action[ids] = 0
            self.residual_delta[ids] = 0
            self.physical_limit_violation[ids] = False


def configure_bfm_action(cfg, contract):
    from mjlab.managers.termination_manager import TerminationTermCfg
    original = cfg.actions["joint_pos"]
    profile = replace(original.profile, kp=tuple(contract["kp"]), kd=tuple(contract["kd"]))
    cfg.actions["joint_pos"] = BFMResidualActionCfg(
        entity_name=original.entity_name, actuator_names=original.actuator_names, profile=profile,
        bfm_default=tuple(contract["default_q"]),
        bfm_action_scale=tuple(.25 * contract["training_effort"] / contract["kp"]),
    )
    previous = cfg.observations["policy"].terms["previous_action"]
    previous.func, previous.params = combined_bfm_previous_action, {}
    # The frozen base was qualified with actual simulator sensor state. Keep
    # that contract for this first residual run; new noise studies are separate.
    cfg.observations["policy"].enable_corruption = False
    cfg.terminations["bfm_physical_limit_failure"] = TerminationTermCfg(func=bfm_physical_limit_failure)
    return cfg


class BFMResidualDriver:
    def __init__(self, env, weights, contract, motion):
        self.env = env
        self.controller = FrozenBFMController(weights, contract, motion, count=env.num_envs, device=env.device)
        env.bfm_controller = self.controller
        self.calls = 0

    @torch.inference_mode()
    def augment(self, obs, dones=None):
        if dones is not None:
            ids = dones.nonzero(as_tuple=False).flatten()
            if len(ids):
                self.controller.reset(ids)
        command = self.env.command_manager.get_term("motion")
        qpos = self.env.sim.data.qpos.clone()
        qpos[:, :3] -= self.env.scene.env_origins
        qvel = self.env.sim.data.qvel
        base, state, history, goal = self.controller.observe(
            qpos, qvel, command.time_steps + 1, command._lifecycle_last_anchor + 1,
        )
        native = pack_observation(obs["policy"], obs["native_goal"])
        target = self.controller.default_q + base * self.controller.action_scale
        obs["bfm_residual"] = torch.cat((native, target), -1)
        self.calls += 1
        return obs
