"""Opt-in corrected reference/action semantics; native23 physics remains intact."""

from dataclasses import dataclass

import torch

from gear_sonic.envs.mjlab.sonic_true23_native_model_actuation import (
    NativeModelActuationAction,
    NativeModelActuationActionCfg,
)
from gear_sonic.utils.g1_true23_release_action_diagnostic import bounded_linear_precompensation_torch
from gear_sonic.utils.g1_true23_source_action_codec import (
    SOURCE_ACTION_CONVENTION,
    source_normalize_native_action_torch,
    source_scaled_precompensation_torch,
)
from gear_sonic.utils.g1_true23_virtual_source_reference import virtual_source_vr_terms


def _reference_terms(env, source_model_path, command_name):
    command = env.command_manager.get_term(command_name)
    cache = getattr(command, "_release_compatible_vr", None)
    if cache is None:
        terms = virtual_source_vr_terms(
            {"joint_pos": command.motion.joint_pos.detach().cpu().numpy()}, source_model_path
        )
        cache = (source_model_path, torch.as_tensor(terms, device=command.time_steps.device))
        command._release_compatible_vr = cache
    if cache[0] != source_model_path:
        raise ValueError("reference geometry changed after environment construction")
    anchor = command.time_steps
    if torch.any(anchor < 0) or torch.any(anchor >= len(cache[1])):
        raise ValueError("release reference anchor outside received motion")
    return cache[1][anchor]


def release_vr_position(env, source_model_path, command_name="motion"):
    return _reference_terms(env, source_model_path, command_name)[:, :9]


def release_vr_orientation(env, source_model_path, command_name="motion"):
    return _reference_terms(env, source_model_path, command_name)[:, 9:]


@dataclass(kw_only=True)
class ReleaseCompatibleActionCfg(NativeModelActuationActionCfg):
    action_convention: str = "released_bounded_linear"

    def build(self, env):
        return ReleaseCompatibleAction(self, env)


class ReleaseCompatibleAction(NativeModelActuationAction):
    def process_actions(self, actions):
        convert = (
            source_scaled_precompensation_torch
            if self.cfg.action_convention == SOURCE_ACTION_CONVENTION
            else bounded_linear_precompensation_torch
        )
        if self.cfg.action_convention not in ("released_bounded_linear", SOURCE_ACTION_CONVENTION):
            raise ValueError("unknown constructed release action convention")
        compensated, projection = convert(actions)
        super().process_actions(compensated)
        # PPO regularization sees the policy's real output; proprioception uses
        # inherited safe_native_action, never the inverse-tanh intermediate.
        self._raw_actions.copy_(actions)
        self.release_projection_delta_rad = projection


def source_previous_action(env):
    from gear_sonic.envs.mjlab.sonic_true23 import pad_native_il23_to_canonical_il29

    action = env.action_manager.get_term("joint_pos")
    if action.cfg.action_convention != SOURCE_ACTION_CONVENTION:
        raise ValueError("source-normalized history requires matching source-scaled action")
    return pad_native_il23_to_canonical_il29(source_normalize_native_action_torch(action.safe_native_action))


def configure_release_compatible_environment(cfg, source_model_path, action_convention="released_bounded_linear"):
    if action_convention not in ("released_bounded_linear", SOURCE_ACTION_CONVENTION):
        raise ValueError("unknown release training action convention")
    original = cfg.actions["joint_pos"]
    if type(original) is not NativeModelActuationActionCfg:
        raise ValueError("release compatibility requires nominal native motor physics")
    cfg.actions["joint_pos"] = ReleaseCompatibleActionCfg(
        entity_name=original.entity_name,
        actuator_names=original.actuator_names,
        profile=original.profile,
        action_convention=action_convention,
    )
    if action_convention == SOURCE_ACTION_CONVENTION:
        # Preserve the action term's native safety buffer. Only the actor's
        # observation is converted before its ten-frame history is collected.
        term = cfg.observations["policy"].terms["previous_action"]
        term.func, term.params = source_previous_action, {}
    terms = cfg.observations["tokenizer"].terms
    for key, function in (
        ("vr_3point_local_target", release_vr_position),
        ("vr_3point_local_orn_target", release_vr_orientation),
    ):
        terms[key].func = function
        terms[key].params = dict(source_model_path=str(source_model_path), command_name="motion")
    return cfg


def verify_executed_release_environment(env, source_model_path):
    """Exercise the actual constructed action term without stepping physics."""
    import numpy as np

    from gear_sonic.utils.g1_23dof_contract import NATIVE_IL23_TO_CANONICAL_IL29
    from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
    from gear_sonic.utils.g1_true23_release_action_diagnostic import bounded_linear_precompensation
    from gear_sonic.utils.g1_true23_source_action_codec import (
        source_action_history_numpy,
        source_scaled_precompensation,
    )

    action = env.action_manager.get_term("joint_pos")
    if not isinstance(action, ReleaseCompatibleAction):
        raise ValueError("executed environment lacks release-compatible action")
    command = env.command_manager.get_term("motion")
    q_before = command.robot_joint_pos.clone()
    raw_before = action.raw_action.clone()
    saved_buffers = {
        name: value.clone() for name, value in vars(action).items() if isinstance(value, torch.Tensor)
    }
    maximum_target = maximum_history = 0.0
    rng = np.random.default_rng(23991)
    try:
        for _ in range(8):
            raw = rng.uniform(-9.9, 9.9, (env.num_envs, 23)).astype(np.float32)
            action.process_actions(torch.as_tensor(raw, device=raw_before.device))
            convert = (
                source_scaled_precompensation
                if action.cfg.action_convention == SOURCE_ACTION_CONVENTION
                else bounded_linear_precompensation
            )
            expected = [safe_target_transform_numpy(convert(row)[0]) for row in raw]
            target_error = np.max(
                np.abs(action.processed_action.cpu().numpy() - np.stack([v[1] for v in expected]))
            )
            history_error = np.max(
                np.abs(action.safe_native_action.cpu().numpy() - np.stack([v[0] for v in expected]))
            )
            maximum_target = max(maximum_target, float(target_error))
            maximum_history = max(maximum_history, float(history_error))
            if action.cfg.action_convention == SOURCE_ACTION_CONVENTION:
                expected_source = []
                for safe, _target in expected:
                    history = np.zeros(930, np.float32)
                    history[610 + np.asarray(NATIVE_IL23_TO_CANONICAL_IL29)] = safe
                    expected_source.append(source_action_history_numpy(history)[610:639])
                actual_history = env.cfg.observations["policy"].terms["previous_action"].func(env)
                source_error = float(np.max(np.abs(actual_history.cpu().numpy() - np.stack(expected_source))))
                maximum_history = max(maximum_history, source_error)
                if source_error > 2e-6:
                    raise ValueError("executed source history differs from independent CPU conversion")
            if target_error > 1e-6 or history_error > 2e-6:
                raise ValueError("executed training action/history diverges from CPU referee")
    finally:
        action.process_actions(raw_before)
        for name, value in saved_buffers.items():
            getattr(action, name).copy_(value)
    if not torch.equal(q_before, command.robot_joint_pos):
        raise ValueError("action parity check unexpectedly mutated physical state")
    reference = _reference_terms(env, str(source_model_path), "motion")
    expected_reference = virtual_source_vr_terms(
        {"joint_pos": command.motion.joint_pos[command.time_steps].cpu().numpy()}, source_model_path
    )
    if not np.array_equal(reference.cpu().numpy(), expected_reference):
        raise ValueError("executed q9 reference diverges from CPU source geometry")
    return dict(
        kind="executed_mjlab_release_compatibility_parity_v1",
        native_joint_count=23,
        probe_count=8 * env.num_envs,
        target_max_abs_error_rad=maximum_target,
        normalized_history_max_abs_error=maximum_history,
        action_convention=action.cfg.action_convention,
        q9_reference_bit_exact=True,
        physical_state_mutated=False,
        physics_steps=0,
        pass_numerical_compatibility=True,
        deployment_ready=False,
    )
