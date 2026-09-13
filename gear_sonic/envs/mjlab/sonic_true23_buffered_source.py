"""Opt-in received-source horizon for simulation, with current measured state."""

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23 import _quat_inverse, _quat_multiply, _rotation_6d
from gear_sonic.utils.g1_true23_buffered_reference import (
    BUFFERED_TIMING,
    lower_horizon_cache,
    reference_profile_contract,
)


def buffered_source_lower_body(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    cached = getattr(command, "_buffered_source_lower240", None)
    if cached is None:
        motion = command.motion
        arrays = {
            name: getattr(motion, name).detach().cpu().numpy()
            for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        }
        parts = []
        cursor = 0
        for row in command._curriculum_spans:
            start, length = row["start"], row["length"]
            if start != cursor:
                raise ValueError("buffered training requires contiguous isolated lifecycle spans")
            part = {name: value[start : start + length] for name, value in arrays.items()}
            parts.append(lower_horizon_cache(part))
            cursor += length
        if cursor != motion.time_step_total:
            raise ValueError("buffered spans do not cover loaded motion")
        cached = torch.as_tensor(np.concatenate(parts), device=command.time_steps.device)
        command._buffered_source_lower240 = cached
    return cached[command.time_steps]


def buffered_source_current_orientation(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return _rotation_6d(_quat_multiply(_quat_inverse(command.robot_anchor_quat_w), command.anchor_quat_w))


def configure_buffered_source_environment(cfg):
    terms = cfg.observations["tokenizer"].terms
    if "causal_history_lower_body" not in terms:
        raise ValueError("buffered configuration requires the explicit causal base task")
    lower = terms["causal_history_lower_body"]
    lower.func, lower.params = buffered_source_lower_body, {"command_name": "motion"}
    cfg.observations["tokenizer"].terms = {
        ("received_source_horizon_lower_body" if name == "causal_history_lower_body" else name): term
        for name, term in terms.items()
    }
    orientation = cfg.observations["tokenizer"].terms["motion_anchor_ori_b"]
    orientation.func, orientation.params = buffered_source_current_orientation, {"command_name": "motion"}
    return cfg


def verify_buffered_source_environment(env):
    """Compare actual constructed terms with the independently fed stream buffer."""
    from gear_sonic.teleop.buffered_source_horizon import ReceivedSourceHorizon
    from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
    from gear_sonic.utils.g1_true23_buffered_reference import continued_standing_source

    command = env.command_manager.get_term("motion")
    terms = env.cfg.observations["tokenizer"].terms
    lower = terms["received_source_horizon_lower_body"].func(env)
    relative = terms["motion_anchor_ori_b"].func(env)
    vr = torch.cat(
        (
            terms["vr_3point_local_target"].func(env, **terms["vr_3point_local_target"].params),
            terms["vr_3point_local_orn_target"].func(env, **terms["vr_3point_local_orn_target"].params),
        ),
        dim=-1,
    )
    source_vr = command._release_compatible_vr[1].cpu().numpy()
    arrays = {
        name: getattr(command.motion, name).cpu().numpy()
        for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    }
    maximum = 0.0
    for env_index, anchor in enumerate(command.time_steps.cpu().tolist()):
        row = next(
            row for row in command._curriculum_spans if row["start"] <= anchor < row["start"] + row["length"]
        )
        start, end = row["start"], row["start"] + row["length"]
        source = continued_standing_source(
            {key: value[start:end] for key, value in arrays.items()}, source_vr[start:end]
        )
        buffer = ReceivedSourceHorizon()
        for i in range(anchor - start, anchor - start + 11):
            window = buffer.push(
                source_timestamp_s=i * 0.02,
                arrival_timestamp_s=i * 0.02,
                joint_names=HARDWARE_23_JOINT_NAMES,
                joint_position23=source["joint_pos"][i],
                root_position_w=source["root_position_w"][i],
                root_quaternion_wxyz=source["root_quaternion_wxyz"][i],
                virtual_source_vr21=source["virtual_vr21"][i],
            )
        expected = window.encoder267(command.robot_anchor_quat_w[env_index].cpu().numpy())
        actual = torch.cat((lower[env_index], vr[env_index], relative[env_index])).cpu().numpy()
        maximum = max(maximum, float(np.abs(actual - expected).max()))
        if not np.array_equal(actual[:261], expected[:261]) or maximum > 2e-6:
            raise ValueError("constructed buffered encoder differs from received-sample implementation")
    return dict(
        kind="g1_true23_buffered_source_executed_parity_v1",
        profile=reference_profile_contract(BUFFERED_TIMING),
        environment_count=env.num_envs,
        maximum_encoder_abs_error=maximum,
        lower_and_vr_bit_exact=True,
        physical_state_mutated=False,
        physics_steps=0,
        deployment_ready=False,
        hardware_authorized=False,
    )
