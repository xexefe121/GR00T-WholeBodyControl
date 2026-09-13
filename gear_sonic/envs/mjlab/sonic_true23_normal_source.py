"""Original-core received reference timing over isolated native23 lifecycles."""

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23_buffered_source import buffered_source_current_orientation
from gear_sonic.utils.g1_true23_normal_reference import normal_lower_horizon_cache, normal_reference_contract

MOTION_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def normal_source_lower_body(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    cache = getattr(command, "_normal_source_lower240", None)
    if cache is None:
        motion = command.motion
        arrays = {name: getattr(motion, name).detach().cpu().numpy() for name in MOTION_KEYS}
        parts, cursor = [], 0
        for row in command._curriculum_spans:
            start, length = row["start"], row["length"]
            if start != cursor:
                raise ValueError("normal source requires contiguous isolated lifecycle spans")
            parts.append(
                normal_lower_horizon_cache({key: value[start : start + length] for key, value in arrays.items()})
            )
            cursor += length
        if cursor != motion.time_step_total:
            raise ValueError("normal source spans do not cover the loaded motion")
        cache = torch.as_tensor(np.concatenate(parts), device=command.time_steps.device)
        command._normal_source_lower240 = cache
    return cache[command.time_steps]


def configure_normal_source_environment(cfg):
    """Replace only lower timing after original29 intent has been configured."""
    from gear_sonic.envs.mjlab.sonic_true23_buffered_source import buffered_source_lower_body

    terms = cfg.observations["tokenizer"].terms
    if "received_normal_horizon_lower_body" in terms or (
        terms.get("received_source_horizon_lower_body") is None
        or terms["received_source_horizon_lower_body"].func is not buffered_source_lower_body
    ):
        raise ValueError("normal source requires a recognized original-intent buffered predecessor")
    if terms["motion_anchor_ori_b"].func is not buffered_source_current_orientation:
        raise ValueError("normal source must retain current measured root orientation")
    terms["received_source_horizon_lower_body"].func = normal_source_lower_body
    terms["received_source_horizon_lower_body"].params = {"command_name": "motion"}
    cfg.observations["tokenizer"].terms = {
        ("received_normal_horizon_lower_body" if name == "received_source_horizon_lower_body" else name): term
        for name, term in terms.items()
    }
    return cfg


def verify_normal_source_environment(env, spec):
    """Read actual tensors and compare with separately admitted47-sample windows."""
    from gear_sonic.envs.mjlab import sonic_true23_original_intent as intent
    from gear_sonic.teleop.normal_source_horizon import ReceivedNormalSourceHorizon
    from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
    from gear_sonic.utils.g1_true23_normal_reference import normal_standing_source
    from gear_sonic.utils.g1_true23_original29_reference import task_targets_from_received_vr
    from gear_sonic.utils.g1_true23_root_feedback import root_feedback_numpy

    command = env.command_manager.get_term("motion")
    cache = intent._cache(command, spec)
    arrays = {name: getattr(command.motion, name).cpu().numpy() for name in MOTION_KEYS}
    terms = env.cfg.observations["tokenizer"].terms
    lower = terms["received_normal_horizon_lower_body"].func(env)
    vr = torch.cat((intent.original_vr_position(env, spec), intent.original_vr_orientation(env, spec)), -1)
    orientation = terms["motion_anchor_ori_b"].func(env)
    root_term = env.cfg.observations["root_feedback"].terms["root_feedback"]
    actual_root9 = root_term.func(env).cpu().numpy()
    desired, desired_quat, _, _ = intent.task_states(env, spec)
    max_encoder = max_world = max_quat = max_root9 = 0.0
    anchors = command.time_steps.cpu().tolist()
    for index, anchor in enumerate(anchors):
        row = next(
            row for row in command._curriculum_spans if row["start"] <= anchor < row["start"] + row["length"]
        )
        start, end = row["start"], row["start"] + row["length"]
        source = normal_standing_source(
            {key: value[start:end] for key, value in arrays.items()}, cache["vr21"][start:end].cpu().numpy()
        )
        buffer = ReceivedNormalSourceHorizon()
        for frame in range(anchor - start, anchor - start + 47):
            window = buffer.push(
                source_timestamp_s=frame * 0.02,
                arrival_timestamp_s=frame * 0.02,
                joint_names=HARDWARE_23_JOINT_NAMES,
                joint_position23=source["joint_pos"][frame],
                root_position_w=source["root_position_w"][frame],
                root_quaternion_wxyz=source["root_quaternion_wxyz"][frame],
                virtual_source_vr21=source["virtual_vr21"][frame],
            )
        expected = window.encoder267(command.robot_anchor_quat_w[index].cpu().numpy())
        actual = torch.cat((lower[index], vr[index], orientation[index])).cpu().numpy()
        np.testing.assert_array_equal(actual[:261], expected[:261])
        max_encoder = max(max_encoder, float(np.abs(actual - expected).max()))
        q1 = anchor - start + 1
        world, quaternion = task_targets_from_received_vr(
            source["root_position_w"][q1 : q1 + 1],
            source["root_quaternion_wxyz"][q1 : q1 + 1],
            source["virtual_vr21"][q1 : q1 + 1],
        )
        world += env.scene.env_origins[index].cpu().numpy()
        max_world = max(max_world, float(np.abs(world[0] - desired[index].cpu().numpy()).max()))
        max_quat = max(max_quat, float(np.abs(quaternion[0] - desired_quat[index].cpu().numpy()).max()))
        expected_root9 = root_feedback_numpy(
            source["root_position_w"][q1] + env.scene.env_origins[index].cpu().numpy(),
            command.robot_anchor_pos_w[index].cpu().numpy(),
            (source["root_position_w"][q1] - source["root_position_w"][q1 - 1]) / np.float32(0.02),
            command.robot_anchor_lin_vel_w[index].cpu().numpy(),
            command.robot_anchor_quat_w[index].cpu().numpy(),
        )
        max_root9 = max(max_root9, float(np.abs(expected_root9 - actual_root9[index]).max()))
    if max(max_encoder, max_world, max_quat, max_root9) > 2e-6:
        raise ValueError("actual normal reference diverges from received-source timing/geometry")
    return dict(
        kind="native23_normal_executed_source_environment_parity_v1",
        reference=normal_reference_contract(),
        environment_count=env.num_envs,
        anchors=anchors,
        lower_and_original_vr_bit_exact=True,
        maximum_encoder_abs_error=max_encoder,
        original_q1_task_position_component_max_error_m=max_world,
        original_q1_task_quaternion_component_max_error=max_quat,
        original_q1_root9_component_max_abs_error=max_root9,
        physical_state_mutated=False,
        physics_steps=0,
        hardware_authorized=False,
        deployment_ready=False,
    )
