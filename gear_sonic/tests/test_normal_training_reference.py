from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from gear_sonic.teleop.normal_source_horizon import ReceivedNormalSourceHorizon
from gear_sonic.trl.mjlab.native23_normal_lora_actor import normal_adapter_contract, normal_conditioned_forward
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_normal_reference import (
    normal_lower_horizon_cache,
    normal_reference_contract,
    normal_root_feedback_contract,
    normal_standing_source,
)


def motion_fixture():
    count = 72
    joint = np.sin(np.arange(count, dtype=np.float32)[:, None] * 0.08) * np.linspace(
        -0.3, 0.3, 23, dtype=np.float32
    )
    joint[-10:] = joint[-11]
    quat = np.zeros((count, 1, 4), np.float32)
    quat[..., 0] = 1
    return dict(
        joint_pos=joint,
        joint_vel=np.zeros_like(joint),
        body_pos_w=np.zeros((count, 1, 3), np.float32),
        body_quat_w=quat,
        body_lin_vel_w=np.zeros((count, 1, 3), np.float32),
        body_ang_vel_w=np.zeros((count, 1, 3), np.float32),
    )


@pytest.mark.parametrize("anchor", [0, 18, 60, 71])
def test_cache_matches_received47_sample_windows_including_terminal_tail(anchor):
    motion = motion_fixture()
    before = {key: value.copy() for key, value in motion.items()}
    vr = np.zeros((72, 21), np.float32)
    vr[:, [9, 13, 17]] = 1
    source = normal_standing_source(motion, vr)
    buffer = ReceivedNormalSourceHorizon()
    for i in range(anchor, anchor + 47):
        window = buffer.push(
            source_timestamp_s=i * 0.02,
            arrival_timestamp_s=i * 0.02,
            joint_names=HARDWARE_23_JOINT_NAMES,
            joint_position23=source["joint_pos"][i],
            root_position_w=source["root_position_w"][i],
            root_quaternion_wxyz=source["root_quaternion_wxyz"][i],
            virtual_source_vr21=source["virtual_vr21"][i],
        )
    np.testing.assert_array_equal(window.lower_body240, normal_lower_horizon_cache(motion)[anchor])
    assert len(source["joint_pos"]) == 72 + 46
    for key, value in before.items():
        np.testing.assert_array_equal(motion[key], value)


def test_normal_velocity_is_adjacent20ms_not100ms_spacing():
    motion = motion_fixture()
    cache = normal_lower_horizon_cache(motion)
    source = motion["joint_pos"]
    expected = ((source[np.arange(10) * 5 + 1, :12] - source[np.arange(10) * 5, :12]) / np.float32(0.02)).ravel()
    np.testing.assert_array_equal(cache[0, 120:], expected)
    assert not np.array_equal(cache[0, 120:132], (source[5, :12] - source[0, :12]) / np.float32(0.1))


def test_nonstanding_eof_is_rejected_not_clamped():
    motion = motion_fixture()
    motion["joint_pos"][-1, 0] += 0.01
    with pytest.raises(ValueError, match="constant terminal standing"):
        normal_lower_horizon_cache(motion)


def test_normal_contracts_are_not_low_latency_labels():
    reference, root, adapters = (
        normal_reference_contract(),
        normal_root_feedback_contract(),
        normal_adapter_contract(),
    )
    assert reference["received_buffer"]["received_samples_required"] == 47
    assert reference["received_buffer"]["reference_anchor_age_s"] == 0.92
    assert reference["root_setpoint_age_s"] == 0.90
    assert root["physical_world_state_estimator_required"] is True
    assert len(adapters["dimensions"]) == 8 and adapters["dimensions"][1] == 2048
    assert adapters["decoder_outputs"] == 23 and adapters["phantom_action_outputs"] is False


def tiny_actor():
    dims = (5, 8, 8, 7, 7, 6, 6, 3)
    layers, a, b = [], [], []
    for index, (left, right) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        layers.append(nn.Linear(left, right).requires_grad_(False))
        a.append(nn.Parameter(torch.randn(2, left)))
        b.append(nn.Parameter(torch.zeros(right, 2)))
        if index < 6:
            layers.append(nn.SiLU())
    root = nn.Linear(9, 8, bias=False)
    nn.init.zeros_(root.weight)
    return SimpleNamespace(
        core=SimpleNamespace(decoder=SimpleNamespace(module=nn.Sequential(*layers))),
        root_conditioner=root,
        lora_a=a,
        lora_b=b,
    )


def test_all_seven_zero_effect_adapters_preserve_mean_and_receive_gradients():
    actor = tiny_actor()
    x, root = torch.randn(4, 5), torch.randn(4, 9)
    actual = normal_conditioned_forward(actor, x, root)
    assert torch.equal(actual, actor.core.decoder.module(x))
    actual.square().sum().backward()
    assert all(p.grad is None for p in actor.core.decoder.module.parameters())
    assert torch.count_nonzero(actor.root_conditioner.weight.grad)
    assert all(torch.count_nonzero(p.grad) for p in actor.lora_b)
    assert all(torch.count_nonzero(p.grad) == 0 for p in actor.lora_a)
