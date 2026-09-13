"""Execute original upstream observation/encoder assembly as the layout oracle."""

import __future__

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from gear_sonic.scripts.simulate_g1_sonic_library_motions import ISAAC_TO_MUJOCO_INDEX
from gear_sonic.utils.g1_sonic_full_pose_reference import full_pose_encoder640, full_pose_reference_contract

ORIGINAL = Path("/mnt/z/codex/GR00T-WholeBodyControl")


def upstream_function(path, function_name, class_name=None):
    tree = ast.parse((ORIGINAL / path).read_text())
    parent = (
        tree
        if class_name is None
        else next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    )
    node = next(node for node in parent.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = {"torch": torch}
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), scope)
    return scope[function_name]


def poses_and_measured():
    rng = np.random.default_rng(640)
    poses = rng.uniform(-0.4, 0.4, (11, 36)).astype(np.float32)
    poses[:, 3:7] = Rotation.from_rotvec(rng.uniform(-0.8, 0.8, (11, 3))).as_quat()[:, [3, 0, 1, 2]]
    measured = Rotation.from_euler("xyz", [0.13, -0.21, 0.48]).as_quat()[[3, 0, 1, 2]].astype(np.float32)
    return poses, measured


def test_actual_upstream_command_observation_and_encoder_assembly():
    poses, measured = poses_and_measured()
    q = poses[:, 7:][:, ISAAC_TO_MUJOCO_INDEX]
    dq = (q[1:] - q[:-1]) / np.float32(0.02)
    command = SimpleNamespace(
        num_envs=1,
        num_future_frames=10,
        joint_pos_multi_future=torch.tensor(q[:-1].reshape(1, 290)),
        joint_vel_multi_future=torch.tensor(dq.reshape(1, 290)),
    )
    command.command_multi_future = upstream_function(
        "gear_sonic/envs/manager_env/mdp/commands.py", "command_multi_future", "TrackingCommand"
    )(command)
    env = SimpleNamespace(num_envs=1, command_manager=SimpleNamespace(get_term=lambda name: command))
    obs_command = upstream_function("gear_sonic/envs/manager_env/mdp/observations.py", "command_multi_future")
    motion = obs_command(env, "motion", non_flatten=True)
    relative = Rotation.from_quat(measured[[1, 2, 3, 0]]).inv() * Rotation.from_quat(poses[:-1, [4, 5, 6, 3]])
    orientation = relative.as_matrix()[:, :, :2].reshape(1, 10, 6).astype(np.float32)
    command.root_rot_dif_l_multi_future = torch.tensor(orientation)
    obs_orientation = upstream_function(
        "gear_sonic/envs/manager_env/mdp/observations.py", "motion_anchor_ori_b_mf"
    )
    orient = obs_orientation(env, "motion", non_flatten=True)
    base_forward = upstream_function("gear_sonic/trl/modules/base_module.py", "forward", "BaseModule")
    base = SimpleNamespace(
        num_input_temporal_dims=10, num_output_temporal_dims=None, input_dim=640, module=torch.nn.Identity()
    )
    encoder_forward = upstream_function(
        "gear_sonic/trl/modules/universal_token_modules.py", "_encode_single", "UniversalTokenModule"
    )
    encoder = SimpleNamespace(
        encoders={"g1": lambda value: base_forward(base, value)},
        encoder_input_features={"g1": ["motion", "orientation"]},
    )
    actual = encoder_forward(encoder, "g1", {"motion": motion[:, None], "orientation": orient[:, None]})
    expected = full_pose_encoder640(poses, measured)
    np.testing.assert_allclose(actual.numpy().reshape(640), expected, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(actual.numpy().reshape(10, 64)[:, :58], expected.reshape(10, 64)[:, :58])
    ordinary_interleaved = np.concatenate((q[:-1], dq, orientation[0]), -1).reshape(640)
    assert not np.allclose(ordinary_interleaved, expected)


def test_all_29_source_axes_retained_and_source_not_mutated():
    poses, measured = poses_and_measured()
    before = poses.copy()
    original = full_pose_encoder640(poses, measured)
    for joint in range(29):
        changed = poses.copy()
        changed[5, 7 + joint] += 0.1
        assert not np.array_equal(full_pose_encoder640(changed, measured), original)
    np.testing.assert_array_equal(poses, before)
    assert full_pose_reference_contract()["hardware_authorized"] is False


@pytest.mark.parametrize("bad", ["count", "dtype", "nan", "quaternion", "measured"])
def test_invalid_horizon_rejected(bad):
    poses, measured = poses_and_measured()
    if bad == "count":
        poses = poses[:10]
    elif bad == "dtype":
        poses = poses.astype(np.float64)
    elif bad == "nan":
        poses[5, 14] = np.nan
    elif bad == "quaternion":
        poses[2, 3:7] = 0
    else:
        measured = measured.astype(np.float64)
    with pytest.raises(ValueError):
        full_pose_encoder640(poses, measured)
