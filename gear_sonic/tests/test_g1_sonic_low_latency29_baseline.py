import ast
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from gear_sonic.scripts import simulate_g1_sonic_library_motions as stock
from gear_sonic.scripts.audit_g1_sonic_low_latency29_baseline import record_case, reference_features
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_step1b_mujoco import ReferenceKinematics


@pytest.fixture
def material():
    model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).resolve().parents[2] / "gear_sonic_deploy/g1/scene_29dof.xml")
    )
    model.opt.timestep = 0.002
    poses = np.tile(np.r_[[0, 0, 0.78, 1, 0, 0, 0], stock.DEFAULT_ANGLES], (12, 1))
    poses[:, 7:] += np.arange(12)[:, None] * np.arange(29)[None, :] * 0.0001
    velocity = np.tile(np.arange(29) * 0.005, (12, 1))
    return model, poses, velocity


def test_reference_future_order_and_fk_matches_independent_existing_builder(material):
    model, poses, velocity = material
    before = compiled_model_sha256(model)
    features = reference_features(model, poses, velocity)
    assert features.shape == (12, 267) and features.dtype == np.float32
    keep = np.arange(12)
    np.testing.assert_allclose(features[0, :120], poses[:10, 7:][:, keep].reshape(-1), atol=1e-7)
    np.testing.assert_allclose(features[0, 120:240], velocity[:10, keep].reshape(-1), atol=1e-7)
    np.testing.assert_allclose(features[-1, :120], np.tile(poses[-1, 7:][keep], 10), atol=1e-7)
    # Populate the independent legacy FK cache directly: its target-model
    # constructor is irrelevant to source-only teleop features.
    reference = object.__new__(ReferenceKinematics)
    reference.clip = SimpleNamespace(
        source_root_pos=poses[:, :3],
        source_root_quat_wxyz=poses[:, 3:7],
        source_joint_pos_hardware=poses[:, 7:],
        source_joint_vel_hardware=velocity,
        future_indices=lambda f: np.minimum(f + np.arange(10), len(poses) - 1),
        frame_index=lambda f: min(f, len(poses) - 1),
    )
    reference._source_positions, reference._source_quaternions = {}, {}
    data = mujoco.MjData(model)
    for name in ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"):
        position, orientation = [], []
        for pose in poses:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            position.append(data.xpos[model.body(name).id].copy())
            orientation.append(data.xquat[model.body(name).id].copy())
        reference._source_positions[name] = np.asarray(position)
        reference._source_quaternions[name] = np.asarray(orientation)
    for frame in (0, 5, 11):
        expected = reference.teleop_encoder_input(frame, poses[frame, 3:7])
        # Legacy helper has a separate wrong lower-body order; only compare FK.
        np.testing.assert_allclose(features[frame, 240:261], expected[240:261], atol=1e-7, rtol=0)
    assert compiled_model_sha256(model) == before


def test_lower_body_order_executes_original_tracking_command_property(material):
    model, poses, velocity = material
    features = reference_features(model, poses, velocity)
    path = Path(__file__).resolve().parents[1] / "envs/manager_env/mdp/commands.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "joint_pos_lower_body_multi_future"
    )
    function.decorator_list = []
    namespace = {"torch": torch}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    il_values = torch.as_tensor(poses[:10, 7:][:, stock.ISAAC_TO_MUJOCO_INDEX])
    command = SimpleNamespace(
        num_envs=1,
        future_motion_ids=None,
        future_time_steps=None,
        lower_joint_isaaclab_indices=stock.MUJOCO_TO_ISAAC_INDEX[:12].tolist(),
        motion_lib=SimpleNamespace(get_dof_pos=lambda *_: il_values),
    )
    expected = namespace["joint_pos_lower_body_multi_future"](command).numpy().reshape(-1)
    np.testing.assert_allclose(features[0, :120], expected, atol=1e-7, rtol=0)


@pytest.mark.parametrize("change", ("short", "nonunit", "nonfinite", "velocity_shape"))
def test_invalid_reference_rejected(material, change):
    model, poses, velocity = material
    if change == "short":
        poses, velocity = poses[:5], velocity[:5]
    elif change == "nonunit":
        poses[0, 3] = 0.5
    elif change == "nonfinite":
        poses[0, 7] = np.nan
    else:
        velocity = velocity[:, :23]
    with pytest.raises(ValueError):
        reference_features(model, poses, velocity)


def test_rollout_retains_every_state_and_torque_with_no_resets(material):
    model, poses, velocity = material
    features = reference_features(model, poses, velocity)
    parameters = SimpleNamespace(
        default_angles=stock.DEFAULT_ANGLES,
        kps=stock.KPS,
        kds=stock.KDS,
        effort=stock.EFFORT_LIMITS,
        target=lambda _: stock.DEFAULT_ANGLES.astype(float),
    )
    teacher = SimpleNamespace(infer=lambda semantic, history: np.zeros(29, dtype=np.float32))
    before = compiled_model_sha256(model)
    arrays, metrics = record_case(model, parameters, teacher, poses, features, poses[0])
    assert metrics["completed_controls"] == metrics["requested_controls"] == 12
    assert arrays["physics_torque29"].shape == (120, 29)
    np.testing.assert_array_equal(arrays["pre_qpos"][1:], arrays["qpos"][:-1])
    np.testing.assert_array_equal(arrays["pre_qpos"][0], poses[0])
    assert metrics["state_resets_after_initialization"] == 0
    assert not metrics["native23_qualification"]
    assert compiled_model_sha256(model) == before
