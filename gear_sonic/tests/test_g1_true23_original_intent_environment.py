"""Original source tasks, physical23 state, timing and recipe separation."""

import copy
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab import sonic_true23_original_intent as module
from gear_sonic.scripts.prepare_g1_true23_contact_step_lifecycle import motion_from_poses
from gear_sonic.scripts.record_g1_sonic_public29_baseline import stock
from gear_sonic.utils.g1_23dof_contract import SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_true23_original29_reference import build_original29_reference


@pytest.fixture
def reference(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source_path = root / "gear_sonic/data/robots/g1/g1_29dof.xml"
    native_path = root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
    source = mujoco.MjModel.from_xml_path(str(source_path))
    native = mujoco.MjModel.from_xml_path(str(native_path))
    poses = np.tile(np.r_[[0, 0, 0.76, 1, 0, 0, 0], stock.DEFAULT_ANGLES], (14, 1))
    poses[:, 0] = np.arange(len(poses)) * 0.01
    poses[:, 20] = 0.45  # Original waist roll, not a native physical axis.
    poses[:, 27] = 0.30  # Original left wrist pitch.
    original = build_original29_reference(source, poses)
    native_poses = np.c_[poses[:, :7], poses[:, 7:][:, SOURCE_MJ29_KEEP_INDICES]]
    motion = motion_from_poses(native, native_poses)
    source_file, motion_file = tmp_path / "source.npz", tmp_path / "native.npz"
    np.savez(source_file, **original.arrays())
    np.savez(motion_file, **motion)
    spec = module.make_spec(
        original_reference=source_file,
        native_motion=motion_file,
        source_model=source_path,
        native_model=native_path,
    )
    names = tuple(native.body(i).name for i in range(1, native.nbody))
    tensors = SimpleNamespace(**{k: torch.tensor(v, dtype=torch.float32) for k, v in motion.items()})
    tensors.time_step_total = len(poses)
    origins = torch.tensor([[0.0, 0, 0], [3.0, -2, 0]])
    command = SimpleNamespace(
        motion=tensors,
        time_steps=torch.tensor([9, 9]),
        cfg=SimpleNamespace(body_names=names),
        motion_anchor_body_index=0,
        _curriculum_spans=[dict(start=0, length=len(poses))],
        _env=SimpleNamespace(scene=SimpleNamespace(env_origins=origins)),
        robot_joint_pos=tensors.joint_pos[[10, 10]].clone(),
        robot_body_pos_w=tensors.body_pos_w[[10, 10]].clone() + origins[:, None],
        robot_body_quat_w=tensors.body_quat_w[[10, 10]].clone(),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command), num_envs=2)
    return spec, command, env


def test_missing_axes_preserved_in_source_not_robot_and_native_perfection_is_not_task_perfection(reference):
    spec, command, env = reference
    vr = torch.cat((module.original_vr_position(env, spec), module.original_vr_orientation(env, spec)), -1)
    loaded = module.load_reference(spec)
    np.testing.assert_array_equal(vr.numpy(), loaded["reference"].virtual_vr21[[9, 9]])
    assert command.robot_joint_pos.shape[-1] == 23
    assert torch.all(module.original_task_position_l2(env, spec) > 0.1)
    assert torch.all(module.original_task_orientation_l2(env, spec) > 0.1)
    torch.testing.assert_close(module.lower_posture_l2(env), torch.zeros(2))


def test_fixed_world_targets_and_environment_origins(reference):
    spec, command, env = reference
    before = module.task_states(env, spec)
    torch.testing.assert_close(before[0][1] - before[0][0], torch.tensor([[3.0, -2, 0]]).expand(3, -1))
    command.robot_body_pos_w[:, :, 0] += 2
    after = module.task_states(env, spec)
    torch.testing.assert_close(after[0], before[0], rtol=0, atol=0)
    torch.testing.assert_close(after[2][..., 0] - before[2][..., 0], torch.full((2, 3), 2.0))


def test_actual_task_error_decreases_when_measured_points_improve(reference):
    spec, command, env = reference
    target, _, measured, _ = module.task_states(env, spec)
    before = module.original_task_position_l2(env, spec)
    indices = command._original_intent_cache["task_body_indices"]
    command.robot_body_pos_w[:, indices] += (target - measured) * 0.5
    torch.testing.assert_close(module.original_task_position_l2(env, spec), before * 0.25, rtol=2e-5, atol=1e-6)


def test_q0_observation_q1_reward_no_q2_read(reference):
    spec, command, env = reference
    before = module.original_task_position_l2(env, spec).clone()
    vr = module.original_vr_position(env, spec).clone()
    critic = module.original_critic_vr(env, spec).clone()
    cached = command._original_intent_cache
    cached["position_w"][11:] += 1000
    cached["vr21"][11:] += 1000
    torch.testing.assert_close(module.original_task_position_l2(env, spec), before, rtol=0, atol=0)
    torch.testing.assert_close(module.original_vr_position(env, spec), vr, rtol=0, atol=0)
    torch.testing.assert_close(module.original_critic_vr(env, spec), critic, rtol=0, atol=0)
    cached["vr21"][10] += 1
    assert not torch.equal(module.original_critic_vr(env, spec), critic)
    torch.testing.assert_close(module.original_vr_position(env, spec), vr, rtol=0, atol=0)
    cached["position_w"][10] += 1
    assert torch.all(module.original_task_position_l2(env, spec) > before)


def test_upper_joint_imitation_removed_without_removing_leg_posture(reference):
    _, command, env = reference
    command.robot_joint_pos[:, 13:] += 0.9
    torch.testing.assert_close(module.lower_posture_l2(env), torch.zeros(2))
    command.robot_joint_pos[:, 1] += 0.3
    assert torch.all(module.lower_posture_l2(env) > 0)


def test_orientation_sign_invariant(reference):
    spec, command, env = reference
    cost = module.original_task_orientation_l2(env, spec)
    command.robot_body_quat_w.neg_()
    torch.testing.assert_close(module.original_task_orientation_l2(env, spec), cost)


def test_ee_height_termination_and_barrier_keep_existing_thresholds(reference):
    spec, command, env = reference
    target, _, actual, _ = module.task_states(env, spec)
    indices = command._original_intent_cache["task_body_indices"]
    command.robot_body_pos_w[:, indices, 2] += target[:, :, 2] - actual[:, :, 2]
    assert not module.bad_ee_height(env, spec).any()
    command.robot_body_pos_w[:, indices[1], 2] += 0.2
    assert not module.bad_ee_height(env, spec).any()
    torch.testing.assert_close(
        module.right_hand_height_barrier(env, spec), torch.full((2,), 0.5), rtol=1e-4, atol=1e-5
    )
    command.robot_body_pos_w[:, indices[1], 2] += 0.051
    assert module.bad_ee_height(env, spec).all()


@pytest.mark.parametrize("reason", ("nan", "missing_q1", "span_crossing", "spec_changed", "motion_mismatch"))
def test_invalid_runtime_rejected(reference, reason):
    spec, command, env = reference
    if reason == "nan":
        command.robot_body_pos_w[:] = float("nan")
    elif reason == "missing_q1":
        command.time_steps[:] = 13
    elif reason == "span_crossing":
        module.original_vr_position(env, spec)
        command._curriculum_spans = [dict(start=0, length=10), dict(start=10, length=4)]
    elif reason == "spec_changed":
        module.original_vr_position(env, spec)
        spec["objective"]["name"] = "wrong"
    else:
        command.motion.joint_pos[0, 0] += 0.1
    with pytest.raises(ValueError):
        module.original_task_position_l2(env, spec)


def make_cfg(spec):
    from gear_sonic.envs.mjlab.sonic_true23_buffered_source import configure_buffered_source_environment
    from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import make_causal_multimotion_v14_env_cfg
    from gear_sonic.envs.mjlab.sonic_true23_native_model_actuation import apply_native_model_actuation_profile
    from gear_sonic.envs.mjlab.sonic_true23_release_compatible import configure_release_compatible_environment
    from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile

    root = Path(__file__).resolve().parents[2]
    profile = NativeModelActuationProfile.from_sim_config(
        root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
    )
    cfg = make_causal_multimotion_v14_env_cfg(motion_file=spec["files"]["native_motion"]["path"], num_envs=2)
    cfg = apply_native_model_actuation_profile(cfg, profile)
    module.base.configure_root_feedback_environment(
        cfg, [dict(timeline=dict(total_requested_controls=3))], objective_profile="root_and_upper_feet_world_v4"
    )
    cfg = configure_release_compatible_environment(
        cfg, spec["files"]["source_model"]["path"], module.SOURCE_ACTION_CONVENTION
    )
    return configure_buffered_source_environment(cfg)


def test_recipe_installation_preserves_physics_and_explicitly_replaces_conflicts(reference):
    spec, _, _ = reference
    original = make_cfg(spec)
    cfg = module.configure_original_intent_environment(original, spec)
    assert "measured_upper_body_posture_l2" in original.rewards
    assert "measured_upper_body_posture_l2" not in cfg.rewards
    assert cfg.actions == original.actions
    for name in (
        "joint_limit",
        "self_collisions",
        "action_target_soft_limit_barrier",
        "requested_effort_excess",
        "root_world_tracking_error",
        "measured_feet_world_position_l2",
        "alive",
        "non_timeout_termination",
    ):
        assert cfg.rewards[name] == original.rewards[name]
    for name in ("anchor_pos", "anchor_ori", "invalid_native_model_actuation", "time_out", "corpus_clip_time_out"):
        assert cfg.terminations[name] == original.terminations[name]
    assert cfg.rewards["motion_body_pos"].params["body_names"] == module.LOWER_BODIES
    assert cfg.rewards["original_task_position_l2"].weight == -1
    assert cfg.terminations["ee_body_pos"].func is module.bad_ee_height
    assert cfg.terminations["ee_body_pos"].params["threshold"] == 0.25
    assert cfg.observations["critic"] == original.observations["critic"]
    assert (
        cfg.observations["original_intent_value_reference"].terms["original_q1_vr21"].func
        is module.original_critic_vr
    )


@pytest.mark.parametrize("reason", ("action", "threshold", "barrier", "native_body_reward", "upper_reward"))
def test_unknown_predecessor_or_relaxed_gate_rejected(reference, reason):
    spec, _, _ = reference
    cfg = make_cfg(spec)
    if reason == "action":
        cfg.actions["joint_pos"].action_convention = "released_bounded_linear"
    elif reason == "threshold":
        cfg.terminations["ee_body_pos"].params["threshold"] = 0.5
    elif reason == "barrier":
        cfg.rewards["right_wrist_prethreshold_barrier"].params["onset_m"] = 0.1
    elif reason == "native_body_reward":
        cfg.rewards["motion_body_pos"].params["body_names"] = ("pelvis",)
    else:
        cfg.rewards["measured_upper_body_posture_l2"].weight = -2
    with pytest.raises(ValueError):
        module.configure_original_intent_environment(cfg, spec)


def test_spec_and_file_tampering_rejected(reference):
    spec, _, _ = reference
    tampered = copy.deepcopy(spec)
    tampered["objective"]["hardware_authorized"] = True
    with pytest.raises(ValueError):
        module.load_reference(tampered)
    Path(spec["files"]["native_motion"]["path"]).write_bytes(b"not the reference")
    with pytest.raises(ValueError, match="bytes changed"):
        module.load_reference(spec)
