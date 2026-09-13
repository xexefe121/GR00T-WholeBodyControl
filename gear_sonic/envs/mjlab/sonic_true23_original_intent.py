"""Explicit original-source task objectives for native23 research training.

The source retains all29 axes; the simulated plant and action stay native23.
This is a distinct recipe, not an update/relabel of an old checkpoint. Targets
are fixed-world received q1, observations use received q0, and measured body
state retains the existing MJLab reward phase. No engine/interlock is changed.
"""

from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np
import torch

from gear_sonic.envs.mjlab import sonic_true23_root_feedback as base
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_ACTION_SCALE
from gear_sonic.utils.g1_true23_buffered_reference import BUFFERED_TIMING
from gear_sonic.utils.g1_true23_generalist_corpus import canonical_digest, sha256_file
from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks
from gear_sonic.utils.g1_true23_original29_reference import (
    REFERENCE_KIND,
    build_original29_reference,
    verify_unmodified_native_pair,
)
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_ACTION_CONVENTION

RECIPE = "native23_original_source_tasks_v1"
FILES = ("original_reference", "native_motion", "source_model", "native_model")
TASK_NAMES = ("left_hand", "right_hand", "head_proxy")
LOWER_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
)


def objective_contract():
    return dict(
        name=RECIPE,
        source_reference_kind=REFERENCE_KIND,
        observation_target="received_q0_original29_VR21",
        critic_extra_input="received_q1_original29_VR21_separate21_value_only_group",
        reward_target="received_q1_original29_world_hand_hand_head",
        original_task_position=dict(weight=-1.0, normalization_m=0.1, reduction="mean_3_points_sum_xyz_squared"),
        original_task_orientation=dict(
            weight=-1.0, normalization_rad=0.4, reduction="mean_3_shortest_angle_squared"
        ),
        native_body_reference_rewards_restricted_to=list(LOWER_BODIES),
        measured_and_requested_joint_posture_restricted_to_hardware_indices=list(range(13)),
        upper_joint_imitation_removed=True,
        old_upper_posture_cost_removed=True,
        original_hand_height_targets_replace_native_wrist_origin_targets=True,
        existing_ee_height_threshold_m=0.25,
        existing_right_hand_barrier=dict(onset_m=0.15, termination_m=0.25, denominator_floor=0.05),
        root_feet_world_rewards_unchanged=True,
        physical_penalties_limits_gains_and_acceptance_criteria_unchanged=True,
        reward_measured_state_phase="existing_MJLab_last_preintegration_substep_2ms_stale",
        actor_measured_state_phase="existing_synchronized_post_forward",
        dynamic_feasibility_or_tracking_improvement_proven=False,
        hardware_authorized=False,
        deployment_ready=False,
    )


def make_spec(**paths):
    if set(paths) != set(FILES):
        raise ValueError("original-intent spec requires source/native reference and model files")
    bindings = {}
    for name, value in paths.items():
        path = Path(value).resolve(strict=True)
        bindings[name] = dict(path=str(path), sha256=sha256_file(path))
    payload = dict(kind=RECIPE, files=bindings, objective=objective_contract())
    return {**payload, "sha256": canonical_digest(payload)}


def validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {"kind", "files", "objective", "sha256"}:
        raise ValueError("invalid original-intent spec")
    if spec["kind"] != RECIPE or spec["objective"] != objective_contract() or set(spec["files"]) != set(FILES):
        raise ValueError("original-intent recipe differs")
    if canonical_digest({k: v for k, v in spec.items() if k != "sha256"}) != spec["sha256"]:
        raise ValueError("original-intent spec hash differs")
    for entry in spec["files"].values():
        if set(entry) != {"path", "sha256"} or sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("original-intent input bytes changed")
    return spec


def load_reference(spec):
    validate_spec(spec)
    with np.load(spec["files"]["original_reference"]["path"], allow_pickle=False) as archive:
        recorded = {name: archive[name].copy() for name in archive.files}
    with np.load(spec["files"]["native_motion"]["path"], allow_pickle=False) as archive:
        native = {name: archive[name].copy() for name in archive.files}
    source = mujoco.MjModel.from_xml_path(spec["files"]["source_model"]["path"])
    model = mujoco.MjModel.from_xml_path(spec["files"]["native_model"]["path"])
    reference = build_original29_reference(source, recorded["source_qpos29"])
    for name, value in reference.arrays().items():
        if name not in recorded or not np.array_equal(value, recorded[name]):
            raise ValueError(f"original-intent reference fails independent source reconstruction: {name}")
    pair = verify_unmodified_native_pair(reference, native)
    tasks, hand_frame = neutral_wrist_hand_tasks(source, model)
    tasks = tuple(next(task for task in tasks if task.name == name) for name in TASK_NAMES)
    validate_spec(spec)
    return dict(
        reference=reference,
        native=native,
        tasks=tasks,
        pair=pair,
        hand_frame=hand_frame,
        body_names=tuple(model.body(i).name for i in range(1, model.nbody)),
    )


def _cache(command, spec):
    cached = getattr(command, "_original_intent_cache", None)
    if cached is not None:
        if cached["spec"] != spec:
            raise ValueError("original-intent reference changed after environment construction")
        return cached
    loaded = load_reference(spec)
    names = tuple(command.cfg.body_names)
    indices = [loaded["body_names"].index(name) for name in names]
    motion = command.motion

    def tensor(value):
        return torch.as_tensor(value.copy(), dtype=motion.joint_pos.dtype, device=command.time_steps.device)

    for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        expected = loaded["native"][name]
        if name.startswith("body_"):
            expected = expected[:, indices]
        if not torch.equal(getattr(motion, name), tensor(expected)):
            raise ValueError(f"executed native motion differs from paired original source: {name}")
    cursor = 0
    for row in command._curriculum_spans:
        if row["start"] != cursor or row["length"] < 12:
            raise ValueError("original-intent references need contiguous isolated lifecycle spans")
        cursor += row["length"]
    if cursor != motion.time_step_total:
        raise ValueError("original-intent lifecycle coverage differs")
    reference = loaded["reference"]
    cached = dict(
        spec=copy.deepcopy(spec),
        vr21=tensor(reference.virtual_vr21),
        position_w=tensor(reference.source_task_position_w),
        quaternion_wxyz=tensor(reference.source_task_quaternion_wxyz),
        task_body_indices=[names.index(task.target_body) for task in loaded["tasks"]],
        task_offsets=tensor(np.array([task.target_point for task in loaded["tasks"]])),
        pair=loaded["pair"],
        hand_frame=loaded["hand_frame"],
    )
    command._original_intent_cache = cached
    return cached


def _indices(command):
    anchor, proof = base._causal_indices(command)
    valid = torch.zeros_like(anchor, dtype=torch.bool)
    for row in command._curriculum_spans:
        valid |= (anchor >= row["start"]) & (proof < row["start"] + row["length"])
    if not valid.all():
        raise ValueError("original-intent target crosses a source lifecycle boundary")
    return anchor, proof


def original_vr_position(env, spec, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return _cache(command, spec)["vr21"][_indices(command)[0], :9]


def original_vr_orientation(env, spec, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return _cache(command, spec)["vr21"][_indices(command)[0], 9:]


def original_critic_vr(env, spec, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return _cache(command, spec)["vr21"][_indices(command)[1]]


def task_states(env, spec, command_name="motion"):
    from mjlab.utils.lab_api.math import quat_apply

    command = env.command_manager.get_term(command_name)
    cached = _cache(command, spec)
    proof = _indices(command)[1]
    target = cached["position_w"][proof] + command._env.scene.env_origins[:, None, :]
    target_quat = cached["quaternion_wxyz"][proof]
    indices = cached["task_body_indices"]
    actual_quat = command.robot_body_quat_w[:, indices]
    offsets = cached["task_offsets"][None].expand_as(target)
    actual = command.robot_body_pos_w[:, indices] + quat_apply(actual_quat, offsets)
    if any(not value.isfinite().all() for value in (target, target_quat, actual, actual_quat)):
        raise ValueError("nonfinite original-intent target or measured state")
    return target, target_quat, actual, actual_quat


def original_task_position_l2(env, spec, command_name="motion"):
    target, _, measured, _ = task_states(env, spec, command_name)
    return ((measured - target) / 0.1).square().sum(-1).mean(-1)


def original_task_orientation_l2(env, spec, command_name="motion"):
    from mjlab.utils.lab_api.math import quat_error_magnitude

    _, target, _, measured = task_states(env, spec, command_name)
    return (quat_error_magnitude(target, measured) / 0.4).square().mean(-1)


def lower_posture_l2(env, command_name="motion", action_name=None):
    command = env.command_manager.get_term(command_name)
    target = command.motion.joint_pos[_indices(command)[1], :13]
    if action_name is None:
        measured = command.robot_joint_pos[:, :13]
    else:
        from gear_sonic.envs.mjlab.sonic_true23_low_latency_recovery import _processed_target

        measured = _processed_target(env, action_name)[:, :13]
    if target.shape != measured.shape or not measured.isfinite().all() or not target.isfinite().all():
        raise ValueError("lower posture requires finite native23 measured/requested joint state")
    return ((measured - target) / measured.new_tensor(HARDWARE_23_ACTION_SCALE[:13])).square().mean(-1)


def ee_height_error(env, spec, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    target, _, measured, _ = task_states(env, spec, command_name)
    feet = base._body_indices(command, ("left_ankle_roll_link", "right_ankle_roll_link"))
    foot_error = (base._q10_body_position(command)[:, feet, 2] - command.robot_body_pos_w[:, feet, 2]).abs()
    return torch.cat((foot_error, (target[:, :2, 2] - measured[:, :2, 2]).abs()), -1)


def worst_ee_height_cost(env, spec, command_name="motion"):
    return (ee_height_error(env, spec, command_name).amax(-1) / 0.25).square()


def right_hand_height_barrier(env, spec, command_name="motion"):
    error = ee_height_error(env, spec, command_name)[:, 3]
    x = torch.relu((error - 0.15) / (0.25 - 0.15))
    return x.square() / (1 - x).clamp_min(0.05)


def bad_ee_height(env, spec, command_name="motion", threshold=0.25):
    if threshold != 0.25:
        raise ValueError("original-intent EE height threshold must remain0.25m")
    return (ee_height_error(env, spec, command_name) > threshold).any(-1)


def configure_original_intent_environment(cfg, spec):
    """Explicitly replace all conflicting upper-pose targets, preserving safety.

    Requires the known v4 reward inventory, buffered timing and source-scaled motor
    action. The separate trainer must bind this recipe to a new actor/checkpoint.
    """
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg

    from gear_sonic.envs.mjlab.sonic_true23_buffered_source import buffered_source_lower_body
    from gear_sonic.envs.mjlab.sonic_true23_release_compatible import (
        ReleaseCompatibleActionCfg,
        release_vr_orientation,
        release_vr_position,
    )

    validate_spec(spec)
    cfg = copy.deepcopy(cfg)
    action = cfg.actions["joint_pos"]
    if type(action) is not ReleaseCompatibleActionCfg or action.action_convention != SOURCE_ACTION_CONVENTION:
        raise ValueError("original-intent training requires nominal source-scaled native23 actions")
    observations = cfg.observations["tokenizer"].terms
    if observations["received_source_horizon_lower_body"].func is not buffered_source_lower_body:
        raise ValueError(f"original-intent training requires {BUFFERED_TIMING}")
    for name, before, after in (
        ("vr_3point_local_target", release_vr_position, original_vr_position),
        ("vr_3point_local_orn_target", release_vr_orientation, original_vr_orientation),
    ):
        if observations[name].func is not before:
            raise ValueError("unrecognized original-intent observation predecessor")
        observations[name].func, observations[name].params = after, dict(spec=copy.deepcopy(spec))
    if "original_intent_value_reference" in cfg.observations:
        raise ValueError("original-intent critic input already installed")
    cfg.observations["original_intent_value_reference"] = ObservationGroupCfg(
        terms={
            "original_q1_vr21": ObservationTermCfg(func=original_critic_vr, params=dict(spec=copy.deepcopy(spec)))
        },
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="error",
    )
    body_functions = {
        "motion_body_pos": base.q10_body_position_reward,
        "motion_body_ori": base.q10_body_orientation_reward,
        "motion_body_lin_vel": base.q10_body_linear_velocity_reward,
        "motion_body_ang_vel": base.q10_body_angular_velocity_reward,
    }
    for name, function in body_functions.items():
        term = cfg.rewards[name]
        if term.func is not function or term.params.get("body_names") is not None:
            raise ValueError("unrecognized native-body reward predecessor")
        term.params["body_names"] = LOWER_BODIES
    replacements = {
        "measured_joint_posture_l2": (base.q10_measured_joint_position_l2, lower_posture_l2),
        "action_target_reference_l2": (base.q10_action_target_reference_l2, lower_posture_l2),
    }
    for name, (before, after) in replacements.items():
        term = cfg.rewards[name]
        if term.func is not before:
            raise ValueError("unrecognized joint-posture reward predecessor")
        term.func = after
    upper = cfg.rewards.pop("measured_upper_body_posture_l2")
    if upper.func is not base.q10_measured_upper_body_posture_l2 or upper.weight != -20:
        raise ValueError("unrecognized upper posture objective predecessor")
    from gear_sonic.trl.mjlab.sonic_task_space_ppo_runner import (
        EE_TERMINATION_BODY_NAMES,
        right_wrist_prethreshold_barrier,
        worst_ee_z_normalized_squared,
    )

    expected_barrier_params = {
        "worst_ee_z_normalized_squared": dict(
            command_name="motion", body_names=EE_TERMINATION_BODY_NAMES, normalization_m=0.25
        ),
        "right_wrist_prethreshold_barrier": dict(
            command_name="motion",
            body_name="right_wrist_roll_rubber_hand",
            onset_m=0.15,
            termination_m=0.25,
            denominator_floor=0.05,
        ),
    }
    for name, before, after, weight in (
        ("worst_ee_z_normalized_squared", worst_ee_z_normalized_squared, worst_ee_height_cost, -20),
        ("right_wrist_prethreshold_barrier", right_wrist_prethreshold_barrier, right_hand_height_barrier, -25),
    ):
        term = cfg.rewards[name]
        if term.func is not before or term.weight != weight or term.params != expected_barrier_params[name]:
            raise ValueError("unrecognized hand-height barrier predecessor")
        term.func, term.params = after, dict(spec=copy.deepcopy(spec))
    term = cfg.terminations["ee_body_pos"]
    if term.func is not base.q10_bad_body_height or term.params["threshold"] != 0.25:
        raise ValueError("original-intent training may not relax EE termination threshold")
    term.func, term.params = bad_ee_height, dict(spec=copy.deepcopy(spec), threshold=0.25)
    for name, function in (
        ("original_task_position_l2", original_task_position_l2),
        ("original_task_orientation_l2", original_task_orientation_l2),
    ):
        if name in cfg.rewards:
            raise ValueError("original task objective already installed")
        cfg.rewards[name] = RewardTermCfg(func=function, weight=-1.0, params=dict(spec=copy.deepcopy(spec)))
    return cfg
