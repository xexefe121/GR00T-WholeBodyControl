import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_buffered_source import (
    buffered_source_current_orientation,
    buffered_source_lower_body,
)
from gear_sonic.scripts.train_g1_true23_root_feedback import feedback_training_contract, validate_bounds
from gear_sonic.teleop.buffered_source_horizon import ReceivedSourceHorizon
from gear_sonic.teleop.buffered_source_simulation import BufferedSourceSimulationAdapter
from gear_sonic.tests.test_train_g1_true23_root_feedback import arguments
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES, SOURCE_IL29_JOINT_NAMES
from gear_sonic.utils.g1_true23_buffered_reference import (
    BUFFERED_TIMING,
    continued_standing_source,
    lower_horizon_cache,
    reference_profile_contract,
)
from gear_sonic.utils.g1_true23_release_compatibility import (
    release_compatibility_contract,
    validate_release_compatibility,
)
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract


def material(count=32):
    rng = np.random.default_rng(341)
    q = rng.uniform(-0.2, 0.2, (count, 23)).astype(np.float32)
    pos = rng.uniform(-0.1, 0.1, (count, 1, 3)).astype(np.float32)
    pos[:, 0, 2] = 0.76
    quat = np.zeros((count, 1, 4), np.float32)
    quat[:, :, 0] = 1
    vr = np.zeros((count, 21), np.float32)
    vr[:, [9, 13, 17]] = 1
    vr[:, :9] = rng.uniform(-0.2, 0.2, (count, 9))
    for v in (q, pos, quat, vr):
        v[-10:] = v[-1]
    motion = dict(
        joint_pos=q,
        joint_vel=np.zeros_like(q),
        body_pos_w=pos,
        body_quat_w=quat,
        body_lin_vel_w=np.zeros_like(pos),
        body_ang_vel_w=np.zeros_like(pos),
    )
    return motion, vr


def test_lower_horizon_matches_executed_original_source_properties():
    motion, _ = material()
    q29 = np.zeros((11, 29), np.float32)
    for index, name in enumerate(HARDWARE_23_JOINT_NAMES):
        q29[:, SOURCE_IL29_JOINT_NAMES.index(name)] = motion["joint_pos"][:11, index]
    dq29 = (q29[1:] - q29[:-1]) / np.float32(0.02)
    path = Path(__file__).resolve().parents[1] / "envs/manager_env/mdp/commands.py"
    tree = ast.parse(path.read_text())
    expected = []
    for component in ("pos", "vel"):
        name = f"joint_{component}_lower_body_multi_future"
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
        function.decorator_list = []
        namespace = {"torch": torch}
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"),
            namespace,
        )
        command = SimpleNamespace(
            num_envs=1,
            future_motion_ids=None,
            future_time_steps=None,
            lower_joint_isaaclab_indices=[
                SOURCE_IL29_JOINT_NAMES.index(name) for name in HARDWARE_23_JOINT_NAMES[:12]
            ],
            motion_lib=SimpleNamespace(
                get_dof_pos=lambda *_: torch.from_numpy(q29[:-1]), get_dof_vel=lambda *_: torch.from_numpy(dq29)
            ),
        )
        expected.append(namespace[name](command).numpy().reshape(-1))
    np.testing.assert_array_equal(lower_horizon_cache(motion)[0], np.concatenate(expected))


def test_every_cache_anchor_equals_received_only_buffer_including_generated_tail():
    motion, vr = material()
    original = copy.deepcopy(motion)
    source = continued_standing_source(motion, vr)
    lower = lower_horizon_cache(motion)
    buffer = ReceivedSourceHorizon()
    for i in range(len(source["joint_pos"])):
        window = buffer.push(
            source_timestamp_s=i * 0.02,
            arrival_timestamp_s=i * 0.02,
            joint_names=HARDWARE_23_JOINT_NAMES,
            joint_position23=source["joint_pos"][i],
            root_position_w=source["root_position_w"][i],
            root_quaternion_wxyz=source["root_quaternion_wxyz"][i],
            virtual_source_vr21=source["virtual_vr21"][i],
        )
        if i >= 10:
            np.testing.assert_array_equal(window.lower_body240, lower[i - 10])
    for name in motion:
        np.testing.assert_array_equal(motion[name], original[name])
    assert len(source["joint_pos"]) == len(motion["joint_pos"]) + 10


@pytest.mark.parametrize(
    "field", ("joint_pos", "body_pos_w", "body_quat_w", "joint_vel", "body_lin_vel_w", "body_ang_vel_w")
)
def test_moving_terminal_source_cannot_be_extended(field):
    motion, vr = material()
    motion[field][-2].flat[0] += 0.01
    with pytest.raises(ValueError, match="terminal"):
        continued_standing_source(motion, vr)


def test_training_cache_is_clip_isolated_and_uses_current_robot_orientation():
    left, _ = material()
    right = copy.deepcopy(left)
    right["joint_pos"] += 0.25
    motion = SimpleNamespace(
        **{key: torch.from_numpy(np.concatenate((left[key], right[key]))) for key in left}, time_step_total=64
    )
    command = SimpleNamespace(
        motion=motion,
        time_steps=torch.tensor([0, 30, 32, 62]),
        _curriculum_spans=[dict(start=0, length=32), dict(start=32, length=32)],
        robot_anchor_quat_w=torch.tensor([[1.0, 0, 0, 0], [0.70710677, 0, 0, 0.70710677]]),
        anchor_quat_w=torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    expected = np.stack(
        (
            lower_horizon_cache(left)[0],
            lower_horizon_cache(left)[30],
            lower_horizon_cache(right)[0],
            lower_horizon_cache(right)[30],
        )
    )
    np.testing.assert_array_equal(buffered_source_lower_body(env).numpy(), expected)
    np.testing.assert_allclose(
        buffered_source_current_orientation(env), [[1, 0, 0, 1, 0, 0], [0, 1, -1, 0, 0, 0]], atol=2e-7
    )


def test_runtime_never_reads_unreceived_sample_or_delays_measured_state():
    motion, vr = material()
    adapter = BufferedSourceSimulationAdapter(motion, vr)
    adapter.source["joint_pos"][20] = np.nan  # Not received until control 1.
    observed = []

    def infer(encoder, history, feedback):
        observed.append((encoder, history, feedback))
        return np.zeros(23, np.float32), np.zeros(994, np.float32)

    state = dict(
        desired_position_w=motion["body_pos_w"][10, 0],
        previous_desired_position_w=motion["body_pos_w"][9, 0],
        measured_qpos=np.r_[np.array([2, 3, 0.8, 1, 0, 0, 0]), np.zeros(23)],
        measured_qvel=np.zeros(29),
    )
    adapter.infer(
        SimpleNamespace(infer=infer),
        np.zeros(267, np.float32),
        np.zeros(930, np.float32),
        control_index=0,
        **state,
    )
    np.testing.assert_array_equal(observed[0][0][:240], lower_horizon_cache(motion)[9])
    np.testing.assert_allclose(observed[0][2][:3], motion["body_pos_w"][10, 0] - [2, 3, 0.8], atol=2e-7)
    assert adapter.timestamps == [(0.38, 0.18, 0.2)]
    with pytest.raises(ValueError, match="finite"):
        adapter.infer(
            SimpleNamespace(infer=infer),
            np.zeros(267, np.float32),
            np.zeros(930, np.float32),
            control_index=1,
            **state,
        )


def test_old_contract_hashes_unchanged_new_contract_rejects_relabelling():
    geometry = "386b1bb9ea5b69ccd6fd0283a73ffea1ee052df95564e23a780125fbcbe2c645"
    old = release_compatibility_contract(geometry, "released29_scale_bounded_linear_v2")
    assert old["contract_sha256"] == "dc12148498fafa30f409522a299042d0441d73a60f33e4299b92a150149a5334"
    new = release_compatibility_contract(geometry, "released29_scale_bounded_linear_v2", BUFFERED_TIMING)
    validate_release_compatibility(new)
    assert old != new
    assert root_feedback_contract() != root_feedback_contract(BUFFERED_TIMING)
    old["reference_timing"] = BUFFERED_TIMING
    with pytest.raises(ValueError):
        validate_release_compatibility(old)
    contract = reference_profile_contract(BUFFERED_TIMING)
    assert (
        not contract["future_samples_relative_to_emission"] and not contract["released_profile_relabel_permitted"]
    )


def test_buffered_training_requires_fresh_nominal_lifecycle_and_distinct_metadata():
    args = arguments(
        "regression",
        "--reference-timing",
        BUFFERED_TIMING,
        "--release-source-geometry",
        "source.xml",
        "--release-action-convention",
        "released29_scale_bounded_linear_v2",
        "--training-physics-profile",
        "pinned_cpu_referee_scene_v1",
        "--curriculum-stage",
        "lifecycle",
    )
    validate_bounds(args)
    contract = feedback_training_contract(args, {})
    assert contract["reference_timing"] == BUFFERED_TIMING
    assert contract["feature_contract"] == root_feedback_contract(BUFFERED_TIMING)
    for change in (
        dict(resume="old.pt"),
        dict(curriculum_stage="acquisition"),
        dict(release_source_geometry=None),
        dict(training_physics_profile="legacy_training_asset"),
    ):
        changed = copy.copy(args)
        for name, value in change.items():
            setattr(changed, name, value)
        with pytest.raises(ValueError, match="fresh lifecycle"):
            validate_bounds(changed)
    campaign = copy.copy(args)
    campaign.mode, campaign.continue_from, campaign.continuation_evaluation = (
        "campaign",
        "parent.pt",
        "report.json",
    )
    validate_bounds(campaign)


def replay_trace():
    motion, vr = material()
    adapter = BufferedSourceSimulationAdapter(motion, vr)
    qpos = np.r_[[0.3, 0, 0.76, 1, 0, 0, 0], np.zeros(23)]
    qvel = np.zeros(29)
    count = len(motion["joint_pos"]) - 11
    policy = SimpleNamespace(infer=lambda *_: (np.zeros(23, np.float32), np.zeros(994, np.float32)))
    for frame in range(count):
        adapter.infer(
            policy,
            np.zeros(267, np.float32),
            np.zeros(930, np.float32),
            control_index=frame,
            desired_position_w=motion["body_pos_w"][10 + frame, 0],
            previous_desired_position_w=motion["body_pos_w"][9 + frame, 0],
            measured_qpos=qpos,
            measured_qvel=qvel,
        )
    trace = dict(
        qpos=np.repeat(qpos[None], count + 1, axis=0),
        qvel=np.repeat(qvel[None], count + 1, axis=0),
        actual_policy_encoder267=np.asarray(adapter.actual_encoder_inputs),
        source_emission_anchor_setpoint_timestamps_s=np.asarray(adapter.timestamps),
        root_feedback9=adapter.root_adapter.arrays()["root_feedback9"],
    )
    return adapter.source, trace


def test_full_saved_stream_reverification():
    from gear_sonic.utils.g1_true23_root_feedback_campaign import validate_buffered_replay_arrays

    source, trace = replay_trace()
    result = validate_buffered_replay_arrays(source, trace)
    assert result["controls_verified"] == 21
    assert result["maximum_encoder_abs_error"] == 0
    assert result["root_feedback_bit_exact"]


@pytest.mark.parametrize(
    "field", ("qpos", "root_feedback9", "actual_policy_encoder267", "source_emission_anchor_setpoint_timestamps_s")
)
def test_saved_stream_mismatch_cannot_continue(field):
    from gear_sonic.utils.g1_true23_root_feedback_campaign import validate_buffered_replay_arrays

    source, trace = replay_trace()
    trace[field][0, 0] += 0.001
    with pytest.raises(ValueError, match="buffered replay"):
        validate_buffered_replay_arrays(source, trace)
