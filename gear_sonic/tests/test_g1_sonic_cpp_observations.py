from collections import deque
import copy
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from gear_sonic.scripts import simulate_g1_sonic_library_motions as legacy
from gear_sonic.scripts.record_g1_sonic_cpp_observation_replay import validate_baseline_rows
from gear_sonic.utils.g1_sonic_cpp_observation_trace import (
    PROFILE,
    audit_engine_trace,
    same_state_observation_comparison,
    trace_cpp_observation_motion,
)
from gear_sonic.utils.g1_sonic_cpp_observations import (
    CppObservations,
    capture_cpp_observations,
    cpp_method,
    validate_observation_layout,
)
from gear_sonic.utils.g1_sonic_cpp_parameters import capture_cpp_parameters, file_sha256
from gear_sonic.utils.g1_sonic_original29_trace import trace_original29_motion
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref"


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cpp_observation_tests")
    capture = capture_cpp_observations(ROOT, directory / "capture")
    params, _ = capture_cpp_parameters(DEPLOY / "include/policy_parameters.hpp", directory / "parameters")
    return capture, params


def reference():
    rng = np.random.default_rng(901)
    quaternions = Rotation.from_rotvec(rng.normal(0, 0.1, (60, 3))).as_quat()[:, [3, 0, 1, 2]]
    return quaternions, rng.normal(0, 0.2, (60, 29)), rng.normal(0, 0.2, (60, 29))


def test_captured_cpp_zero_quaternion_padding_is_not_zero_gravity(captured):
    capture, params = captured
    quaternions, joints, velocity = reference()
    pose = np.r_[0, 0, 0.8, 1, 0, 0, 0, params.default_angles]
    with CppObservations(capture, quaternions, joints, velocity, pose[3:7]) as observation:
        observation.append(pose, np.zeros(35), np.zeros(29))
        _, history = observation.gather(0)
    expected = np.tile([0, 0, 1], (10, 1))
    expected[-1] = [0, 0, -1]
    np.testing.assert_array_equal(history[900:].reshape(10, 3), expected)
    np.testing.assert_array_equal(history[0:30], np.zeros(30))
    assert not np.array_equal(legacy._zero_history_frame()["gravity"], history[900:903])
    assert capture["sdk_or_deployment_application_built"] is False
    assert capture["csv_logging_enabled"] is False
    assert all(file_sha256(path) == digest for path, digest in capture["inputs"].items())


def test_full_history_order_float_sensor_boundary_and_all_encoder_fields(captured):
    capture, params = captured
    quaternions, joints, velocities = reference()
    initial = quaternions[0]
    expected = deque(maxlen=10)
    rng = np.random.default_rng(92)
    with CppObservations(capture, quaternions, joints, velocities, initial) as observation:
        for frame in range(22):
            pose = np.r_[0, 0, 0.8, quaternions[frame], params.default_angles + rng.normal(0, 0.3, 29)]
            velocity = rng.normal(0, 0.2, 35)
            action = rng.normal(0, 0.2, 29)
            observation.append(pose, velocity, action)
            encoder, history = observation.gather(frame)
            quantized = SimpleNamespace(
                qpos=pose.astype(np.float32).astype(float), qvel=velocity.astype(np.float32).astype(float)
            )
            entry = legacy._current_history_frame(quantized, action.astype(np.float32).astype(float))
            entry["joint_position"] = (quantized.qpos[7:] - params.default_angles)[legacy.ISAAC_TO_MUJOCO_INDEX]
            expected.append(entry)
            if frame >= 9:
                np.testing.assert_allclose(history, legacy._policy_observation(expected), atol=3e-7, rtol=2e-6)
            heading = legacy._quat_multiply(
                legacy._heading_quaternion(initial.astype(np.float32).astype(float)),
                legacy._quat_conjugate(legacy._heading_quaternion(initial)),
            )
            python_encoder = legacy._encoder_observation(
                quaternions, joints, velocities, frame, quantized.qpos[3:7], heading
            )
            np.testing.assert_allclose(encoder, python_encoder, atol=3e-7, rtol=2e-6)
    assert history.shape == (930,) and encoder.shape == (1762,)


def test_paused_pose_repeats_zero_velocity_and_last_future_frame_clamps(captured):
    capture, params = captured
    quaternions, joints, velocity = reference()
    pose = np.r_[0, 0, 0.8, quaternions[0], params.default_angles]
    with CppObservations(capture, quaternions, joints, velocity, pose[3:7]) as observation:
        observation.append(pose, np.zeros(35), np.zeros(29))
        paused, _ = observation.gather(15, play=False)
        np.testing.assert_array_equal(
            paused[4:294].reshape(10, 29), np.tile(joints[15].astype(np.float32), (10, 1))
        )
        np.testing.assert_array_equal(paused[294:584], np.zeros(290))
        np.testing.assert_array_equal(paused[601:661].reshape(10, 6), np.tile(paused[601:607], (10, 1)))
        last, _ = observation.gather(59)
        np.testing.assert_array_equal(last[4:294].reshape(10, 29), np.tile(joints[-1].astype(np.float32), (10, 1)))
        np.testing.assert_array_equal(
            last[294:584].reshape(10, 29), np.tile(velocity[-1].astype(np.float32), (10, 1))
        )


def test_handles_and_input_admission_do_not_treat_invalid_state_as_padding(captured):
    capture, params = captured
    quaternions, joints, velocity = reference()
    pose = np.r_[0, 0, 0.8, quaternions[0], params.default_angles]
    with CppObservations(capture, quaternions, joints, velocity, pose[3:7]) as observation:
        with pytest.raises(RuntimeError, match="missing state"):
            observation.gather(0)
        for bad in (np.zeros(36), np.full(36, np.nan), np.full(36, 1e300), np.zeros(35)):
            with pytest.raises(ValueError):
                observation.append(bad, np.zeros(35), np.zeros(29))
        observation.append(pose, np.zeros(35), np.zeros(29))
        for bad in (-1, 60, True):
            with pytest.raises(ValueError):
                observation.gather(bad)
        with pytest.raises(ValueError):
            observation.gather(0, play=1)
    observation.close()
    with pytest.raises(ValueError, match="closed"):
        observation.append(pose, np.zeros(35), np.zeros(29))


@pytest.mark.parametrize("damage", ["promoted", "hash", "quaternion", "shape"])
def test_changed_or_mislabelled_capture_or_reference_is_rejected(captured, damage):
    capture, _ = captured
    capture = copy.deepcopy(capture)
    q, joint, velocity = reference()
    if damage == "promoted":
        capture["deployment_ready"] = True
    elif damage == "hash":
        capture["binary_sha256"] = "wrong"
    elif damage == "quaternion":
        q[0] = 0
    else:
        joint = joint[:-1]
    with pytest.raises(ValueError):
        CppObservations(capture, q, joint, velocity, [1, 0, 0, 0])


def test_complete_method_extraction_handles_defaults_comments_and_strings():
    method = (
        '    bool Example(std::vector<int> values = {}) { /* } */\n const char* x = "}"; // }\n return true; }'
    )
    assert cpp_method(method + "\n    bool Other() { return false; }", "Example") == method
    for source in (
        method + "\n" + method,
        "    bool Example() { return true;",
        "    bool Example();\n    bool Other() {}",
        '    bool Example() { auto a = R"x({})x"; }',
    ):
        with pytest.raises(ValueError):
            cpp_method(source, "Example")


@pytest.mark.parametrize("damage", ["history_order", "encoder_mode", "encoder_field", "dimension"])
def test_changed_yaml_cannot_silently_reuse_witness_offsets(damage):
    config = yaml.safe_load((ROOT / "gear_sonic_deploy/policy/release/observation_config.yaml").read_text())
    source = (DEPLOY / "src/g1_deploy_onnx_ref.cpp").read_text()
    assert validate_observation_layout(source, config)["encoder_size"] == 1762
    if damage == "history_order":
        config["observations"].reverse()
    elif damage == "encoder_mode":
        config["encoder"]["encoder_modes"][0]["required_observations"].pop()
    elif damage == "encoder_field":
        config["encoder"]["encoder_observations"][0]["enabled"] = False
    else:
        config["encoder"]["dimension"] = 32
    with pytest.raises(ValueError):
        validate_observation_layout(source, config)


def test_cpp_observation_trace_preserves_pd_and_records_actual_engine_time(captured):
    capture, parameters = captured
    model = mujoco.MjModel.from_xml_path(str(ROOT / "gear_sonic_deploy/g1/scene_29dof.xml"))
    model.opt.timestep = 0.002
    raw = np.tile(np.r_[0, 0, 0.78, 1, 0, 0, 0, legacy.DEFAULT_ANGLES], (9, 1))

    class Encoder:
        def run(self, names, feeds):
            return [np.zeros((1, 64), dtype=np.float32)]

    class Decoder:
        def run(self, names, feeds):
            return [np.full((1, 29), 0.01, dtype=np.float32)]

    original_function = legacy._current_history_frame
    model_hash = compiled_model_sha256(model)
    old, _ = trace_original29_motion(model, Encoder(), Decoder(), raw, parameters=parameters)
    arrays, report = trace_cpp_observation_motion(
        model, Encoder(), Decoder(), raw, parameters=parameters, capture=capture
    )
    for key in (
        "qpos",
        "pre_qpos",
        "physics_post_qvel",
        "target_positions_hardware29",
        "physics_requested_torque_hardware29",
    ):
        np.testing.assert_array_equal(arrays[key], old[key])
    np.testing.assert_array_equal(arrays["decoder_inputs"][0, 964:991].reshape(9, 3), np.tile([0, 0, 1], (9, 1)))
    assert report["profile"] == PROFILE
    assert report["complete_original_clip"] and report["actual_engine_audit"]["passed"]
    assert len(arrays["physics_engine_post_time_s"]) == 150
    assert arrays["physics_engine_post_time_s"][-1] == pytest.approx(0.3, abs=1e-12)
    assert arrays["physics_engine_warning_counts"].shape == (150, 8)
    assert report["actual_cpp_observation_gatherers_executed"] is True
    assert report["complete_cpp_deployment_equivalence_proven"] is False
    assert report["live_receding_horizon_planner_executed"] is False
    assert legacy._current_history_frame is original_function and compiled_model_sha256(model) == model_hash
    for flag in ("teacher_accepted", "deployment_ready", "hardware_authorized"):
        assert report[flag] is False
    same, comparison = same_state_observation_comparison(raw, old, Encoder(), Decoder(), capture=capture)
    assert comparison["inference_calls_compared"] == 15
    assert comparison["rejected_final_baseline_inference_retained"] is False
    assert comparison["counterfactual_actions_executed"] is False
    assert comparison["history_segment_differences"]["gravity"]["maximum_initial_nine_calls"] == 1
    assert comparison["history_segment_differences"]["gravity"]["maximum_after_nine_calls"] < 3e-7
    np.testing.assert_array_equal(same["decoder_outputs"], old["decoder_outputs"])
    partial = copy.deepcopy(old)
    for key in ("pre_qpos", "qpos", "pre_qvel", "post_qvel", "raw_actions"):
        partial[key] = partial[key][:12].copy()
    for key in ("encoder_inputs", "encoder_outputs", "decoder_inputs", "decoder_outputs"):
        partial[key] = partial[key][:13].copy()
    partial["decoder_outputs"][-1, -1] = 10.5
    saved = copy.deepcopy(partial)
    same, comparison = same_state_observation_comparison(raw, partial, Encoder(), Decoder(), capture=capture)
    assert comparison["inference_calls_compared"] == len(same["decoder_inputs"]) == 13
    assert comparison["completed_baseline_control_steps"] == 12
    assert comparison["rejected_final_baseline_inference_retained"] is True
    assert comparison["differences"]["decoder_outputs"]["maximum_absolute_difference"] > 10
    for key in partial:
        np.testing.assert_array_equal(saved[key], partial[key])
    for key in ("planned_qpos50", "pre_qpos", "pre_qvel", "raw_actions", "encoder_outputs"):
        damaged = copy.deepcopy(partial)
        damaged[key][1, 0] += 0.01
        with pytest.raises(ValueError, match="lineage changed"):
            same_state_observation_comparison(raw, damaged, Encoder(), Decoder(), capture=capture)
    partial["encoder_inputs"] = old["encoder_inputs"]
    with pytest.raises(ValueError, match="trim or invent"):
        same_state_observation_comparison(raw, partial, Encoder(), Decoder(), capture=capture)


@pytest.mark.parametrize("damage", ["time_reset", "warning", "missing_step", "wrong_dt"])
def test_engine_reset_or_warning_cannot_be_reported_as_success(damage):
    arrays = {
        "physics_dt": np.array([0.002]),
        "physics_engine_pre_time_s": np.arange(30) * 0.002,
        "physics_engine_post_time_s": np.arange(1, 31) * 0.002,
        "physics_engine_warning_counts": np.zeros((30, 8), dtype=np.int64),
    }
    assert audit_engine_trace(arrays)["passed"]
    if damage == "time_reset":
        arrays["physics_engine_post_time_s"][15:] -= 0.02
    elif damage == "warning":
        arrays["physics_engine_warning_counts"][12:, 6] = 1
    elif damage == "missing_step":
        arrays["physics_engine_pre_time_s"] = arrays["physics_engine_pre_time_s"][:-1]
    else:
        arrays["physics_dt"][0] = 0
    if damage in ("missing_step", "wrong_dt"):
        with pytest.raises(ValueError):
            audit_engine_trace(arrays)
    else:
        assert not audit_engine_trace(arrays)["passed"]


@pytest.mark.parametrize("damage", ["kind", "promoted", "missing", "duplicate", "order", "trace", "failure"])
def test_full_source_comparison_never_filters_or_relabels_baseline_failures(damage):
    report = {
        "kind": "g1_sonic_original29_recorded_comparison_v1",
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
        "records": [
            {
                "name": name,
                "profile": "cpp_parameters_and_float32_targets",
                "failure": None,
                "trace_path": "record.npz",
                "trace_sha256": "hash",
            }
            for name in ("hand_crawling", "elbow_crawling", "happy_dance")
        ],
    }
    assert len(validate_baseline_rows(report)) == 3
    collision = copy.deepcopy(report)
    collision["kind"] = "g1_sonic_original29_full_hand_collision_recorded_diagnostic_v1"
    assert len(validate_baseline_rows(collision, hand_collisions=True)) == 3
    if damage == "kind":
        report["kind"] = collision["kind"]
    elif damage == "promoted":
        report["teacher_accepted"] = True
    elif damage == "missing":
        report["records"].pop(1)
    elif damage == "duplicate":
        report["records"][1] = report["records"][0]
    elif damage == "order":
        report["records"].reverse()
    elif damage == "trace":
        report["records"][1]["trace_path"] = None
    else:
        report["records"][1]["failure"] = "missing attempt"
    with pytest.raises(ValueError):
        validate_baseline_rows(report)
