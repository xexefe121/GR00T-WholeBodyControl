from pathlib import Path

import mujoco
import numpy as np
import pytest

from gear_sonic.scripts import simulate_g1_sonic_library_motions as stock
from gear_sonic.utils.g1_sonic_cpp_parameters import capture_cpp_parameters
from gear_sonic.utils.g1_sonic_original29_trace import trace_original29_motion
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


class Encoder:
    def run(self, names, values):
        assert names == ["encoded_tokens"] and values["obs_dict"].shape == (1, 1762)
        return [np.zeros((1, 64), dtype=np.float32)]


class Decoder:
    def run(self, names, values):
        assert names == ["action"] and values["obs_dict"].shape == (1, 994)
        return [np.full((1, 29), 0.01, dtype=np.float32)]


@pytest.fixture
def material():
    path = Path(__file__).resolve().parents[2] / "gear_sonic_deploy/g1/scene_29dof.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.timestep = 0.002
    planner = np.tile(np.r_[[0, 0, 0.78, 1, 0, 0, 0], stock.DEFAULT_ANGLES], (9, 1))
    return model, planner


def test_trace_is_exact_original_execution_without_shared_patching(material):
    model, planner = material
    model_hash, module = compiled_model_sha256(model), stock.mujoco
    expected_diagnostics = {}
    expected, expected_metrics = stock.simulate_motion(model, Encoder(), Decoder(), planner, expected_diagnostics)
    arrays, report = trace_original29_motion(model, Encoder(), Decoder(), planner)
    np.testing.assert_array_equal(arrays["qpos"], expected)
    for name, values in expected_diagnostics.items():
        np.testing.assert_array_equal(arrays[name], values)
    assert report["stock_metrics"] == expected_metrics
    assert report["complete_original_clip"]
    assert stock.mujoco is module and compiled_model_sha256(model) == model_hash
    assert report["physics_steps_recorded"] == len(expected) * 10
    np.testing.assert_array_equal(arrays["pre_qpos"][1:], arrays["qpos"][:-1])
    np.testing.assert_array_equal(arrays["qpos"][-1], arrays["physics_post_qpos"][-1])
    assert arrays["post_control_time_s"][-1] == pytest.approx(len(expected) * 0.02)
    assert arrays["encoder_inputs"].shape == (len(expected), 1762)
    assert arrays["decoder_inputs"].shape == (len(expected), 994)
    assert arrays["physics_actual_actuated_generalized_force"].shape == (10 * len(expected), 29)
    assert all(report[key] is False for key in ("teacher_accepted", "hardware_authorized", "deployment_ready"))


@pytest.mark.parametrize("change", ["short", "nan", "nonunit", "wrong_dt"])
def test_bad_source_or_timestep_rejected(material, change):
    model, planner = material
    if change == "short":
        planner = planner[:1]
    elif change == "nan":
        planner[3, 8] = np.nan
    elif change == "nonunit":
        planner[3, 3] = 0.5
    else:
        model.opt.timestep = 0.005
    with pytest.raises(ValueError):
        trace_original29_motion(model, Encoder(), Decoder(), planner)


def test_cpp_variant_applies_exact_command_and_pd_every_substep(material, tmp_path):
    model, planner = material
    root = Path(__file__).resolve().parents[2]
    parameters, _ = capture_cpp_parameters(
        root / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp", tmp_path / "cpp"
    )
    old = {name: getattr(stock, name).copy() for name in ("KPS", "KDS", "DEFAULT_ANGLES", "ACTION_SCALE")}
    arrays, report = trace_original29_motion(model, Encoder(), Decoder(), planner, parameters=parameters)
    target = parameters.target(np.full(29, 0.01, dtype=np.float32))
    np.testing.assert_array_equal(arrays["target_positions_hardware29"], np.tile(target, (len(arrays["qpos"]), 1)))
    requested = np.clip(
        parameters.kps * (target - arrays["physics_pre_qpos"][:, 7:])
        - parameters.kds * arrays["physics_pre_qvel"][:, 6:],
        -parameters.effort,
        parameters.effort,
    )
    np.testing.assert_array_equal(arrays["physics_requested_torque_hardware29"], requested)
    for name, before in old.items():
        np.testing.assert_array_equal(getattr(stock, name), before)
    assert report["profile"] == "cpp_parameters_and_float32_targets"
    assert not report["stock_simulator_bytecode_executed_without_control_law_changes"]
    assert not report["complete_cpp_deployment_equivalence_proven"]
    assert not report["firmware_torque_clipping_equivalence_proven"]
    assert report["complete_original_clip"]
