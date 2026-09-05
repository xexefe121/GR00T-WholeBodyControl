from dataclasses import replace
import hashlib
from pathlib import Path
import re

import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_stage_one_actuation import (
    native_support_pd_step,
    requested_stage_one_target,
    stage_one_pd_step,
)
from gear_sonic.scripts.evaluate_g1_true23_deployment_envelope import apply_target_envelope
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_actuation_profile import (
    HEADER,
    SIM_CONFIG,
    NativeSupportActuationProfile,
    StageOneActuationProfile,
    read_joint_amplitude_scale,
)

ROOT = Path(__file__).resolve().parents[2]


def test_native_support_profile_is_hash_bound_simulation_only():
    profile = NativeSupportActuationProfile.from_sim_config(ROOT / SIM_CONFIG)
    assert profile.source_sha256 == hashlib.sha256((ROOT / SIM_CONFIG).read_bytes()).hexdigest()
    assert profile.kp[3] == pytest.approx(99.098427782)
    assert profile.fraction == 1.0 and profile.slew_rad_s == 5.0
    assert all(profile.effort[index] <= 35 for index in (4, 5, 10, 11))
    contract = profile.contract()
    assert contract["effort_target_projection"] is True
    assert not contract["gain_review_for_hardware_complete"] and not contract["hardware_authorized"]


def test_native_support_torch_matches_numpy_effort_projection_on_valid_rows():
    from gear_sonic.utils.g1_true23_sim_acquisition import effort_feasible_target

    profile = NativeSupportActuationProfile.from_sim_config(ROOT / SIM_CONFIG)
    generator = np.random.default_rng(2301)
    q = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE) + generator.normal(0, 0.01, (100, 23))
    dq = generator.normal(0, 0.1, (100, 23))
    previous = q.copy()
    requested = q + generator.normal(0, 0.1, (100, 23))
    target, torque, invalid = native_support_pd_step(*map(torch.tensor, (requested, previous, q, dq)), profile)
    assert not invalid.any()
    for index in range(100):
        expected = effort_feasible_target(
            requested[index],
            previous[index],
            q[index],
            dq[index],
            np.asarray(profile.kp),
            np.asarray(profile.kd),
            np.asarray(profile.effort),
            dt=0.002,
            slew_rate=5.0,
        )
        np.testing.assert_allclose(target[index].numpy(), expected, rtol=0, atol=1e-15)
        predicted = np.asarray(profile.kp) * (expected - q[index]) - np.asarray(profile.kd) * dq[index]
        np.testing.assert_allclose(torque[index].numpy(), predicted, rtol=0, atol=1e-13)
    assert np.all(np.abs(torque.numpy()) <= 0.95 * 0.25 * np.asarray(profile.effort) + 1e-12)


def test_native_support_infeasible_row_fails_without_poisoning_other_envs():
    profile = NativeSupportActuationProfile.from_sim_config(ROOT / SIM_CONFIG)
    q = torch.tensor(SAFE_TARGET_DEFAULT_Q_HARDWARE)[None].repeat(2, 1)
    dq = torch.zeros_like(q)
    dq[0] = 1000.0
    target, torque, invalid = native_support_pd_step(q, q, q, dq, profile)
    assert invalid.tolist() == [True, False]
    assert torch.count_nonzero(torque) == 0
    torch.testing.assert_close(target, q)


def test_training_rejects_raw_clip_boundary_as_runtime_does(profile):
    from types import SimpleNamespace

    from gear_sonic.envs.mjlab.sonic_true23 import _MJLAB_IMPORT_ERROR

    if _MJLAB_IMPORT_ERROR is not None:
        pytest.skip("MJLab unavailable")
    from gear_sonic.envs.mjlab.sonic_true23_stage_one_actuation import StageOneActuationAction

    action = StageOneActuationAction.__new__(StageOneActuationAction)
    action.cfg = SimpleNamespace(profile=profile, record_requested_projection=False)
    action._raw_actions = torch.zeros(2, 23)
    action._safe_native_actions = torch.zeros(2, 23)
    action._requested_targets = torch.zeros(2, 23)
    action.envelope_violation = torch.zeros(2, dtype=torch.bool)
    raw = torch.zeros(2, 23)
    raw[0, 0] = 10.0
    action.process_actions(raw)
    assert action.envelope_violation.tolist() == [True, False]


@pytest.fixture
def profile():
    return StageOneActuationProfile.from_cpp(ROOT / HEADER)


def test_profile_binds_current_cpp_bytes_without_authorizing_robot(profile):
    assert profile.source_sha256 == hashlib.sha256((ROOT / HEADER).read_bytes()).hexdigest()
    assert profile.kp[3] == 16.0 and profile.kp[9] == 16.0
    assert profile.kp[3] * profile.hold_kp_fraction == 4.0
    source = (ROOT / HEADER).read_text()
    assert profile.fraction == float(re.search(r"kStageOneActionFraction = ([0-9.]+);", source)[1])
    assert profile.slew_rad_s == float(re.search(r"kStageOneTargetRateRadPerSecond = ([0-9.]+);", source)[1])
    contract = profile.contract()
    assert contract["hardware_motor_slots"] == list(range(13)) + list(range(15, 20)) + list(range(22, 27))
    assert not contract["hardware_authorized"] and not contract["deployment_ready"]


def test_historical_pre_taper_profile_uses_identity_but_malformed_taper_fails():
    assert read_joint_amplitude_scale("// pre-taper implementation") == (1.0,) * 23
    with pytest.raises(ValueError, match=r"missing C\+\+ array"):
        read_joint_amplitude_scale("raw_target *= kStageOneJointAmplitudeScale[compact];")


def test_changed_cpp_effort_guard_is_not_silently_misrepresented(tmp_path):
    source = (ROOT / HEADER).read_text()
    changed = re.sub(r"0\.25\s*\*\s*kHardwareEffortLimitNm", "0.30 * kHardwareEffortLimitNm", source)
    assert changed != source
    path = tmp_path / "controller.hpp"
    path.write_text(changed)
    with pytest.raises(ValueError, match=r"unsupported C\+\+ predicted-effort"):
        StageOneActuationProfile.from_cpp(path)


@pytest.mark.parametrize(
    "change",
    [
        {"kp": (1.0,) * 22},
        {"kd": (float("nan"),) * 23},
        {"effort": (-1.0,) * 23},
        {"fraction": 1.01},
        {"timestep_s": 0.02},
        {"joint_scale": (2.0,) * 23},
        {"source_sha256": "missing"},
    ],
)
def test_invalid_profile_rejected(profile, change):
    with pytest.raises(ValueError):
        replace(profile, **change)


def test_torch_substeps_match_numpy_deployment_loop(profile):
    generator = np.random.default_rng(23)
    previous = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)
    state_q = previous + generator.normal(0.0, 0.03, 23)
    dq = generator.normal(0.0, 0.05, 23)
    for _ in range(50):
        full = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE) + generator.normal(0.0, 0.1, 23)
        requested = requested_stage_one_target(torch.tensor(full)[None], profile)
        for _ in range(10):
            expected, _ = apply_target_envelope(
                full,
                previous,
                fraction=profile.fraction,
                joint_scale=np.asarray(profile.joint_scale),
                slew_rate=profile.slew_rad_s,
                dt=profile.timestep_s,
            )
            target, torque, guard = stage_one_pd_step(
                requested,
                torch.tensor(previous)[None],
                torch.tensor(state_q)[None],
                torch.tensor(dq)[None],
                profile,
            )
            np.testing.assert_allclose(target[0].numpy(), expected, rtol=0.0, atol=1e-15)
            predicted = np.asarray(profile.kp) * (expected - state_q) - np.asarray(profile.kd) * dq
            np.testing.assert_allclose(
                torque[0].numpy(),
                np.clip(predicted, -np.asarray(profile.effort), profile.effort),
                rtol=0.0,
                atol=1e-13,
            )
            assert guard.shape == (1,)
            previous = expected


def test_predicted_effort_guard_does_not_hide_saturation(profile):
    q = torch.tensor(SAFE_TARGET_DEFAULT_Q_HARDWARE)[None]
    previous = q.clone()
    _, torque, guard = stage_one_pd_step(q, previous, q, torch.full_like(q, 1000), profile)
    assert guard.item()
    assert torch.all(torque.abs() <= torque.new_tensor(profile.effort))


def test_profiled_config_is_isolated_and_every_motor_is_explicit(profile):
    from gear_sonic.envs.mjlab.sonic_true23 import _MJLAB_IMPORT_ERROR

    if _MJLAB_IMPORT_ERROR is not None:
        pytest.skip("MJLab/Unitree package unavailable")
    from mjlab.actuator import BuiltinMotorActuatorCfg, BuiltinPositionActuatorCfg

    from gear_sonic.envs.mjlab.sonic_true23_causal_multimotion_v14 import make_causal_multimotion_v14_env_cfg
    from gear_sonic.envs.mjlab.sonic_true23_stage_one_actuation import apply_stage_one_actuation_profile
    from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES

    original = make_causal_multimotion_v14_env_cfg(motion_file="test.npz", num_envs=2, play=True)
    updated = apply_stage_one_actuation_profile(original, profile)
    assert all(
        isinstance(actuator, BuiltinPositionActuatorCfg)
        for actuator in original.scene.entities["robot"].articulation.actuators
    )
    actuators = updated.scene.entities["robot"].articulation.actuators
    assert len(actuators) == 23 and all(isinstance(actuator, BuiltinMotorActuatorCfg) for actuator in actuators)
    assert tuple(actuator.target_names_expr[0] for actuator in actuators) == HARDWARE_23_JOINT_NAMES
    assert tuple(actuator.effort_limit for actuator in actuators) == profile.effort
    assert updated.sim.mujoco.integrator == "euler" and updated.decimation == 10
    assert "stage_one_actuation_guard" in updated.terminations
    assert "stage_one_actuation_guard" not in original.terminations
    spec = updated.scene.entities["robot"].spec_fn()
    for name, effort in zip(HARDWARE_23_JOINT_NAMES, profile.effort, strict=True):
        np.testing.assert_array_equal(spec.joint(name).actfrcrange, [-effort, effort])
