from dataclasses import replace
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_stage_one_actuation import (
    requested_stage_one_target,
    stage_one_pd_step,
)
from gear_sonic.scripts.evaluate_g1_true23_deployment_envelope import apply_target_envelope
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_actuation_profile import (
    HEADER,
    StageOneActuationProfile,
    read_joint_amplitude_scale,
)


ROOT = Path(__file__).resolve().parents[2]


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
