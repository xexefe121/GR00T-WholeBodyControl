"""Selected-native observation/action semantics, independent of SONIC labels."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    HOME_Q_HARDWARE,
    ACTION_SCALE_HARDWARE,
    Native124Checkpoint21204Policy,
    load_checkpoint21204_binding,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
    safe_target_transform_numpy,
)
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL
from gear_sonic.utils.g1_true23_native124_pico_comparison import (
    Native124PicoComparator,
    relative_rotation6,
    target_to_existing_envelope,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent / "GR00T-WholeBodyControl"


def test_actual_readonly_artifact_binding_and_selected_onnx():
    binding = load_checkpoint21204_binding(ASSETS)
    policy = Native124Checkpoint21204Policy(binding)
    observation = np.zeros((1, 124), np.float32)
    action = policy.run(observation)
    assert action.shape == (23,) and action.dtype == np.float32
    np.testing.assert_array_equal(action, policy.run(observation))


@pytest.fixture
def motion():
    model = mujoco.MjModel.from_xml_path(str(ASSETS / MODEL))
    data = mujoco.MjData(model)
    poses = np.tile(np.r_[0, 0, 0.76, 1, 0, 0, 0, SAFE_TARGET_DEFAULT_Q_HARDWARE], (30, 1))
    poses[:, 7 + 12] = np.arange(30) * 0.003
    poses[:, 7 + 2] = np.arange(30) * 0.001
    positions, quaternions = [], []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos[1:].copy())
        quaternions.append(data.xquat[1:].copy())
    return dict(
        fps=np.array([50.0]),
        joint_pos=poses[:, 7:],
        joint_vel=np.gradient(poses[:, 7:], 0.02, axis=0),
        body_pos_w=np.asarray(positions),
        body_quat_w=np.asarray(quaternions),
        body_lin_vel_w=np.zeros((30, 24, 3)),
        body_ang_vel_w=np.zeros((30, 24, 3)),
    )


class Policy:
    def __init__(self, output=None):
        self.output = np.zeros(23, np.float32) if output is None else output

    def run(self, observation):
        self.observation = observation.copy()
        return self.output.copy()


def state(motion, frame=10):
    return dict(
        measured_qpos=np.r_[
            motion["body_pos_w"][frame, 0], motion["body_quat_w"][frame, 0], motion["joint_pos"][frame]
        ],
        measured_qvel=np.linspace(-0.1, 0.1, 29),
    )


@pytest.mark.parametrize("phase,anchor", (("causal_q9", 9), ("current_q10", 10)))
def test_observation_exact_hardware_order_and_causal_phase(motion, phase, anchor):
    adapter = Native124PicoComparator(motion, phase=phase, root=ROOT, assets=ASSETS)
    adapter.preview.filter = lambda raw, q, v: raw.copy()
    history = np.arange(930, dtype=np.float32) / 100
    policy, measured = Policy(), state(motion)
    before = {k: v.copy() for k, v in measured.items()}
    raw, stub = adapter.infer(policy, np.zeros(267), history, control_index=0, **measured)
    obs = policy.observation[0]
    np.testing.assert_array_equal(obs[:23], motion["joint_pos"][anchor].astype(np.float32))
    np.testing.assert_array_equal(
        obs[23:46],
        (motion["joint_pos"][10].astype(np.float32) - motion["joint_pos"][9].astype(np.float32))
        / np.float32(0.02),
    )
    np.testing.assert_allclose(obs[46:52], [1, 0, 0, 1, 0, 0], atol=1e-7)
    np.testing.assert_array_equal(obs[52:55], history[27:30])
    np.testing.assert_array_equal(obs[55:78], measured["measured_qpos"][7:].astype(np.float32) - HOME_Q_HARDWARE)
    np.testing.assert_array_equal(obs[78:101], measured["measured_qvel"][6:].astype(np.float32))
    expected_previous = (
        np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE, np.float32) - HOME_Q_HARDWARE
    ) / ACTION_SCALE_HARDWARE
    np.testing.assert_array_equal(obs[101:], expected_previous)
    for k, v in before.items():
        np.testing.assert_array_equal(measured[k], v)
    np.testing.assert_array_equal(stub[:64], np.zeros(64, np.float32))
    np.testing.assert_array_equal(stub[64:], history)
    _, target = safe_target_transform_numpy(raw)
    np.testing.assert_allclose(target, HOME_Q_HARDWARE, atol=1e-7, rtol=0)
    adapter.infer(policy, np.zeros(267), history, control_index=1, **state(motion, 11))
    np.testing.assert_array_equal(policy.observation[0, 101:], (target - HOME_Q_HARDWARE) / ACTION_SCALE_HARDWARE)
    assert adapter.contract()["policy_is_sonic"] is False
    assert adapter.contract()["teacher_label_admitted"] is False


def test_torso_q9_is_actual_previous_measurement_not_reference(motion):
    adapter = Native124PicoComparator(motion, phase="causal_q9", root=ROOT, assets=ASSETS)
    adapter.preview.filter = lambda raw, q, v: raw.copy()
    policy, measured = Policy(), state(motion)
    measured["measured_qpos"][19] += 0.4
    adapter.infer(policy, np.zeros(267), np.zeros(930, np.float32), control_index=0, **measured)
    adapter.infer(policy, np.zeros(267), np.zeros(930, np.float32), control_index=1, **state(motion, 11))
    expected = Rotation.from_euler("z", -0.4).as_matrix()[:, :2].reshape(6)
    np.testing.assert_allclose(policy.observation[0, 46:52], expected, atol=1e-7)


def test_relative_rotation_matches_independent_scipy():
    r = Rotation.from_rotvec([0.25, -0.17, 0.4])
    t = Rotation.from_rotvec([-0.1, 0.4, -0.15])
    actual = relative_rotation6(r.as_quat()[[3, 0, 1, 2]], t.as_quat()[[3, 0, 1, 2]])
    np.testing.assert_allclose(actual, (r.inv() * t).as_matrix()[:, :2].reshape(6), atol=6e-8, rtol=0)


def test_envelope_preserves_interior_hardware_targets():
    for alpha in (-0.2, 0, 0.2):
        target = HOME_Q_HARDWARE + np.float32(alpha) * ACTION_SCALE_HARDWARE
        raw, projection = target_to_existing_envelope(target)
        _, actual = safe_target_transform_numpy(raw)
        np.testing.assert_allclose(actual, target, atol=2e-7, rtol=0)
        np.testing.assert_allclose(projection, np.zeros(23, np.float32), atol=1e-15, rtol=0)


@pytest.mark.parametrize("output", (np.full(23, np.nan, np.float32), np.full(23, 10, np.float32)))
def test_invalid_actor_latches_rejection_and_preserves_attempt(motion, output):
    adapter = Native124PicoComparator(motion, phase="causal_q9", root=ROOT, assets=ASSETS)
    with pytest.raises(ValueError):
        adapter.infer(Policy(output), np.zeros(267), np.zeros(930, np.float32), control_index=0, **state(motion))
    assert adapter.failed and len(adapter.attempts) == 1
    assert "applied_target23" not in adapter.attempts[0]
    with pytest.raises(ValueError, match="restart"):
        adapter.infer(Policy(), np.zeros(267), np.zeros(930, np.float32), control_index=0, **state(motion))


def test_preview_failure_never_returns_an_action(motion):
    adapter = Native124PicoComparator(motion, phase="current_q10", root=ROOT, assets=ASSETS)

    def reject(*args):
        raise ValueError("range rejection")

    adapter.preview.filter = reject
    with pytest.raises(ValueError, match="range rejection"):
        adapter.infer(Policy(), np.zeros(267), np.zeros(930, np.float32), control_index=0, **state(motion))
    assert adapter.failed and adapter.next_control == 0
    assert "proposed_target23" in adapter.attempts[0]
    assert "accepted23" not in adapter.attempts[0]
