"""BFM interface and referee checks; synthetic physics is not robot qualification."""

import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import torch

from gear_sonic.scripts import evaluate_g1_true23_bfmzero as evaluator
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_bfmzero_inference import (
    BFMHistory,
    reference_features,
    state_and_terms,
)


BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_rubber_hand",
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_rubber_hand",
)


def yaw_quaternion(angle):
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def reference(count=5, *, yaw=0.0):
    motion = {
        "body_pos_w": np.zeros((count, 24, 3)),
        "body_quat_w": np.tile(yaw_quaternion(yaw), (count, 24, 1)),
        "body_lin_vel_w": np.zeros((count, 24, 3)),
        "body_ang_vel_w": np.zeros((count, 24, 3)),
        "joint_pos": np.zeros((count, 23)),
        "joint_vel": np.zeros((count, 23)),
    }
    motion["body_pos_w"][:, :, 2] = 0.8
    contract = {"default_q": np.zeros(23), "body_names": BODY_NAMES}
    return motion, contract


class GoalProbe:
    """Capture real backward-map inputs; return a nonconstant test embedding."""

    def __init__(self):
        self.inputs = []

    def backward(self, state, privileged):
        self.inputs.append((state.numpy().copy(), privileged.numpy().copy()))
        result = torch.zeros((len(state), 256), dtype=torch.float32)
        result[:, 0] = 1.0
        result[:, 1:4] = privileged[:, 223:226]
        result[:, 4] = state[:, -1]
        return 16 * torch.nn.functional.normalize(result, dim=-1)


def test_physical_joint_and_body_order_matches_native_model():
    """The embodiment XML, not a second policy index list, supplies the oracle."""
    root = Path(__file__).resolve().parents[1]
    xml = ET.parse(root / "data/robots/g1/g1_23dof_rev_1_0.xml")
    nodes = xml.findall(".//worldbody//joint")
    joints = tuple(node.attrib["name"] for node in nodes if node.attrib.get("type") != "free")
    bodies = tuple(node.attrib["name"] for node in xml.findall(".//worldbody//body"))
    motors = tuple(node.attrib["joint"] for node in xml.findall("./actuator/motor"))
    assert joints == tuple(HARDWARE_23_JOINT_NAMES) == motors
    assert sum(node.attrib.get("type") == "free" for node in nodes) == 1
    assert bodies == BODY_NAMES
    assert len(joints) == 23 and len(bodies) == 24
    assert joints[12] == "waist_yaw_joint"
    assert joints[13] == "left_shoulder_pitch_joint"
    assert joints[18] == "right_shoulder_pitch_joint"
    assert not any(name.endswith(("wrist_pitch_joint", "wrist_yaw_joint")) for name in joints)


def test_sensor_order_scales_only_body_gyro_and_keeps_previous_action_units():
    half = np.sqrt(0.5)
    default = np.full(23, 0.5)
    q = default + np.arange(23) / 10
    dq = -np.arange(23) / 5
    previous = np.linspace(-5, 5, 23)
    # Positive 90-degree pitch maps world gravity to positive body X.
    state, terms = state_and_terms(q, dq, [half, 0, half, 0], [4, -8, 12], previous, default)
    np.testing.assert_allclose(state[:23], np.arange(23) / 10, atol=1e-7)
    np.testing.assert_allclose(state[23:46], dq, atol=2e-7)
    np.testing.assert_allclose(state[46:49], [1, 0, 0], atol=1e-7)
    np.testing.assert_array_equal(state[49:52], [1, -2, 3])
    np.testing.assert_allclose(terms["actions"], previous, atol=2e-7)


def test_history_contains_previous_observations_newest_first_without_aliasing():
    history = BFMHistory()
    samples = []
    for sample in range(6):
        samples.append({
            "actions": np.full(23, 10 + sample, np.float32),
            "base_ang_vel": np.full(3, 20 + sample, np.float32),
            "dof_pos": np.full(23, 30 + sample, np.float32),
            "dof_vel": np.full(23, 40 + sample, np.float32),
            "projected_gravity": np.full(3, 50 + sample, np.float32),
        })
    first = history.before_update(samples[0])
    np.testing.assert_array_equal(first, np.zeros(300))
    second = history.before_update(samples[1])
    np.testing.assert_array_equal(second[:23], np.full(23, 10))
    np.testing.assert_array_equal(second[23:92], np.zeros(69))
    for sample in samples[2:]:
        latest = history.before_update(sample)
    # At sample 5, history contains samples 4,3,2,1; sample 5 is not yet visible.
    for start, width, values in (
        (0, 23, [14, 13, 12, 11]), (92, 3, [24, 23, 22, 21]),
        (104, 23, [34, 33, 32, 31]), (196, 23, [44, 43, 42, 41]),
        (288, 3, [54, 53, 52, 51]),
    ):
        np.testing.assert_array_equal(latest[start:start + 4 * width].reshape(4, width),
                                      np.repeat(np.array(values)[:, None], width, axis=1))
    np.testing.assert_array_equal(first, np.zeros(300))


def test_reference_features_use_heading_local_x_z_axes_and_world_expert_gyro():
    motion, contract = reference(1, yaw=np.pi / 2)
    motion["body_pos_w"][:, 1] += [0, 2, 0]
    motion["body_lin_vel_w"][:] = [0, 2, 0]
    motion["body_ang_vel_w"][:] = [0, 0, 3]
    state, privileged = reference_features(motion, contract)
    assert state.shape == (1, 52) and privileged.shape == (1, 373)
    np.testing.assert_allclose(privileged[0, 1:4], [2, 0, 0], atol=1e-7)
    np.testing.assert_allclose(privileged[0, 73:79], [1, 0, 0, 0, 0, 1], atol=1e-7)
    np.testing.assert_allclose(privileged[0, 223:226], [2, 0, 0], atol=1e-7)
    np.testing.assert_array_equal(state[0, -3:], [0, 0, 3])
    np.testing.assert_allclose(privileged[0, 70:73], [0, 0, 0.35], atol=1e-7)


def test_world_position_feedback_uses_measured_heading_and_preserves_articulation():
    motion, contract = reference(2, yaw=np.pi / 2)
    motion["body_pos_w"][:, :, 0] = 1.0
    motion["body_lin_vel_w"][:] = [0, 0.2, 0]
    # First nonroot body's articulation adds world +Z, independent of translation.
    motion["body_lin_vel_w"][:, 1, 2] = 0.1
    state, privileged = reference_features(motion, contract)
    old_state, old_privileged = state.copy(), privileged.copy()
    qpos = np.r_[[0, 0, 0.8], yaw_quaternion(0), np.zeros(23)]
    probe = GoalProbe()
    evaluator.corrected_goal(probe, state, privileged, motion, 0, qpos, 1, 1.0, 0.0)
    observed_state, observed_privileged = probe.inputs[-1]
    # Desired world +Y .2m/s plus bounded world +X .6m/s, robot yaw zero.
    np.testing.assert_allclose(observed_privileged[0, 223:226], [0.6, 0.2, 0], atol=1e-7)
    np.testing.assert_allclose(observed_privileged[0, 226:229], [0.6, 0.2, 0.1], atol=1e-7)
    np.testing.assert_array_equal(observed_state, state[:1])
    np.testing.assert_array_equal(observed_privileged[0, :223], privileged[0, :223])
    np.testing.assert_array_equal(state, old_state)
    np.testing.assert_array_equal(privileged, old_privileged)


def test_yaw_feedback_uses_shortest_wrapped_error_and_consistent_rigid_twist():
    motion, contract = reference(1, yaw=np.deg2rad(-179))
    # A marker one metre along reference-local X.
    motion["body_pos_w"][:, 1, :2] = [np.cos(np.deg2rad(-179)), np.sin(np.deg2rad(-179))]
    state, privileged = reference_features(motion, contract)
    qpos = np.r_[[0, 0, 0.8], yaw_quaternion(np.deg2rad(179)), np.zeros(23)]
    probe = GoalProbe()
    evaluator.corrected_goal(probe, state, privileged, motion, 0, qpos, 1, 0.0, 2.0)
    observed_state, observed_privileged = probe.inputs[-1]
    omega = np.deg2rad(4)  # +2 degree shortest error times gain 2.
    np.testing.assert_allclose(observed_state[0, -3:], [0, 0, omega], atol=1e-7)
    np.testing.assert_allclose(observed_privileged[0, 298:301], [0, 0, omega], atol=1e-7)
    np.testing.assert_allclose(observed_privileged[0, 226:229], [0, omega, 0], atol=1e-7)


@pytest.mark.parametrize("horizon", [1, 8])
@pytest.mark.parametrize("actual_yaw", [0.0, 0.7])
def test_zero_gain_goal_matches_unmodified_source_mean(horizon, actual_yaw):
    motion, contract = reference(10)
    motion["body_lin_vel_w"][:, :, 0] = np.arange(10)[:, None] / 20
    state, privileged = reference_features(motion, contract)
    qpos = np.r_[[4, -3, 0.8], yaw_quaternion(actual_yaw), np.zeros(23)]
    probe = GoalProbe()
    expected = probe.backward(torch.from_numpy(state[1:1 + horizon]),
                              torch.from_numpy(privileged[1:1 + horizon])).mean(0, keepdim=True)
    expected = 16 * torch.nn.functional.normalize(expected, dim=-1)
    actual = evaluator.corrected_goal(probe, state, privileged, motion, 1, qpos, horizon, 0.0, 0.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture
def synthetic_referee(monkeypatch, tmp_path):
    """Real tiny MuJoCo integration; only policy, data input and model loader substituted."""
    children = "".join(
        f'<body name="{body}"><inertial pos="0 0 0" mass=".1" diaginertia=".1 .1 .1"/>'
        f'<joint name="{joint}" axis="0 0 1" range="-2 2"/></body>'
        for body, joint in zip(BODY_NAMES[1:], HARDWARE_23_JOINT_NAMES, strict=True)
    )
    motors = "".join(f'<motor joint="{joint}"/>' for joint in HARDWARE_23_JOINT_NAMES)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><option gravity="0 0 0" timestep=".01"/>'
        '<worldbody><body name="pelvis"><freejoint/>'
        '<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>'
        f'{children}</body></worldbody><actuator>{motors}</actuator></mujoco>'
    )
    motion, contract = reference(16)
    contract.update(kp=np.full(23, 24.0), kd=np.ones(23), training_effort=np.full(23, 32.0))
    timeline = {"phases": [
        {"name": "source_motion", "control_start": 0, "control_stop": 3, "requested_controls": 3},
        {"name": "returned_standing", "control_start": 3, "control_stop": 5, "requested_controls": 2},
    ]}

    class ActorProbe(GoalProbe):
        def __init__(self):
            super().__init__()
            self.actor_inputs = []

        def actor(self, state, action, history, goal):
            self.actor_inputs.append((state.numpy().copy(), action.numpy().copy(), history.numpy().copy()))
            result = torch.zeros((1, 23))
            if len(self.actor_inputs) == 1:
                result[0, 0], result[0, 22] = 0.12, -0.06
            elif len(self.actor_inputs) == 2:
                result[0, 0], result[0, 22] = -0.04, 0.08
            return result

    policy = ActorProbe()
    np.savez(tmp_path / "source.npz", **motion)
    (tmp_path / "physics.json").write_text(json.dumps({"physics": {"velocity_limit_hardware_radps": [100] * 23}}))
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "PHYSICS", "physics.json")
    monkeypatch.setattr(evaluator, "load_contract", lambda _: contract)
    monkeypatch.setattr(evaluator, "BFMZeroInference", lambda _: policy)
    monkeypatch.setattr(evaluator, "load_motion", lambda _: (motion, timeline, tmp_path / "source.npz"))
    monkeypatch.setattr(evaluator, "prepare_true23_model", lambda *_: (
        None, model, SimpleNamespace(decimation=2, effort=np.full(23, 25.0)),
    ))

    def unexpected_feedback(*args):
        raise AssertionError("zero gains must preserve the baseline goal path")

    monkeypatch.setattr(evaluator, "corrected_goal", unexpected_feedback)
    args = SimpleNamespace(threads=1, output=tmp_path / "result", clip="walk002", goal_horizon=8,
                           max_controls=None, position_gain=0.0, yaw_gain=0.0, arm_reference=False)
    return args, policy


def test_referee_scaling_previous_action_and_complete_source_plus_tail(synthetic_referee):
    args, policy = synthetic_referee
    evaluator.run(args)
    report = json.loads((args.output / "report.json").read_text())
    with np.load(args.output / "trace.npz") as trace:
        assert len(trace["action"]) == 5 and len(trace["qpos"]) == 6
        # Known upstream contract: a=.12 -> previous action=.6 -> target=.2rad.
        np.testing.assert_allclose(trace["action"][0, [0, 22]], [0.6, -0.3], atol=1e-7)
        np.testing.assert_allclose(trace["target"][0, [0, 22]], [0.2, -0.1], atol=1e-7)
        np.testing.assert_array_equal(policy.actor_inputs[0][1], np.zeros((1, 23)))
        np.testing.assert_array_equal(policy.actor_inputs[1][1][0], trace["action"][0])
        np.testing.assert_array_equal(policy.actor_inputs[1][2][0, :92], np.zeros(92))
        np.testing.assert_array_equal(policy.actor_inputs[2][2][0, :23], trace["action"][0])
    assert report["completed"] == report["requested"] == report["available"] == 5
    assert report["source_metrics"]["source_controls"] == report["source_metrics"]["source_requested"] == 3
    assert report["full_lifecycle_completed"] and report["failure"] is None
    assert report["hardware_authorized"] is False and report["deployment_ready"] is False


def test_cropped_debug_request_cannot_claim_full_lifecycle(synthetic_referee):
    args, _ = synthetic_referee
    args.max_controls = 2
    evaluator.run(args)
    report = json.loads((args.output / "report.json").read_text())
    assert report["completed"] == report["requested"] == 2
    assert report["available"] == 5 and report["full_lifecycle_completed"] is False
    assert report["source_metrics"]["source_controls"] == 2
    assert report["source_metrics"]["source_requested"] == 3
