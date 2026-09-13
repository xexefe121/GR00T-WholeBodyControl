import numpy as np
import pytest

from gear_sonic.scripts.diagnose_g1_true23_leg_error_feedback import LegFeedbackPolicy, leg_contract
from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_SCALE_NATIVE_IL23


class Probe:
    def infer(self, encoder, history, feedback):
        self.feedback = feedback.copy()
        return np.linspace(-0.3, 0.3, 23, dtype=np.float32), np.zeros(994, dtype=np.float32)


def inputs():
    encoder = np.zeros(267, dtype=np.float32)
    encoder[:12] = -0.5
    encoder[12:24] = 0.3
    encoder[24:120] = 0.8
    qpos = np.zeros(30)
    qpos[3] = 1
    qpos[7:19] = 0.1
    return encoder, np.zeros(930, dtype=np.float32), np.ones(9, dtype=np.float32), qpos


def test_current_state_and_received_q1_target_not_q0_or_q2():
    policy = LegFeedbackPolicy(Probe(), 0.5, 4)
    encoder, history, feedback, qpos = inputs()
    policy.set_measured_joints(qpos)
    changed, _ = policy.infer(encoder, history, feedback)
    records = policy.take_records()
    np.testing.assert_array_equal(records["leg_received_reference12"][0], encoder[12:24])
    np.testing.assert_allclose(records["leg_measured_position12"][0], 0.1, rtol=0, atol=1e-8)
    np.testing.assert_allclose(records["leg_target_correction12"][0], 0.1, rtol=0, atol=1e-8)
    order = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
    mask = order < 12
    raw = records["uncorrected_model_raw23"][0]
    np.testing.assert_array_equal(changed[~mask], raw[~mask])
    np.testing.assert_allclose((changed - raw)[mask] * np.asarray(SOURCE_SCALE_NATIVE_IL23)[mask], 0.1, atol=2e-8)
    np.testing.assert_array_equal(feedback, np.ones(9, dtype=np.float32))
    np.testing.assert_array_equal(records["actual_policy_root_feedback9"][0], [4, 4, 4, 1, 1, 1, 1, 1, 1])


def test_zero_leg_gain_is_exact_network_output_control():
    policy = LegFeedbackPolicy(Probe(), 0, 4)
    encoder, history, feedback, qpos = inputs()
    policy.set_measured_joints(qpos)
    changed, _ = policy.infer(encoder, history, feedback)
    records = policy.take_records()
    np.testing.assert_array_equal(changed, records["uncorrected_model_raw23"][0])
    assert not records["leg_target_correction12"].any()


def test_correction_is_bounded_and_cannot_reuse_stale_state():
    policy = LegFeedbackPolicy(Probe(), 1, 1)
    encoder, history, feedback, qpos = inputs()
    encoder[12:24] = np.arange(12, dtype=np.float32) - 6
    with pytest.raises(ValueError, match="fresh state"):
        policy.infer(encoder, history, feedback)
    policy.set_measured_joints(qpos)
    with pytest.raises(ValueError, match="exactly one fresh"):
        policy.set_measured_joints(qpos)
    policy.infer(encoder, history, feedback)
    assert abs(policy.take_records()["leg_target_correction12"]).max() <= np.float32(0.1)
    with pytest.raises(ValueError, match="fresh state"):
        policy.infer(encoder, history, feedback)


@pytest.mark.parametrize("gain", [-1, 2, float("inf"), float("nan"), True])
def test_undeclared_gain_rejected(gain):
    with pytest.raises(ValueError):
        leg_contract(gain, 4)


def test_hybrid_effective_feedback_disclosed_without_hardware_or_training_claims():
    proof = leg_contract(1, 4)
    assert proof["effective_closed_loop_joint_feedback_modified"]
    assert proof["always_on_hybrid_controller"]
    assert not proof["eligible_parent_for_training_continuation"]
    assert not proof["physical_pd_gains_modified"]
    assert not proof["hardware_authorized"]
    assert not proof["deployment_ready"]
