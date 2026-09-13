import numpy as np
import pytest

from gear_sonic.scripts.diagnose_g1_true23_swing_reference_targets import (
    KIND,
    SwingReferencePolicy,
    swing_contract,
    swing_weights,
)
from gear_sonic.tests.test_diagnose_g1_true23_leg_error_feedback import Probe, inputs
from gear_sonic.tests.test_g1_true23_root_feedback_campaign import receipt, validate
from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_SCALE_NATIVE_IL23


def test_only_one_swing_leg_uses_current_received_q1_pose():
    probe = Probe()
    policy = SwingReferencePolicy(probe, 1)
    encoder, history, feedback, _ = inputs()
    original = encoder.copy()
    policy.set_received_weights(np.array([1, 0]))
    changed, _ = policy.infer(encoder, history, feedback)
    records = policy.take_records()
    order = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
    left = order < 6
    raw = records["uncorrected_model_raw23"][0]
    np.testing.assert_array_equal(changed[~left], raw[~left])
    target = np.asarray(SAFE_TARGET_DEFAULT_Q_HARDWARE)[order] + changed * np.asarray(SOURCE_SCALE_NATIVE_IL23)
    np.testing.assert_allclose(target[left], original[12:24][order[left]], atol=3e-8)
    np.testing.assert_array_equal(probe.feedback, feedback)
    np.testing.assert_array_equal(encoder, original)
    np.testing.assert_array_equal(records["swing_received_reference12"][0], original[12:24])


@pytest.mark.parametrize("strength,weights", [(0, [1, 1]), (1, [0, 0])])
def test_zero_strength_or_stance_is_bit_exact_network_output(strength, weights):
    policy = SwingReferencePolicy(Probe(), strength)
    encoder, history, feedback, _ = inputs()
    policy.set_received_weights(np.asarray(weights))
    changed, _ = policy.infer(encoder, history, feedback)
    np.testing.assert_array_equal(changed, policy.take_records()["uncorrected_model_raw23"][0])


def test_only_generated_phases_and_all_sole_corners_allow_blending():
    timeline = dict(
        total_frames=8,
        phases=[
            dict(name="initial_standing", frame_start=0, frame_stop=1),
            dict(name="acquisition_ramp", frame_start=1, frame_stop=3),
            dict(name="source_motion", frame_start=3, frame_stop=5),
            dict(name="return_ramp", frame_start=5, frame_stop=7),
            dict(name="standing_proof_margin", frame_start=7, frame_stop=8),
        ],
    )
    gaps = np.full((8, 2, 4), 0.04)
    gaps[1, 0, 3] = 0.0
    gaps[2, 1] = 0.011
    weights = swing_weights(gaps, timeline)
    np.testing.assert_array_equal(weights[[0, 3, 4, 7]], np.zeros((4, 2)))
    assert weights[1, 0] == 0 and weights[1, 1] == 1
    assert weights[2, 1] == pytest.approx(0.5)
    gaps[5, 0, 0] = np.nan
    with pytest.raises(ValueError):
        swing_weights(gaps, timeline)


def test_stale_or_duplicate_reference_activation_rejected():
    policy = SwingReferencePolicy(Probe(), 1)
    encoder, history, feedback, _ = inputs()
    with pytest.raises(ValueError, match="fresh received"):
        policy.infer(encoder, history, feedback)
    policy.set_received_weights(np.ones(2))
    with pytest.raises(ValueError, match="one fresh"):
        policy.set_received_weights(np.ones(2))
    policy.infer(encoder, history, feedback)
    with pytest.raises(ValueError, match="fresh received"):
        policy.infer(encoder, history, feedback)


@pytest.mark.parametrize("strength", [-1, 0.5, 2, float("inf"), float("nan"), True])
def test_only_declared_probe_strengths_allowed(strength):
    with pytest.raises(ValueError):
        swing_contract(strength)


def test_counterfactual_cannot_become_normal_training_parent():
    report = receipt()
    report["kind"] = KIND
    report["swing_reference_counterfactual"] = swing_contract(1)
    assert report["swing_reference_counterfactual"]["modified_controller_not_pure_sonic_policy"]
    assert not report["swing_reference_counterfactual"]["eligible_parent_for_training_continuation"]
    with pytest.raises(ValueError):
        validate(report)
