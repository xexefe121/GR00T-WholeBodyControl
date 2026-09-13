"""The offline future-frame ablation must never masquerade as causal input."""

import numpy as np
import pytest

from gear_sonic.scripts.diagnose_g1_true23_release_semantics import ReferenceAblation, preview_encoder


def motion():
    return dict(
        joint_pos=np.arange(40 * 23, dtype=np.float64).reshape(40, 23),
        joint_vel=np.full((40, 23), 17.0),
    )


def test_preview_uses_future_lower_positions_and_stored_velocities_only():
    original = np.arange(267, dtype=np.float32)
    reference = motion()
    actual = preview_encoder(original, reference, 2)
    np.testing.assert_array_equal(actual[:120], reference["joint_pos"][11:21, :12].reshape(-1))
    np.testing.assert_array_equal(actual[120:240], 17.0)
    np.testing.assert_array_equal(actual[240:], original[240:])
    np.testing.assert_array_equal(original, np.arange(267, dtype=np.float32))


def test_end_of_nonconstant_clip_is_rejected_not_padded():
    with pytest.raises(ValueError, match="constant terminal hold"):
        preview_encoder(np.zeros(267, dtype=np.float32), motion(), 24)


def test_constant_terminal_hold_can_be_repeated_without_trimming_evaluation():
    reference = motion()
    reference["joint_pos"][-10:] = reference["joint_pos"][-1]
    reference["joint_vel"][-10:] = 0
    actual = preview_encoder(np.zeros(267, dtype=np.float32), reference, 29)
    np.testing.assert_array_equal(actual[:120].reshape(10, 12), reference["joint_pos"][-10:, :12])
    np.testing.assert_array_equal(actual[120:240], 0)


@pytest.mark.parametrize("mode", ["causal_past", "released_future_preview"])
def test_adapter_records_consumed_encoder_and_passes_history_unchanged(mode):
    class Policy:
        def infer(self, encoder, history):
            return encoder.copy(), history

    adapter = ReferenceAblation(motion(), mode)
    encoder, history = np.zeros(267, np.float32), np.ones(930, np.float32)
    actual, passed_history = adapter.infer(Policy(), encoder, history, control_index=0)
    assert passed_history is history
    np.testing.assert_array_equal(actual, adapter.actual_encoder_inputs[0])
    assert adapter.contract()["future_reference_consumed"] == (mode == "released_future_preview")
    assert adapter.contract()["live_teleoperation_compatible"] is False
    assert adapter.contract()["deployment_ready"] is False
    np.testing.assert_array_equal(adapter.external_force_world(0), np.zeros(3))


def test_virtual_geometry_is_explicit_and_touches_only_21_reference_values():
    class Policy:
        def infer(self, encoder, history):
            return encoder, history

    reference = motion()
    virtual = np.full((40, 21), 42, np.float32)
    adapter = ReferenceAblation(reference, "causal_past_virtual_source", virtual)
    encoder = np.arange(267, dtype=np.float32)
    actual, _ = adapter.infer(Policy(), encoder, np.zeros(930, np.float32), control_index=0)
    np.testing.assert_array_equal(actual[:240], encoder[:240])
    np.testing.assert_array_equal(actual[240:261], virtual[9])
    np.testing.assert_array_equal(actual[261:], encoder[261:])
    np.testing.assert_array_equal(encoder, np.arange(267, dtype=np.float32))
    assert adapter.contract()["future_reference_consumed"] is False
    with pytest.raises(ValueError, match="explicitly constructed"):
        ReferenceAblation(reference, "causal_past_virtual_source")


@pytest.mark.parametrize("case,force", [("standing_push_x", [40, 0, 0]), ("standing_push_y", [0, 40, 0])])
def test_perturbation_is_exactly_the_same_bounded_100ms_pulse(case, force):
    adapter = ReferenceAblation(motion(), "causal_past", case=case)
    np.testing.assert_array_equal(adapter.external_force_world(499), np.zeros(3))
    np.testing.assert_array_equal(adapter.external_force_world(500), force)
    np.testing.assert_array_equal(adapter.external_force_world(549), force)
    np.testing.assert_array_equal(adapter.external_force_world(550), np.zeros(3))
    assert adapter.contract()["external_force_used"] is True


def test_existing_root_policy_receives_real_copied_state_with_corrected_reference():
    class Policy:
        def infer(self, encoder, history, feedback):
            self.inputs = encoder.copy(), history.copy(), feedback.copy()
            return np.zeros(23, np.float32), np.concatenate((np.zeros(64, np.float32), history))

    policy = Policy()
    reference = motion()
    virtual = np.full((40, 21), 3, np.float32)
    adapter = ReferenceAblation(reference, "causal_past_virtual_source", virtual, root_feedback=True)
    qpos = np.zeros(30)
    qpos[3] = 1
    qvel = np.zeros(29)
    qvel[0] = 0.25
    before = qpos.copy(), qvel.copy()
    adapter.infer(
        policy,
        np.zeros(267, np.float32),
        np.zeros(930, np.float32),
        control_index=0,
        desired_position_w=np.array([1, 0, 0]),
        previous_desired_position_w=np.array([0.98, 0, 0]),
        measured_qpos=qpos,
        measured_qvel=qvel,
    )
    np.testing.assert_array_equal(policy.inputs[0][240:261], virtual[9])
    np.testing.assert_allclose(policy.inputs[2], [1, 0, 0, 1, 0, 0, 0.25, 0, 0], atol=1e-6)
    np.testing.assert_array_equal(qpos, before[0])
    np.testing.assert_array_equal(qvel, before[1])
    assert adapter.root_adapter.arrays()["root_feedback9"].shape == (1, 9)
    assert adapter.contract()["root_feedback"]["hardware_state_estimation_validated"] is False
