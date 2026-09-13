import numpy as np
import pytest

from gear_sonic.teleop.buffered_source_horizon import ReceivedSourceHorizon, buffered_horizon_contract
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES


def packet(frame):
    vr = np.zeros(21, np.float32)
    vr[[9, 13, 17]] = 1
    vr[:9] = frame * 0.01
    return dict(
        source_timestamp_s=frame * 0.02,
        arrival_timestamp_s=frame * 0.02,
        joint_names=HARDWARE_23_JOINT_NAMES,
        joint_position23=(np.arange(23) * 0.01 + frame * 0.001).astype(np.float32),
        root_position_w=np.array([frame * 0.02, 0, 0.76], np.float32),
        root_quaternion_wxyz=np.array([1, 0, 0, 0], np.float32),
        virtual_source_vr21=vr,
    )


def full_window():
    buffer = ReceivedSourceHorizon()
    for i in range(10):
        assert buffer.push(**packet(i)) is None
    return buffer, buffer.push(**packet(10))


def test_emission_requires_received_proof_and_exposes_exact_latency():
    buffer, window = full_window()
    assert window.encoder_anchor_timestamp_s == 0
    assert window.emission_source_timestamp_s == pytest.approx(0.2)
    assert window.root_setpoint_timestamp_s == pytest.approx(0.02)
    next_window = buffer.push(**packet(11))
    assert next_window.encoder_anchor_timestamp_s == pytest.approx(0.02)
    assert next_window.emission_source_timestamp_s - next_window.encoder_anchor_timestamp_s == pytest.approx(0.2)
    assert next_window.emission_source_timestamp_s - next_window.root_setpoint_timestamp_s == pytest.approx(0.18)


def test_original_horizon_leg_order_forward_differences_and_oldest_vr():
    _, window = full_window()
    q = np.stack([packet(i)["joint_position23"][:12] for i in range(11)])
    np.testing.assert_array_equal(window.lower_body240[:120], q[:-1].reshape(-1))
    np.testing.assert_array_equal(window.lower_body240[120:], ((q[1:] - q[:-1]) / np.float32(0.02)).reshape(-1))
    np.testing.assert_array_equal(window.virtual_vr21, packet(0)["virtual_source_vr21"])
    encoded = window.encoder267(np.array([1, 0, 0, 0], np.float32))
    assert encoded.shape == (267,) and encoded.dtype == np.float32
    np.testing.assert_array_equal(encoded[261:], [1, 0, 0, 1, 0, 0])


def test_measured_state_is_current_not_delayed_with_reference():
    _, window = full_window()
    measured = np.array([0.5, 0, 0.76], np.float32)
    quat = np.array([1, 0, 0, 0], np.float32)
    feedback = window.root_feedback9(measured, np.array([2, 0, 0], np.float32), quat)
    np.testing.assert_allclose(feedback, [-0.48, 0, 0, 1, 0, 0, 2, 0, 0], atol=1e-7)
    measured[0] = 0.75
    assert window.root_feedback9(measured, np.zeros(3, np.float32), quat)[0] == pytest.approx(-0.73)
    turned = np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)], np.float32)
    np.testing.assert_allclose(window.encoder267(turned)[261:], [0, 1, -1, 0, 0, 0], atol=1e-7)


@pytest.mark.parametrize(
    "change",
    ["duplicate", "gap", "stale", "future", "nan", "wrong_order", "wrong_dtype", "nonunit", "arrival_reversal"],
)
def test_rejection_latches_without_replacement_frames(change):
    buffer, _ = full_window()
    next_packet = packet(11)
    if change == "duplicate":
        next_packet["source_timestamp_s"] = 0.2
    elif change == "gap":
        next_packet = packet(12)
    elif change == "stale":
        next_packet["arrival_timestamp_s"] += 0.041
    elif change == "future":
        next_packet["arrival_timestamp_s"] -= 0.003
    elif change == "nan":
        next_packet["joint_position23"][0] = np.nan
    elif change == "wrong_order":
        next_packet["joint_names"] = tuple(reversed(HARDWARE_23_JOINT_NAMES))
    elif change == "wrong_dtype":
        next_packet["joint_position23"] = next_packet["joint_position23"].astype(float)
    elif change == "nonunit":
        next_packet["root_quaternion_wxyz"][0] = 0.5
    else:
        buffer.reset()
        first = packet(0)
        first["arrival_timestamp_s"] = 0.039
        assert buffer.push(**first) is None
        next_packet = packet(1)
    with pytest.raises(ValueError):
        buffer.push(**next_packet)
    with pytest.raises(RuntimeError, match="explicit reset"):
        buffer.push(**packet(12))
    buffer.reset()
    assert buffer.push(**packet(0)) is None


def test_input_output_and_later_packets_do_not_alias_buffer_history():
    buffer = ReceivedSourceHorizon()
    first = packet(0)
    buffer.push(**first)
    first["joint_position23"][:] = 99
    first["virtual_source_vr21"][:] = 99
    for i in range(1, 11):
        window = buffer.push(**packet(i))
    np.testing.assert_array_equal(window.lower_body240[:12], packet(0)["joint_position23"][:12])
    window.virtual_vr21[:] = 22
    next_window = buffer.push(**packet(11))
    np.testing.assert_array_equal(next_window.virtual_vr21, packet(1)["virtual_source_vr21"])


def test_contract_does_not_relabel_old_causal_policies_or_claim_readiness():
    contract = buffered_horizon_contract()
    assert contract["encoder_anchor_age_s"] == 0.2
    assert contract["integration_and_policy_retraining_required"]
    for key in (
        "old_causal_artifacts_may_be_relabelled",
        "unreceived_samples_or_prediction",
        "terminal_padding_or_automatic_hold",
        "hardware_authorized",
        "deployment_ready",
    ):
        assert contract[key] is False
