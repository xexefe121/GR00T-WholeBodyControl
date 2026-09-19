import numpy as np

from gear_sonic.scripts.stream_g1_true23_pico_bfm_packets import (
    BFMPicoPacketBridge,
    CausalFloorLift,
    FLOOR_BUFFER_M,
    FLOOR_GUARD_M,
)
from gear_sonic.tests.test_g1_true23_pico_body_adapter import fixture


class _Clearance:
    """Synthetic exact-clearance stand-in for the bridge unit boundary."""

    def minimum(self, fields):
        return float(np.min(fields["body_pos_w"][:, 2]))


def _wire(frame, z):
    contract, wire = fixture(frame)
    body = wire["native23_body_pose"]
    body["body_position_w"] = np.asarray(body["body_position_w"], dtype=float).tolist()
    for position in body["body_position_w"]:
        position[2] = z
    return contract, wire


def test_bridge_applies_v4_floor_lift_and_rederives_backward_velocities():
    contract, first = _wire(20, -0.03)
    bridge = BFMPicoPacketBridge(contract, CausalFloorLift(_Clearance()))
    packet0, source0 = bridge.adapt(first, received_monotonic_ns=first["control_monotonic_ns"])
    assert source0["floor"]["after_m"] >= FLOOR_GUARD_M
    assert source0["floor"]["applied_lift_m"] >= 0.03 + FLOOR_GUARD_M
    np.testing.assert_array_equal(packet0.fields["joint_vel"], 0.0)
    np.testing.assert_array_equal(packet0.fields["body_lin_vel_w"], 0.0)
    _, second = _wire(21, -0.02)
    packet1, source1 = bridge.adapt(second, received_monotonic_ns=second["control_monotonic_ns"])
    assert packet1.sequence == 1 and packet1.source_time == 0.02
    np.testing.assert_allclose(
        packet1.fields["body_lin_vel_w"],
        (packet1.fields["body_pos_w"] - packet0.fields["body_pos_w"]) / 0.02,
    )
    assert source1["derivative"] == "current_minus_previous_corrected_pose_over_20ms"


def test_floor_filter_adds_fixed_buffer_without_preview():
    floor = CausalFloorLift(_Clearance())
    fields = {
        "joint_pos": np.zeros(23), "joint_vel": np.zeros(23),
        "body_pos_w": np.zeros((24, 3)), "body_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (24, 1)),
        "body_lin_vel_w": np.zeros((24, 3)), "body_ang_vel_w": np.zeros((24, 3)),
    }
    fields["body_pos_w"][:, 2] = 0.1
    corrected, evidence = floor.apply(fields)
    assert evidence["applied_lift_m"] >= FLOOR_BUFFER_M
    assert np.min(corrected["body_pos_w"][:, 2]) >= 0.1 + FLOOR_BUFFER_M - 1e-12
