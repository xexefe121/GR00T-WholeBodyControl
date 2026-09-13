import numpy as np
import pytest

from gear_sonic.teleop.normal_source_horizon import ReceivedNormalSourceHorizon, normal_horizon_contract
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES


def sample(index):
    return dict(
        source_timestamp_s=index * 0.02,
        arrival_timestamp_s=index * 0.02,
        joint_names=HARDWARE_23_JOINT_NAMES,
        joint_position23=(np.arange(23) + index**2 * 0.001).astype(np.float32),
        root_position_w=np.asarray([index * 0.02, 0, 0.8], np.float32),
        root_quaternion_wxyz=np.asarray([1, 0, 0, 0], np.float32),
        virtual_source_vr21=np.asarray([0] * 9 + [1, 0, 0, 0] * 3, np.float32),
    )


def test_exact_step5_positions_adjacent_velocities_and_920ms_delay():
    buffer = ReceivedNormalSourceHorizon()
    for i in range(46):
        assert buffer.push(**sample(i)) is None
    result = buffer.push(**sample(46))
    positions = np.stack([sample(i)["joint_position23"][:12] for i in range(0, 46, 5)])
    following = np.stack([sample(i + 1)["joint_position23"][:12] for i in range(0, 46, 5)])
    np.testing.assert_array_equal(result.lower_body240[:120], positions.reshape(-1))
    np.testing.assert_array_equal(
        result.lower_body240[120:], ((following - positions) / np.float32(0.02)).reshape(-1)
    )
    assert result.emission_source_timestamp_s == 0.92
    assert result.encoder_anchor_timestamp_s == 0
    assert result.root_setpoint_timestamp_s == 0.02
    assert result.encoder267(np.asarray([1, 0, 0, 0], np.float32)).shape == (267,)
    next_result = buffer.push(**sample(47))
    assert next_result.encoder_anchor_timestamp_s == 0.02
    np.testing.assert_array_equal(next_result.lower_body240[:12], sample(1)["joint_position23"][:12])


@pytest.mark.parametrize("change", ["skip", "arrival", "joint_names", "nan", "quaternion", "dtype"])
def test_rejection_latches_and_requires_reset(change):
    buffer = ReceivedNormalSourceHorizon()
    buffer.push(**sample(0))
    value = sample(2 if change == "skip" else 1)
    if change == "arrival":
        value["arrival_timestamp_s"] += 0.05
    elif change == "joint_names":
        value["joint_names"] = tuple(reversed(HARDWARE_23_JOINT_NAMES))
    elif change == "nan":
        value["joint_position23"][0] = np.nan
    elif change == "quaternion":
        value["root_quaternion_wxyz"] *= 2
    elif change == "dtype":
        value["joint_position23"] = value["joint_position23"].astype(np.float64)
    with pytest.raises(ValueError):
        buffer.push(**value)
    with pytest.raises(RuntimeError, match="explicit reset"):
        buffer.push(**sample(2))
    buffer.reset()
    assert buffer.push(**sample(0)) is None


def test_copied_input_and_output_no_unreceived_tail():
    buffer = ReceivedNormalSourceHorizon()
    first = sample(0)
    buffer.push(**first)
    first["joint_position23"][:] = -999
    for i in range(1, 47):
        result = buffer.push(**sample(i))
    np.testing.assert_array_equal(result.lower_body240[:12], sample(0)["joint_position23"][:12])
    result.lower_body240[:] = -999
    np.testing.assert_array_equal(
        buffer.push(**sample(47)).lower_body240[:12], sample(1)["joint_position23"][:12]
    )
    assert normal_horizon_contract()["automatic_terminal_padding"] is False
    assert normal_horizon_contract()["deployment_ready"] is False
