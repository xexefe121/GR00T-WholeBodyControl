from pathlib import Path

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_contract import SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_DEFAULT_Q_HARDWARE
from gear_sonic.utils.g1_true23_continuous_reference_alignment import temporal_bounds
from gear_sonic.utils.g1_true23_correction_braking_alignment import CorrectionBrakingAlignment


@pytest.fixture(scope="module")
def models():
    path = Path(__file__).resolve().parents[2] / "gear_sonic/data/robots/g1"
    return tuple(
        mujoco.MjModel.from_xml_path(str(path / name)) for name in ("g1_29dof.xml", "g1_23dof_rev_1_0.xml")
    )


def initial():
    pose = np.zeros(36)
    pose[2:4] = [0.8, 1]
    pose[7 + np.asarray(SOURCE_MJ29_KEEP_INDICES)] = SAFE_TARGET_DEFAULT_Q_HARDWARE
    return pose


def test_correction_box_brakes_before_previous_algorithm_becomes_infeasible():
    q, v = np.array([1.0]), np.array([3.0])
    low, physical_high, correction_high = np.array([-1.9]), np.array([1.9]), np.array([1.14])
    vmax, amax = np.array([4.975]), np.array([79.6])
    old_low, old_high = temporal_bounds(q, v, low, physical_high, vmax, amax)
    new_low, new_high = temporal_bounds(q, v, low, correction_high, vmax, amax)
    assert new_high[0] < old_high[0]
    # Old final clipping did not constrain a stopping distance to1.14rad.
    old_q = np.minimum(old_high, correction_high)
    with pytest.raises(ValueError, match="not relaxed"):
        temporal_bounds(old_q, (old_q - q) / 0.02, low, correction_high, vmax, amax)
    new_velocity = (new_high - q) / 0.02
    temporal_bounds(new_high, new_velocity, low, correction_high, vmax, amax)
    np.testing.assert_allclose(new_low, old_low, atol=1e-15, rtol=0)


def test_current_correction_bounds_are_used_then_static_limits_restored(models, monkeypatch):
    import gear_sonic.utils.g1_true23_continuous_reference_alignment as module

    pose = initial()
    obj = CorrectionBrakingAlignment(*models, pose)
    old_low, old_high = obj.static_lower.copy(), obj.static_upper.copy()
    actual = module.temporal_bounds
    seen = []

    def capture(previous, velocity, lower, upper, vmax, amax, **kwargs):
        seen.append((lower.copy(), upper.copy()))
        return actual(previous, velocity, lower, upper, vmax, amax, **kwargs)

    monkeypatch.setattr(module, "temporal_bounds", capture)
    result, evidence = obj.push(pose, frame_index=0)
    np.testing.assert_array_equal(seen[0][0][6:], np.maximum(old_low[6:], pose[7 + obj.keep] - 0.597))
    np.testing.assert_array_equal(seen[0][1][6:], np.minimum(old_high[6:], pose[7 + obj.keep] + 0.597))
    np.testing.assert_array_equal(obj.static_lower, old_low)
    np.testing.assert_array_equal(obj.static_upper, old_high)
    np.testing.assert_array_equal(result, np.r_[pose[:7], pose[7 + obj.keep]])
    assert evidence["current_source_correction_in_braking_distance"]


def test_failed_source_step_restores_bounds_and_latches_without_advancing(models):
    pose = initial()
    obj = CorrectionBrakingAlignment(*models, pose)
    lower, upper, previous = obj.static_lower.copy(), obj.static_upper.copy(), obj.previous.copy()
    pose[7 + obj.keep[-1]] += 0.7
    with pytest.raises(ValueError):
        obj.push(pose, frame_index=0)
    assert obj.failed and obj.index == 0
    np.testing.assert_array_equal(obj.previous, previous)
    np.testing.assert_array_equal(obj.static_lower, lower)
    np.testing.assert_array_equal(obj.static_upper, upper)
    with pytest.raises(RuntimeError, match="latched"):
        obj.push(initial(), frame_index=0)


def test_contract_does_not_claim_future_source_guarantee(models):
    contract = CorrectionBrakingAlignment(*models, initial()).contract()
    assert contract["current_source_correction_in_braking_distance"]
    assert not contract["temporal_feasibility_for_arbitrary_next_source_guaranteed"]
    assert not contract["weights_limits_iterations_or_source_timing_changed_from_v1"]
    assert not contract["deployment_ready"]
