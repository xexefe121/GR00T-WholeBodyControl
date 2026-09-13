import numpy as np
import pytest

from gear_sonic.utils.g1_true23_continuous_reference_alignment import temporal_bounds
from gear_sonic.utils.g1_true23_reference_correction_bounds import source_correction_temporal_bounds


def material():
    previous, velocity = np.zeros(29), np.zeros(29)
    lower, upper = np.full(29, -2.0), np.full(29, 2.0)
    vmax, amax = np.full(29, 4.975), np.full(29, 79.6)
    source = np.zeros(23)
    return previous, velocity, lower, upper, vmax, amax, source


def test_saved_v2_failure_has_a_feasible_next_step_without_relaxing_limits():
    previous, velocity, low, high, vmax, amax, source = material()
    previous[24], velocity[24], source[18] = -0.11561864532650426, 0.597942477267184, -0.7159172672372582
    moving_low, moving_high = low.copy(), high.copy()
    moving_low[6:], moving_high[6:] = source - 0.597, source + 0.597
    with pytest.raises(ValueError, match="invalid reference state"):
        temporal_bounds(previous, velocity, moving_low, moving_high, vmax, amax)
    lo, hi = source_correction_temporal_bounds(
        previous, velocity, low, high, vmax, amax, current_source_joints23=source, correction_limit_rad=0.597
    )
    assert lo[24] <= hi[24] <= source[18] + 0.597
    next_pose = (lo + hi) / 2
    new_velocity = (next_pose - previous) / 0.02
    assert np.max(np.abs(new_velocity)) <= 4.975
    assert np.max(np.abs(new_velocity - velocity)) <= 79.6 * 0.02
    assert np.max(np.abs(next_pose[6:] - source)) <= 0.597


def test_unreachable_moving_box_still_fails():
    previous, velocity, low, high, vmax, amax, source = material()
    source[0] = 1.0
    with pytest.raises(ValueError, match="intersection is empty"):
        source_correction_temporal_bounds(
            previous, velocity, low, high, vmax, amax, current_source_joints23=source, correction_limit_rad=0.597
        )


def test_previous_outside_actual_static_bounds_is_not_accepted():
    previous, velocity, low, high, vmax, amax, source = material()
    previous[0] = 3.0
    with pytest.raises(ValueError, match="invalid reference state"):
        source_correction_temporal_bounds(
            previous, velocity, low, high, vmax, amax, current_source_joints23=source, correction_limit_rad=0.597
        )


def test_stationary_box_matches_existing_braking_exactly():
    previous, velocity, low, high, vmax, amax, source = material()
    source[:] = 0.1
    moving_low, moving_high = low.copy(), high.copy()
    moving_low[6:], moving_high[6:] = source - 0.597, source + 0.597
    expected = temporal_bounds(previous, velocity, moving_low, moving_high, vmax, amax)
    actual = source_correction_temporal_bounds(
        previous, velocity, low, high, vmax, amax, current_source_joints23=source, correction_limit_rad=0.597
    )
    for old, new in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("correction", [0, 0.61, float("nan"), True])
def test_correction_limits_not_expanded(correction):
    previous, velocity, low, high, vmax, amax, source = material()
    with pytest.raises(ValueError):
        source_correction_temporal_bounds(
            previous,
            velocity,
            low,
            high,
            vmax,
            amax,
            current_source_joints23=source,
            correction_limit_rad=correction,
        )
