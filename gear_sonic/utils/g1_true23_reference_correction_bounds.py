"""Current source-correction braking without a mismatched-time precondition.

The preceding accepted reference must obey STATIC physical-reference limits.
It need not belong to the next source frame's shifted correction box. Only the
next reference is required to lie in that current box. Future source motion is
unknown: these are current-step constraints, not general recursive feasibility.
"""

import numpy as np

from gear_sonic.utils.g1_true23_continuous_reference_alignment import temporal_bounds


def source_correction_temporal_bounds(
    previous,
    velocity,
    static_lower,
    static_upper,
    maximum_velocity,
    maximum_acceleration,
    *,
    current_source_joints23,
    correction_limit_rad,
    dt=0.02,
):
    """Intersect unchanged static/temporal constraints and current-box braking."""
    previous, velocity, static_lower, static_upper, maximum_velocity, maximum_acceleration = (
        np.asarray(value, dtype=float)
        for value in (previous, velocity, static_lower, static_upper, maximum_velocity, maximum_acceleration)
    )
    source = np.asarray(current_source_joints23, dtype=float)
    if (
        previous.shape != (29,)
        or source.shape != (23,)
        or not np.isfinite(source).all()
        or isinstance(correction_limit_rad, bool)
        or not np.isfinite(correction_limit_rad)
        or not 0 < correction_limit_rad <= 0.6
    ):
        raise ValueError("correction bounds require complete finite29-variable/23-source reference and <=.6rad")
    # Validates the accepted past against its genuinely static bounds, and
    # preserves the existing velocity/acceleration/braking checks unchanged.
    lower, upper = temporal_bounds(
        previous,
        velocity,
        static_lower,
        static_upper,
        maximum_velocity,
        maximum_acceleration,
        dt=dt,
    )
    current_lower, current_upper = static_lower.copy(), static_upper.copy()
    current_lower[6:] = np.maximum(current_lower[6:], source - correction_limit_rad)
    current_upper[6:] = np.minimum(current_upper[6:], source + correction_limit_rad)
    if np.any(current_lower > current_upper):
        raise ValueError("current source correction envelope is empty")
    a = maximum_acceleration * dt * dt
    distance_lower = np.maximum(previous - current_lower, 0)
    distance_upper = np.maximum(current_upper - previous, 0)
    lower_step = -a + np.sqrt(a**2 + 2 * a * distance_lower)
    upper_step = -a + np.sqrt(a**2 + 2 * a * distance_upper)
    # Previous outside a shifted box is not by itself a violation. These next
    # bounds still require position and acceleration feasibility; no clipping
    # of past velocity, enlarged box or removed source frame is permitted.
    lower = np.maximum.reduce((lower, current_lower, previous - lower_step))
    upper = np.minimum.reduce((upper, current_upper, previous + upper_step))
    if np.any(lower > upper):
        raise ValueError("current source correction/temporal/braking intersection is empty")
    return lower, upper
