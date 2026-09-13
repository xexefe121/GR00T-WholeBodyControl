import numpy as np
from numpy.polynomial import Polynomial
import pytest

from gear_sonic.scripts.refine_g1_true23_lifecycle_ramps import ramp_candidates, shoulder_clearance_bump


def test_ramp_offset_is_symmetric_bounded_and_endpoint_flat_through_acceleration():
    bump = shoulder_clearance_bump(100)
    assert bump.shape == (100,) and np.all(bump > 0) and bump.max() <= 1
    np.testing.assert_allclose(bump, bump[::-1], atol=1e-14)
    polynomial = Polynomial([0, 0, 0, 64, -192, 192, -64])
    np.testing.assert_allclose(bump, polynomial(np.arange(1, 101) / 101), atol=2e-14)
    for order in range(3):
        np.testing.assert_array_equal(polynomial.deriv(order)([0.0, 1.0]), [0.0, 0.0])


@pytest.mark.parametrize("count", [0, 24, 201, True, 100.0])
def test_ramp_offset_cannot_change_unbounded_or_unknown_timing(count):
    with pytest.raises(ValueError):
        shoulder_clearance_bump(count)


def test_discrete_search_is_fixed_smallest_first_and_does_not_change_motor_bounds():
    pairs = ramp_candidates()
    assert len(pairs) == len(set(pairs)) == 64 and pairs[0] == (0.0, 0.0)
    assert all(0 <= x <= 0.3 and 0 <= y <= 0.3 for x, y in pairs)
    cost = [x * x + y * y for x, y in pairs]
    assert cost == sorted(cost)
