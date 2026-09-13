from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_pd_substep_limits import (
    JOINT_PLANNING_INSET_RAD,
    SPEED_PLANNING_INSET_RAD_S,
    substep_bound_jacobian,
    substep_bound_margins,
)


def fixture_plant():
    ranges = np.tile([-1.0, 1.0], (24, 1))
    return SimpleNamespace(model=SimpleNamespace(jnt_range=ranges), profile=SimpleNamespace(velocity=np.ones(23)))


@pytest.mark.parametrize("kind", ["lower", "upper", "positive_speed", "negative_speed"])
def test_mid_control_joint_or_speed_limit_is_detected(kind):
    poses, velocities = np.zeros((10, 30)), np.zeros((10, 29))
    offset = {"lower": 0, "upper": 230, "positive_speed": 460, "negative_speed": 690}[kind]
    if kind in ("lower", "upper"):
        poses[4, 7 + 8] = -1.001 if kind == "lower" else 1.001
        inset = JOINT_PLANNING_INSET_RAD
    else:
        velocities[4, 6 + 8] = -1.001 if kind == "negative_speed" else 1.001
        inset = SPEED_PLANNING_INSET_RAD_S
    margins = substep_bound_margins(fixture_plant(), poses, velocities)
    assert np.flatnonzero(margins < 0).tolist() == [offset + 4 * 23 + 8]
    assert margins.min() == pytest.approx(-0.001 - inset)
    assert margins[9 * 23 + 8] > 0  # Final substep alone would miss the violation.


def test_bound_jacobian_matches_independent_all_state_perturbation():
    rng = np.random.default_rng(45109)
    poses, velocities = rng.normal(0, 0.1, (10, 30)), rng.normal(0, 0.1, (10, 29))
    q, v, direction = rng.normal(size=(10, 29, 23)), rng.normal(size=(10, 29, 23)), rng.normal(size=23)
    analytic, epsilon = substep_bound_jacobian(q, v) @ direction, 1e-6
    plus, minus = poses.copy(), poses.copy()
    plus[:, 7:] += epsilon * (q[:, 6:] @ direction)
    minus[:, 7:] -= epsilon * (q[:, 6:] @ direction)
    plant = fixture_plant()
    finite = (
        substep_bound_margins(plant, plus, velocities + epsilon * (v @ direction))
        - substep_bound_margins(plant, minus, velocities - epsilon * (v @ direction))
    ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, finite, atol=3e-10, rtol=1e-9)


def test_bound_helpers_reject_incomplete_or_nonfinite_state():
    with pytest.raises(ValueError, match="ten"):
        substep_bound_margins(fixture_plant(), np.zeros((9, 30)), np.zeros((9, 29)))
    with pytest.raises(ValueError, match="finite"):
        substep_bound_jacobian(np.full((10, 29, 23), np.nan), np.zeros((10, 29, 23)))
