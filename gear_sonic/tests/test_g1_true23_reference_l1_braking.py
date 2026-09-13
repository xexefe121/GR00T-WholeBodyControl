import numpy as np
import pytest

from gear_sonic.utils.g1_true23_reference_l1_braking import l1_stopping_path


def settings():
    return dict(maximum_velocity=np.full(3, 1.4925), maximum_acceleration=np.full(3, 11.94), limit_rad=0.44775)


def test_saved_dead_end_would_be_rejected_before_rotation_budget_is_exhausted():
    candidate = np.array([-0.07557206272145284, 0.016583338804833845, -0.3555945984737133])
    velocity = np.array([-1.2234366940904504, -0.06230474343189331, -0.778781411926055])
    previous = candidate - 0.02 * velocity
    assert np.abs(candidate).sum() <= 0.44775 + 1e-15
    margin, _, evidence = l1_stopping_path(candidate, previous, **settings())
    assert margin.min() < -0.02
    assert np.max(np.sum(np.abs(evidence["reference_coordinate_path"]), axis=1)) > 0.45


def test_one_common_stop_respects_every_halfspace_velocity_and_acceleration():
    candidate = np.array([0.1, -0.07, 0.03])
    previous = candidate - 0.02 * np.array([0.5, -0.7, 1.2])
    margins, _, evidence = l1_stopping_path(candidate, previous, **settings())
    assert margins.min() >= 0
    positions, velocity = evidence["reference_coordinate_path"], evidence["reference_coordinate_velocity"]
    np.testing.assert_allclose(np.diff(positions, axis=0), velocity[1:] * 0.02, atol=1e-16, rtol=0)
    assert np.max(np.abs(velocity)) <= 1.4925
    assert np.max(np.abs(np.diff(velocity, axis=0))) <= 11.94 * 0.02 + 1e-15
    np.testing.assert_array_equal(velocity[-1], np.zeros(3))
    assert np.max(np.sum(np.abs(positions), axis=1)) <= 0.44775
    assert evidence["shared_braking_schedule"]
    assert not evidence["deployment_ready"]


def test_exact_stopping_path_jacobian_away_from_active_set_changes():
    candidate = np.array([0.1, -0.07, 0.03])
    previous = candidate - 0.02 * np.array([0.5, -0.7, 1.2])
    _, jacobian, _ = l1_stopping_path(candidate, previous, **settings())
    numerical = []
    for column in range(3):
        delta = np.eye(3)[column] * 1e-7
        plus = l1_stopping_path(candidate + delta, previous, **settings())[0]
        minus = l1_stopping_path(candidate - delta, previous, **settings())[0]
        numerical.append((plus - minus) / 2e-7)
    np.testing.assert_allclose(jacobian, np.asarray(numerical).T, atol=1e-8, rtol=1e-8)


def test_static_boundary_pose_with_zero_velocity_is_admissible():
    pose = np.array([0.1, -0.2, 0.14775])
    margins, _, evidence = l1_stopping_path(pose, pose, **settings())
    assert margins.min() >= -1e-15
    np.testing.assert_array_equal(evidence["reference_coordinate_path"], np.tile(pose, (8, 1)))


@pytest.mark.parametrize(
    "changes",
    [
        {"limit_rad": 0.46},
        {"maximum_velocity": np.full(3, 1.51)},
        {"maximum_acceleration": np.full(3, 12.1)},
        {"dt": 0},
    ],
)
def test_limits_not_expanded(changes):
    kwargs = {**settings(), **changes}
    with pytest.raises(ValueError):
        l1_stopping_path(np.zeros(3), np.zeros(3), **kwargs)


def test_candidate_outside_velocity_contract_rejected():
    with pytest.raises(ValueError, match="candidate velocity"):
        l1_stopping_path(np.array([0.1, 0, 0]), np.zeros(3), **settings())


def test_computational_horizon_is_bounded():
    with pytest.raises(ValueError, match="computational budget"):
        l1_stopping_path(np.zeros(3), np.zeros(3), **{**settings(), "maximum_acceleration": np.full(3, 0.001)})
