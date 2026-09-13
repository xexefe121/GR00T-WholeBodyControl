import numpy as np
import pytest

from gear_sonic.scripts.audit_g1_true23_source_action_reachability import measure_source_reachability
from gear_sonic.trl.mjlab.native23_projected_target_ppo import reachable_source_bounds


def test_interior_means_have_no_projected_or_dead_exploration_controls():
    low, high = reachable_source_bounds()
    report = measure_source_reachability(np.tile((low + high) / 2, (10, 1)), 0.1)
    assert report["any_mean_projected_fraction"] == 0
    assert report["projection_loss_radian_squared"] == 0
    assert report["any_joint_less_than_one_percent_reentry_probability_fraction"] == 0


def test_gaussian_tail_bound_detects_both_sides_without_changing_inputs():
    low, high = reachable_source_bounds()
    means = np.tile((low + high) / 2, (10, 1))
    means[0, 15], means[1, 16] = low[15] - 3, high[16] + 2.2
    before = means.copy()
    report = measure_source_reachability(means, 0.1)
    assert report["any_mean_projected_fraction"] == 0.2
    assert report["any_joint_less_than_one_percent_reentry_probability_fraction"] == 0.2
    assert report["joints"]["left_ankle_pitch_joint"][
        "maximum_outside_distance_in_std_upper_units"
    ] == pytest.approx(30)
    np.testing.assert_array_equal(means, before)


@pytest.mark.parametrize("std", [0, 0.51, float("nan")])
def test_invalid_noise_cannot_make_reachability_claim(std):
    with pytest.raises(ValueError, match="exploration bound"):
        measure_source_reachability(np.zeros((3, 23)), std)
