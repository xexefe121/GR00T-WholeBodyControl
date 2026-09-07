from dataclasses import asdict

import pytest

from gear_sonic.utils.g1_true23_generalist_retarget import AdaptationLimits, retained_candidate_limits


def report(*, excursion=0.9, actual=2.0, **limit_changes):
    limits = asdict(AdaptationLimits(duration_scales=(2.0,), excursion_scales=(excursion,)))
    limits.update(limit_changes)
    return {
        "accepted": False,
        "source_role": "requested_choreography",
        "limits": limits,
        "attempts": [
            {
                "requested_duration_scale": 2.0,
                "requested_excursion_scale": excursion,
                "actual_duration_scale": actual,
            }
        ],
    }


def test_original_fixed_candidate_stays_unchanged():
    assert retained_candidate_limits(report()) == AdaptationLimits(duration_scales=(2.0,), excursion_scales=(0.9,))


def test_120hz_source_uses_bounded_exact50hz_duration_and_unreduced_excursion():
    duration = 796 / 120
    actual = 663 / 50 / duration
    limits = retained_candidate_limits(report(excursion=1.0, actual=actual, maximum_output_frames=2500))
    assert limits.excursion_scales == (1.0,)
    assert 1.99 < actual < 2


@pytest.mark.parametrize(
    "change",
    [
        {"foot_p95_m": 0.06},
        {"hand_head_p95_m": 0.11},
        {"max_joint_velocity_rad_s": 9},
        {"max_joint_acceleration_rad_s2": 81},
        {"max_duration_scale": 2.1},
        {"max_excursion_reduction": 0.3},
        {"target_fps": 25},
    ],
)
def test_generalization_cannot_weaken_gates(change):
    with pytest.raises(ValueError):
        retained_candidate_limits(report(**change))


@pytest.mark.parametrize("actual", [float("nan"), float("inf"), 0.99, 2.01, True])
def test_invalid_retained_duration_rejected(actual):
    with pytest.raises(ValueError):
        retained_candidate_limits(report(actual=actual))


def test_declared_scale_and_parent_must_match():
    value = report()
    value["attempts"][0]["requested_excursion_scale"] = 1.0
    with pytest.raises(ValueError, match="differ"):
        retained_candidate_limits(value)
    value = report()
    value["source_role"] = "robot_state"
    with pytest.raises(ValueError, match="source role"):
        retained_candidate_limits(value)
