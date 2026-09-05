import pytest

from gear_sonic.utils.g1_true23_motion_fidelity import MAXIMUM_ERRORS, assess_motion_fidelity


def evaluate(**changes):
    values = dict(metrics=dict(MAXIMUM_ERRORS), completed=535, requested=535, available=535, failure=None)
    values.update(changes)
    return assess_motion_fidelity(**values)


def test_exact_thresholds_pass_without_authorizing_hardware():
    result = evaluate()
    assert result["passed"] and result["full_clip_completed"]
    assert not result["hardware_authorized"] and not result["deployment_ready"]


@pytest.mark.parametrize("changes", [dict(completed=534), dict(completed=10, requested=10), dict(failure={"gate": "height"}), dict(available=True)])
def test_partial_or_failed_clip_cannot_qualify(changes):
    assert "full_clip_completion" in evaluate(**changes)["failed_checks"]


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), -0.01, 5.0104])
def test_invalid_or_historical_five_metre_drift_fails(value):
    metrics = dict(MAXIMUM_ERRORS, maximum_pelvis_position_error_m=value)
    assert "maximum_pelvis_position_error_m" in evaluate(metrics=metrics)["failed_checks"]


def test_historical_happy_completion_is_not_fidelity():
    result = evaluate(metrics={
        "maximum_pelvis_position_error_m": 5.0104,
        "maximum_pelvis_orientation_error_rad": 0.57,
        "maximum_relative_tracked_body_position_error_m": 0.607,
        "maximum_joint_tracking_rmse_rad": 0.71697,
    })
    assert result["full_clip_completed"] and not result["passed"]


def test_missing_metrics_fail_closed():
    assert set(evaluate(metrics={})["failed_checks"]) == set(MAXIMUM_ERRORS)
