import numpy as np
import pytest

from gear_sonic.scripts.verify_g1_true23_world_tracking_update import audit_world_arrays


def fixture_arrays():
    measured = np.zeros((2, 4, 3), np.float32)
    measured[0, 1, 0] = 0.5
    error = np.linalg.norm(measured, axis=-1)
    failure = error > 0.3
    return dict(
        world_tracking_desired_position_w=np.zeros_like(measured),
        world_tracking_measured_position_w=measured,
        world_tracking_error_m=error,
        world_tracking_failure=failure.copy(),
        world_tracking_reference_q0=np.zeros((2, 4), np.int64),
        world_tracking_common_step_counter=np.arange(1, 3, dtype=np.int64),
        terminated=failure.copy(), stored_done=failure.copy(), timeouts=np.zeros((2, 4), bool),
    )


def test_actual_positions_condition_and_done_union_match():
    report = audit_world_arrays(fixture_arrays(), 2, 4)
    assert report["actual_transitions"] == 8
    assert report["world_failure_transitions"] == 1


@pytest.mark.parametrize("key", ["world_tracking_failure", "terminated", "stored_done"])
def test_omitted_actual_failure_rejected(key):
    arrays = fixture_arrays()
    arrays[key][0, 1] = False
    with pytest.raises(ValueError):
        audit_world_arrays(arrays, 2, 4)


def test_forged_error_rejected():
    arrays = fixture_arrays()
    arrays["world_tracking_error_m"][0, 1] = 0.1
    with pytest.raises(ValueError):
        audit_world_arrays(arrays, 2, 4)


def test_duplicate_or_missing_capture_rejected():
    arrays = fixture_arrays()
    arrays["world_tracking_common_step_counter"][1] = 1
    with pytest.raises(AssertionError):
        audit_world_arrays(arrays, 2, 4)


def test_other_failure_and_timeout_flags_are_preserved():
    arrays = fixture_arrays()
    arrays["terminated"][1, 3] = True
    arrays["stored_done"][1, 3] = True
    arrays["timeouts"][0, 1] = True
    report = audit_world_arrays(arrays, 2, 4)
    assert report["other_failure_transitions_without_world_failure"] == 1
    assert report["simultaneous_world_failure_and_timeout"] == 1
