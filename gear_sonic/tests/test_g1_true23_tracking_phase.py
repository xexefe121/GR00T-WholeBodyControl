import numpy as np
import pytest

from gear_sonic.scripts.measure_g1_true23_tracking_phase import measure_phase


def material():
    reference = np.zeros((160, 23))
    reference[:, :12] = np.sin(np.arange(160)[:, None] * np.arange(1, 13)[None] * 0.01)
    qpos = np.zeros((150, 30))
    qpos[1:, 7:] = reference[11:160]
    relative = np.zeros((149, 5))
    return dict(qpos=qpos, relative_landmark_error_m=relative), reference


def test_unshifted_perfect_tracking():
    trace, reference = material()
    result = measure_phase(trace, reference, 20, 120)
    assert result["leg_rmse_rad"] == 0
    assert result["lag_diagnostic"]["best_reference_offset_frames"] == 0


def test_delay_diagnosis_never_replaces_unshifted_score():
    trace, reference = material()
    trace["qpos"][1:, 7:] = reference[1:150]
    result = measure_phase(trace, reference, 20, 120)
    assert result["leg_rmse_rad"] > 0.1
    assert result["lag_diagnostic"]["estimated_measured_leg_lag_s"] == 0.2
    assert result["lag_diagnostic"]["shifted_interior_leg_rmse_rad"] == 0
    assert not result["lag_diagnostic"]["reference_or_accepted_metrics_time_shifted"]
    assert not result["deployment_ready"]


def test_incomplete_dance_phase_rejected():
    trace, reference = material()
    with pytest.raises(ValueError, match="complete"):
        measure_phase(trace, reference, 20, 180)
