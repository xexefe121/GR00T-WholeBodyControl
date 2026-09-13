import numpy as np
import pytest

from gear_sonic.scripts.measure_g1_true23_original_source_tracking import point_statistics, source_phase_poses


def material():
    poses = np.zeros((101, 30))
    poses[:, 0] = np.arange(101)
    timeline = {
        "total_requested_controls": 100,
        "phases": [
            {"name": "source_motion", "control_start": 20, "control_stop": 80, "frame_start": 31, "frame_stop": 91}
        ],
    }
    result = {"completed_controls": 100, "requested_controls": 100, "failure": None}
    return {"qpos": poses}, timeline, result


def test_complete_source_uses_postcontrol_states_without_lag_shift():
    trace, timeline, result = material()
    poses, _ = source_phase_poses(trace, timeline, result)
    np.testing.assert_array_equal(poses[:, 0], np.arange(21, 81))


@pytest.mark.parametrize("defect", ["prefix", "failure", "nonfinite", "short_trace", "shifted_reference"])
def test_incomplete_or_shifted_data_rejected(defect):
    trace, timeline, result = material()
    if defect == "prefix":
        result["completed_controls"] -= 1
    elif defect == "failure":
        result["failure"] = "fallen"
    elif defect == "nonfinite":
        trace["qpos"][30, 7] = np.nan
    elif defect == "short_trace":
        trace["qpos"] = trace["qpos"][:-1]
    else:
        timeline["phases"][0]["frame_start"] += 1
    with pytest.raises(ValueError, match="complete unshifted"):
        source_phase_poses(trace, timeline, result)


def test_world_error_not_erased_by_relative_diagnostic():
    original = np.zeros((10, 2, 3))
    actual = original.copy()
    actual[..., 0] = 2
    root = np.zeros((10, 3))
    shifted = root.copy()
    shifted[:, 0] = 2
    result = point_statistics(actual, original, shifted, root, ["foot", "hand"])
    assert result["world"]["foot"]["p95_m"] == 2
    assert result["pelvis_relative"]["foot"]["p95_m"] == 0


def test_mismatched_task_samples_rejected():
    with pytest.raises(ValueError, match="complete finite"):
        point_statistics(
            np.zeros((10, 2, 3)), np.zeros((9, 2, 3)), np.zeros((10, 3)), np.zeros((10, 3)), ["a", "b"]
        )
