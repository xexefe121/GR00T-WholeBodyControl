from copy import deepcopy

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_ramp_rate_projection import project_generated_ramp_rates


def materials():
    poses = np.zeros((80, 30))
    poses[:, 3] = 1.0
    poses[5:20, 7] = np.linspace(0.0, 0.2, 17)[1:-1]
    poses[20:57, 7] = 0.2
    poses[57:60, 7] = [0.22, 0.26, 0.32]
    poses[60:75, 7] = np.linspace(0.32, 0.0, 17)[1:-1]
    timeline = {
        "phases": [
            {"name": "acquisition_ramp", "frame_start": 5, "frame_stop": 20},
            {"name": "source_motion", "frame_start": 20, "frame_stop": 60},
            {"name": "return_ramp", "frame_start": 60, "frame_stop": 75},
        ]
    }
    return poses, timeline, np.full(23, -1.0), np.ones(23)


def test_ramp_rate_projection_preserves_exact_source_standing_and_root():
    poses, timeline, low, high = materials()
    original = poses.copy()
    result, proof = project_generated_ramp_rates(poses, timeline, low, high)
    np.testing.assert_array_equal(poses, original)
    np.testing.assert_array_equal(result[20:60], poses[20:60])
    np.testing.assert_array_equal(result[:5], poses[:5])
    np.testing.assert_array_equal(result[75:], poses[75:])
    np.testing.assert_array_equal(result[:, :7], poses[:, :7])
    assert proof["path_bounds"]["passed"]
    assert proof["path_bounds"]["max_acceleration_abs"] <= 80.0005
    assert proof["maximum_actual_joint_change_rad"] <= 0.1
    assert proof["source_and_standing_poses_bit_exact"]
    assert not proof["training_reference_accepted"]


def test_impossible_frozen_source_boundary_rejected_before_projection(monkeypatch):
    poses, timeline, low, high = materials()
    poses[58, 7], poses[59, 7] = 0.94, 0.99
    monkeypatch.setattr(
        "gear_sonic.utils.g1_true23_ramp_rate_projection.project_nearest_trajectory",
        lambda *a, **kw: pytest.fail("must reject necessary terminal failure before optimization"),
    )
    with pytest.raises(ValueError, match="retarget endpoint first"):
        project_generated_ramp_rates(poses, timeline, low, high)


@pytest.mark.parametrize("mutation", ["source_overlap", "missing_return", "unbounded_delta", "nonfinite"])
def test_projection_cannot_change_source_or_enlarge_edit_scope(mutation):
    poses, timeline, low, high = materials()
    timeline = deepcopy(timeline)
    kwargs = {}
    if mutation == "source_overlap":
        timeline["phases"][0]["frame_stop"] = 21
    elif mutation == "missing_return":
        timeline["phases"].pop()
    elif mutation == "unbounded_delta":
        kwargs["maximum_joint_change_rad"] = 0.2
    else:
        poses[5, 8] = np.nan
    with pytest.raises(ValueError):
        project_generated_ramp_rates(poses, timeline, low, high, **kwargs)
