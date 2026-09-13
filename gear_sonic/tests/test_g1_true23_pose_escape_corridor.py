import numpy as np
import pytest

from gear_sonic.utils.g1_true23_pose_escape_corridor import pose_escape_objective_target


def test_guidance_is_local_exact_at_anchor_and_never_declared_a_feasible_motion():
    initial = np.arange(300 * 29).reshape(300, 29) * 1e-6
    before = initial.copy()
    escaped = initial[150] + 0.1
    result, report = pose_escape_objective_target(initial, 150, escaped, radius_controls=50)
    np.testing.assert_array_equal(initial, before)
    np.testing.assert_allclose(result[150], escaped, atol=1e-16)
    np.testing.assert_array_equal(result[:101], before[:101])
    np.testing.assert_array_equal(result[200:], before[200:])
    assert report["affected_objective_frames"] == list(range(101, 200))
    assert report["numerical_objective_target_only"]
    assert not report["guidance_is_an_accepted_motion"]
    assert not report["original29_task_targets_replaced"]


@pytest.mark.parametrize("frame,radius", [(-1, 100), (300, 100), (True, 100), (150, 0), (150, 151)])
def test_guidance_rejects_unknown_frame_or_unbounded_neighborhood(frame, radius):
    with pytest.raises(ValueError):
        pose_escape_objective_target(np.zeros((300, 29)), frame, np.zeros(29), radius_controls=radius)


def test_last_frame_guidance_keeps_whole_timeline_without_padding_or_dropping_knots():
    initial = np.zeros((200, 29))
    result, report = pose_escape_objective_target(initial, 199, np.ones(29))
    assert result.shape == initial.shape
    np.testing.assert_array_equal(result[:100], initial[:100])
    np.testing.assert_array_equal(result[-1], np.ones(29))
    assert report["affected_objective_frames"][-1] == 199
