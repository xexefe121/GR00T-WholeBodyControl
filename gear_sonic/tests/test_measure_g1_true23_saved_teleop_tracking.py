import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.measure_g1_true23_saved_teleop_tracking import align_reference_once, source_indices


def material():
    poses = np.zeros((5, 30))
    poses[:, 2:4] = [0.75, 1]
    trace = dict(qpos=poses, simulation_time=np.arange(5) * 0.02)
    packets = [dict(control_source_frame_index=10 + i, pico_anchor_source_frame_index=9 + i) for i in range(4)]
    report = dict(
        requested_controls=4,
        recorded_post_attempt_states=4,
        controller_completed_controls=4,
        fallback_first_transition=None,
        fallback_active=False,
        successful_controls=4,
        failure=None,
        passed=True,
    )
    return trace, packets, report


def test_elapsed_source_frame_is_q11_after_first_step_not_q10():
    trace, packets, report = material()
    indexes, sonic = source_indices(trace, packets, report, 15)
    np.testing.assert_array_equal(indexes, np.arange(10, 15))
    assert sonic == 4


@pytest.mark.parametrize("defect", ["shift", "time", "duplicate", "nan", "quaternion", "short", "count", "pass"])
def test_changed_timeline_or_outcome_rejected(defect):
    trace, packets, report = material()
    count = 15
    if defect == "shift":
        packets[0]["pico_anchor_source_frame_index"] += 1
    elif defect == "time":
        trace["simulation_time"] += 0.02
    elif defect == "duplicate":
        packets[1] = packets[0]
    elif defect == "nan":
        trace["qpos"][1, 0] = np.nan
    elif defect == "quaternion":
        trace["qpos"][1, 3] = 2
    elif defect == "short":
        count = 14
    elif defect == "count":
        report["controller_completed_controls"] -= 1
    else:
        report["passed"] = False
    with pytest.raises(ValueError):
        source_indices(trace, packets, report, count)


def test_failed_prefix_and_fallback_not_counted_as_sonic_success():
    trace, packets, report = material()
    packets += [dict(control_source_frame_index=14, pico_anchor_source_frame_index=13)]
    report.update(
        requested_controls=5,
        fallback_active=True,
        fallback_first_transition=2,
        failure="fallen",
        successful_controls=3,
        passed=False,
    )
    indexes, sonic = source_indices(trace, packets, report, 16)
    assert len(indexes) == 5
    assert sonic == 2


def test_preintegration_rejection_keeps_failure_without_extra_physics_sample():
    trace, packets, report = material()
    packets += [dict(control_source_frame_index=14, pico_anchor_source_frame_index=13)]
    report.update(
        requested_controls=5,
        failure="action rejected before integration",
        failed_attempt_integrated=False,
        attempted_controls=5,
        passed=False,
    )
    indexes, sonic = source_indices(trace, packets, report, 16)
    assert len(indexes) == 5 and sonic == 4
    report["failed_attempt_integrated"] = True
    with pytest.raises(ValueError, match="failure/success count"):
        source_indices(trace, packets, report, 16)


def test_alignment_preserves_height_motion_turns_and_joints():
    reference = np.zeros((3, 30))
    reference[:, :3] = [[2, 3, 0.70], [3, 3, 0.75], [3, 4, 0.73]]
    reference[:, 3:7] = Rotation.from_euler("z", [0.5, 0.8, 1.0]).as_quat()[:, [3, 0, 1, 2]]
    reference[:, 7:] = np.arange(23)
    untouched = reference.copy()
    actual = reference[0].copy()
    actual[:3], actual[3:7] = [0, 0, 0.9], [1, 0, 0, 0]
    aligned, calibration = align_reference_once(reference, actual)
    np.testing.assert_allclose(aligned[0, :2], 0, atol=1e-15)
    np.testing.assert_array_equal(aligned[:, 2], reference[:, 2])
    np.testing.assert_array_equal(aligned[:, 7:], reference[:, 7:])
    np.testing.assert_allclose(
        Rotation.from_quat(aligned[:, [4, 5, 6, 3]]).as_euler("xyz")[:, 2],
        [0, 0.3, 0.5],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(aligned[:, :3], axis=0), axis=1),
        np.linalg.norm(np.diff(reference[:, :3], axis=0), axis=1),
    )
    np.testing.assert_array_equal(reference, untouched)
    assert calibration["translation_xyz_m"][2] == 0
    assert calibration["height_aligned"] is False
