import numpy as np
import pytest

from gear_sonic.scripts.prepare_g1_true23_motion_requests import (
    CLIPS,
    PICO_NAMES,
    assemble_clips,
    full_request_rows,
)


def report():
    return {
        "kind": "g1_true23_interior_effort_full_request_comparison_v1",
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
        "full_eight_clip_qualification": False,
        "records": [
            {
                "name": name,
                "inner_fraction": 1.0,
                "historical_start": False,
                "not_executed": name == "elbow_crawling",
            }
            for name in [*CLIPS, *PICO_NAMES]
        ],
    }


def motion(frames=500):
    return dict(
        fps=np.array([50.0]),
        joint_pos=np.zeros((frames, 23)),
        joint_vel=np.zeros((frames, 23)),
        body_pos_w=np.zeros((frames, 24, 3)),
        body_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (frames, 24, 1)),
        body_lin_vel_w=np.zeros((frames, 24, 3)),
        body_ang_vel_w=np.zeros((frames, 24, 3)),
    )


def test_full_request_ledger_includes_unavailable_elbow():
    rows = full_request_rows(report())
    assert len(rows) == 8
    assert [row["name"] for row in rows if row["not_executed"]] == ["elbow_crawling"]


def test_pico_request_cannot_disappear_or_become_unavailable():
    source = report()
    source["records"].pop()
    with pytest.raises(ValueError, match="five PICO"):
        full_request_rows(source)
    source = report()
    source["records"][-1]["not_executed"] = True
    with pytest.raises(ValueError, match="only the already unavailable"):
        full_request_rows(source)


def test_assembly_preserves_whole_clips_and_exact_spans_after_float32_cast():
    first, second = motion(), motion(511)
    first["joint_pos"][:, 0] = np.arange(500) / 5000
    joined, spans = assemble_clips([("full_motion", first), ("additional_standing", second)])
    assert spans == [
        {"name": "full_motion", "start": 0, "length": 500},
        {"name": "additional_standing", "start": 500, "length": 511},
    ]
    assert joined["joint_pos"].shape == (1011, 23)
    np.testing.assert_array_equal(joined["joint_pos"][:500], first["joint_pos"].astype(np.float32))
    np.testing.assert_array_equal(joined["joint_pos"][500:], second["joint_pos"].astype(np.float32))


def test_short_and_duplicate_clips_are_errors_not_silently_dropped():
    with pytest.raises(ValueError, match="do not silently drop"):
        assemble_clips([("short", motion(499))])
    with pytest.raises(ValueError, match="unique"):
        assemble_clips([("same", motion()), ("same", motion())])
