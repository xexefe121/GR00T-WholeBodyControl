from pathlib import Path

import pytest

from gear_sonic.scripts.evaluate_g1_true23_motion_ppo import evaluation_plan
from gear_sonic.scripts.prepare_g1_true23_motion_requests import CLIPS, PICO_NAMES


def suite():
    rows = []
    for name in [*CLIPS, *PICO_NAMES]:
        for historical in (False, True) if name == "happy_dance" else (False,):
            rows.append(
                {
                    "name": name,
                    "inner_fraction": 1,
                    "historical_start": historical,
                    "not_executed": name == "elbow_crawling",
                    "result": {"requested_transitions": 500},
                    "source_motion_path": f"{name}.npz",
                    "source_motion_sha256": "a" * 64,
                }
            )
    return {
        "kind": "g1_true23_interior_effort_full_request_comparison_v1",
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
        "full_eight_clip_qualification": False,
        "records": rows,
    }


def test_full_suite_and_two_distinct_stationary_regressions():
    plan = evaluation_plan(suite(), Path("stationary.npz"))
    assert len(plan) == 11
    assert len({row["label"] for row in plan}) == 11
    assert sum(row["unavailable"] for row in plan) == 1
    assert sum(row["stationary_prerequisite_only"] for row in plan) == 2
    assert sum(row["lifecycle"] for row in plan) == 2
    assert [row["name"] for row in plan if row["unavailable"]] == ["elbow_crawling"]


def test_historical_dance_return_cannot_be_removed():
    report = suite()
    report["records"] = [row for row in report["records"] if not row["historical_start"]]
    with pytest.raises(ValueError, match="historical dance lifecycle"):
        evaluation_plan(report, Path("stationary.npz"))
