import copy

import numpy as np
import pytest

from gear_sonic.scripts.record_g1_sonic_original29_baseline import CLIPS, validate_source_report
from gear_sonic.scripts.retarget_g1_true23_original29_trace import PROFILE, completed_trace_source


def source_material():
    count = 5
    pre = np.zeros((count, 36))
    pre[:, 3] = 1
    pre[:, 0] = np.arange(count) * 0.001
    post = pre.copy()
    post[:-1] = pre[1:]
    post[-1, 0] += 0.001
    record = {
        "profile": PROFILE,
        "failure": None,
        "details": {
            "profile": PROFILE,
            "complete_original_clip": True,
            "pre_qpos_is_same_command_time_state": True,
            "frames_requested": count,
            "frames_completed": count,
            "teacher_accepted": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        },
    }
    arrays = {
        "pre_qpos": pre,
        "qpos": post,
        "planned_qpos50": pre.copy(),
        "control_dt": np.array([0.02]),
        "command_time_s": np.arange(count) * 0.02,
        "post_control_time_s": (np.arange(count) + 1) * 0.02,
        "physics_post_qpos": np.repeat(post, 10, axis=0),
    }
    return record, arrays


def test_complete_source_preserves_every_frame_and_sampling_phase_without_alias():
    record, arrays = source_material()
    source = completed_trace_source(record, arrays)
    np.testing.assert_array_equal(source, arrays["pre_qpos"])
    source[0, 0] = 99
    assert arrays["pre_qpos"][0, 0] == 0


@pytest.mark.parametrize(
    "damage", ["partial", "promoted", "legacy", "phase", "crop", "nan", "time", "last", "gap", "quaternion"]
)
def test_incomplete_or_mislabelled_trace_cannot_become_whole_original_reference(damage):
    record, arrays = source_material()
    if damage == "partial":
        record["details"]["frames_requested"] += 1
        record["details"]["complete_original_clip"] = False
    elif damage == "promoted":
        record["details"]["teacher_accepted"] = True
    elif damage == "legacy":
        record["profile"] = "legacy_python"
    elif damage == "phase":
        arrays["pre_qpos"] = arrays["qpos"].copy()
    elif damage == "crop":
        arrays["planned_qpos50"] = arrays["planned_qpos50"][:-1]
    elif damage == "nan":
        arrays["pre_qpos"][0, 7] = np.nan
    elif damage == "time":
        arrays["command_time_s"] *= 2
    elif damage == "last":
        arrays["physics_post_qpos"] = arrays["physics_post_qpos"][:-1]
    elif damage == "gap":
        arrays["pre_qpos"][2, 0] += 0.1
    else:
        arrays["pre_qpos"][0, 3] = 0.5
    with pytest.raises(ValueError):
        completed_trace_source(record, arrays)


def test_source_report_keeps_failures_and_rejects_omission_or_replacement():
    report = {
        "kind": "g1_released_sonic_planner_motion_suite",
        "fps": 30,
        "passed": False,
        "records": [
            {"name": name, "mode": mode, "npz": f"{name}.npz"}
            for name, mode in zip(CLIPS, (8, 14, 23), strict=True)
        ],
    }
    assert list(validate_source_report(report)) == list(CLIPS)
    for mutate in (
        lambda value: value["records"].pop(),
        lambda value: value["records"].append(value["records"][0]),
        lambda value: value["records"][0].update(npz="../replacement.npz"),
        lambda value: value["records"][0].update(mode=23),
    ):
        broken = copy.deepcopy(report)
        mutate(broken)
        with pytest.raises(ValueError):
            validate_source_report(broken)
