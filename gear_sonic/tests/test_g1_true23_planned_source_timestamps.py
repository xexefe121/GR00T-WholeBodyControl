from pathlib import Path

import pytest

from gear_sonic.scripts.audit_g1_true23_planned_source_timestamps import validate_planned_bindings


def material():
    paths = {key: Path("/bound") / key for key in ("trace", "source_model", "target_model", "adapted", "named")}
    bindings = {str(path): key * 2 for key, path in paths.items()}
    report = {
        "accepted": True,
        "source_field": "planned_qpos50",
        "recorded_policy_pose_used_as_choreography": False,
        "adapted_motion_sha256": bindings[str(paths["adapted"])],
        "named_source_sha256": bindings[str(paths["named"])],
        "input_bindings": {
            str(paths[key]): bindings[str(paths[key])] for key in ("trace", "source_model", "target_model")
        },
    }
    return report, paths, bindings


def test_accepted_original_planner_bindings():
    validate_planned_bindings(*material())


@pytest.mark.parametrize("key", ["trace", "source_model", "target_model", "adapted", "named"])
def test_any_changed_motion_or_model_binding_rejected(key):
    report, paths, bindings = material()
    bindings[str(paths[key])] = "changed"
    with pytest.raises(ValueError, match="hash-bound original"):
        validate_planned_bindings(report, paths, bindings)


@pytest.mark.parametrize(
    "key,value",
    [("accepted", False), ("source_field", "post_qpos"), ("recorded_policy_pose_used_as_choreography", True)],
)
def test_rejected_or_recorded_policy_path_cannot_become_original_source(key, value):
    report, paths, bindings = material()
    report[key] = value
    with pytest.raises(ValueError, match="not recorded policy poses"):
        validate_planned_bindings(report, paths, bindings)
