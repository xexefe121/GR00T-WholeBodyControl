import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_world_tracking_termination import termination_contract
from gear_sonic.scripts.audit_g1_true23_world_tracking_lifecycle import audit_world_measurements
from gear_sonic.scripts.record_g1_true23_world_tracking_lifecycle import (
    actor_observations,
    captured_world_record,
    scored_world,
    selected_inference_rows,
    training_inference_batch,
    verified_update_checkpoint,
)


def update_report(run):
    checkpoint = (run / "train/checkpoints/world_tracking_model_100.pt").resolve()
    return dict(
        kind="native23_world_tracking_actual_update_audit_v1", passed=True,
        completed_updates=100, actual_transitions=204800,
        initial_actor_and_critic_equal_matched_bounded_predecessor=True,
        additional_training_failure=termination_contract(), inputs={str(checkpoint): "a" * 64},
    )


def test_checkpoint_requires_actual_audit_binding(tmp_path):
    report = update_report(tmp_path)
    path, digest = verified_update_checkpoint(tmp_path, report)
    assert path.name == "world_tracking_model_100.pt" and digest == "a" * 64


@pytest.mark.parametrize("key,value", [
    ("kind", "native23_bounded_progress_actual_update_audit_v1"), ("passed", False),
    ("completed_updates", 2), ("actual_transitions", 64),
    ("initial_actor_and_critic_equal_matched_bounded_predecessor", False),
    ("additional_training_failure", {}), ("inputs", {}),
])
def test_wrong_audit_contract_rejected(tmp_path, key, value):
    report = update_report(tmp_path)
    report[key] = value
    with pytest.raises(ValueError):
        verified_update_checkpoint(tmp_path, report)


def test_cpu_reader_route_removed_once_and_each_actual_world_retained():
    original = dict(tokenizer=torch.arange(4 * 268).reshape(4, 268).float(),
                    policy=torch.zeros(4, 930), root_feedback=torch.zeros(4, 9))
    semantic = actor_observations(original)
    assert original["tokenizer"].shape == (4, 268)
    torch.testing.assert_close(semantic["tokenizer"], original["tokenizer"][:, 1:])
    batch = training_inference_batch(semantic)
    np.testing.assert_array_equal(selected_inference_rows(batch["tokenizer"].numpy()),
                                  semantic["tokenizer"].numpy())
    rows = np.arange(4).astype(np.float64) * 1e-12
    for index in range(4):
        assert scored_world(rows, index) == rows[index]


def test_capture_records_actual_call_without_modifying_tensor():
    entry = dict(common_step_counter=1, error_m=torch.arange(4).float())
    env = SimpleNamespace(common_step_counter=1, _world_root_failure_capture=[entry])
    result = captured_world_record(env)
    result["error_m"][0] = 9
    assert entry["error_m"][0] == 0
    assert "common_step_counter" not in result


@pytest.mark.parametrize("step,entries", [(0, []), (2, []), (2, [{"common_step_counter": 2}]),
                                        (1, [{"common_step_counter": 0}])])
def test_missing_or_duplicate_termination_capture_rejected(step, entries):
    env = SimpleNamespace(common_step_counter=step, _world_root_failure_capture=entries)
    with pytest.raises(ValueError):
        captured_world_record(env)


def world_trace():
    source = np.zeros((14, 36), dtype=np.float64)
    source[10:13, 0] = [0.2, 0.3, 0.4]
    desired = source[10:13, :3].astype(np.float32)
    error = desired[:, 0].copy()
    failure = error > np.float32(0.30)
    trace = dict(
        world_tracking_desired_position_w=desired,
        world_tracking_measured_position_w=np.zeros((3, 3), dtype=np.float32),
        world_tracking_error_m=error, world_tracking_failure=failure,
        world_tracking_reference_q0=np.arange(9, 12, dtype=np.int64),
        world_tracking_common_step_counter=np.arange(1, 4, dtype=np.int64),
        training_episode_terminated=failure.copy(),
        physics_pre_qpos=np.zeros((30, 30), dtype=np.float32),
    )
    return trace, source


def test_world_calls_match_original_q1_actual_q9_and_strict_failure_union():
    trace, source = world_trace()
    result = audit_world_measurements(trace, 3, 0, source)
    assert result["world_failure_controls"] == [2]
    assert result["measured_pelvis_matches_pre_last_substep_exact"] is True
    assert result["actual_world_failure_calls"] == 3


@pytest.mark.parametrize("key", ["world_tracking_error_m", "world_tracking_failure",
                                "training_episode_terminated", "world_tracking_reference_q0",
                                "world_tracking_common_step_counter", "physics_pre_qpos"])
def test_altered_termination_or_measurement_rejected(key):
    trace, source = world_trace()
    altered = copy.deepcopy(trace)
    if key == "world_tracking_failure":
        altered[key][1] = True
    elif key == "training_episode_terminated":
        altered[key][-1] = False
    else:
        altered[key] += 1
    with pytest.raises((ValueError, AssertionError)):
        audit_world_measurements(altered, 3, 0, source)


def test_changed_original_source_rejected():
    trace, source = world_trace()
    source[10, 0] += 0.001
    with pytest.raises(AssertionError):
        audit_world_measurements(trace, 3, 0, source)


def test_missing_calls_rejected():
    trace, source = world_trace()
    trace["world_tracking_error_m"] = trace["world_tracking_error_m"][:-1]
    with pytest.raises(ValueError):
        audit_world_measurements(trace, 3, 0, source)
