"""Recheck full/prefix identity, every return substep and all retained evidence."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_23dof_safe_target_transform import SAFE_TARGET_HARD_LOWER_HARDWARE, SAFE_TARGET_HARD_UPPER_HARDWARE
from gear_sonic.utils.g1_sonic_cpp_observation_trace import audit_engine_trace
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_recovery_window import require_full_replay_identity, require_prefix_identity, summarize_window

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = HERE.parent / "v14_native_ieee_20260906_v1"
inputs, candidates, traces = {}, [], []


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    if digest != inputs.get(str(path), digest) or (expected is not None and digest != expected):
        raise ValueError(f"recovery integrity mismatch: {path}")
    inputs[str(path)] = digest
    return path


def read(path):
    return json.loads(bind(path).read_text())


def merge(report):
    for path, digest in report["inputs"].items():
        bind(path, digest)


def load_arrays(path):
    with np.load(bind(path), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


bind(Path(__file__))
old = read(bind(PREVIOUS / "integrity_report.json", "ef8ecd40ceaa24d2af0f1888783fc9b4507ec5ff9b0ee2b5155730f034c96b22"))
assert old["checked_files"] == 1066 and old["mismatches"] == []
merge(old)
experiment, regression = read(HERE / "experiment_report.json"), read(HERE / "regression_report.json")
merge(experiment)
merge(regression)
assert len(experiment["stages"]) == 3 and all(row["return_code"] == 0 for row in experiment["stages"])
assert regression["exit_code"] == 0 and len(regression["modules"]) == 49
suite = ET.parse(bind(HERE / "regression.xml")).getroot().find("testsuite")
assert int(suite.attrib["tests"]) == 641
assert all(int(suite.attrib[key]) == 0 for key in ("errors", "failures", "skipped"))
lower = np.asarray(SAFE_TARGET_HARD_LOWER_HARDWARE) + .05
upper = np.asarray(SAFE_TARGET_HARD_UPPER_HARDWARE) - .05
total_successes = total_prefixes = return_steps = prefix_steps = 0
for name in ("lora100", "lora300", "v14_100"):
    report = read(HERE / name / "report.json")
    merge(report)
    parent = read(bind(report["full_request_parent"], report["full_request_parent_sha256"]))
    assert report["original_full_request_records"] == parent["records"]
    assert report["original_full_request_plan"] == parent["plan"]
    assert report["pair"] == parent["pair"]
    assert report["full_source_requested_controls"] == 535
    assert report["full_eight_clip_qualification"] is False and report["default_candidate_selected"] is False
    case = next(row for row in parent["records"] if row["label"] == "happy_dance.historical")
    baseline = load_arrays(case["trace_path"])
    expected_count = case["result"]["completed_transitions"]
    assert len(report["records"]) == expected_count + 1 == len(report["plan"])
    outcomes = []
    for entry, plan in zip(report["records"], report["plan"], strict=True):
        assert all(entry[key] == value for key, value in plan.items())
        arrays = load_arrays(entry["trace_path"])
        result = entry["result"]
        audit = audit_engine_trace(arrays)
        assert audit["passed"] and audit == result["actual_engine_audit"]
        assert result["motion_fidelity"]["passed"] is False
        assert result["lifecycle_simulator_screen_passed"] is False
        assert result["compiled_native_model_sha256"] == case["result"]["compiled_native_model_sha256"]
        np.testing.assert_array_equal(arrays["physics_pre_qpos"][1:], arrays["physics_post_qpos"][:-1])
        np.testing.assert_array_equal(arrays["physics_pre_qvel"][1:], arrays["physics_post_qvel"][:-1])
        np.testing.assert_array_equal(arrays["physics_effort"], arrays["physics_generalized_actuator_force"])
        active = np.flatnonzero(arrays["physics_phase"] == 1)
        returning = np.flatnonzero(arrays["physics_phase"] == 2)
        assert len(active) == result["completed_active_physics_steps"]
        assert len(returning) == result["return_hold"]["completed_physics_steps"]
        assert int(np.sum(arrays["physics_phase"] == 0)) == 2500
        assert result["return_hold"]["requested_transitions"] == 250
        for key in ("gain_kp_hardware", "gain_kd_hardware", "target_slew_rad_s", "effort_target_projection"):
            assert result["return_hold"][key] == case["result"]["return_hold"][key]
        kp, kd = np.asarray(result["gain_kp_hardware"]), np.asarray(result["gain_kd_hardware"])
        np.testing.assert_array_equal(arrays["physics_effort"][active], arrays["actuation_effort"])
        np.testing.assert_allclose(arrays["actuation_effort"], kp * (arrays["actuation_target"] - arrays["actuation_q"])
                                   - kd * arrays["actuation_dq"], atol=1e-12, rtol=0)
        effort = np.asarray(result["effort_limit_hardware_nm"])
        assert np.max(np.abs(arrays["physics_effort"]) / effort) <= .2375 + 1e-12
        maximum_slew = None
        if len(returning):
            kp_return = np.asarray(result["return_hold"]["gain_kp_hardware"])
            kd_return = np.asarray(result["return_hold"]["gain_kd_hardware"])
            q = arrays["physics_pre_qpos"][returning, 7:]
            dq = arrays["physics_pre_qvel"][returning, 6:]
            # The legacy return trace does not separately record targets.
            # Reconstruct from its recorded effort/state and known PD gains.
            target = q + (arrays["physics_effort"][returning] + kd_return * dq) / kp_return
            previous_target = np.vstack((arrays["terminal_active_target"], target[:-1]))
            maximum_slew = float(np.max(np.abs(target - previous_target)) / .002)
            assert maximum_slew <= 5 + 1e-9
            assert np.all(target >= lower - 1e-12) and np.all(target <= upper + 1e-12)
        if entry["full_request_control"]:
            assert entry["recovery_outcome"] is None and result == case["result"]
            identity = require_full_replay_identity(arrays, baseline)
        else:
            count = entry["stop_after_controls"]
            identity = require_prefix_identity(arrays, baseline, count)
            outcome = entry["recovery_outcome"]
            assert result["requested_transitions"] == result["completed_transitions"] == count
            assert len(active) == count * 10 and result["failure"] is None
            returned = result["return_hold"]
            expected_success = (returned["completed_transitions"] == 250 and len(returning) == 2500
                                and returned["existing_guard_screen_passed"] is True and returned["failure"] is None)
            assert outcome["recovery_screen_passed"] is expected_success
            assert outcome["complete_source_passed"] is False and outcome["full_source_requested_controls"] == 535
            assert outcome["return_completed_physics_steps"] == len(returning)
            assert outcome["native_unitree_mode_handoff_proven"] is False
            outcomes.append(outcome)
            total_prefixes += 1
            total_successes += int(expected_success)
            prefix_steps += len(active)
            return_steps += len(returning)
        assert entry["identity"] == identity
        traces.append(dict(candidate=name, probe=entry["label"], physics_steps=len(arrays["physics_phase"]),
                           active_steps=len(active), return_steps=len(returning), engine_audit=audit,
                           maximum_reconstructed_return_slew_rad_s=maximum_slew,
                           return_target_reconstruction_not_separate_recorded_channel=True))
    summary = summarize_window(outcomes, expected_count)
    assert summary == report["summary"]
    failures_with_immediate_room = sum(not row["recovery_screen_passed"] and row["handoff_state"]["immediate_return_interval_nonempty"] for row in outcomes)
    successes = [row for row in outcomes if row["recovery_screen_passed"]]
    failures = [row for row in outcomes if not row["recovery_screen_passed"]]
    candidates.append(dict(
        candidate=name, summary=summary, report_sha256=file_sha256(HERE / name / "report.json"),
        first_failed_probe=failures[0] if failures else None, last_successful_probe=successes[-1] if successes else None,
        failed_returns_despite_nonempty_immediate_interval=failures_with_immediate_room,
        lowest_handoff_tilt_failed_probe=min(failures, key=lambda row: row["handoff_state"]["tilt_rad"]) if failures else None,
        incomplete_return_failure_types=sorted({str(row["return_failure"]) for row in failures}),
    ))
assert total_prefixes == 140 and len(traces) == 143
for path in HERE.glob("*.json"):
    bind(path)
for path in HERE.glob("*.py"):
    bind(path)
for path in list(inputs):
    bind(path)
dump(HERE / "integrity_report.json", dict(
    kind="g1_true23_recovery_window_integrity_v1", inputs=inputs, checked_files=len(inputs),
    previous_pinned_files=1066, mismatches=[], regression_tests=641, full_request_control_replays=3,
    exact_pre_return_prefixes=total_prefixes, exact_active_prefix_physics_steps=prefix_steps,
    successful_5s_return_probes=total_successes, failed_return_probes=total_prefixes-total_successes,
    actual_return_physics_steps=return_steps, actual_all_physics_steps=sum(row["physics_steps"] for row in traces),
    traces=traces, candidates=candidates, full_source_dance_requested_controls=535,
    no_training_or_policy_change=True, full_dance_qualified=False, full_eight_clip_qualification=False,
    automatic_live_recovery_guard_qualified=False, native_unitree_mode_handoff_proven=False,
    physical_bit30_failure_explained=False, hardware_authorized=False, deployment_ready=False))
print(json.dumps(dict(checked_files=len(inputs), mismatches=0, regression_tests=641,
                       exact_prefixes=total_prefixes, successful_returns=total_successes,
                       physics_steps=sum(row["physics_steps"] for row in traces))), flush=True)
