"""Compare complete requests and independently check the new continuous traces."""

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
inputs = {}


def bind(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != inputs.get(str(path), digest):
        raise ValueError(f"comparison input changed: {path}")
    inputs[str(path)] = digest
    return path


def read(path):
    return json.loads(bind(path).read_text())


bind(Path(__file__))
evaluation = read(HERE / "interrupted_export_evaluation_report.json")
assert evaluation["requested_training_updates"] == 500
assert evaluation["completed_updates_logged"] == 410
assert evaluation["completed_update_rollout_transitions"] == 209920
assert evaluation["last_saved_update"] == 400
assert evaluation["experiment_completed"] is False
assert evaluation["original_evaluation_plan"] == [100, 500]
assert evaluation["unavailable_planned_checkpoint"]["update"] == 500
assert evaluation["unavailable_planned_checkpoint"]["not_executed"] is True
assert evaluation["evaluation_updates"] == [100, 400]
assert all(row["return_code"] == 0 for row in evaluation["stages"])
for path, expected in evaluation["inputs"].items():
    bind(path)
    assert inputs[str(Path(path).resolve())] == expected

reports = {
    "original_v14_100": read(HERE.parent / "v14_native_ieee_20260906_v1/model_100/evaluation/report.json"),
    "prior_lora_100": read(HERE.parent / "ieee_motion_ppo_20260906_v2/model_100/evaluation/report.json"),
    "curriculum_plus100": read(HERE / "model_100/evaluation/report.json"),
    "curriculum_plus400_last_saved": read(HERE / "model_400/evaluation/report.json"),
}
baseline = reports["prior_lora_100"]["records"]
rows, trace_count, physics_steps = [], 0, 0
for candidate, report in reports.items():
    assert len(report["records"]) == 11 and report["original_eight_request_set_preserved"] is True
    for old, record in zip(baseline, report["records"], strict=True):
        for key in ("label", "name", "source", "source_sha256", "expected_transitions", "lifecycle", "historical_start"):
            assert record[key] == old[key], (candidate, record["label"], key)
        if record.get("not_executed"):
            assert old.get("not_executed") is True
            rows.append(dict(candidate=candidate, label=record["label"], not_executed=True, reason=record["reason"]))
            continue
        result, prior = record["result"], old["result"]
        for key in ("requested_transitions", "compiled_native_model_sha256", "gain_kp_hardware",
                    "gain_kd_hardware", "effort_limit_hardware_nm", "stateful_native_controller",
                    "previous_action_semantics", "observation_timing", "body_tracking_timing"):
            assert result[key] == prior[key], (candidate, record["label"], key)
        for key in ("startup_hold", "return_hold"):
            assert result[key].get("requested_transitions") == prior[key].get("requested_transitions")
        if candidate.startswith("curriculum_"):
            with np.load(bind(record["trace_path"]), allow_pickle=False) as data:
                assert all(np.isfinite(data[key]).all() for key in data.files)
                assert float(data["physics_dt"][0]) == 0.002
                for kind in ("qpos", "qvel"):
                    np.testing.assert_array_equal(data[f"physics_post_{kind}"][:-1], data[f"physics_pre_{kind}"][1:])
                assert not data["physics_engine_warning_counts"].any()
                np.testing.assert_array_equal(data["physics_generalized_actuator_force"], data["physics_effort"])
                np.testing.assert_allclose(
                    data["physics_engine_post_time_s"] - data["physics_engine_pre_time_s"], 0.002, atol=1e-12, rtol=0
                )
                np.testing.assert_array_equal(data["physics_engine_post_time_s"][:-1], data["physics_engine_pre_time_s"][1:])
                cap = 0.2375 * np.asarray(result["effort_limit_hardware_nm"])
                assert np.all(np.abs(data["physics_effort"]) <= cap + 1e-10)
                kp, kd = np.asarray(result["gain_kp_hardware"]), np.asarray(result["gain_kd_hardware"])
                np.testing.assert_allclose(
                    kp * (data["actuation_target"] - data["actuation_q"]) - kd * data["actuation_dq"],
                    data["actuation_effort"], atol=1e-12, rtol=0,
                )
                active_steps = len(data["actuation_effort"])
                assert active_steps == result["completed_active_physics_steps"]
                assert active_steps == 10 * result["completed_transitions"] + result["active_partial_transition_substeps"]
                expected = active_steps + sum(result[key].get("completed_physics_steps", 0) for key in ("startup_hold", "return_hold"))
                assert len(data["physics_effort"]) == expected
                trace_count += 1
                physics_steps += expected
        rows.append(dict(
            candidate=candidate, label=record["label"],
            completed=result["completed_transitions"], requested=result["requested_transitions"],
            startup_completed=result["startup_hold"].get("completed_transitions"),
            return_completed=result["return_hold"].get("completed_transitions"),
            return_requested=result["return_hold"].get("requested_transitions"),
            full_motion_fidelity=result["motion_fidelity"]["passed"],
            lifecycle_simulator_screen_passed=result["lifecycle_simulator_screen_passed"],
            failure=None if result["failure"] is None else result["failure"]["type"],
        ))
summary = dict(
    inputs=inputs, records=rows, new_continuous_traces_verified=trace_count,
    new_physics_steps_verified=physics_steps, same_complete_requests_and_limits=True,
    curriculum_initial_actor_is_prior_lora100=True,
    equal_total_training_budget_comparison=False,
    planned_update500_unavailable_not_replaced=evaluation["unavailable_planned_checkpoint"],
    requested_500_update_experiment_failed=True,
    training_reward_used_for_checkpoint_selection=False,
    native_unitree_fsm_handoff_proven=False, physical_damping_cause_proven=False,
    hardware_authorized=False, deployment_ready=False,
)
with (HERE / "full_result_comparison.json").open("x") as stream:
    json.dump(summary, stream, indent=2)
for row in rows:
    print(json.dumps(row))
print(json.dumps(dict(new_continuous_traces_verified=trace_count, new_physics_steps_verified=physics_steps)))
