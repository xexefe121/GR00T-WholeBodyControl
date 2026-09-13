"""Compare a completed 100-update experiment, without shortening failed requests."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE / "train100"
inputs = {}


def bind(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    assert actual == inputs.get(str(path), actual), path
    inputs[str(path)] = actual
    return path


def read(path):
    return json.loads(bind(path).read_text())


bind(Path(__file__))
experiment = read(HERE / "train100_experiment_report.json")
assert experiment["stage"]["return_code"] == 0 and experiment["actual_new_updates"] == 100
report = read(RUN / "report.json")
assert report["actual_new_updates"] == 100 and len(report["learning"]) == 100
for path, expected in report["inputs"].items():
    assert inputs[str(bind(path))] == expected, path
initial = torch.load(bind(RUN / "lifecycle_ppo_model_0.pt"), map_location="cpu", weights_only=True)
final = torch.load(bind(RUN / "lifecycle_ppo_model_100.pt"), map_location="cpu", weights_only=True)
assert initial["new_update_count"] == 0 and not initial["optimizer_state_dict"]["state"]
assert initial["adapter_contract"] == final["adapter_contract"]
assert torch.equal(initial["source_std"], final["source_std"])
assert any(not torch.equal(initial["adapter_state_dict"][key], final["adapter_state_dict"][key])
           for key in initial["adapter_state_dict"])
assert any(not torch.equal(initial["critic_state_dict"][key], final["critic_state_dict"][key])
           for key in initial["critic_state_dict"])
assert final["kind"] == "g1_true23_cpu_lifecycle_lora_ppo_checkpoint_v1" and final["new_update_count"] == 100
assert not final["mjlab_resume_checkpoint"]
for flag in ("hardware_authorized", "deployment_ready", "promotion_eligible"):
    assert final[flag] is False and report[flag] is False
assert report["deployment_artifacts_emitted"] is False

actions = minibatches = 0
training = []
for index, learning in enumerate(report["learning"], 1):
    assert learning["update"] == index and len(learning["episodes"]) == 8
    actual = 0
    for row in learning["episodes"]:
        result, case = row["result"], row["case"]
        assert result["startup_hold"]["completed_transitions"] == 250
        assert result["requested_transitions"] == case["expected_transitions"]
        with np.load(bind(row["trace_path"]), allow_pickle=False) as arrays:
            n = len(arrays["policy_raw23"])
            assert n in (result["completed_transitions"], result["completed_transitions"] + 1)
            assert n > 0 and arrays["policy_inference_returned"].all()
            assert len(arrays["ppo_rewards"]) == n and np.isfinite(arrays["ppo_rewards"]).all()
            actual += n
        training.append(dict(update=index, case=case["label"], completed=result["completed_transitions"],
            requested=result["requested_transitions"], returned=result["return_hold"].get("completed_transitions", 0),
            fidelity=result["motion_fidelity"]["passed"],
            terminal_assessment=result["lifecycle_training_assessment"]["terminal_assessment"]))
    assert actual == learning["actual_active_actions"]
    actions += actual
    minibatches += 4 * math.ceil(actual / 128)
    assert learning["total_active_actions"] == actions and learning["minibatches"] == minibatches
assert actions == final["actual_active_actions"] == report["actual_active_policy_actions"]
assert minibatches == final["actual_minibatches"] == report["actual_minibatches"]
assert {int(state["step"]) for state in final["optimizer_state_dict"]["state"].values()} == {minibatches}

base = HERE.parent
reports = {
    "original_v14_100": read(base / "v14_native_ieee_20260906_v1/model_100/evaluation/report.json"),
    "prior_lora_100": read(base / "ieee_motion_ppo_20260906_v2/model_100/evaluation/report.json"),
    "lifecycle_plus2_smoke": read(HERE / "smoke_validated/final_evaluation/report.json"),
    "lifecycle_initial": read(RUN / "initial_evaluation/report.json"),
    "lifecycle_plus100": read(RUN / "final_evaluation/report.json"),
}
reference = reports["prior_lora_100"]["records"]
outcomes = []
traces = steps = 0
for candidate, evaluation in reports.items():
    assert len(evaluation["records"]) == 11
    for previous, row in zip(reference, evaluation["records"], strict=True):
        for key in ("label", "name", "source", "source_sha256", "expected_transitions", "lifecycle", "historical_start"):
            assert row[key] == previous[key], (candidate, key)
        if row.get("not_executed"):
            assert previous.get("not_executed")
            outcomes.append(dict(candidate=candidate, label=row["label"], not_executed=True))
            continue
        result = row["result"]
        for key in ("requested_transitions", "compiled_native_model_sha256", "gain_kp_hardware", "gain_kd_hardware",
                    "effort_limit_hardware_nm", "stateful_native_controller", "previous_action_semantics",
                    "observation_timing", "body_tracking_timing"):
            assert result[key] == previous["result"][key], (candidate, key)
        for phase in ("startup_hold", "return_hold"):
            assert result[phase].get("requested_transitions") == previous["result"][phase].get("requested_transitions")
        if candidate in ("lifecycle_initial", "lifecycle_plus100"):
            with np.load(bind(row["trace_path"]), allow_pickle=False) as arrays:
                assert all(np.isfinite(arrays[key]).all() for key in arrays.files)
                assert not arrays["physics_engine_warning_counts"].any()
                for kind in ("qpos", "qvel"):
                    np.testing.assert_array_equal(arrays[f"physics_pre_{kind}"][1:], arrays[f"physics_post_{kind}"][:-1])
                np.testing.assert_array_equal(arrays["physics_engine_pre_time_s"][1:], arrays["physics_engine_post_time_s"][:-1])
                np.testing.assert_allclose(arrays["physics_engine_post_time_s"] - arrays["physics_engine_pre_time_s"],
                                           .002, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(arrays["physics_effort"], arrays["physics_generalized_actuator_force"])
                cap = .2375 * np.asarray(result["effort_limit_hardware_nm"])
                assert np.all(np.abs(arrays["physics_effort"]) <= cap + 1e-10)
                kp, kd = np.asarray(result["gain_kp_hardware"]), np.asarray(result["gain_kd_hardware"])
                np.testing.assert_allclose(kp * (arrays["actuation_target"] - arrays["actuation_q"])
                                           - kd * arrays["actuation_dq"], arrays["actuation_effort"], atol=1e-12, rtol=0)
                actual_steps = result["completed_active_physics_steps"]
                assert actual_steps == 10 * result["completed_transitions"] + result["active_partial_transition_substeps"]
                actual_steps += sum(result[phase].get("completed_physics_steps", 0) for phase in ("startup_hold", "return_hold"))
                assert len(arrays["physics_effort"]) == actual_steps
                traces += 1
                steps += actual_steps
        outcomes.append(dict(candidate=candidate, label=row["label"], completed=result["completed_transitions"],
            requested=result["requested_transitions"], acquired=result["startup_hold"].get("completed_transitions"),
            returned=result["return_hold"].get("completed_transitions"),
            full_motion_fidelity=result["motion_fidelity"]["passed"],
            lifecycle_passed=result["lifecycle_simulator_screen_passed"], failure=result["failure"]))

with (HERE / "completed_100_comparison.json").open("x") as stream:
    json.dump(dict(inputs=inputs, outcomes=outcomes, training_episodes=training,
        new_updates=100, actual_active_policy_actions=actions, actual_minibatches=minibatches,
        actual_training_episodes=len(training), independently_checked_evaluation_traces=traces,
        independently_checked_evaluation_physics_steps=steps,
        same_full_requests_limits_and_cpu_physics=True, equal_total_training_budget=False,
        training_reward_used_for_checkpoint_selection=False,
        physical_damping_cause_proven=False, hardware_authorized=False, deployment_ready=False), stream, indent=2)
for row in outcomes:
    if row["candidate"] == "lifecycle_plus100":
        print(json.dumps(row))
print(json.dumps(dict(updates=100, actions=actions, minibatches=minibatches, training_episodes=len(training),
                     evaluation_traces=traces, evaluation_physics_steps=steps)))
