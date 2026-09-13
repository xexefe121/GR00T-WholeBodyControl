"""Independently recount actual lifecycle learning and complete-request outcomes."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUN = HERE / "smoke_validated"
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
report = read(RUN / "report.json")
assert report["actual_new_updates"] == 2
for path, expected in report["inputs"].items():
    assert inputs.get(str(bind(path))) == expected, path
saved = [torch.load(bind(RUN / f"lifecycle_ppo_model_{step}.pt"), map_location="cpu", weights_only=True)
         for step in range(3)]
prior_path = Path(report["contract"]["initial_actor"]["path"])
prior = torch.load(bind(prior_path), map_location="cpu", weights_only=True)
assert saved[0]["adapter_contract"] == prior["adapter_contract"]
for key, value in prior["adapter_state_dict"].items():
    assert torch.equal(saved[0]["adapter_state_dict"][key], value), key
assert saved[0]["merged_true23_policy_sha256"] == prior["merged_true23_policy_sha256"]
assert not saved[0]["optimizer_state_dict"]["state"]
assert any(not torch.equal(saved[0]["adapter_state_dict"][key], saved[2]["adapter_state_dict"][key])
           for key in saved[0]["adapter_state_dict"])
assert any(not torch.equal(saved[0]["critic_state_dict"][key], saved[2]["critic_state_dict"][key])
           for key in saved[0]["critic_state_dict"])
for index, payload in enumerate(saved):
    assert payload["kind"] == "g1_true23_cpu_lifecycle_lora_ppo_checkpoint_v1"
    assert payload["new_update_count"] == index
    assert payload["mjlab_resume_checkpoint"] is False
    assert not any(payload[key] for key in ("hardware_authorized", "deployment_ready", "promotion_eligible"))
    assert torch.equal(payload["source_std"], saved[0]["source_std"])
    if index:
        assert {int(state["step"]) for state in payload["optimizer_state_dict"]["state"].values()} == {
            payload["actual_minibatches"]}

trace_count = physics_steps = 0


def trace(row):
    global trace_count, physics_steps
    result = row["result"]
    with np.load(bind(row["trace_path"]), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    assert all(np.isfinite(value).all() for value in data.values())
    for kind in ("qpos", "qvel"):
        np.testing.assert_array_equal(data[f"physics_post_{kind}"][:-1], data[f"physics_pre_{kind}"][1:])
    np.testing.assert_array_equal(data["physics_generalized_actuator_force"], data["physics_effort"])
    np.testing.assert_array_equal(data["physics_engine_post_time_s"][:-1], data["physics_engine_pre_time_s"][1:])
    np.testing.assert_allclose(data["physics_engine_post_time_s"] - data["physics_engine_pre_time_s"],
                               .002, atol=1e-12, rtol=0)
    assert not data["physics_engine_warning_counts"].any()
    cap = .2375 * np.asarray(result["effort_limit_hardware_nm"])
    assert np.all(np.abs(data["physics_effort"]) <= cap + 1e-10)
    kp, kd = np.asarray(result["gain_kp_hardware"]), np.asarray(result["gain_kd_hardware"])
    np.testing.assert_allclose(kp * (data["actuation_target"] - data["actuation_q"])
                               - kd * data["actuation_dq"], data["actuation_effort"], atol=1e-12, rtol=0)
    active = result["completed_active_physics_steps"]
    assert active == 10 * result["completed_transitions"] + result["active_partial_transition_substeps"]
    expected = active + sum(result[key].get("completed_physics_steps", 0) for key in ("startup_hold", "return_hold"))
    assert len(data["physics_effort"]) == expected
    trace_count += 1
    physics_steps += expected
    return data


actions = minibatches = 0
episodes = []
for step, learning in enumerate(report["learning"], 1):
    actual = 0
    assert len(learning["episodes"]) == 8
    for row in learning["episodes"]:
        result, case = row["result"], row["case"]
        data = trace(row)
        completed, requested = result["completed_transitions"], result["requested_transitions"]
        n = len(data["policy_raw23"])
        assert n in (completed, completed + 1) and n > 0 and requested == case["expected_transitions"]
        assert result["startup_hold"]["completed_transitions"] == 250
        assert int(np.sum(data["physics_phase"] == 0)) == 2500
        assert data["policy_inference_returned"].all()
        returned = result["return_hold"].get("completed_transitions", 0)
        full_motion = completed == requested and result["failure"] is None and result["motion_fidelity"]["passed"]
        full_return = returned == 250 and result["return_hold"].get("existing_guard_screen_passed") is True
        terminal = (25 if full_motion else -50) + 10 * returned / 250 + (15 if full_motion and full_return else 0)
        rewards = np.zeros(n, np.float64)
        terms = [np.exp(-(data[key] / scale)**2) for key, scale in (
            ("joint_rmse_rad", .35), ("relative_body_error_m", .2),
            ("pelvis_orientation_error_rad", .5), ("pelvis_position_error_m", .25))]
        rewards[:completed] = 1 + np.mean(terms, axis=0)
        rewards[-1] += terminal
        np.testing.assert_allclose(data["ppo_rewards"], rewards, atol=5e-6, rtol=0)
        std = saved[0]["source_std"].numpy().astype(np.float64)
        logp = (-.5 * ((data["policy_raw23"].astype(np.float64) - data["ppo_mean23"]) / std)**2
                - np.log(std) - .5 * math.log(2 * math.pi)).sum(-1)
        np.testing.assert_allclose(data["ppo_log_probability"], logp, atol=2e-5, rtol=0)
        actual += n
        episodes.append(dict(update=step, case=case["label"], actions=n, completed=completed,
                             requested=requested, returned=returned, terminal_reward=terminal))
    assert actual == learning["actual_active_actions"]
    actions += actual
    minibatches += 4 * math.ceil(actual / 128)
    assert learning["minibatches"] == saved[step]["actual_minibatches"] == minibatches
    assert learning["total_active_actions"] == saved[step]["actual_active_actions"] == actions
assert actions == report["actual_active_policy_actions"] == 656
assert minibatches == report["actual_minibatches"] == 24

base = HERE.parent
comparators = {
    "original_v14_100": read(base / "v14_native_ieee_20260906_v1/model_100/evaluation/report.json"),
    "prior_lora_100": read(base / "ieee_motion_ppo_20260906_v2/model_100/evaluation/report.json"),
    "lifecycle_initial": read(RUN / "initial_evaluation/report.json"),
    "lifecycle_plus2": read(RUN / "final_evaluation/report.json"),
}
reference = comparators["prior_lora_100"]["records"]
outcomes = []
for candidate, evaluation in comparators.items():
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
        if candidate.startswith("lifecycle"):
            trace(row)
        if candidate == "lifecycle_initial":
            assert result["completed_transitions"] == previous["result"]["completed_transitions"]
            assert result["return_hold"].get("completed_transitions") == previous["result"]["return_hold"].get("completed_transitions")
        outcomes.append(dict(candidate=candidate, label=row["label"], completed=result["completed_transitions"],
            requested=result["requested_transitions"], returned=result["return_hold"].get("completed_transitions"),
            full_motion_fidelity=result["motion_fidelity"]["passed"],
            lifecycle_passed=result["lifecycle_simulator_screen_passed"]))
with (HERE / "smoke_comparison.json").open("x") as stream:
    json.dump(dict(inputs=inputs, outcomes=outcomes, training_episodes=episodes, new_updates=2,
        actual_active_policy_actions=actions, actual_minibatches=minibatches, continuous_traces=trace_count,
        physics_steps=physics_steps, exact_prior_adapter_initialization=True,
        initial_completion_counts_equal_prior_onnx=True, complete_trajectory_bit_identity_claimed=False,
        same_full_requests_limits_and_cpu_physics=True, equal_total_training_budget=False,
        full_motion_qualified=False, physical_damping_cause_proven=False,
        hardware_authorized=False, deployment_ready=False), stream, indent=2)
print(json.dumps(dict(updates=2, actions=actions, minibatches=minibatches, traces=trace_count, physics_steps=physics_steps)))
