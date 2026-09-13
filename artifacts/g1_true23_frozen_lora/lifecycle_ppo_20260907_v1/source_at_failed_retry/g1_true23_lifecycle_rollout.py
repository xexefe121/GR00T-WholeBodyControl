"""Complete motion requests between actual simulated standing transitions."""

import numpy as np

from gear_sonic.utils.g1_true23_lifecycle_ppo import terminal_episode_rewards
from gear_sonic.utils.g1_true23_policy_input_trace import run_recorded_case


def lifecycle_training_plan(plan):
    if len(plan) != 11 or len({row["label"] for row in plan}) != 11:
        raise ValueError("lifecycle training requires the complete original evaluation plan")
    selected = [row for row in plan if not row["historical_start"] and row["label"] != "standing.acquired"]
    if len(selected) != 9 or sum(row["unavailable"] for row in selected) != 1:
        raise ValueError("lifecycle plan must retain eight original requests and separate standing")
    if sum(row["stationary_prerequisite_only"] for row in selected) != 1:
        raise ValueError("stationary training must not replace an original motion")
    return [
        {
            **row,
            "training_initial_state": "historical_after_actual_five_second_acquisition",
            "training_return_requested_controls": 250,
        }
        for row in selected
    ]


def case_options(*, root, assets, policy, profile, measured, balance, case, training):
    if case["unavailable"]:
        raise ValueError("unavailable original motion cannot be fabricated or replaced")
    with np.load(case["source"], allow_pickle=False) as archive:
        motion = {key: archive[key].copy() for key in archive.files}
    lifecycle = training or case["lifecycle"]
    historical = training or case["historical_start"]
    return dict(
        fraction=1.0,
        root=root,
        asset_root=assets,
        policy=policy,
        motion=motion,
        kp=np.asarray(profile.kp),
        kd=np.asarray(profile.kd),
        joint_scale=np.ones(23),
        ankle_effort=35.0,
        slew_rate=5.0,
        initial_state="measured" if historical else "reference",
        maximum_steps=None,
        measured_state=measured if historical else None,
        startup_hold_s=5.0 if lifecycle else 0.0,
        return_hold_s=5.0 if lifecycle else 0.0,
        transition_policy=balance if lifecycle else None,
        align_reference_start=lifecycle,
        project_transition_effort=lifecycle,
        project_active_effort=True,
        stateful_native_controller=True,
        trace_active_actuation=True,
    )


def collect_attempt(*, options, case, policy_update, compiled_model_sha256):
    if options["startup_hold_s"] != 5.0 or options["return_hold_s"] != 5.0 or options["maximum_steps"] is not None:
        raise ValueError("lifecycle PPO cannot truncate acquisition, motion or return")
    policy = options["policy"]
    policy.records = []
    report, arrays = run_recorded_case(**options)
    if report["requested_transitions"] != case["expected_transitions"]:
        raise ValueError("lifecycle rollout shortened its source motion")
    if (
        report["compiled_native_model_sha256"] != compiled_model_sha256
        or not report["actual_engine_audit"]["passed"]
    ):
        raise ValueError("lifecycle training changed native evaluation physics or recorded invalid integration")
    if report["startup_hold"].get("completed_transitions") != 250:
        raise ValueError("failed acquisition is retained as failure, not silently used as active PPO data")
    rewards, assessment = terminal_episode_rewards(report, arrays)
    if len(policy.records) != len(rewards):
        raise ValueError("PPO inference records differ from actual controller calls")
    np.testing.assert_array_equal(np.asarray([row["action"] for row in policy.records]), arrays["policy_raw23"])
    arrays.update(
        ppo_rewards=rewards,
        ppo_value=np.asarray([row["value"] for row in policy.records]),
        ppo_log_probability=np.asarray([row["log_prob"] for row in policy.records]),
        ppo_mean23=np.asarray([row["mean"] for row in policy.records]),
    )
    episode = dict(records=list(policy.records), rewards=rewards, policy_update_at_collection=policy_update)
    return episode, {**report, "lifecycle_training_assessment": assessment}, arrays
