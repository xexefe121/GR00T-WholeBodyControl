"""Audit early standing-return probes without relabelling dance completion.

Probe only already observed 50-Hz boundaries. Re-run acquisition and the
unchanged policy to each boundary; do not synthesize an MjData state from
q/qvel or reseed controller targets. Preserve the full failed source request.
"""

from __future__ import annotations

import numpy as np

PHYSICS_FIELDS = (
    "physics_pre_qpos",
    "physics_post_qpos",
    "physics_pre_qvel",
    "physics_post_qvel",
    "physics_effort",
    "physics_generalized_actuator_force",
    "physics_engine_pre_time_s",
    "physics_engine_post_time_s",
    "physics_engine_warning_counts",
    "physics_engine_warning_lastinfo",
    "physics_phase",
)
ACTUATION_FIELDS = tuple(
    "actuation_" + name
    for name in (
        "q",
        "dq",
        "previous_target",
        "requested",
        "target",
        "effort",
        "qacc",
    )
) + ("interior_changed_joints", "interior_relaxed_joints", "interior_outer_effort_ratio")


def probe_plan(case):
    result = case["result"]
    completed, requested = result["completed_transitions"], result["requested_transitions"]
    if (
        case.get("label") != "happy_dance.historical"
        or case.get("historical_start") is not True
        or case.get("lifecycle") is not True
        or type(completed) is not int
        or type(requested) is not int
        or not 0 < completed < requested
        or result.get("failure") is None
        or result.get("motion_fidelity", {}).get("passed") is not False
        or result.get("startup_hold", {}).get("requested_transitions") != 250
        or result["startup_hold"].get("completed_transitions") != 250
        or result["startup_hold"].get("existing_guard_screen_passed") is not True
        or result.get("return_hold", {}).get("requested_transitions") != 250
    ):
        raise ValueError("recovery probes require the original failed full-dance acquired lifecycle")
    return [
        dict(label="full_request_control", stop_after_controls=None, full_request_control=True),
        *(
            dict(label=f"return_after_{step:03d}", stop_after_controls=step, full_request_control=False)
            for step in range(1, completed + 1)
        ),
    ]


def require_full_replay_identity(actual, baseline):
    if set(actual) != set(baseline):
        raise ValueError("full control replay array set changed")
    for name in baseline:
        if not np.array_equal(actual[name], baseline[name]):
            raise ValueError(f"full control replay changed: {name}")
    return dict(exact=True, arrays=len(baseline), physics_steps=len(baseline["physics_phase"]))


def require_prefix_identity(actual, baseline, controls):
    if type(controls) is not int or controls <= 0:
        raise ValueError("recovery prefix must contain positive complete controls")
    phase, old_phase = actual["physics_phase"], baseline["physics_phase"]
    if (
        not np.isin(phase, (0, 1, 2)).all()
        or not np.isin(old_phase, (0, 1, 2)).all()
        or np.any(np.diff(phase) < 0)
        or np.any(np.diff(old_phase) < 0)
    ):
        raise ValueError("recovery phases must be contiguous acquisition, active, return")
    active = np.flatnonzero(phase == 1)
    acquired = np.flatnonzero(phase == 0)
    old_acquired = np.flatnonzero(old_phase == 0)
    old_active = np.flatnonzero(old_phase == 1)
    if (
        len(active) != controls * 10
        or len(old_active) < len(active)
        or len(acquired) != len(old_acquired)
        or len(acquired) == 0
    ):
        raise ValueError("prefix lost acquisition or a complete 500-Hz active interval")
    before_return = len(acquired) + len(active)
    for name in PHYSICS_FIELDS:
        if not np.array_equal(actual[name][:before_return], baseline[name][:before_return]):
            raise ValueError(f"recovery prefix changed before return: {name}")
    for name in ACTUATION_FIELDS:
        if not np.array_equal(actual[name], baseline[name][: len(active)]):
            raise ValueError(f"recovery prefix changed controller history: {name}")
    for name in ("qpos", "startup_hold_qpos"):
        if not np.array_equal(actual[name], baseline[name][: len(actual[name])]):
            raise ValueError(f"recovery prefix changed sampled state: {name}")
    for terminal, recorded in (
        ("terminal_active_qpos", "physics_post_qpos"),
        ("terminal_active_qvel", "physics_post_qvel"),
    ):
        if not np.array_equal(actual[terminal], baseline[recorded][before_return - 1]):
            raise ValueError(f"recovery handoff state does not match its full-run boundary: {terminal}")
    if not np.array_equal(actual["terminal_active_target"], baseline["actuation_target"][len(active) - 1]):
        raise ValueError("recovery handoff target was reseeded")
    returning = np.flatnonzero(phase == 2)
    if len(returning):
        for terminal, recorded in (
            ("terminal_active_qpos", "physics_pre_qpos"),
            ("terminal_active_qvel", "physics_pre_qvel"),
        ):
            if not np.array_equal(actual[terminal], actual[recorded][returning[0]]):
                raise ValueError("return reset the robot state at its first physics step")
    return dict(
        exact=True,
        acquisition_physics_steps=len(acquired),
        active_physics_steps=len(active),
        source_prefix_controls=controls,
        terminal_state_and_previous_target_equal=True,
        first_return_state_equal=True if len(returning) else None,
        return_physics_steps=len(returning),
        reconstructed_mjdata_used=False,
    )


def probe_outcome(result, controls, full_requested):
    if (
        type(controls) is not int
        or type(full_requested) is not int
        or not 0 < controls < full_requested
        or result["requested_transitions"] != controls
        or result["completed_transitions"] != controls
        or result["completed_active_physics_steps"] != controls * 10
        or result["failure"] is not None
        or result["motion_fidelity"]["passed"] is not False
        or result["motion_fidelity"]["full_clip_completed"] is not False
        or result["lifecycle_simulator_screen_passed"] is not False
        or result["actual_engine_audit"]["passed"] is not True
    ):
        raise ValueError("early return cannot hide changed prefix execution or claim full dance")
    returned = result["return_hold"]
    if returned.get("requested_transitions") != 250:
        raise ValueError("recovery probe cannot shorten the standing return")
    success = (
        returned.get("completed_transitions") == 250
        and returned.get("completed_physics_steps") == 2500
        and returned.get("existing_guard_screen_passed") is True
        and returned.get("failure") is None
    )
    return dict(
        stop_after_controls=controls,
        stop_after_s=controls * 0.02,
        full_source_requested_controls=full_requested,
        complete_source_passed=False,
        recovery_screen_passed=success,
        return_completed_controls=returned.get("completed_transitions"),
        return_completed_physics_steps=returned.get("completed_physics_steps"),
        return_failure=returned.get("failure"),
        standing_screen_passed=returned.get("standing_screen_passed"),
        native_unitree_mode_handoff_proven=False,
        hardware_authorized=False,
        deployment_ready=False,
    )


def summarize_window(rows, expected_controls):
    if type(expected_controls) is not int or expected_controls <= 0:
        raise ValueError("recovery window requires its complete observed boundary count")
    if [row["stop_after_controls"] for row in rows] != list(range(1, expected_controls + 1)):
        raise ValueError("recovery window is incomplete, reordered or duplicated")
    if any(
        type(row.get("recovery_screen_passed")) is not bool or row.get("complete_source_passed") is not False
        for row in rows
    ):
        raise ValueError("recovery outcomes cannot be missing or count as source completion")
    passed = [row["stop_after_controls"] for row in rows if row["recovery_screen_passed"]]
    initial_run = 0
    for row in rows:
        if not row["recovery_screen_passed"]:
            break
        initial_run += 1
    return dict(
        tested_boundaries=expected_controls,
        successful_return_boundaries=passed,
        failed_return_boundaries=[row["stop_after_controls"] for row in rows if not row["recovery_screen_passed"]],
        initial_contiguous_success_controls=initial_run,
        successful_boundary_after_an_earlier_failure=any(step > initial_run for step in passed),
        latest_observed_success_controls=max(passed, default=None),
        interpolation_between_boundaries_proven=False,
        unseen_states_covered=False,
        automatic_live_guard_qualified=False,
        full_dance_qualified=False,
        native_unitree_mode_handoff_proven=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
