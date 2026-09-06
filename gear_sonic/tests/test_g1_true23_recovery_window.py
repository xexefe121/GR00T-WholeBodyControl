import copy

import numpy as np
import pytest

from gear_sonic.utils import g1_true23_recovery_window as window


def case():
    return dict(
        label="happy_dance.historical",
        historical_start=True,
        lifecycle=True,
        result=dict(
            completed_transitions=3,
            requested_transitions=535,
            failure={"type": "TargetIntersectionError"},
            motion_fidelity={"passed": False},
            startup_hold=dict(
                requested_transitions=250, completed_transitions=250, existing_guard_screen_passed=True
            ),
            return_hold=dict(requested_transitions=250),
        ),
    )


def test_plan_retains_full_control_and_every_observed_boundary():
    plan = window.probe_plan(case())
    assert [row["stop_after_controls"] for row in plan] == [None, 1, 2, 3]
    assert [row["full_request_control"] for row in plan] == [True, False, False, False]


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed_transitions", True),
        ("completed_transitions", 0),
        ("completed_transitions", 535),
        ("failure", None),
        ("motion_fidelity", {"passed": True}),
        ("startup_hold", {}),
        ("return_hold", {"requested_transitions": 25}),
    ],
)
def test_unqualified_or_changed_parent_cannot_be_used(field, value):
    row = case()
    row["result"][field] = value
    with pytest.raises(ValueError, match="original failed full-dance"):
        window.probe_plan(row)


def arrays():
    # Synthetic identity fixture, not a simulated or physically valid motion.
    n, acquisition, controls = 32, 2, 3
    result = {}
    for name in window.PHYSICS_FIELDS:
        width = 30 if "qpos" in name else 29 if "qvel" in name else 23
        result[name] = np.repeat(np.arange(n, dtype=float)[:, None], width, axis=1)
    result["physics_phase"] = np.array([0] * acquisition + [1] * (controls * 10))
    for prefix in ("qpos", "qvel"):
        result[f"physics_post_{prefix}"] = result[f"physics_pre_{prefix}"] + 1
    for name in window.ACTUATION_FIELDS:
        result[name] = np.repeat(np.arange(controls * 10, dtype=float)[:, None], 23, axis=1)
    result["qpos"] = result["physics_post_qpos"][[1, 11, 21, 31]].copy()
    result["startup_hold_qpos"] = result["physics_post_qpos"][:acquisition].copy()
    result["terminal_active_qpos"] = result["physics_post_qpos"][-1].copy()
    result["terminal_active_qvel"] = result["physics_post_qvel"][-1].copy()
    result["terminal_active_target"] = result["actuation_target"][-1].copy()
    return result


def prefix(base):
    stop, controls = 22, 2
    result = {name: value.copy() for name, value in base.items()}
    for name in window.PHYSICS_FIELDS:
        result[name] = result[name][: stop + 4]
    result["physics_phase"][stop:] = 2
    for name in window.ACTUATION_FIELDS:
        result[name] = result[name][: controls * 10]
    result["qpos"] = result["qpos"][: controls + 1]
    result["terminal_active_qpos"] = base["physics_post_qpos"][stop - 1].copy()
    result["terminal_active_qvel"] = base["physics_post_qvel"][stop - 1].copy()
    result["terminal_active_target"] = base["actuation_target"][controls * 10 - 1].copy()
    return result


def test_full_control_requires_every_array_and_exact_values():
    base = arrays()
    assert window.require_full_replay_identity(copy.deepcopy(base), base)["exact"]
    changed = copy.deepcopy(base)
    changed["physics_effort"][0, 0] += 1e-15
    with pytest.raises(ValueError, match="full control replay changed"):
        window.require_full_replay_identity(changed, base)
    changed.pop("physics_phase")
    with pytest.raises(ValueError, match="array set"):
        window.require_full_replay_identity(changed, base)


def test_prefix_replays_original_state_history_and_carries_terminal_state():
    base = arrays()
    report = window.require_prefix_identity(prefix(base), base, 2)
    assert report == dict(
        exact=True,
        acquisition_physics_steps=2,
        active_physics_steps=20,
        source_prefix_controls=2,
        terminal_state_and_previous_target_equal=True,
        first_return_state_equal=True,
        return_physics_steps=4,
        reconstructed_mjdata_used=False,
    )


@pytest.mark.parametrize(
    "name,index",
    [
        ("physics_pre_qpos", (3, 0)),
        ("actuation_previous_target", (1, 0)),
        ("terminal_active_target", 0),
        ("terminal_active_qvel", 0),
        ("physics_pre_qpos", (22, 0)),
    ],
)
def test_changed_prefix_or_reseeded_handoff_is_rejected(name, index):
    base = arrays()
    changed = prefix(base)
    changed[name][index] += 1e-6
    with pytest.raises(ValueError):
        window.require_prefix_identity(changed, base, 2)


def result():
    return dict(
        requested_transitions=2,
        completed_transitions=2,
        completed_active_physics_steps=20,
        failure=None,
        motion_fidelity=dict(passed=False, full_clip_completed=False),
        lifecycle_simulator_screen_passed=False,
        actual_engine_audit=dict(passed=True),
        return_hold=dict(
            requested_transitions=250,
            completed_transitions=250,
            completed_physics_steps=2500,
            existing_guard_screen_passed=True,
            standing_screen_passed=True,
            failure=None,
        ),
    )


def test_successful_return_does_not_become_full_dance_or_fsm_success():
    outcome = window.probe_outcome(result(), 2, 535)
    assert outcome["recovery_screen_passed"] is True
    assert outcome["full_source_requested_controls"] == 535
    for flag in (
        "complete_source_passed",
        "native_unitree_mode_handoff_proven",
        "hardware_authorized",
        "deployment_ready",
    ):
        assert outcome[flag] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("requested_transitions", 1),
        ("completed_active_physics_steps", 19),
        ("failure", {}),
        ("motion_fidelity", dict(passed=True, full_clip_completed=True)),
        ("lifecycle_simulator_screen_passed", True),
        ("actual_engine_audit", dict(passed=False)),
    ],
)
def test_execution_change_or_completion_relabel_is_rejected(field, value):
    row = result()
    row[field] = value
    with pytest.raises(ValueError, match="cannot hide"):
        window.probe_outcome(row, 2, 535)


def test_complete_duration_without_standing_screen_stays_failed():
    row = result()
    row["return_hold"]["existing_guard_screen_passed"] = False
    assert window.probe_outcome(row, 2, 535)["recovery_screen_passed"] is False
    row["return_hold"]["requested_transitions"] = 25
    with pytest.raises(ValueError, match="shorten"):
        window.probe_outcome(row, 2, 535)


def outcomes():
    return [
        dict(stop_after_controls=i, recovery_screen_passed=passed, complete_source_passed=False)
        for i, passed in enumerate([True, False, True], 1)
    ]


def test_recoverability_is_not_assumed_monotone_or_live_qualified():
    summary = window.summarize_window(outcomes(), 3)
    assert summary["successful_return_boundaries"] == [1, 3]
    assert summary["initial_contiguous_success_controls"] == 1
    assert summary["latest_observed_success_controls"] == 3
    assert summary["successful_boundary_after_an_earlier_failure"] is True
    assert summary["automatic_live_guard_qualified"] is False
    assert summary["full_dance_qualified"] is False


@pytest.mark.parametrize("rows", [outcomes()[:2], outcomes()[::-1], [*outcomes(), outcomes()[0]]])
def test_window_cannot_omit_reorder_or_duplicate_failures(rows):
    with pytest.raises(ValueError, match="incomplete"):
        window.summarize_window(rows, 3)
