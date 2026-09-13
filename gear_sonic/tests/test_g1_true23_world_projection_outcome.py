"""No failed prefix or mean improvement may conceal an individual regression."""

import copy

import pytest

from gear_sonic.utils.g1_true23_world_projection_outcome import NAMES, PHASE, compare_backend, compare_outcomes


def report(root=1.0, leg=0.2):
    return dict(
        cases=[
            dict(
                name=name,
                completed_controls=1000,
                requested_controls=1000,
                full_lifecycle_completed=True,
                failure=None,
                original_q2_metrics_after=dict(root_world_position_p95_m=root, leg_joint_rmse_rad=leg),
                maximum_actual_range_excess_rad=0.0,
                maximum_actual_applied_effort_excess_nm=0.0,
                maximum_actual_engine_effort_excess_nm=0.0,
                maximum_target_jump_rad=0.5,
                lifecycle_after=dict(
                    source_motion_tracking=dict(provisional_reference_landmark_screen_passed=False)
                ),
            )
            for name in NAMES
        ]
    )


def test_improvement_allows_follow_on_not_world_qualification():
    value = compare_backend(report(0.89), report(), backend="cpu")
    assert value["continuation_gate_passed"] is True
    assert value["absolute_world_screens_all_passed"] is False


def test_prefix_never_averaged():
    candidate = report(0.1)
    candidate["cases"][0].update(
        completed_controls=100, full_lifecycle_completed=False, failure={"message": "guard"}
    )
    value = compare_backend(candidate, report(), backend="cpu")
    assert value["continuation_gate_passed"] is False
    assert value["mean_root_p95_ratio"] is None
    assert value["cases"][0]["root_p95_m"] is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("original_q2_metrics_after", {"root_world_position_p95_m": 0.1, "leg_joint_rmse_rad": 0.211}),
        ("maximum_actual_range_excess_rad", 1e-9),
        ("maximum_actual_engine_effort_excess_nm", 1e-9),
    ],
)
def test_single_regression_or_physical_excess_rejects(key, value):
    candidate = report(0.1)
    candidate["cases"][0][key] = value
    assert compare_backend(candidate, report(), backend="cpu")["continuation_gate_passed"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "length", "nan"])
def test_bad_evidence_rejected(mutation):
    candidate = copy.deepcopy(report(0.8))
    if mutation == "missing":
        candidate["cases"].pop()
    elif mutation == "duplicate":
        candidate["cases"][1]["name"] = NAMES[0]
    elif mutation == "length":
        candidate["cases"][0]["requested_controls"] = 1001
    else:
        candidate["cases"][0]["original_q2_metrics_after"]["root_world_position_p95_m"] = float("nan")
    with pytest.raises(ValueError):
        compare_backend(candidate, report(), backend="cpu")


def gpu_report(root=1.0, leg=0.2):
    return dict(
        cases=[
            dict(
                name=row["name"],
                result={key: row[key] for key in ("completed_controls", "requested_controls", "failure")},
                trace_audit=dict(
                    actual_range_excess_max_rad=0.0, actual_effort_excess_max_nm=0.0, target_step_max_rad=0.5
                ),
                original_task_metrics={PHASE: row["original_q2_metrics_after"]},
                lifecycle=row["lifecycle_after"],
            )
            for row in report(root, leg)["cases"]
        ]
    )


def test_cpu_improvement_cannot_cancel_gpu_regression():
    value = compare_outcomes(report(0.1), gpu_report(1.01), report(), gpu_report())
    assert value["backends"]["cpu"]["continuation_gate_passed"] is True
    assert value["backends"]["gpu"]["continuation_gate_passed"] is False
    assert value["continuation_gate_passed"] is False


def test_both_relative_gates_still_do_not_grant_deployment():
    value = compare_outcomes(report(0.8), gpu_report(0.8), report(), gpu_report())
    assert value["continuation_gate_passed"] is True
    assert value["deployment_ready"] is False
    assert value["simulator_qualified"] is False
    assert value["hardware_authorized"] is False


def test_gpu_guard_stop_rejects_even_if_metric_improves():
    candidate = gpu_report(0.1)
    candidate["cases"][0]["result"].update(completed_controls=900, failure={"message": "guard"})
    value = compare_backend(candidate, gpu_report(), backend="gpu")
    assert value["mean_root_p95_ratio"] is None
    assert value["continuation_gate_passed"] is False
