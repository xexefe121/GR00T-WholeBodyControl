"""Continuation criteria are not survival or partial-source promotion."""

from copy import deepcopy

from gear_sonic.scripts.audit_g1_sonic_public29_full_pose import continuation_gate


def cases():
    row = dict(
        complete_lifecycle=True,
        tracking=dict(full_source=True, root_p95_m=0.8, leg_rmse_rad=0.1),
        baseline_tracking=dict(root_p95_m=1.0, leg_rmse_rad=0.1),
    )
    return [deepcopy(row) for _ in range(3)]


def test_pass_is_not_hardware_or_native23_authorization():
    result = continuation_gate(cases())
    assert result["passed"] is True
    assert result["authorizes_native23_or_hardware"] is False


def test_leg_regression_rejects_better_root_tracking():
    rows = cases()
    rows[1]["tracking"]["leg_rmse_rad"] = 0.106
    assert continuation_gate(rows)["passed"] is False


def test_survival_with_insufficient_root_improvement_rejected():
    rows = cases()
    for row in rows:
        row["tracking"]["root_p95_m"] = 0.95
    assert continuation_gate(rows)["passed"] is False


def test_partial_motion_never_compared_as_full():
    rows = cases()
    rows[1]["tracking"]["full_source"] = False
    result = continuation_gate(rows)
    assert result["passed"] is False
    assert "mean_root_p95_ratio" not in result
