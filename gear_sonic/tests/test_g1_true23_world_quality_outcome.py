import pytest

from gear_sonic.utils.g1_true23_world_quality_outcome import NAMES, PHASE, compare_outcomes


def reports():
    cpu = dict(
        kind="native23_world_quality_singleton_outcome_input_network_physics_audit_v1",
        recorded_outcomes_independently_verified=True,
        cases=[],
    )
    gpu = dict(
        kind="native23_world_quality_saved_trace_audit_v1",
        checked_trace_structure_not_independent_physics_reexecution=True,
        recording_mode="full",
        cases=[],
    )
    baseline = dict(kind="native23_world_tracking_full_lifecycle_diagnostic_v1", cases=[])
    for name in NAMES:
        result = dict(failure=None, completed_controls=1000, requested_controls=1000)
        cpu["cases"].append(
            dict(
                name=name,
                full_lifecycle_completed=True,
                completed_controls=1000,
                requested_controls=1000,
                maximum_actual_range_excess_rad=0.0,
                maximum_actual_applied_effort_excess_nm=0.0,
                maximum_actual_engine_effort_excess_nm=0.0,
            )
        )
        gpu["cases"].append(
            dict(
                name=name,
                result=result.copy(),
                trace_audit=dict(actual_range_excess_max_rad=0.0, actual_effort_excess_max_nm=0.0),
                source_tracking=dict(
                    full_source_completed=True, root_world_position_p95_m=0.8, leg_joint_rmse_rad=0.1
                ),
            )
        )
        baseline["cases"].append(
            dict(
                name=name,
                result=result.copy(),
                original_task_metrics={PHASE: dict(root_world_position_p95_m=1.0, leg_joint_rmse_rad=0.1)},
            )
        )
    return cpu, gpu, baseline


def test_relative_pass_is_not_deployment_qualification():
    result = compare_outcomes(*reports())
    assert result["continuation_gate_passed"]
    assert result["deployment_ready"] is False
    assert result["hardware_authorized"] is False


@pytest.mark.parametrize(
    "reason", ["cpu_prefix", "gpu_prefix", "leg_regression", "root_regression", "range", "effort"]
)
def test_incomplete_worse_or_limit_exceeding_candidate_rejected(reason):
    cpu, gpu, baseline = reports()
    if reason == "cpu_prefix":
        cpu["cases"][0]["completed_controls"] = 999
    elif reason == "gpu_prefix":
        gpu["cases"][0]["source_tracking"]["full_source_completed"] = False
    elif reason == "leg_regression":
        gpu["cases"][0]["source_tracking"]["leg_joint_rmse_rad"] = 0.106
    elif reason == "root_regression":
        gpu["cases"][0]["source_tracking"]["root_world_position_p95_m"] = 1.06
    elif reason == "range":
        cpu["cases"][0]["maximum_actual_range_excess_rad"] = 1e-10
    else:
        gpu["cases"][0]["trace_audit"]["actual_effort_excess_max_nm"] = 1e-10
    assert not compare_outcomes(cpu, gpu, baseline)["continuation_gate_passed"]


def test_nonfinite_measurement_rejected():
    cpu, gpu, baseline = reports()
    gpu["cases"][0]["source_tracking"]["root_world_position_p95_m"] = float("nan")
    with pytest.raises(ValueError):
        compare_outcomes(cpu, gpu, baseline)
