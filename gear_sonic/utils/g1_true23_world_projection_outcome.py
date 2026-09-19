"""Predeclared matched full-motion continuation gate; never deployment approval."""

import math

NAMES = ("walk002", "walk003", "walk008", "dance")
PHASE = "unchanged_referee_post_control_q2"


def normalized_cases(report, *, backend):
    rows = report["cases"]
    if [row["name"] for row in rows] != list(NAMES):
        raise ValueError("requires all four predeclared cases in order")
    out = []
    for row in rows:
        if backend == "cpu":
            completed, requested = row["completed_controls"], row["requested_controls"]
            complete = row["full_lifecycle_completed"] and row["failure"] is None and completed == requested
            metric = row["original_q2_metrics_after"]
            range_excess = row["maximum_actual_range_excess_rad"]
            effort_excess = max(
                row["maximum_actual_applied_effort_excess_nm"], row["maximum_actual_engine_effort_excess_nm"]
            )
            lifecycle = row["lifecycle_after"]
            jump = row["maximum_target_jump_rad"]
        elif backend == "gpu":
            result, trace = row["result"], row["trace_audit"]
            completed, requested = result["completed_controls"], result["requested_controls"]
            complete = result["failure"] is None and completed == requested
            metric = row["original_task_metrics"][PHASE]
            range_excess, effort_excess = (
                trace["actual_range_excess_max_rad"],
                trace["actual_effort_excess_max_nm"],
            )
            lifecycle, jump = row["lifecycle"], trace["target_step_max_rad"]
        else:
            raise ValueError("unknown backend")
        for value in (range_excess, effort_excess, jump):
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid measured physical statistic")
        if complete:
            root, leg = metric["root_world_position_p95_m"], metric["leg_joint_rmse_rad"]
            if any(not math.isfinite(value) or value < 0 for value in (root, leg)):
                raise ValueError("invalid complete-source metric")
        else:
            root = leg = None
        out.append(
            dict(
                name=row["name"],
                complete=bool(complete),
                completed=completed,
                requested=requested,
                root_p95_m=root,
                leg_rmse_rad=leg,
                range_excess_rad=range_excess,
                effort_excess_nm=effort_excess,
                target_step_max_rad=jump,
                provisional_world_screen_passed=bool(
                    complete
                    and lifecycle["source_motion_tracking"]["provisional_reference_landmark_screen_passed"]
                ),
            )
        )
    return out


def compare_backend(candidate, baseline, *, backend):
    after, before = normalized_cases(candidate, backend=backend), normalized_cases(baseline, backend=backend)
    if not all(row["complete"] for row in before):
        raise ValueError("matched quality baseline must contain all complete lifecycles")
    rows = []
    for new, old in zip(after, before, strict=True):
        if new["requested"] != old["requested"]:
            raise ValueError("candidate source/lifecycle length changed")
        root_ratio = leg_ratio = None
        if new["complete"]:
            if old["root_p95_m"] <= 0 or old["leg_rmse_rad"] <= 0:
                raise ValueError("undefined relative comparison against zero baseline")
            root_ratio = new["root_p95_m"] / old["root_p95_m"]
            leg_ratio = new["leg_rmse_rad"] / old["leg_rmse_rad"]
        rows.append(
            dict(
                **new,
                before_root_p95_m=old["root_p95_m"],
                before_leg_rmse_rad=old["leg_rmse_rad"],
                root_ratio=root_ratio,
                leg_ratio=leg_ratio,
                individual_nonregression_passed=bool(new["complete"] and root_ratio <= 1.05 and leg_ratio <= 1.05),
            )
        )
    all_complete = all(row["complete"] for row in rows)
    ratio = (
        sum(row["root_p95_m"] for row in rows) / sum(row["root_p95_m"] for row in before) if all_complete else None
    )
    checks = dict(
        all_four_complete=all_complete,
        mean_full_source_root_improves_at_least_ten_percent=bool(all_complete and ratio <= 0.9),
        no_individual_root_or_leg_regresses_over_five_percent=all(
            row["individual_nonregression_passed"] for row in rows
        ),
        zero_actual_range_and_effort_excess=all(
            row["range_excess_rad"] == 0 and row["effort_excess_nm"] == 0 for row in rows
        ),
    )
    return dict(
        backend=backend,
        cases=rows,
        mean_root_p95_ratio=ratio,
        checks=checks,
        continuation_gate_passed=all(checks.values()),
        incomplete_prefixes_never_averaged=True,
        absolute_world_screens_all_passed=all(row["provisional_world_screen_passed"] for row in rows),
        maximum_target_step_rad=max(row["target_step_max_rad"] for row in rows),
    )


def compare_outcomes(cpu, gpu, baseline_cpu, baseline_gpu):
    devices = dict(
        cpu=compare_backend(cpu, baseline_cpu, backend="cpu"),
        gpu=compare_backend(gpu, baseline_gpu, backend="gpu"),
    )
    passed = all(row["continuation_gate_passed"] for row in devices.values())
    return dict(
        kind="native23_world_projection_predeclared_full_motion_decision_v1",
        backends=devices,
        continuation_gate_passed=passed,
        disposition="eligible_for_separate_follow_on_design_not_promotion"
        if passed
        else "recipe_rejected_no_extension",
        source_speed_or_physical_limits_relaxed=False,
        deployment_ready=False,
        simulator_qualified=False,
        hardware_authorized=False,
        live_estimator_timing_transport_and_physical_transition_gates_unproven=True,
    )
