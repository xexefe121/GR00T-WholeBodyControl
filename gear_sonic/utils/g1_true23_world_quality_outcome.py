"""Predeclared research continuation gate; never a live-deployment certificate."""

import numpy as np

NAMES = ("walk002", "walk003", "walk008", "dance")
PHASE = "unchanged_referee_post_control_q2"


def indexed(rows):
    if len(rows) != 4 or {row["name"] for row in rows} != set(NAMES):
        raise ValueError("comparison requires each of the four declared full motions exactly once")
    return {row["name"]: row for row in rows}


def finite_nonnegative(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
        raise ValueError("comparison requires finite nonnegative actual measurements")
    return float(value)


def compare_outcomes(cpu, gpu, baseline):
    if (
        cpu.get("kind") != "native23_world_quality_singleton_outcome_input_network_physics_audit_v1"
        or cpu.get("recorded_outcomes_independently_verified") is not True
        or gpu.get("kind") != "native23_world_quality_saved_trace_audit_v1"
        or gpu.get("checked_trace_structure_not_independent_physics_reexecution") is not True
        or gpu.get("recording_mode") != "full"
        or baseline.get("kind") != "native23_world_tracking_full_lifecycle_diagnostic_v1"
    ):
        raise ValueError("continuation requires the declared completed independent evidence")
    cpus, gpus, before = (indexed(report["cases"]) for report in (cpu, gpu, baseline))
    rows = []
    for name in NAMES:
        c, g, b = cpus[name], gpus[name], before[name]
        br, gr = b["result"], g["result"]
        if br["failure"] is not None or br["completed_controls"] != br["requested_controls"]:
            raise ValueError("incomplete baseline cannot supply a full-motion comparison")
        physical = g["trace_audit"]
        limits = [
            finite_nonnegative(c["maximum_actual_range_excess_rad"]),
            finite_nonnegative(c["maximum_actual_applied_effort_excess_nm"]),
            finite_nonnegative(c["maximum_actual_engine_effort_excess_nm"]),
            finite_nonnegative(physical["actual_range_excess_max_rad"]),
            finite_nonnegative(physical["actual_effort_excess_max_nm"]),
        ]
        complete = (
            c["full_lifecycle_completed"] is True
            and c["completed_controls"] == c["requested_controls"] == br["requested_controls"]
            and gr["failure"] is None
            and gr["completed_controls"] == gr["requested_controls"] == br["requested_controls"]
        )
        source = g["source_tracking"]
        row = dict(
            name=name,
            both_full_lifecycles_completed=complete,
            actual_range_and_effort_excess_zero=max(limits) == 0,
        )
        if source["full_source_completed"] is True:
            old = b["original_task_metrics"][PHASE]
            for label, key in (("root", "root_world_position_p95_m"), ("leg", "leg_joint_rmse_rad")):
                prior, current = finite_nonnegative(old[key]), finite_nonnegative(source[key])
                if prior == 0:
                    raise ValueError("relative comparison cannot divide by a zero baseline")
                row[label + "_baseline"] = prior
                row[label + "_candidate"] = current
                row[label + "_ratio"] = current / prior
        else:
            row.update(root_ratio=None, leg_ratio=None, partial_source_not_compared=True)
        rows.append(row)
    complete_metrics = all(row["root_ratio"] is not None for row in rows)
    average_ratio = ratio_of_averages = None
    relative = False
    if complete_metrics:
        average_ratio = float(np.mean([row["root_ratio"] for row in rows]))
        ratio_of_averages = float(
            np.mean([row["root_candidate"] for row in rows]) / np.mean([row["root_baseline"] for row in rows])
        )
        relative = (
            average_ratio <= 0.90
            and ratio_of_averages <= 0.90
            and all(row["root_ratio"] <= 1.05 and row["leg_ratio"] <= 1.05 for row in rows)
        )
    passed = relative and all(
        row["both_full_lifecycles_completed"] and row["actual_range_and_effort_excess_zero"] for row in rows
    )
    return dict(
        kind="native23_world_quality_predeclared_continuation_decision_v1",
        cases=rows,
        GPU_mean_of_root_ratios=average_ratio,
        GPU_ratio_of_mean_root_p95=ratio_of_averages,
        GPU_relative_tracking_gate_passed=relative,
        continuation_gate_passed=bool(passed),
        disposition="further_research_only" if passed else "reject_recipe_extension",
        partial_baseline_CPU_metrics_used=False,
        hardware_authorized=False,
        deployment_ready=False,
        absolute_world_limb_contact_standing_target_rate_live_timing_still_required=True,
    )
