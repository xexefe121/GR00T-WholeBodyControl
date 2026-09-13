"""Read saved hardware logs only; do not infer a firmware cause from FSM flags."""

from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
inputs = {}


def bind(path):
    with path.open("rb") as stream:
        inputs[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    return path


def keys_in(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from keys_in(child)
    elif isinstance(value, list):
        for child in value:
            yield from keys_in(child)


rows = []
bind(Path(__file__))
for path in sorted((BASE / "physical_dance_v1").glob("*.execution.jsonl")):
    records = [json.loads(line) for line in bind(path).read_text().splitlines() if line.strip()]
    terminal = [row for row in records if row.get("event") in ("session_complete", "session_failed")]
    last = terminal[-1] if terminal else {}
    keys = set(keys_in(records))
    motor_fields = sorted(keys & {"motorstate", "motor_status_hardware", "motor_modes_hardware", "motorstate_bit30_slots"})
    rows.append(dict(path=str(path), records=len(records), terminal_records=len(terminal),
        terminal_event=last.get("event"), reported_pass=last.get("passed"), fault=last.get("final_fault"),
        fault_joint_compact23=last.get("fault_joint"), fault_value=last.get("fault_value"),
        post_stop_damping_frames=last.get("damping_frames_after_stop"),
        reported_mode_restored=last.get("motion_mode_restored"),
        excursion_rad=last.get("measured_armed_excursion_rad"),
        terminal_knee_flexion_delta_rad=last.get("armed_knee_flexion_delta_rad"),
        explicit_raw_motor_fields=motor_fields,
        terminal_reason=last.get("reason", last.get("writer_error")),
        first_event_monotonic_ns=records[0].get("monotonic_ns"),
        last_event_monotonic_ns=records[-1].get("monotonic_ns")))

health_path = BASE / "readiness_audit_20260905_v1/motor_health.json"
health = json.loads(bind(health_path).read_text())
summary = dict(
    archived_execution_logs=len(rows),
    logs_with_explicit_raw_motor_status_or_modes=sum(bool(row["explicit_raw_motor_fields"]) for row in rows),
    logs_with_positive_reported_post_stop_damping=sum((row["post_stop_damping_frames"] or 0) > 0 for row in rows),
    logs_with_zero_reported_post_stop_damping=sum(row["post_stop_damping_frames"] == 0 for row in rows),
    logs_without_reported_post_stop_damping=sum(row["post_stop_damping_frames"] is None for row in rows),
    terminal_faults=dict(Counter(row["fault"] if row["fault"] is not None else "not_recorded" for row in rows)),
    later_read_only_snapshot=dict(path=str(health_path), started=health["started_utc"],
        completed=health["completed_utc"], enabled_count=health["nonzero_mode_count"],
        bit30_slots=health["motorstate_bit30_slots"], commands_published=health["robot_commands_published"]),
)
with (HERE / "historical_incident_evidence.json").open("x") as stream:
    json.dump(dict(inputs=inputs, summary=summary, sessions=rows,
        scope="76 archived physical_dance_v1 execution JSONL files and one later read-only health snapshot",
        raw_driver_fault_edge_captured_in_these_execution_logs=False,
        firmware_meaning_of_bit30_established=False,
        later_healthy_snapshot_rules_out_transient_power_or_temperature_cause=False,
        fsm_or_reported_pass_proves_physical_standing=False,
        physical_damping_cause_proven=False, new_robot_observation=False,
        hardware_authorized=False, deployment_ready=False), stream, indent=2)
print(json.dumps(summary))
