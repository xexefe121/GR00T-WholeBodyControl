"""Characterize current runtime defects; passing these tests is not readiness."""

import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from gear_sonic.scripts.audit_g1_true23_restore_rpc_no_robot import (
    SOURCE,
    SPANS,
    LEGACY_SPANS,
    extract_runtime,
    has_physical_gate,
    run_audit,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("G1_TRUE23_RESTORE_TEST_RUNTIME_ROOT", str(ROOT)))
SOURCE_TEXT = (SOURCE_ROOT / SOURCE).read_text(encoding="utf-8")
PHYSICAL = has_physical_gate(SOURCE_TEXT)


def test_extraction_changes_only_clock_and_preserves_exact_bodies():
    source = SOURCE_TEXT
    extracted, bindings = extract_runtime(source)
    original = []
    for (start, end), binding in zip(SPANS if PHYSICAL else LEGACY_SPANS, bindings, strict=True):
        begin = source.index(start)
        finish = source.index(end, begin + len(start))
        span = source[begin:finish]
        original.append(span)
        assert binding["source_span_sha256"] == hashlib.sha256(span.encode()).hexdigest()
    assert extracted.replace("fixture::SleepFor(", "std::this_thread::sleep_for(") == "\n".join(original)
    assert sum(row["sleep_substitutions"] for row in bindings) == (6 if PHYSICAL else 1)
    assert "ChannelFactory" not in extracted and "ChannelPublisher" not in extracted


def test_extraction_rejects_missing_or_duplicate_marker():
    source = SOURCE_TEXT
    with pytest.raises(ValueError, match="expected one runtime marker"):
        extract_runtime(source + SPANS[0][0])
    with pytest.raises(ValueError, match="missing runtime end marker"):
        extract_runtime(source.replace(SPANS[0][1], "changed_marker"))


@pytest.fixture(scope="module")
def audit(tmp_path_factory):
    if shutil.which("g++") is None or not Path("/usr/include/nlohmann/json.hpp").is_file():
        pytest.skip("offline C++ compiler and nlohmann headers required")
    output = tmp_path_factory.mktemp("restore_rpc") / "audit"
    return run_audit(ROOT, output, runtime_source_root=SOURCE_ROOT), output


def test_no_hardware_or_standing_claims_and_exact_source_binding(audit):
    report, output = audit
    assert report["experiment_completed"]
    assert not report["deployment_ready"]
    assert not report["hardware_authorized"]
    assert not report["physical_damping_cause_proven"]
    assert not report["dds_opened"]
    assert not report["robot_commands_published"]
    assert not report["writer_and_monitor_threads_executed"]
    assert report["no_damp_rpc_contract_met_for_all_cases"] is not PHYSICAL
    for path, digest in report["source_files"].items():
        assert sha256(Path(path)) == digest
    for name, digest in report["artifacts"].items():
        assert sha256(output / name) == digest


def test_lifecycle_qualifier_rejects_actual_handoff_defects(audit):
    from gear_sonic.scripts.qualify_g1_true23_active_lifecycle_no_robot import restore_rpc_contract_checks

    checks = restore_rpc_contract_checks(audit[0])
    assert not all(checks.values())
    assert not checks["crouched_state_not_labelled_normal_standing"]
    assert checks["unhealthy_final_states_rejected"] is PHYSICAL
    assert checks["normal_restore_never_requests_damp_fsm"] is not PHYSICAL
    assert checks["no_stand_commands_after_mock_motor_disable"] is not PHYSICAL


def test_lifecycle_qualifier_rejects_missing_evidence():
    from gear_sonic.scripts.qualify_g1_true23_active_lifecycle_no_robot import restore_rpc_contract_checks

    assert not any(restore_rpc_contract_checks({}).values())


def test_refuses_overwriting_evidence(audit):
    _, output = audit
    before = sha256(output / "report.json")
    with pytest.raises(FileExistsError):
        run_audit(ROOT, output)
    assert sha256(output / "report.json") == before


@pytest.mark.parametrize("name", ["already_801", "completed_4", "walkready_500"])
def test_powered_fsm_paths_do_not_request_damp(audit, name):
    row = audit[0]["scenarios"][name]
    assert row["restore_returned_success"] is (PHYSICAL or name != "completed_4")
    assert row["damp_rpc_requests"] == 0
    expected_id = 801 if PHYSICAL or name == "already_801" else (4 if name == "completed_4" else 500)
    assert row["final_mock_fsm_id"] == expected_id
    assert row["stable_samples"] == (100 if row["restore_returned_success"] else 0)


@pytest.mark.parametrize("name", ["zero_fsm", "damped_fsm"])
def test_actual_normal_return_contains_damp_recovery(audit, name):
    # A regression witness of the current no-damp contract violation, not an
    # endorsement of this sequence or a firmware behavior prediction.
    row = audit[0]["scenarios"][name]
    assert row["restore_returned_success"] is PHYSICAL
    assert row["fsm_commands"] == ([1, 4, 801] if PHYSICAL else [])
    assert row["damp_rpc_accepted"] == int(PHYSICAL)
    assert row["maximum_restore_health_read_gap_seconds"] == pytest.approx(6.5 if PHYSICAL else 30)


@pytest.mark.parametrize(
    "name",
    [
        "disabled_801",
        "bit30_801",
        "missing_state_801",
        "stale_state_801",
        "frozen_tick_801",
        "bad_crc_801",
        "nonfinite_801",
        "zero_torque_801",
    ],
)
def test_unhealthy_final_state_exposes_legacy_false_success(audit, name):
    row = audit[0]["scenarios"][name]
    assert row["restore_returned_success"] is not PHYSICAL
    assert row["fsm_commands"] == []
    assert (row["stable_samples"] < 100) is PHYSICAL


def test_stand_rpc_failure_can_leave_actual_restore_in_damp(audit):
    row = audit[0]["scenarios"]["stand_rpc_rejected"]
    assert not row["restore_returned_success"]
    assert row["fsm_commands"] == ([1, 4, 1, 4] if PHYSICAL else [])
    assert row["final_mock_fsm_id"] == (1 if PHYSICAL else 0)
    assert row["damp_rpc_accepted"] == (2 if PHYSICAL else 0)


@pytest.mark.parametrize(
    "name", ["stand_transition_incomplete", "upright_rpc_rejected", "upright_rpc_accepted_but_ignored"]
)
def test_incomplete_upright_transition_is_not_success(audit, name):
    row = audit[0]["scenarios"][name]
    assert not row["restore_returned_success"]
    assert row["final_mock_fsm_id"] == (4 if PHYSICAL else 0)


def test_restore_function_does_not_read_health_during_blocking_kick(audit):
    rows = audit[0]["scenarios"]
    persistent = rows["disabled_during_kick"]
    transient = rows["transient_disabled_during_kick"]
    assert persistent["stand_commands_while_mock_disabled"] == (2 if PHYSICAL else 0)
    assert not persistent["restore_returned_success"]
    assert transient["stand_commands_while_mock_disabled"] == 0
    assert transient["health_reads_observing_mock_disabled"] == 0
    assert transient["restore_returned_success"] is PHYSICAL
    # This says nothing about the separate DDS monitor thread, which is not
    # executed by the fixture, or whether real firmware permits recovery.


def test_enabled_motor_gate_is_not_normal_posture_gate(audit):
    row = audit[0]["scenarios"]["crouched_801"]
    assert row["restore_returned_success"]
    assert row["physical_state_verified"] is PHYSICAL
    result = json.loads((audit[1] / "crouched_801.json").read_text())
    if PHYSICAL:
        assert result["physical_state"]["knee_q_rad"] == [1.5, 1.5]
    else:
        assert result["physical_state"]["missing_runtime_gate"]


def test_rpc_errors_keep_bounded_failures_and_specific_causes(audit):
    rows = audit[0]["scenarios"]
    assert rows["checkmode_retry"]["restore_returned_success"] is PHYSICAL
    assert rows["select_error_but_applied"]["restore_returned_success"]
    for name in ("checkmode_unavailable", "service_unavailable", "internal_rpc_rejected", "fsm_unreadable"):
        assert not rows[name]["restore_returned_success"]
        assert rows[name]["fsm_commands"] == []
    assert "CheckMode" in rows["checkmode_unavailable"]["error"]
    assert rows["checkmode_unavailable"]["elapsed_virtual_seconds"] == (10 if PHYSICAL else 0)
    assert "SelectMode" in rows["service_unavailable"]["error"]
    assert "SwitchToInternalCtrl" in rows["internal_rpc_rejected"]["error"]
