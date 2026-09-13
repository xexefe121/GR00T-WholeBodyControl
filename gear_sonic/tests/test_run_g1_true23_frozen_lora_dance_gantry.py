from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gear_sonic.scripts import replay_g1_true23_pico_packets_zmq as replay
from gear_sonic.scripts.run_g1_true23_frozen_lora_dance_gantry import (
    _DIRECT_EVENTS,
    _active_command,
    _wsl_path,
    validate_direct_dance_execution_evidence,
)


def _passing_execution_records() -> list[dict]:
    records = [
        {
            "schema_version": 1,
            "kind": "g1_true23_stage1_gantry_execution_evidence",
            "event": event,
            "authorization_id": "gantry-session-1",
            "monotonic_ns": index + 1,
        }
        for index, event in enumerate(_DIRECT_EVENTS)
    ]
    records[0].update(
        operator_contract="bounded_direct_dance_command_v1",
        post_arm_duration_seconds=1,
    )
    records[3].update(writes_before_event=0, motion_mode_released=False)
    records[5].update(
        pre_release_lowcmd_writes=0,
        kp_fraction=0.25,
        feedforward_tau_zero=True,
    )
    records[6].update(
        captured_pre_release_form="g1",
        captured_pre_release_name="normal",
        captured_pre_release_fsm_id=801,
        captured_pre_release_fsm_mode=0,
        pre_release_lowcmd_writes=0,
        first_post_release_command="sampled_posture_hold",
    )
    records[7].update(
        pre_arm_hold_frames=25,
        required_pre_arm_hold_frames=25,
        startup_damping_frames=0,
        release_to_first_hold_write_ns=2_000_000,
        maximum_first_hold_write_delay_ns=20_000_000,
        kp_positive=True,
        feedforward_tau_zero=True,
    )
    records[8].update(command="DANCE", policy_ready=True)
    records[9].update(feedforward_tau_zero=True)
    records[10].update(
        kp_fraction=0.25,
        feedforward_tau_zero=True,
        damping_frames_before_return=0,
    )
    records[11].update(
        restored_form="g1",
        restored_name="normal",
        restored_fsm_id=801,
        restored_fsm_mode=0,
        normal_return_hold_frames=250,
        required_normal_return_hold_frames=250,
        startup_damping_frames=0,
        damping_frames_after_stop=0,
        writer_quiesced_before_select=True,
        lowcmd_publisher_closed_before_select=True,
        select_mode_attempts=1,
        internal_control_handoff="walkrun",
        internal_control_attempts=1,
        restore_poll_attempts=2,
        stable_restore_samples=100,
        required_stable_restore_samples=100,
    )
    records[12].update(
        passed=True,
        policy_prewarmed_before_motion_release=True,
        pre_release_lowcmd_writes=0,
        pre_arm_hold_gate_open=True,
        pre_arm_hold_frames=25,
        startup_damping_frames=0,
        rejected_non_positive_gain_commands=0,
        release_to_first_hold_write_ns=2_000_000,
        maximum_abs_feedforward_tau_nm=0.0,
        final_fault="none",
        stop_reason="reviewed_post_arm_duration_complete",
        required_post_arm_duration_ns=1_000_000_000,
        post_arm_elapsed_ns=1_000_000_000,
        publisher_write_failed=False,
        writer_quiesced_before_restore=True,
        lowcmd_publisher_closed_before_restore=True,
        restore_select_mode_attempts=1,
        restore_internal_control_handoff="walkrun",
        restore_internal_control_attempts=1,
        restore_poll_attempts=2,
        stable_restore_samples=100,
        required_stable_restore_samples=100,
        damping_frames_after_stop=0,
        required_damping_frames_after_stop=0,
        normal_return_hold_frames=250,
        required_normal_return_hold_frames=250,
        motion_mode_restored=True,
        restored_motion_mode_name="normal",
        restored_locomotion_fsm_id=801,
        restored_locomotion_fsm_mode=0,
        measured_armed_excursion_rad=0.25,
        armed_q_seen=True,
        armed_q_first_rad=[0.3] * 23,
        armed_q_last_rad=[0.3] * 23,
        armed_knee_flexion_delta_rad=0.0,
        direct_dance_terminal_posture_screen_passed=True,
        maximum_direct_dance_terminal_knee_flexion_rad=0.5,
    )
    physical_state = {
        "verified": True,
        "fresh": True,
        "sampled_controlled_joints": 23,
        "enabled_motor_count": 23,
        "observed_latched_disabled_motor_count": 0,
        "crc_valid": True,
        "mode_machine": 4,
        "maximum_abs_tau_est_nm": 7.0,
        "knee_q_rad": [0.29, 0.33],
        "imu_rpy_rad": [0.0, 0.01, 0.0],
        "tick": 100,
        "received_monotonic_ns": 1,
        "q_hardware_rad": [0.0] * 23,
        "motor_modes_hardware": [1] * 23,
        "motor_status_hardware": [0] * 23,
    }
    physical_state["q_hardware_rad"][3] = 0.29
    physical_state["q_hardware_rad"][9] = 0.33
    records[5]["physical_state"] = copy.deepcopy(physical_state)
    records[6]["physical_state"] = copy.deepcopy(physical_state)
    records[6]["sample_check_monotonic_ns"] = records[6]["monotonic_ns"]
    records[11]["physical_state"] = copy.deepcopy(physical_state)
    records[11]["sample_check_monotonic_ns"] = records[11]["monotonic_ns"]
    records[12]["pre_release_physical_state"] = copy.deepcopy(physical_state)
    records[12]["restored_physical_state"] = copy.deepcopy(physical_state)
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_active_command_binds_direct_frozen_dance() -> None:
    command = _active_command(
        distro="Ubuntu-22.04",
        binary="/repo/g1_true23_active_gantry",
        encoder="/repo/encoder.onnx",
        decoder="/repo/decoder.onnx",
        metadata="/repo/decoder.json",
        promotion="/repo/promotion.json",
        active_promotion="/repo/active.json",
        live_shadow_evidence="/repo/shadow.jsonl",
        authorization_id="gantry-session-1",
        network="eth0",
        endpoint="tcp://127.0.0.1:5557",
        evidence="/repo/execution.jsonl",
        duration_seconds=5,
        gantry_authorize="I_CONFIRM_G1_TRUE23_STAGE1_GANTRY",
        control_cpu_set="0-3",
        direct_dance_command="DANCE",
    )
    assert command[:11] == [
        "wsl.exe",
        "-d",
        "Ubuntu-22.04",
        "--",
        "taskset",
        "-c",
        "0-3",
        "stdbuf",
        "-oL",
        "-eL",
        "/repo/g1_true23_active_gantry",
    ]
    assert "--frozen-lora-policy" in command
    assert command[command.index("--network") + 1] == "eth0"
    assert command[command.index("--pico-endpoint") + 1] == (
        "tcp://127.0.0.1:5557"
    )
    assert command[command.index("--post-arm-duration-seconds") + 1] == "5"
    assert command[command.index("--gantry-authorize") + 1] == (
        "I_CONFIRM_G1_TRUE23_STAGE1_GANTRY"
    )
    assert command[command.index("--direct-dance-command") + 1] == "DANCE"


def test_repeat_reference_packets_keeps_values_and_advances_indices(
    monkeypatch,
) -> None:
    monkeypatch.setattr(replay, "validate_reference_terms", lambda _packet: {})
    packets = [
        {
            "control_source_frame_index": index,
            "pico_anchor_source_frame_index": index - 1,
            "value": f"pose-{index}",
        }
        for index in range(10, 13)
    ]
    repeated = replay.repeat_reference_packets(packets, 2)
    assert len(repeated) == 6
    assert repeated[3]["control_source_frame_index"] == 13
    assert repeated[3]["pico_anchor_source_frame_index"] == 12
    first_again = copy.deepcopy(repeated[3])
    first_again["control_source_frame_index"] = 10
    first_again["pico_anchor_source_frame_index"] = 9
    assert first_again == packets[0]


def test_wsl_path_maps_windows_drive_without_shell() -> None:
    assert _wsl_path(Path("Z:/repo/file"), "Ubuntu") == "/mnt/z/repo/file"
    assert _wsl_path(Path("Z:\\repo\\some file"), "Ubuntu") == "/mnt/z/repo/some file"


@pytest.mark.parametrize("path", ["repo/file", "Z:relative", "/mnt/z/repo", "//host/share/file", "Z:/repo/../file"])
def test_wsl_path_rejects_nonabsolute_or_unresolved_windows_path(path) -> None:
    with pytest.raises(ValueError, match="Windows drive path"):
        _wsl_path(Path(path), "Ubuntu")


@pytest.mark.parametrize("delta", [0.500001, 1.24])
def test_apparent_dance_success_rejected_when_both_knees_sag(tmp_path, delta) -> None:
    records = _passing_execution_records()
    terminal = records[-1]
    terminal["measured_armed_excursion_rad"] = 1.491
    for joint in (3, 9):
        terminal["armed_q_last_rad"][joint] += delta
    terminal["armed_knee_flexion_delta_rad"] = delta
    evidence = tmp_path / "collapse.jsonl"
    _write_jsonl(evidence, records)
    with pytest.raises(ValueError, match="terminal posture screen"):
        validate_direct_dance_execution_evidence(evidence, authorization_id="gantry-session-1", duration_seconds=1)


@pytest.mark.parametrize("field,value", [
    ("armed_q_seen", False), ("armed_q_first_rad", None),
    ("armed_q_last_rad", [float("nan")] * 23),
    ("armed_knee_flexion_delta_rad", 0.1),
    ("direct_dance_terminal_posture_screen_passed", False),
    ("maximum_direct_dance_terminal_knee_flexion_rad", 1.5),
])
def test_dance_posture_claim_requires_consistent_measured_evidence(tmp_path, field, value) -> None:
    records = _passing_execution_records()
    records[-1][field] = value
    evidence = tmp_path / "invalid_posture.jsonl"
    _write_jsonl(evidence, records)
    with pytest.raises(ValueError, match="posture"):
        validate_direct_dance_execution_evidence(evidence, authorization_id="gantry-session-1", duration_seconds=1)


def test_direct_execution_validator_accepts_hold_first_startup(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, _passing_execution_records())
    terminal = validate_direct_dance_execution_evidence(
        evidence,
        authorization_id="gantry-session-1",
        duration_seconds=1,
    )
    assert terminal["startup_damping_frames"] == 0


@pytest.mark.parametrize("captured_fsm", [4, 500, 801])
def test_direct_execution_validator_accepts_completed_801_restore(
    tmp_path: Path, captured_fsm: int,
) -> None:
    records = _passing_execution_records()
    records[6]["captured_pre_release_fsm_id"] = captured_fsm
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, records)

    terminal = validate_direct_dance_execution_evidence(
        evidence,
        authorization_id="gantry-session-1",
        duration_seconds=1,
    )

    assert terminal["restored_locomotion_fsm_id"] == 801


@pytest.mark.parametrize("fsm, mode", [(0, 0), (1, 0), (4, 0), (500, 0), (801, 1)])
def test_direct_execution_validator_rejects_incomplete_stand(
    tmp_path: Path, fsm: int, mode: int,
) -> None:
    records = _passing_execution_records()
    records[11].update(restored_fsm_id=fsm, restored_fsm_mode=mode)
    records[12].update(restored_locomotion_fsm_id=fsm, restored_locomotion_fsm_mode=mode)
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, records)
    with pytest.raises(ValueError, match="terminal evidence did not pass"):
        validate_direct_dance_execution_evidence(
            evidence, authorization_id="gantry-session-1", duration_seconds=1,
        )


@pytest.mark.parametrize("field, value", [
    ("verified", False), ("fresh", False), ("enabled_motor_count", 0),
    ("enabled_motor_count", 22), ("observed_latched_disabled_motor_count", 1),
    ("crc_valid", False), ("mode_machine", 5),
    ("motor_modes_hardware", [1] * 22 + [0]),
    ("motor_status_hardware", [0] * 22 + [1 << 30]),
    ("q_hardware_rad", [0] * 23),
    ("maximum_abs_tau_est_nm", 0.0), ("maximum_abs_tau_est_nm", float("nan")),
    ("knee_q_rad", [float("inf"), 0.3]), ("imu_rpy_rad", [0, 0]),
])
@pytest.mark.parametrize("record_index, terminal_key", [
    (5, "pre_release_physical_state"), (6, "pre_release_physical_state"),
    (11, "restored_physical_state"),
])
def test_direct_execution_rejects_dead_or_invalid_actuators_despite_801(
    tmp_path: Path, field: str, value: object, record_index: int, terminal_key: str,
) -> None:
    records = _passing_execution_records()
    records[record_index]["physical_state"][field] = value
    records[12][terminal_key][field] = value
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, records)
    with pytest.raises(ValueError, match="physical motor evidence|invalid measured"):
        validate_direct_dance_execution_evidence(
            evidence, authorization_id="gantry-session-1", duration_seconds=1,
        )


@pytest.mark.parametrize("excursion", [0.0, 0.01, True, float("nan"), float("inf")])
def test_direct_execution_rejects_absent_or_nonfinite_motion(
    tmp_path: Path, excursion: object,
) -> None:
    records = _passing_execution_records()
    records[12]["measured_armed_excursion_rad"] = excursion
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, records)
    with pytest.raises(ValueError, match="terminal evidence did not pass"):
        validate_direct_dance_execution_evidence(
            evidence, authorization_id="gantry-session-1", duration_seconds=1,
        )


def test_direct_execution_rejects_historical_fsm_only_success(tmp_path: Path) -> None:
    records = _passing_execution_records()
    del records[5]["physical_state"]
    del records[11]["physical_state"]
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, records)
    with pytest.raises(ValueError, match="physical motor evidence missing"):
        validate_direct_dance_execution_evidence(
            evidence, authorization_id="gantry-session-1", duration_seconds=1,
        )


@pytest.mark.parametrize("record_index", [6, 11])
@pytest.mark.parametrize("checked_ns", [None, True, 0, 14, 40_000_002])
def test_direct_execution_rejects_invalid_physical_check_timestamp(
    tmp_path: Path, record_index: int, checked_ns: object,
) -> None:
    records = _passing_execution_records()
    records[record_index]["sample_check_monotonic_ns"] = checked_ns
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, records)
    with pytest.raises(ValueError, match="physical state check timestamp"):
        validate_direct_dance_execution_evidence(
            evidence, authorization_id="gantry-session-1", duration_seconds=1,
        )


def test_direct_execution_allows_post_release_rpc_delay(tmp_path: Path) -> None:
    records = _passing_execution_records()
    for record in records[6:]:
        record["monotonic_ns"] += 100_000_000
    records[11]["physical_state"]["received_monotonic_ns"] = 100_000_001
    records[11]["sample_check_monotonic_ns"] += 100_000_000
    records[12]["restored_physical_state"] = copy.deepcopy(records[11]["physical_state"])
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, records)
    validate_direct_dance_execution_evidence(
        evidence, authorization_id="gantry-session-1", duration_seconds=1,
    )


def test_direct_execution_rejects_changed_terminal_physical_summary(tmp_path: Path) -> None:
    records = _passing_execution_records()
    records[12]["pre_release_physical_state"]["tick"] += 1
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, records)
    with pytest.raises(ValueError, match="physical evidence summary mismatch"):
        validate_direct_dance_execution_evidence(
            evidence, authorization_id="gantry-session-1", duration_seconds=1,
        )


def test_direct_execution_validator_accepts_reacquisition_diagnostics(
    tmp_path: Path,
) -> None:
    records = _passing_execution_records()
    records.insert(
        11,
        {
            "schema_version": 1,
            "kind": "g1_true23_stage1_gantry_execution_evidence",
            "event": "pre_arm_causal_reacquisition",
            "authorization_id": "gantry-session-1",
            "monotonic_ns": 0,
            "frame_index": 1339,
            "reason": "lowstate_covered_range_invalid",
        },
    )
    for index, record in enumerate(records):
        record["monotonic_ns"] = index + 1
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, records)

    terminal = validate_direct_dance_execution_evidence(
        evidence,
        authorization_id="gantry-session-1",
        duration_seconds=1,
    )

    assert terminal["motion_mode_restored"] is True


def test_direct_execution_validator_rejects_unknown_diagnostic(
    tmp_path: Path,
) -> None:
    records = _passing_execution_records()
    records.insert(
        11,
        {
            "schema_version": 1,
            "kind": "g1_true23_stage1_gantry_execution_evidence",
            "event": "unexpected_diagnostic",
            "authorization_id": "gantry-session-1",
            "monotonic_ns": 0,
        },
    )
    for index, record in enumerate(records):
        record["monotonic_ns"] = index + 1
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, records)

    with pytest.raises(ValueError, match="event sequence is not exact"):
        validate_direct_dance_execution_evidence(
            evidence,
            authorization_id="gantry-session-1",
            duration_seconds=1,
        )


def test_direct_execution_validator_rejects_old_damping_startup(
    tmp_path: Path,
) -> None:
    records = _passing_execution_records()
    records[7]["startup_damping_frames"] = 1
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, records)
    with pytest.raises(ValueError, match="pre-arm hold gate failed"):
        validate_direct_dance_execution_evidence(
            evidence,
            authorization_id="gantry-session-1",
            duration_seconds=1,
        )


def test_direct_execution_validator_rejects_post_dance_dump(
    tmp_path: Path,
) -> None:
    records = _passing_execution_records()
    records[11]["damping_frames_after_stop"] = 250
    records[12]["damping_frames_after_stop"] = 250
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, records)
    with pytest.raises(ValueError, match="terminal evidence did not pass"):
        validate_direct_dance_execution_evidence(
            evidence,
            authorization_id="gantry-session-1",
            duration_seconds=1,
        )


def test_direct_execution_validator_rejects_transient_mode_restore(
    tmp_path: Path,
) -> None:
    records = _passing_execution_records()
    records[11]["stable_restore_samples"] = 1
    records[12]["stable_restore_samples"] = 1
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, records)

    with pytest.raises(ValueError, match="terminal evidence did not pass"):
        validate_direct_dance_execution_evidence(
            evidence,
            authorization_id="gantry-session-1",
            duration_seconds=1,
        )


def test_direct_execution_validator_rejects_blocked_dump_packet(
    tmp_path: Path,
) -> None:
    records = _passing_execution_records()
    records[12]["rejected_non_positive_gain_commands"] = 1
    evidence = tmp_path / "execution.jsonl"
    _write_jsonl(evidence, records)
    with pytest.raises(ValueError, match="terminal evidence did not pass"):
        validate_direct_dance_execution_evidence(
            evidence,
            authorization_id="gantry-session-1",
            duration_seconds=1,
        )
