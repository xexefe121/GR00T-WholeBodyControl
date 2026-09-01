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
        restore_poll_attempts=2,
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
        restore_poll_attempts=2,
        damping_frames_after_stop=0,
        required_damping_frames_after_stop=0,
        normal_return_hold_frames=250,
        required_normal_return_hold_frames=250,
        motion_mode_restored=True,
        restored_motion_mode_name="normal",
        restored_locomotion_fsm_id=801,
        restored_locomotion_fsm_mode=0,
    )
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
        direct_dance_command="DANCE",
    )
    assert command[:8] == [
        "wsl.exe",
        "-d",
        "Ubuntu-22.04",
        "--",
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
