from __future__ import annotations

import copy
from pathlib import Path

from gear_sonic.scripts import replay_g1_true23_pico_packets_zmq as replay
from gear_sonic.scripts.run_g1_true23_frozen_lora_dance_gantry import (
    _active_command,
    _wsl_path,
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
    assert "--frozen-lora-dance" in command
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
