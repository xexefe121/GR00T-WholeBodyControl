from __future__ import annotations

import json

from gear_sonic.scripts.run_g1_true23_frozen_lora_dance_gantry import (
    _active_command,
)
from gear_sonic.scripts.run_g1_true23_frozen_lora_live_gantry import (
    _live_publisher_command,
    _published_reference,
)


def test_live_publisher_command_is_real_pico_and_read_only(tmp_path) -> None:
    evidence = tmp_path / "publisher.jsonl"
    command = _live_publisher_command(
        distro="Ubuntu-22.04",
        publisher_python="/usr/bin/python3",
        workspace="/mnt/z/repo",
        xrt_module_dir="/mnt/z/xrt",
        soma_source_root="/root/soma",
        capture_python="/usr/bin/python3",
        endpoint="tcp://127.0.0.1:5557",
        packets=90_000,
        timeout_seconds=1_800,
        evidence=str(evidence),
        pico_client_apk_sha256="a" * 64,
    )
    assert command[:7] == [
        "wsl.exe",
        "-d",
        "Ubuntu-22.04",
        "--",
        "env",
        "PYTHONPATH=/mnt/z/repo:/mnt/z/xrt",
        "/usr/bin/python3",
    ]
    assert "gear_sonic.scripts.stream_g1_23dof_pico_causal_zmq" in command
    assert "lowcmd" not in " ".join(command).lower()
    assert command[command.index("--packets") + 1] == "90000"


def test_published_reference_requires_packet_event(tmp_path) -> None:
    evidence = tmp_path / "publisher.jsonl"
    evidence.write_text(
        json.dumps({"event": "session_start"}) + "\n",
        encoding="utf-8",
    )
    assert _published_reference(evidence) is False
    with evidence.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "reference_packet_published"}) + "\n")
    assert _published_reference(evidence) is True


def test_live_controller_uses_frozen_policy_without_direct_dance() -> None:
    command = _active_command(
        distro="Ubuntu-22.04",
        binary="/repo/controller",
        encoder="/repo/encoder.onnx",
        decoder="/repo/decoder.onnx",
        metadata="/repo/decoder.json",
        promotion="/repo/promotion.json",
        active_promotion="/repo/live-active.json",
        live_shadow_evidence="/repo/shadow.jsonl",
        authorization_id="live-session-1",
        network="eth0",
        endpoint="tcp://127.0.0.1:5557",
        evidence="/repo/execution.jsonl",
        duration_seconds=10,
        gantry_authorize="I_CONFIRM_G1_TRUE23_STAGE1_GANTRY",
        frozen_lora_policy=True,
    )
    assert "--frozen-lora-policy" in command
    assert "--direct-dance-command" not in command
    assert command[command.index("--post-arm-duration-seconds") + 1] == "10"
