from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from gear_sonic.scripts import preflight_g1_true23_native_runtime as preflight


def _elf(path: Path, machine: int = 183) -> Path:
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    header[18:20] = machine.to_bytes(2, "little")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header)
    path.chmod(0o755)
    return path


def _healthy_report() -> dict:
    return {
        "kind": "g1_true23_pico_tracking_health_probe_v1",
        "passed": True,
        "xrt_binding_sha256": preflight.EXPECTED_XRT_SHA256,
        "service_binary_sha256": preflight.EXPECTED_SERVICE_SHA256,
        "authorization": {"read_only": True, "robot_commands_published": False},
        "latest_health": {
            "health_available": True,
            "health_valid": True,
            "health_supported": True,
            "health_connect_state_result": 0,
            "health_calibrated": True,
            "health_calibration_result": 0,
            "health_is_tracking": True,
            "health_tracking_state_code": 0,
            "health_body_data_result": 0,
            "health_body_state_result": 0,
            "health_body_error_code": 0,
            "health_body_role_count": 24,
            "health_connected_band_count": 2,
            "health_unique_tracker_count": 2,
        },
    }


def test_wrong_architecture_never_runs_binary_or_dependency_loader(tmp_path) -> None:
    def forbidden(command):
        pytest.fail(f"command must not run: {command}")

    result = preflight.inspect_binary(
        _elf(tmp_path / "controller", machine=62), system="Linux", machine="aarch64", run=forbidden
    )
    assert result["ready"] is False
    assert result["error"] == "binary is x86_64; host requires aarch64"


@pytest.mark.parametrize(
    "data", [b"libddsc.so.0.10.2", b"version https://git-lfs.github.com/spec/v1\n", b"MZ" + b"0" * 80]
)
def test_elf_rejects_checkout_stubs_and_wrong_formats(tmp_path, data) -> None:
    path = tmp_path / "binary"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="materialized"):
        preflight.elf_architecture(path)


def test_missing_native_dependencies_fail_without_launch(tmp_path) -> None:
    calls = []

    def dependency_only(command):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "libonnxruntime.so.1 => not found\n", "")

    binary = _elf(tmp_path / "controller")
    result = preflight.inspect_binary(binary, system="Linux", machine="aarch64", run=dependency_only)
    assert result["ready"] is False
    assert calls == [["ldd", str(binary)]]


def test_health_requires_recent_calibrated_matching_source(tmp_path) -> None:
    path = tmp_path / "health.json"
    value = _healthy_report()
    path.write_text(json.dumps(value), encoding="utf-8")
    now = path.stat().st_mtime
    assert preflight.inspect_headset_report(path, now=now + 1)["ready"] is True
    assert preflight.inspect_headset_report(path, now=now + 301)["ready"] is False
    value["latest_health"]["health_calibrated"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    assert preflight.inspect_headset_report(path, now=path.stat().st_mtime)["ready"] is False
    value["latest_health"]["health_calibrated"] = True
    value["xrt_binding_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    assert preflight.inspect_headset_report(path, now=path.stat().st_mtime)["ready"] is False


def _artifact_fixture(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    paths = {
        "encoder": root / "encoder.onnx",
        "decoder_report": root
        / preflight.ARTIFACT_ROOT
        / "original_sonic_happy_residual_v1/candidate.plus_0p002.decoder.json",
        "packet_bundle": root
        / preflight.ARTIFACT_ROOT
        / "physical_dance_v1/original_sonic_happy.true23.causal_packets.json",
        "promotion": root
        / preflight.ARTIFACT_ROOT
        / "physical_dance_v1/candidate.plus_0p002.dance_shadow_promotion.v2.json",
    }
    paths["decoder"] = paths["decoder_report"].with_name("candidate.plus_0p002.decoder.onnx")
    pins = {}
    for role in preflight.SELECTED_PINS:
        paths[role].parent.mkdir(parents=True, exist_ok=True)
        paths[role].write_bytes(role.encode())
        pins[role] = hashlib.sha256(role.encode()).hexdigest()
    monkeypatch.setattr(preflight, "SELECTED_PINS", pins)
    promotion = {
        "kind": "g1_true23_frozen_lora_dance_shadow_admission_v1",
        "native_action_dof": 23,
        "deployment_bytes_authorized_for_shadow": True,
        "active_motor_control_authorized": False,
        "free_standing_authorized": False,
        "source_artifacts": {f"{role}_sha256": digest for role, digest in pins.items()},
    }
    encoded = (json.dumps(promotion, sort_keys=True, separators=(",", ":")) + "\n").encode()
    promotion["promotion_payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    paths["promotion"].write_text(json.dumps(promotion), encoding="utf-8")
    return root, paths


def test_artifact_tamper_is_rejected_despite_existing_report(tmp_path, monkeypatch) -> None:
    _, paths = _artifact_fixture(tmp_path, monkeypatch)
    assert all(item["ready"] for item in preflight.inspect_selected_artifacts(paths).values())
    paths["decoder"].write_bytes(b"different policy")
    assert preflight.inspect_selected_artifacts(paths)["decoder"]["ready"] is False
    promotion = json.loads(paths["promotion"].read_text())
    promotion["free_standing_authorized"] = True
    paths["promotion"].write_text(json.dumps(promotion), encoding="utf-8")
    assert preflight.inspect_selected_artifacts(paths)["promotion"]["ready"] is False


def test_native_shadow_plan_uses_same_host_clock_and_never_starts_controller(tmp_path, monkeypatch) -> None:
    root, paths = _artifact_fixture(tmp_path, monkeypatch)
    for name in ("g1_true23_live_shadow", "g1_true23_active_gantry"):
        _elf(root / "gear_sonic_deploy/target/release" / name)
    monkeypatch.setattr(preflight.platform, "system", lambda: "Linux")
    monkeypatch.setattr(preflight.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(preflight.platform, "release", lambda: "5.10.0-tegra")
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _: object())
    real_is_dir = Path.is_dir
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda path: str(path).replace("\\", "/").endswith("/sys/class/net/eth0") or real_is_dir(path),
    )
    calls = []

    def dependency_only(command):
        calls.append(command)
        assert command[0] == "ldd"
        return subprocess.CompletedProcess(command, 0, "libc.so.6 => /lib/libc.so.6\n", "")

    args = preflight.parser().parse_args(
        [
            "--repository-root",
            str(root),
            "--encoder",
            str(paths["encoder"]),
            "--service-binary",
            str(root / "missing_service"),
        ]
    )
    result = preflight.collect(args, run=dependency_only)
    assert result["saved_clip_shadow_runtime_ready"] is True
    assert result["runtime"] == "native_linux"
    assert result["physical_teleop_ready"] is False
    assert result["live_headset_prerequisites_present"] is False
    commands = result["same_host_shadow_commands"]
    publisher = commands["publisher_argv"]
    assert publisher[publisher.index("--timestamp-clock") + 1] == "local"
    assert "--execute-stage-one" not in commands["shadow_argv"]
    assert len(calls) == 2
    assert not (root / preflight.ARTIFACT_ROOT / "native_runtime_preflight").exists()


def test_windows_inspection_does_not_call_elf_or_wsl(tmp_path) -> None:
    def forbidden(command):
        pytest.fail(f"unexpected invocation: {command}")

    result = preflight.inspect_binary(
        _elf(tmp_path / "controller", machine=62), system="Windows", machine="AMD64", run=forbidden
    )
    assert result["ready"] is False
    assert "inside WSL" in result["error"]
