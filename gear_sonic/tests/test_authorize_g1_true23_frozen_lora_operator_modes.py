from __future__ import annotations

import argparse
import json
from pathlib import Path

from gear_sonic.scripts import (
    authorize_g1_true23_frozen_lora_dance_gantry as authorize,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    promotion = tmp_path / "promotion.json"
    encoder = tmp_path / "encoder.onnx"
    decoder = tmp_path / "decoder.onnx"
    report = tmp_path / "decoder.json"
    shadow = tmp_path / "shadow.jsonl"
    promotion.write_text("{}", encoding="utf-8")
    encoder.write_bytes(b"encoder")
    decoder.write_bytes(b"decoder")
    report.write_text(
        json.dumps({"decoder": {"filename": decoder.name}}),
        encoding="utf-8",
    )
    shadow.write_text("{}\n", encoding="utf-8")
    return argparse.Namespace(
        promotion=promotion,
        encoder=encoder,
        decoder_report=report,
        live_shadow_evidence=shadow,
        authorization_id="gantry-session-1",
        gantry_authorize="I_CONFIRM_G1_TRUE23_STAGE1_GANTRY",
        direct_dance_command="DANCE",
    )


def test_operator_modes_get_distinct_bound_envelopes(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(authorize, "promotion_body", lambda _args: {})
    monkeypatch.setattr(
        authorize,
        "validate_causal_live_shadow_evidence",
        lambda *_args, **_kwargs: {
            "action_frames": 100,
            "evidence_sha256": "a" * 64,
        },
    )
    direct = authorize.active_body_for_operator_mode(args, direct_dance=True)
    live = authorize.active_body_for_operator_mode(args, direct_dance=False)

    assert direct["kind"] == authorize.KIND
    assert direct["stage_one_envelope"] == {
        "action_fraction": 0.10,
        "maximum_target_rate_rad_per_second": 0.25,
        "maximum_post_arm_duration_seconds": 5,
        "wireless_deadman_required": False,
        "wireless_stop_required": False,
        "direct_dance_command_required": "DANCE",
        "physical_estop_required": True,
        "process_signal_stop_required": True,
    }
    assert live["kind"] == authorize.LIVE_KIND
    assert live["stage_one_envelope"] == {
        "action_fraction": 0.10,
        "maximum_target_rate_rad_per_second": 0.25,
        "maximum_post_arm_duration_seconds": 10,
        "wireless_deadman_required": True,
        "wireless_stop_required": True,
        "direct_dance_command_required": False,
        "physical_estop_required": True,
        "process_signal_stop_required": True,
    }
    assert direct["promotion_payload_sha256"] != live["promotion_payload_sha256"]
