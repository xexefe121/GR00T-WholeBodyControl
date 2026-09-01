from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from gear_sonic.scripts.evaluate_g1_true23_frozen_lora_happy_residuals import (
    _selection_key,
)
from gear_sonic.scripts.evaluate_g1_true23_frozen_lora_saved_teleop import (
    _load_decoder_report,
)
from gear_sonic.scripts.fit_g1_true23_frozen_lora_happy_residual import (
    _alpha_name,
    _load_base_decoder,
)
from gear_sonic.scripts.summarize_g1_true23_frozen_lora_parity_candidate import (
    build_candidate_summary,
)
from gear_sonic.scripts.summarize_g1_true23_original_sonic_parity import (
    build_parity_summary,
)


def _reports(candidate_completed: int = 89, candidate_requested: int = 100):
    original = {
        "kind": "g1_released_sonic_policy_mujoco_motion_suite",
        "records": [{
            "name": "happy_dance",
            "source_planner_npz_sha256": "source",
            "metrics": {
                "passed": True,
                "completed_control_steps": 110,
                "requested_control_steps": 110,
            },
        }],
    }
    teacher = {
        "kind": "g1_released_sonic_feature_teacher_true23_physical_adapter",
        "records": [{
            "name": "happy_dance",
            "source_sha256": "source",
            "physical_npz_sha256": "teacher",
            "metrics": {
                "passed": True,
                "completed_control_steps": 110,
                "requested_control_steps": 110,
            },
        }],
    }
    manifest = {
        "kind": "g1_true23_physical_rollout_motion_reference_v1",
        "passed": True,
        "physical_dof": 23,
        "source_sha256": "teacher",
        "output_sha256": "reference",
    }
    candidate = {
        "kind": "g1_true23_genuine_sonic_library_motion_mujoco_replay",
        "physical_dof": 23,
        "source_29dof_physics_used": False,
        "motion_sha256": "reference",
        "passed": candidate_completed == candidate_requested,
        "completed_transitions": candidate_completed,
        "requested_transitions": candidate_requested,
        "failure": None if candidate_completed == candidate_requested else {"type": "gate"},
        "metrics": {
            "maximum_joint_tracking_rmse_rad": 0.7,
            "maximum_relative_tracked_body_position_error_m": 0.5,
        },
    }
    return original, teacher, manifest, candidate


def test_summary_measures_normalized_completion_gap() -> None:
    original, teacher, manifest, candidate = _reports()
    value = build_parity_summary(
        motion_name="happy_dance",
        original_policy_report=original,
        true23_teacher_report=teacher,
        reference_manifest=manifest,
        candidate_report=candidate,
    )
    assert value["provenance"]["chain_validated"] is True
    assert value["parity"]["achieved"] is False
    assert value["parity"]["remaining_candidate_transitions"] == 11
    assert value["parity"]["completion_ratio_gap_to_original"] == pytest.approx(0.11)


def test_full_candidate_completion_reaches_parity() -> None:
    original, teacher, manifest, candidate = _reports(100, 100)
    value = build_parity_summary(
        motion_name="happy_dance",
        original_policy_report=original,
        true23_teacher_report=teacher,
        reference_manifest=manifest,
        candidate_report=candidate,
    )
    assert value["parity"]["achieved"] is True


def test_hash_chain_mismatch_is_rejected() -> None:
    original, teacher, manifest, candidate = _reports()
    broken = deepcopy(candidate)
    broken["motion_sha256"] = "wrong"
    with pytest.raises(ValueError, match="reference hash"):
        build_parity_summary(
            motion_name="happy_dance",
            original_policy_report=original,
            true23_teacher_report=teacher,
            reference_manifest=manifest,
            candidate_report=broken,
        )


def test_saved_teleop_decoder_report_is_hash_bound(tmp_path: Path) -> None:
    decoder = tmp_path / "candidate.onnx"
    decoder.write_bytes(b"diagnostic decoder")
    digest = hashlib.sha256(decoder.read_bytes()).hexdigest()
    report = tmp_path / "candidate.json"
    report.write_text(
        json.dumps(
            {
                "kind": "g1_true23_frozen_lora_diagnostic_decoder_onnx",
                "diagnostic_only": True,
                "deployment_ready": False,
                "hardware_authorized": False,
                "active_motor_control_authorized": False,
                "decoder": {"filename": decoder.name, "sha256": digest},
                "source": {"update_count": 12},
            }
        ),
        encoding="utf-8",
    )
    assert _load_decoder_report(report) == (decoder.resolve(), digest, 12)

    decoder.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        _load_decoder_report(report)


def test_happy_residual_base_is_hash_bound_and_diagnostic(tmp_path: Path) -> None:
    decoder = tmp_path / "base.onnx"
    decoder.write_bytes(b"base decoder")
    digest = hashlib.sha256(decoder.read_bytes()).hexdigest()
    report = tmp_path / "base.json"
    report.write_text(
        json.dumps(
            {
                "kind": "g1_true23_frozen_lora_diagnostic_decoder_onnx",
                "diagnostic_only": True,
                "deployment_ready": False,
                "hardware_authorized": False,
                "active_motor_control_authorized": False,
                "decoder": {"filename": decoder.name, "sha256": digest},
            }
        ),
        encoding="utf-8",
    )
    assert _load_base_decoder(report)[:2] == (decoder.resolve(), digest)
    assert _alpha_name(-0.005) == "minus_0p005"
    assert _alpha_name(0.02) == "plus_0p020"

    decoder.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        _load_base_decoder(report)


def test_happy_residual_selection_prefers_closed_loop_survival() -> None:
    def item(name: str, alpha: float, done: int, error: float, passed=False):
        return {
            "name": name,
            "alpha": alpha,
            "report": {
                "passed": passed,
                "completed_transitions": done,
                "requested_transitions": 100,
                "metrics": {
                    "maximum_relative_tracked_body_position_error_m": error
                },
            },
        }

    candidates = [
        item("low_error", 0.001, 80, 0.2),
        item("survival", 0.02, 90, 0.8),
        item("full", 0.05, 100, 0.9, passed=True),
    ]
    assert max(candidates, key=_selection_key)["name"] == "full"


def test_parity_candidate_requires_preservation_and_saved_pico() -> None:
    decoder = {
        "kind": (
            "g1_true23_frozen_lora_happy_residual_diagnostic_decoder_onnx"
        ),
        "diagnostic_only": True,
        "deployment_ready": False,
        "hardware_authorized": False,
        "decoder": {"sha256": "decoder"},
        "source": {"base_update_count": 25, "alpha": 0.002},
    }

    def suite(happy_passed: bool, survival: float):
        return {
            "kind": "g1_true23_frozen_lora_comparison_result_v1",
            "diagnostic_only": True,
            "deployment_ready": False,
            "hardware_authorized": False,
            "decoder_report": {"sha256": "report"},
            "suite": {"sha256": "suite"},
            "second_referee": {"survival_rate": survival},
            "cases": [
                {"label": "walk001", "passed": True},
                {"label": "happy_dance", "passed": happy_passed},
            ],
        }

    base = suite(False, 0.8)
    candidate = suite(True, 0.9)
    saved = {
        "kind": "g1_true23_clean_mujoco_teleop_session",
        "passed": True,
        "physical_dof": 23,
        "decoder_sha256": "decoder",
        "fallback_active": False,
        "completed_transitions": 684,
    }
    parity = {
        "kind": "g1_true23_original_sonic_parity_summary",
        "parity": {"achieved": True},
        "provenance": {"chain_validated": True},
    }
    value = build_candidate_summary(
        decoder_report=decoder,
        decoder_report_sha256="report",
        base_suite=base,
        candidate_suite=candidate,
        saved_teleop=saved,
        parity=parity,
    )
    assert value["simulator_diagnostic_default_candidate"] is True
    assert value["full_suite"]["newly_passing_cases"] == ["happy_dance"]

    regressed = deepcopy(candidate)
    regressed["cases"][0]["passed"] = False
    with pytest.raises(ValueError, match="loses passing"):
        build_candidate_summary(
            decoder_report=decoder,
            decoder_report_sha256="report",
            base_suite=base,
            candidate_suite=regressed,
            saved_teleop=saved,
            parity=parity,
        )
