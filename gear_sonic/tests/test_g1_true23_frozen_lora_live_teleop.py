from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    SupervisedCleanTrue23MujocoController,
    motion_reference_terms,
    validate_reference_terms,
)
from gear_sonic.utils.g1_true23_frozen_lora_live_teleop import (
    EXPECTED_CANDIDATE_SUMMARY_SHA256,
    EXPECTED_DECODER_REPORT_SHA256,
    EXPECTED_DECODER_SHA256,
    LiveTransportFault,
    load_frozen_lora_live_profile,
    validate_live_packet,
)

ROOT = Path(__file__).resolve().parents[2]
SELECTED = ROOT / "artifacts/g1_true23_frozen_lora/original_sonic_happy_residual_v1"
MOTION = ROOT / "external_dependencies/unitree_rl_mjlab/src/assets/motions/g1_23dof/B_DadDance.npz"


def _motion() -> dict[str, np.ndarray]:
    with np.load(MOTION, allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def test_selected_live_profile_is_exact_hash_bound() -> None:
    profile = load_frozen_lora_live_profile(
        decoder_report_path=SELECTED / "candidate.plus_0p002.decoder.json",
        candidate_summary_path=SELECTED / "candidate.plus_0p002.summary.json",
    )
    assert profile.decoder_sha256 == EXPECTED_DECODER_SHA256
    assert profile.decoder_report_sha256 == EXPECTED_DECODER_REPORT_SHA256
    assert profile.candidate_summary_sha256 == EXPECTED_CANDIDATE_SUMMARY_SHA256
    assert profile.base_update_count == 25
    assert profile.residual_alpha == 0.002


def test_live_packet_faults_are_fail_closed_and_classified() -> None:
    motion = _motion()
    first = motion_reference_terms(motion, 9)
    second = motion_reference_terms(motion, 10)
    first_terms = validate_reference_terms(first)
    first_summary = validate_live_packet(
        first,
        previous=None,
        maximum_age_ns=100_000_000,
        now_ns=first_terms["control_monotonic_ns"],
    )
    second_terms = validate_reference_terms(second)
    with pytest.raises(LiveTransportFault, match="packet age") as stale:
        validate_live_packet(
            second,
            previous=first_summary,
            maximum_age_ns=100_000_000,
            now_ns=second_terms["control_monotonic_ns"] + 100_000_001,
        )
    assert stale.value.trigger == "transport_stale"

    skipped = motion_reference_terms(motion, 11)
    skipped_terms = validate_reference_terms(skipped)
    with pytest.raises(LiveTransportFault, match="contiguous") as gap:
        validate_live_packet(
            skipped,
            previous=first_summary,
            maximum_age_ns=100_000_000,
            now_ns=skipped_terms["control_monotonic_ns"],
        )
    assert gap.value.trigger == "transport_gap"

    malformed = copy.deepcopy(second)
    malformed.pop("q_ref23_native")
    with pytest.raises(LiveTransportFault) as payload:
        validate_live_packet(
            malformed,
            previous=first_summary,
            maximum_age_ns=100_000_000,
            now_ns=second_terms["control_monotonic_ns"],
        )
    assert payload.value.trigger == "transport_payload"


def test_transport_fallback_activation_is_latched() -> None:
    activations: list[np.ndarray] = []
    controller = object.__new__(SupervisedCleanTrue23MujocoController)
    controller.fallback_active = False
    controller.fallback_trigger = None
    controller.fallback_transition = None
    controller.completed = 17
    controller.data = SimpleNamespace(qpos=np.zeros(30, dtype=np.float64))
    controller.fallback_policy = SimpleNamespace(
        activate=lambda value: activations.append(value.copy())
    )
    controller.activate_fallback("transport_timeout")
    controller.activate_fallback("transport_gap")
    assert controller.fallback_active is True
    assert controller.fallback_trigger == "transport_timeout"
    assert controller.fallback_transition == 17
    assert len(activations) == 1
    with pytest.raises(ValueError, match="unsupported"):
        fresh = object.__new__(SupervisedCleanTrue23MujocoController)
        fresh.fallback_active = False
        fresh.activate_fallback("unknown")
