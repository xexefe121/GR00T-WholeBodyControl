"""Versioned simulation-only contract for training with corrected SONIC semantics."""

from gear_sonic.utils.g1_true23_buffered_reference import (
    BUFFERED_TIMING,
    CAUSAL_TIMING,
    reference_profile_contract,
)
from gear_sonic.utils.g1_true23_generalist_corpus import canonical_digest
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_ACTION_CONVENTION, source_action_codec_contract
from gear_sonic.utils.g1_true23_virtual_source_reference import VIRTUAL_SOURCE_REFERENCE


def release_compatibility_contract(
    source_geometry_sha256, action_convention="released_bounded_linear", reference_timing=CAUSAL_TIMING
):
    if reference_timing not in (CAUSAL_TIMING, BUFFERED_TIMING):
        raise ValueError("unsupported release reference timing")
    if reference_timing == BUFFERED_TIMING and action_convention != SOURCE_ACTION_CONVENTION:
        raise ValueError("buffered source requires original source action units")
    if action_convention not in ("released_bounded_linear", SOURCE_ACTION_CONVENTION):
        raise ValueError("unsupported release action convention")
    if (
        not isinstance(source_geometry_sha256, str)
        or len(source_geometry_sha256) != 64
        or any(c not in "0123456789abcdef" for c in source_geometry_sha256)
    ):
        raise ValueError("release compatibility requires source29 geometry SHA256")
    result = dict(
        kind="native23_causal_release_compatibility_v1",
        reference_geometry=VIRTUAL_SOURCE_REFERENCE,
        source_geometry_sha256=source_geometry_sha256,
        reference_phase="q9_received_history_only",
        action_convention="released_bounded_linear",
        target_transform="existing_native23_tanh_inverse_precompensation_with_projection",
        previous_action="effective_safe_normalized_target",
        measured_robot_geometry="native23_unchanged",
        evaluation_target_geometry="native23_unchanged",
        deployment_ready=False,
        hardware_authorized=False,
    )
    if action_convention == SOURCE_ACTION_CONVENTION:
        result.update(
            kind="native23_causal_release_compatibility_v2",
            action_convention=action_convention,
            source_action_codec=source_action_codec_contract(),
            previous_action="effective_safe_target_in_original_source_action_units",
        )
    if reference_timing == BUFFERED_TIMING:
        result.update(
            kind="native23_buffered_release_compatibility_v3",
            reference_timing=reference_timing,
            reference_phase="received_horizon_q0_with_current_measured_orientation",
            reference_profile=reference_profile_contract(reference_timing),
        )
    return {**result, "contract_sha256": canonical_digest(result)}


def validate_release_compatibility(contract):
    if not isinstance(contract, dict) or contract != release_compatibility_contract(
        contract.get("source_geometry_sha256"),
        contract.get("action_convention"),
        contract.get("reference_timing", CAUSAL_TIMING),
    ):
        raise ValueError("release compatibility contract mismatch")
    return contract
