"""Bind simulator encoder/decoder exports to the same validated checkpoint.

This verifies export provenance and bytes, not fresh numerical parity, motion
quality, hardware compatibility or authorization. Legacy sidecars without a
paired encoder identity must be re-exported, never silently grandfathered in.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

ENCODER_CONTRACT = {
    "input_shape": [1, 267],
    "output_shape": [1, 64],
    "dtype": "float32",
    "fsq_level": 32,
    "fsq_formula": "fsq_tanh_round_ste_even_levels_v1",
}


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_component(report_path: Path, component: str):
    path = report_path.expanduser().resolve(strict=True)
    report_bytes = path.read_bytes()
    report = json.loads(report_bytes)
    if not isinstance(report, dict) or report.get("kind") != f"g1_true23_frozen_lora_diagnostic_{component}_onnx":
        raise ValueError(f"invalid diagnostic {component} report kind")
    flags = {
        "diagnostic_only": True,
        "deployment_ready": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "active_motor_control_authorized": False,
    }
    if any(report.get(key) is not value for key, value in flags.items()):
        raise ValueError("diagnostic pair authorization contract mismatch")
    if report.get("schema_version") != 1 or report.get("encoder_contract") != ENCODER_CONTRACT:
        raise ValueError("diagnostic pair encoder contract missing or incompatible; re-export both components")
    if not _digest(report.get("paired_encoder_state_sha256")):
        raise ValueError("paired encoder identity missing; re-export both components")
    source = report.get("source")
    if not isinstance(source, dict) or any(
        not _digest(source.get(key)) for key in ("sha256", "policy_state_sha256", "adapter_state_sha256")
    ):
        raise ValueError("diagnostic checkpoint identity missing")
    model = report.get(component)
    if not isinstance(model, dict) or not _digest(model.get("sha256")):
        raise ValueError(f"diagnostic {component} model identity missing")
    filename = model.get("filename")
    if (
        not isinstance(filename, str)
        or not filename
        or filename in (".", "..")
        or any(c in filename for c in ("/", "\\", ":"))
    ):
        raise ValueError("diagnostic model filename must be a local basename")
    model_path = path.with_name(filename).resolve(strict=True)
    if model_path.parent != path.parent:
        raise ValueError("diagnostic model escapes report directory")
    expected_shapes = ([1, 267], [1, 64]) if component == "encoder" else ([1, 994], [1, 23])
    if (model.get("input_shape"), model.get("output_shape")) != expected_shapes or model.get("opset") != 13:
        raise ValueError("diagnostic pair model ABI mismatch")
    if _file_hash(model_path) != model["sha256"]:
        raise ValueError(f"diagnostic {component} bytes changed")
    validation = report.get("validation")
    if not isinstance(validation, dict) or any(
        validation.get(key) is not True
        for key in (
            "weights_only_diagnostic_policy_validated",
            f"exact_merged_{component}_reconstructed",
            "onnx_structure_validated",
        )
    ):
        raise ValueError("diagnostic component validation missing")
    parity = validation.get("onnx_runtime_parity")
    if (
        not isinstance(parity, dict)
        or type(parity.get("parity_case_count")) is not int
        or parity["parity_case_count"] < 3
    ):
        raise ValueError("diagnostic component parity evidence missing")
    for key in ("parity_max_abs_error", "parity_max_rel_error", "parity_atol", "parity_rtol"):
        value = parity.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("diagnostic component parity evidence invalid")
    if parity["parity_atol"] > 1e-5 or parity["parity_rtol"] > 1e-5:
        raise ValueError("diagnostic component parity tolerance widened")
    if component == "encoder" and parity["parity_max_abs_error"] != 0:
        raise ValueError("encoder discrete token parity must be exact")
    if parity.get("onnx_checker_full_check") is not True or parity.get("shape_inference") is not True:
        raise ValueError("diagnostic component ONNX validation missing")
    return report, {
        "path": str(model_path),
        "sha256": model["sha256"],
        "report_path": str(path),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
    }


def load_diagnostic_pair(encoder_report: Path, decoder_report: Path) -> dict:
    """Reject unknown or mixed model pairs before simulator inference or fitting."""
    encoder, encoder_identity = _load_component(encoder_report, "encoder")
    decoder, decoder_identity = _load_component(decoder_report, "decoder")
    if encoder["paired_encoder_state_sha256"] != decoder["paired_encoder_state_sha256"]:
        raise ValueError("diagnostic encoder/decoder training encoder mismatch")
    # Same weights can appear in different adapters. Require same source export
    # anyway so pairing cannot accidentally combine unrelated model generations.
    if any(
        encoder["source"][key] != decoder["source"][key]
        for key in ("sha256", "policy_state_sha256", "adapter_state_sha256")
    ):
        raise ValueError("diagnostic encoder/decoder checkpoint mismatch")
    return {
        "kind": "g1_true23_diagnostic_pair_provenance_v1",
        "validated": True,
        "encoder": encoder_identity,
        "decoder": decoder_identity,
        "paired_encoder_state_sha256": encoder["paired_encoder_state_sha256"],
        "encoder_contract": copy.deepcopy(ENCODER_CONTRACT),
        "source": dict(decoder["source"]),
        "hardware_authorized": False,
        "deployment_ready": False,
        "scope": "same_checkpoint_export_provenance_and_model_bytes_not_new_numerical_or_physical_qualification",
    }


def load_residual_diagnostic_pair(manifest_path: Path, decoder_path: Path) -> dict:
    """Bind a fitted decoder to the exact encoder used to construct its labels."""
    path = manifest_path.expanduser().resolve(strict=True)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if (
        not isinstance(manifest, dict)
        or manifest.get("kind") != "g1_true23_frozen_lora_happy_residual_diagnostic_v1"
    ):
        raise ValueError("invalid residual diagnostic manifest")
    for key, value in {
        "diagnostic_only": True,
        "closed_loop_screening_required": True,
        "deployment_ready": False,
        "hardware_authorized": False,
        "robot_network_commands": False,
    }.items():
        if manifest.get(key) is not value:
            raise ValueError("residual diagnostic authorization contract mismatch")
    source = manifest.get("source")
    saved_pair = source.get("diagnostic_pair") if isinstance(source, dict) else None
    if not isinstance(saved_pair, dict):
        raise ValueError("residual training encoder identity missing; refit with a validated pair")
    try:
        pair = load_diagnostic_pair(
            Path(saved_pair["encoder"]["report_path"]), Path(saved_pair["decoder"]["report_path"])
        )
    except (KeyError, TypeError) as error:
        raise ValueError("residual base pair identity missing") from error
    if saved_pair != pair or source.get("encoder_sha256") != pair["encoder"]["sha256"]:
        raise ValueError("residual base pair provenance changed")
    if (
        source.get("base_decoder_sha256") != pair["decoder"]["sha256"]
        or source.get("base_decoder_report_sha256") != pair["decoder"]["report_sha256"]
    ):
        raise ValueError("residual base decoder identity changed")
    candidate_path = decoder_path.expanduser().resolve(strict=True)
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or candidate_path.parent != path.parent:
        raise ValueError("residual candidate must be beside its manifest")
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("decoder_filename") == candidate_path.name
    ]
    if len(matches) != 1 or not _digest(matches[0].get("decoder_sha256")):
        raise ValueError("residual candidate identity missing or duplicated")
    if _file_hash(candidate_path) != matches[0]["decoder_sha256"]:
        raise ValueError("residual decoder bytes changed")
    return {
        **pair,
        "kind": "g1_true23_residual_diagnostic_pair_provenance_v1",
        "base_decoder": pair["decoder"],
        "decoder": {"path": str(candidate_path), "sha256": matches[0]["decoder_sha256"]},
        "residual_manifest": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()},
        "scope": "fitted_candidate_and_exact_fitting_encoder_provenance_not_numerical_or_physical_qualification",
    }
