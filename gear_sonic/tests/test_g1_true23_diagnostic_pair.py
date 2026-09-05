import copy
import hashlib
import json

import numpy as np
import pytest

from gear_sonic.scripts.audit_g1_true23_runtime_encoder_parity import compare_tokens
from gear_sonic.utils.g1_true23_diagnostic_pair import (
    ENCODER_CONTRACT,
    load_diagnostic_pair,
    load_residual_diagnostic_pair,
)


@pytest.fixture
def pair_files(tmp_path):
    paths = []
    for component, shapes in (("encoder", ([1, 267], [1, 64])), ("decoder", ([1, 994], [1, 23]))):
        model = tmp_path / f"test.diagnostic.{component}.onnx"
        model.write_bytes(component.encode())  # Provenance-only fixture; not an ONNX execution test.
        report = {
            "kind": f"g1_true23_frozen_lora_diagnostic_{component}_onnx",
            "schema_version": 1,
            "encoder_contract": copy.deepcopy(ENCODER_CONTRACT),
            "paired_encoder_state_sha256": "a" * 64,
            "source": {key: "b" * 64 for key in ("sha256", "policy_state_sha256", "adapter_state_sha256")},
            component: {
                "filename": model.name,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "input_shape": shapes[0],
                "output_shape": shapes[1],
                "opset": 13,
            },
            "validation": {
                "weights_only_diagnostic_policy_validated": True,
                f"exact_merged_{component}_reconstructed": True,
                "onnx_structure_validated": True,
                "onnx_runtime_parity": {
                    "parity_case_count": 3,
                    "parity_max_abs_error": 0.0,
                    "parity_max_rel_error": 0.0,
                    "parity_atol": 1e-5,
                    "parity_rtol": 1e-5,
                    "onnx_checker_full_check": True,
                    "shape_inference": True,
                },
            },
            "diagnostic_only": True,
            "deployment_ready": False,
            "promotion_eligible": False,
            "hardware_authorized": False,
            "active_motor_control_authorized": False,
        }
        path = model.with_suffix(".json")
        path.write_text(json.dumps(report))
        paths.append(path)
    return paths


def test_pair_binds_both_models_reports_and_checkpoint_without_authorizing(pair_files):
    pair = load_diagnostic_pair(*pair_files)
    assert pair["validated"] and not pair["hardware_authorized"] and not pair["deployment_ready"]
    assert pair["encoder"]["report_sha256"] == hashlib.sha256(pair_files[0].read_bytes()).hexdigest()
    pair["encoder_contract"]["input_shape"][1] = 999
    assert ENCODER_CONTRACT["input_shape"] == [1, 267]
    assert load_diagnostic_pair(*pair_files)["encoder_contract"]["input_shape"] == [1, 267]


@pytest.mark.parametrize(
    "field,value",
    [
        ("paired_encoder_state_sha256", "c" * 64),
        ("paired_encoder_state_sha256", None),
        ("encoder_contract", None),
        ("hardware_authorized", True),
        ("promotion_eligible", True),
        ("diagnostic_only", 1),
        ("source", {"sha256": "c" * 64, "policy_state_sha256": "b" * 64, "adapter_state_sha256": "b" * 64}),
    ],
)
def test_pair_rejects_missing_mixed_or_authorizing_evidence(pair_files, field, value):
    report = json.loads(pair_files[1].read_text())
    report[field] = value
    pair_files[1].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_diagnostic_pair(*pair_files)


@pytest.mark.parametrize("filename", ["../test.onnx", "..\\test.onnx", "C:test.onnx", "", ".", ".."])
def test_pair_rejects_nonlocal_model_names(pair_files, filename):
    report = json.loads(pair_files[0].read_text())
    report["encoder"]["filename"] = filename
    pair_files[0].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="local basename"):
        load_diagnostic_pair(*pair_files)


def test_pair_rejects_changed_model_bytes(pair_files):
    pair_files[0].with_suffix(".onnx").write_bytes(b"different encoder")
    with pytest.raises(ValueError, match="bytes changed"):
        load_diagnostic_pair(*pair_files)


@pytest.mark.parametrize(
    "key,value",
    [
        ("parity_max_abs_error", 0.0625),
        ("parity_max_abs_error", float("nan")),
        ("parity_case_count", True),
        ("parity_rtol", 0.1),
    ],
)
def test_pair_rejects_invalid_or_inexact_encoder_parity(pair_files, key, value):
    report = json.loads(pair_files[0].read_text())
    report["validation"]["onnx_runtime_parity"][key] = value
    pair_files[0].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_diagnostic_pair(*pair_files)


def test_token_comparison_requires_exact_coordinates():
    expected = np.zeros((1, 64), dtype=np.float32)
    assert compare_tokens(expected, expected.copy()) == (0, 0.0)
    actual = expected.copy()
    actual[0, [0, 10]] = [0.0625, -0.9375]
    assert compare_tokens(expected, actual) == (2, 0.9375)


@pytest.mark.parametrize("value", [np.zeros(64), np.full((1, 64), np.nan), np.full((1, 64), np.inf)])
def test_token_comparison_rejects_bad_evidence(value):
    with pytest.raises(ValueError):
        compare_tokens(np.zeros((1, 64)), value)


@pytest.fixture
def residual_files(pair_files):
    pair = load_diagnostic_pair(*pair_files)
    model = pair_files[0].with_name("candidate.onnx")
    model.write_bytes(b"fitted decoder")
    manifest = {
        "kind": "g1_true23_frozen_lora_happy_residual_diagnostic_v1",
        "diagnostic_only": True,
        "closed_loop_screening_required": True,
        "deployment_ready": False,
        "hardware_authorized": False,
        "robot_network_commands": False,
        "source": {
            "diagnostic_pair": pair,
            "encoder_sha256": pair["encoder"]["sha256"],
            "base_decoder_sha256": pair["decoder"]["sha256"],
            "base_decoder_report_sha256": pair["decoder"]["report_sha256"],
        },
        "candidates": [
            {"decoder_filename": model.name, "decoder_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
        ],
    }
    path = pair_files[0].with_name("manifest.json")
    path.write_text(json.dumps(manifest))
    return path, model


def test_residual_pair_binds_fitting_encoder_and_candidate(residual_files):
    pair = load_residual_diagnostic_pair(*residual_files)
    assert pair["validated"] and not pair["hardware_authorized"]
    assert pair["base_decoder"]["sha256"] != pair["decoder"]["sha256"]


@pytest.mark.parametrize(
    "field,value", [("diagnostic_pair", None), ("encoder_sha256", "c" * 64), ("base_decoder_sha256", "d" * 64)]
)
def test_residual_pair_rejects_missing_or_mixed_provenance(residual_files, field, value):
    path, _ = residual_files
    manifest = json.loads(path.read_text())
    manifest["source"][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_residual_diagnostic_pair(*residual_files)


def test_residual_pair_rejects_changed_candidate(residual_files):
    residual_files[1].write_bytes(b"wrong candidate")
    with pytest.raises(ValueError, match="bytes changed"):
        load_residual_diagnostic_pair(*residual_files)


def test_residual_pair_revalidates_base_report_bytes(residual_files, pair_files):
    # Even semantically harmless changes invalidate the saved provenance hash.
    pair_files[0].write_text(pair_files[0].read_text() + "\n")
    with pytest.raises(ValueError, match="provenance changed"):
        load_residual_diagnostic_pair(*residual_files)
