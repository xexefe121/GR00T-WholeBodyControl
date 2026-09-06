"""Diagnostic-only pair boundaries and hash-bound temporal semantics."""

from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from gear_sonic.envs.mjlab.sonic_true23_causal_history import causal_history_profile_contract
from gear_sonic.scripts.export_g1_true23_generalist import (
    DecoderExport,
    EncoderExport,
    export_pair,
    validate_export_semantics,
)
from gear_sonic.tests.test_native23_generalist_runner import lineage as lineage
from gear_sonic.trl.mjlab.frozen_platform_lora_actor import FrozenPlatformTrue23Core, G1True23AnalyticCodec
from gear_sonic.trl.mjlab.native23_generalist_runner import CHECKPOINT_HEADER
from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes
from gear_sonic.utils.g1_23dof_contract import LOW_LATENCY_RELEASE_SHA256
from gear_sonic.utils.g1_23dof_mjlab_training import build_config_manifest


def checkpoint_with_config(lineage, resolved):
    bound = copy.deepcopy(lineage)
    del bound["lineage_sha256"]
    bound["materials"]["resolved_config"] = build_config_manifest(resolved)
    digest = hashlib.sha256(canonical_json_bytes(bound)).hexdigest()
    bound["lineage_sha256"] = digest
    return {"header": copy.deepcopy(CHECKPOINT_HEADER), "lineage": bound, "lineage_sha256": digest}


def config():
    return {
        "semantic_profile": causal_history_profile_contract(),
        "native23_generalist": {
            "source_checkpoint_sha256": LOW_LATENCY_RELEASE_SHA256,
            "encoder_and_fsq_frozen": True,
            "deployment_ready": False,
        },
    }


def test_export_recovers_causal_semantics_from_validated_lineage(lineage):
    result = validate_export_semantics(checkpoint_with_config(lineage, config()))
    assert result["semantic_profile"] == causal_history_profile_contract()
    assert not result["semantic_profile"]["future_samples_relative_to_emission"]
    assert not result["semantic_profile"]["released_profile_relabel_permitted"]


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_semantics",
        "future_profile",
        "future_samples",
        "missing_generalist",
        "different_source",
        "encoder_trainable",
        "deployment_claim",
    ],
)
def test_no_future_or_nonfrozen_source_relabel_even_with_rehashed_lineage(lineage, tamper):
    resolved = config()
    if tamper == "missing_semantics":
        del resolved["semantic_profile"]
    elif tamper == "future_profile":
        resolved["semantic_profile"]["profile"] = "released_low_latency_step1_0p02s"
    elif tamper == "future_samples":
        resolved["semantic_profile"]["future_samples_relative_to_emission"] = True
    elif tamper == "missing_generalist":
        del resolved["native23_generalist"]
    elif tamper == "different_source":
        resolved["native23_generalist"]["source_checkpoint_sha256"] = "0" * 64
    elif tamper == "encoder_trainable":
        resolved["native23_generalist"]["encoder_and_fsq_frozen"] = False
    else:
        resolved["native23_generalist"]["deployment_ready"] = True
    with pytest.raises(ValueError):
        validate_export_semantics(checkpoint_with_config(lineage, resolved))


def test_decoder_export_applies_exact_training_mask_for_all_input_values():
    torch.manual_seed(19)
    decoder = torch.nn.Linear(994, 23)
    codec = G1True23AnalyticCodec()
    exported = DecoderExport(decoder, codec.proprioception_keep_mask)
    value = torch.randn(3, 994)
    expected = decoder(torch.cat((value[:, :64], codec.encode_proprioception(value[:, 64:])), -1))
    assert torch.equal(exported(value), expected)


def test_encoder_export_keeps_released_fsq_rounding():
    torch.manual_seed(20)
    encoder = torch.nn.Linear(267, 64)
    semantic = torch.randn(2, 267)
    assert torch.equal(EncoderExport(encoder)(semantic), FrozenPlatformTrue23Core._fsq(encoder(semantic)))


def test_existing_export_directory_rejected_before_checkpoint_read(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        export_pair(
            checkpoint_path="missing.pt",
            warm_start_path="missing.pt",
            source_checkpoint_path="missing.pt",
            output_directory=output,
        )
