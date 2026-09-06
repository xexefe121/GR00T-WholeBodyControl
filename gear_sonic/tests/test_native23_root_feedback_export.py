"""Versioned two-input export; no old artifact relabel or missing root graph."""

from copy import deepcopy

import pytest
import torch

from gear_sonic.scripts.export_g1_true23_root_feedback import (
    RootFeedbackDecoderExport,
    export_pair,
    validate_export_semantics,
    validate_root_decoder_onnx,
    validate_root_decoder_parity,
)
from gear_sonic.tests.test_native23_generalist_export import checkpoint_with_config, config
from gear_sonic.tests.test_native23_generalist_runner import lineage as lineage
from gear_sonic.tests.test_native23_root_feedback_actor import actor as actor, observations
from gear_sonic.trl.mjlab.native23_root_feedback_actor import ROOT_FEEDBACK_ARCHITECTURE
from gear_sonic.trl.mjlab.native23_root_feedback_runner import CHECKPOINT_HEADER
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract


def checkpoint(lineage, resolved=None):
    resolved = config() if resolved is None else resolved
    resolved.setdefault(
        "native23_root_feedback",
        {
            "feature_contract": root_feedback_contract(),
            "architecture": ROOT_FEEDBACK_ARCHITECTURE,
            "deployment_ready": False,
        },
    )
    value = checkpoint_with_config(lineage, resolved)
    value["header"] = deepcopy(CHECKPOINT_HEADER)
    return value


def test_export_requires_distinct_root_and_original_sonic_semantics(lineage):
    result = validate_export_semantics(checkpoint(lineage))
    assert result["root_feedback_contract"] == root_feedback_contract()
    assert result["semantic_profile"]["future_samples_relative_to_emission"] is False


@pytest.mark.parametrize(
    "tamper", ["old_header", "root_width", "root_frame", "velocity", "architecture", "promotion", "future"]
)
def test_reject_even_rehashed_wrong_semantic_contract(lineage, tamper):
    base = config()
    root = {
        "feature_contract": root_feedback_contract(),
        "architecture": ROOT_FEEDBACK_ARCHITECTURE,
        "deployment_ready": False,
    }
    base["native23_root_feedback"] = root
    if tamper == "root_width":
        root["feature_contract"]["dimension"] = 8
    elif tamper == "root_frame":
        root["feature_contract"]["coordinate_frame"] = "world"
    elif tamper == "velocity":
        root["feature_contract"]["desired_velocity_definition"] = "future_central_difference"
    elif tamper == "architecture":
        root["architecture"] = "concatenated1003"
    elif tamper == "promotion":
        root["deployment_ready"] = True
    elif tamper == "future":
        base["semantic_profile"]["future_samples_relative_to_emission"] = True
    value = checkpoint(lineage, base)
    if tamper == "old_header":
        from gear_sonic.trl.mjlab.native23_generalist_runner import CHECKPOINT_HEADER as OLD_HEADER

        value["header"] = deepcopy(OLD_HEADER)
    with pytest.raises(ValueError):
        validate_export_semantics(value)


def test_export_wrapper_exactly_matches_training_path_with_nonzero_conditioner(actor):
    saved = actor.root_conditioner.weight.detach().clone()
    try:
        with torch.no_grad():
            actor.root_conditioner.weight.fill_(0.001)
            obs = observations()
            obs["policy"].normal_(std=0.1)
            obs["root_feedback"].normal_()
            wrapper = RootFeedbackDecoderExport(
                actor.core.decoder, actor.root_conditioner, actor.core.codec.proprioception_keep_mask
            )
            decoder_input = torch.cat((actor.core.encode(obs["tokenizer"]), obs["policy"]), -1)
            assert torch.equal(actor(obs), wrapper(decoder_input, obs["root_feedback"]))
    finally:
        with torch.no_grad():
            actor.root_conditioner.weight.copy_(saved)


def test_real_two_input_onnx_parity_and_no_single_input_drop(actor, tmp_path):
    import onnx
    from gear_sonic.utils.g1_23dof_artifact import ONNX_OPSET_VERSION

    path = tmp_path / "root_decoder.onnx"
    saved = actor.root_conditioner.weight.detach().clone()
    try:
        with torch.no_grad():
            actor.root_conditioner.weight.fill_(0.001)
        wrapper = RootFeedbackDecoderExport(
            actor.core.decoder, actor.root_conditioner, actor.core.codec.proprioception_keep_mask
        ).eval()
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (torch.zeros(1, 994), torch.zeros(1, 9)),
                path,
                input_names=["obs_dict", "root_feedback"],
                output_names=["action"],
                opset_version=ONNX_OPSET_VERSION,
                dynamo=False,
            )
        graph = onnx.load(path, load_external_data=False)
        validate_root_decoder_onnx(graph)
        result = validate_root_decoder_parity(wrapper, path)
        assert result["parity_passed"] and result["includes_nonzero_root_feedback"]
        del graph.graph.input[1]
        with pytest.raises(Exception):
            validate_root_decoder_onnx(graph)
    finally:
        with torch.no_grad():
            actor.root_conditioner.weight.copy_(saved)


def test_existing_export_directory_rejected_before_checkpoint_read(tmp_path):
    with pytest.raises(FileExistsError, match="already exists"):
        export_pair(
            checkpoint_path="missing",
            warm_start_path="missing",
            source_checkpoint_path="missing",
            output_directory=tmp_path,
        )
