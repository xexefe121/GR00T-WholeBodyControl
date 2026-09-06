"""Export a diagnostic native23 generalist pair, never deployment promotion."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import gc
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gear_sonic.envs.mjlab.sonic_true23_causal_history import causal_history_profile_contract
from gear_sonic.trl.mjlab.frozen_platform_lora_actor import FrozenPlatformTrue23Core
from gear_sonic.trl.mjlab.native23_generalist_actor import True23Native23GeneralistActorModel
from gear_sonic.trl.mjlab.native23_generalist_runner import CHECKPOINT_HEADER, validate_generalist_checkpoint
from gear_sonic.utils import g1_23dof_artifact as artifact
from gear_sonic.utils.g1_23dof_contract import LOW_LATENCY_RELEASE_SHA256
from gear_sonic.utils.g1_23dof_mjlab_training import validate_mjlab_training_lineage


class EncoderExport(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, semantic: torch.Tensor) -> torch.Tensor:
        return FrozenPlatformTrue23Core._fsq(self.encoder(semantic))


class DecoderExport(nn.Module):
    """Include the exact analytic mask used in training, not an assumed caller."""

    def __init__(self, decoder: nn.Module, proprioception_mask: torch.Tensor) -> None:
        super().__init__()
        self.decoder = decoder
        self.register_buffer("input_keep_mask", torch.cat((torch.ones(64), proprioception_mask.detach().cpu())))

    def forward(self, decoder_input: torch.Tensor) -> torch.Tensor:
        return self.decoder(decoder_input * self.input_keep_mask)


def validate_export_semantics(checkpoint: Any) -> dict[str, Any]:
    """Recover causal semantics from hash-bound config, never from tensor width."""
    if not isinstance(checkpoint, Mapping) or checkpoint.get("header") != CHECKPOINT_HEADER:
        raise ValueError("diagnostic export requires native23 generalist resume header")
    lineage = validate_mjlab_training_lineage(checkpoint.get("lineage"))
    if checkpoint.get("lineage_sha256") != lineage["lineage_sha256"]:
        raise ValueError("diagnostic export checkpoint lineage hash mismatch")
    resolved = lineage["materials"]["resolved_config"]["payload"]
    semantic = resolved.get("semantic_profile")
    if semantic != causal_history_profile_contract():
        raise ValueError("diagnostic export requires exact hash-bound causal-history semantics")
    generalist = resolved.get("native23_generalist")
    if not isinstance(generalist, Mapping) or (
        generalist.get("source_checkpoint_sha256") != LOW_LATENCY_RELEASE_SHA256
        or generalist.get("encoder_and_fsq_frozen") is not True
        or generalist.get("deployment_ready") is not False
    ):
        raise ValueError("diagnostic export requires native23 frozen-tokenizer training lineage")
    return {"semantic_profile": copy.deepcopy(semantic), "training_configuration": copy.deepcopy(generalist)}


def export_pair(
    *,
    checkpoint_path: str | Path,
    warm_start_path: str | Path,
    source_checkpoint_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    import onnx
    from onnx import shape_inference

    checkpoint_file = Path(checkpoint_path).expanduser()
    output = Path(output_directory).expanduser()
    if checkpoint_file.is_symlink():
        raise ValueError("diagnostic source checkpoint may not be a symlink")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"diagnostic output directory already exists: {output}")
    checkpoint_file = checkpoint_file.resolve(strict=True)
    checkpoint_hash = artifact.sha256_file(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    semantics = validate_export_semantics(checkpoint)
    exploration = checkpoint["actor"]["contract"]["exploration"]
    actor = True23Native23GeneralistActorModel(
        {"tokenizer": torch.zeros(1, 267), "policy": torch.zeros(1, 930)},
        {"actor": ["tokenizer", "policy"]},
        "actor",
        23,
        warm_start_path=str(warm_start_path),
        source_checkpoint_path=str(source_checkpoint_path),
        std_min=exploration["std_min"],
        std_max=exploration["std_max"],
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "std_type": "scalar",
            "init_std": exploration["init_std"],
        },
    )
    validate_generalist_checkpoint(checkpoint, actor=actor, lineage=checkpoint["lineage"])
    actor.load_training_artifact(checkpoint["actor"])
    actor.eval()
    source = {
        "checkpoint_filename": checkpoint_file.name,
        "checkpoint_sha256": checkpoint_hash,
        "actor_state_sha256": checkpoint["actor"]["state_sha256"],
        "lineage_sha256": checkpoint["lineage_sha256"],
        "completed_update_count": checkpoint["trainer_state"]["completed_update_count"],
        "architecture_initialization_profile": actor.reference_profile,
        "paired_release_checkpoint_sha256": actor.core.source_checkpoint_sha256,
        "frozen_encoder_state_sha256": actor.core.frozen_encoder_sha256(),
    }
    del checkpoint
    gc.collect()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "g1_native23_generalist_diagnostic_pair",
        "source": source,
        **semantics,
        "actor_contract": actor.artifact_contract(),
        "proprioception_absent_slots_fixed_zero_in_decoder_graph": True,
        "diagnostic_only": True,
        "deployment_ready": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "active_motor_control_authorized": False,
        "completed_motion_qualification": False,
    }
    components = (
        (
            "encoder",
            EncoderExport(actor.core.encoder).eval(),
            267,
            64,
            artifact.ENCODER_ONNX_INPUT_NAME,
            artifact.ENCODER_ONNX_OUTPUT_NAME,
            artifact.validate_encoder_onnx_structure,
        ),
        (
            "decoder",
            DecoderExport(actor.core.decoder, actor.core.codec.proprioception_keep_mask).eval(),
            994,
            23,
            artifact.ONNX_INPUT_NAME,
            artifact.ONNX_OUTPUT_NAME,
            artifact.validate_onnx_structure,
        ),
    )
    for name, model, input_dim, output_dim, input_name, output_name, validator in components:
        path = output / f"{name}.diagnostic.onnx"
        with torch.no_grad():
            torch.onnx.export(
                model,
                torch.zeros(1, input_dim),
                path,
                input_names=[input_name],
                output_names=[output_name],
                opset_version=artifact.ONNX_OPSET_VERSION,
                dynamo=False,
                export_params=True,
                do_constant_folding=True,
                dynamic_axes=None,
            )
        graph = shape_inference.infer_shapes(
            onnx.load(path, load_external_data=False), strict_mode=True, data_prop=True
        )
        onnx.helper.set_model_props(
            graph,
            {
                "artifact_role": f"native23_generalist_diagnostic_{name}",
                "deployment_ready": "false",
                "hardware_authorized": "false",
                "source_checkpoint_sha256": checkpoint_hash,
                "actor_state_sha256": source["actor_state_sha256"],
                "semantic_profile": semantics["semantic_profile"]["profile"],
                "semantic_contract_sha256": semantics["semantic_profile"]["contract_sha256"],
                "architecture_initialization_profile": actor.reference_profile,
            },
        )
        onnx.save_model(graph, path, save_as_external_data=False)
        validator(onnx.load(path, load_external_data=False))
        parity = artifact.validate_ort_parity(
            model, path, input_name=input_name, input_dim=input_dim, output_name=output_name, output_dim=output_dim
        )
        if name == "encoder" and parity["parity_max_abs_error"] != 0:
            raise ValueError("diagnostic SONIC encoder requires exact discrete token parity")
        report[name] = {
            "filename": path.name,
            "sha256": artifact.sha256_file(path),
            "input_name": input_name,
            "output_name": output_name,
            "input_shape": [1, input_dim],
            "output_shape": [1, output_dim],
            "parity": parity,
        }
    if artifact.sha256_file(checkpoint_file) != checkpoint_hash:
        raise ValueError("diagnostic source checkpoint changed during export")
    # Written last: partial ONNX files from an interrupted export have no pair
    # completion report and cannot claim successful pair/parity validation.
    with (output / "generalist.diagnostic.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--warm-start", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args(argv)
    report = export_pair(
        checkpoint_path=args.checkpoint,
        warm_start_path=args.warm_start,
        source_checkpoint_path=args.source_checkpoint,
        output_directory=args.output_directory,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
