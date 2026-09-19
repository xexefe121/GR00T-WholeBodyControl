"""Strict SIM-only reader; ordinary decoder-LoRA readers reject this header."""

from pathlib import Path

import torch

from gear_sonic.envs.mjlab.sonic_true23_world_tracking_termination import termination_contract
from gear_sonic.trl.mjlab.native23_world_tracking_runner import decoder_schema_view, validate_checkpoint
from gear_sonic.utils import g1_true23_decoder_lora_checkpoint as decoder
from gear_sonic.utils.g1_true23_bounded_progress import reward_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def validate_semantics(checkpoint):
    result = decoder.validate_semantics(decoder_schema_view(checkpoint))
    required = {
        "gear_sonic/envs/mjlab/sonic_true23_world_tracking_termination.py",
        "gear_sonic/trl/mjlab/native23_world_tracking_runner.py",
        "gear_sonic/scripts/train_g1_true23_world_tracking.py",
        "gear_sonic/utils/g1_true23_world_tracking_checkpoint.py",
        "gear_sonic/utils/g1_true23_bounded_progress.py",
        "gear_sonic/envs/mjlab/sonic_true23_bounded_progress.py",
        "gear_sonic/trl/mjlab/native23_bounded_progress_runner.py",
        "gear_sonic/scripts/train_g1_true23_bounded_progress.py",
    }
    rows = checkpoint["lineage"]["materials"]["source_files"]["files"]
    if not required.issubset({row["logical_path"] for row in rows}):
        raise ValueError("bounded progress executed implementation missing from lineage")
    result["reward_transform"] = reward_contract()
    result["additional_training_failure"] = termination_contract()
    return result


def load_cpu_actor(checkpoint_path, *, warm_start_path, source_checkpoint_path):
    requested = Path(checkpoint_path)
    if requested.is_symlink():
        raise ValueError("checkpoint may not be a symlink")
    path = requested.resolve(strict=True)
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    semantics = validate_semantics(checkpoint)
    actor = decoder.construct_actor(
        checkpoint, semantics, warm_start_path=warm_start_path, source_checkpoint_path=source_checkpoint_path
    )
    validate_checkpoint(checkpoint, actor=actor, lineage=checkpoint["lineage"])
    actor.load_training_artifact(checkpoint["actor"])
    if sha256_file(path) != digest:
        raise ValueError("checkpoint changed during read")
    identity = {
        "kind": "native23_world_tracking_pytorch_cpu_research_reader_v1",
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "actor_state_sha256": checkpoint["actor"]["state_sha256"],
        "lineage_sha256": checkpoint["lineage_sha256"],
        "completed_updates": checkpoint["trainer_state"]["completed_update_count"],
        "release_compatibility": semantics["compatibility"],
        "reward_transform": reward_contract(),
        "additional_training_failure": termination_contract(),
        "backend": "pytorch_cpu_deterministic_mean_not_ONNX",
        "export_or_transport_acceptance_claimed": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    return decoder.DecoderLoraCPUActor(actor), identity, semantics
