"""Strict read-only SIM reader for world-quality plus reachable-mean loss."""

from pathlib import Path

import torch

from gear_sonic.trl.mjlab.native23_world_projection_runner import (
    quality_schema_view,
    training_contract,
    validate_checkpoint,
)
from gear_sonic.utils import g1_true23_world_quality_checkpoint as quality
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def validate_semantics(checkpoint):
    result = quality.validate_semantics(quality_schema_view(checkpoint))
    required = {
        "gear_sonic/trl/mjlab/native23_world_projection_runner.py",
        "gear_sonic/trl/mjlab/native23_projected_target_ppo.py",
        "gear_sonic/utils/g1_true23_world_projection_checkpoint.py",
        "gear_sonic/scripts/train_g1_true23_world_projection.py",
    }
    rows = checkpoint["lineage"]["materials"]["source_files"]["files"]
    if not required.issubset({row["logical_path"] for row in rows}):
        raise ValueError("world-projection executed source closure incomplete")
    result["world_projection"] = training_contract()
    return result


def load_cpu_actor(checkpoint_path, *, warm_start_path, source_checkpoint_path):
    requested = Path(checkpoint_path)
    if requested.is_symlink():
        raise ValueError("world-projection checkpoint may not be a symlink")
    path = requested.resolve(strict=True)
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    semantics = validate_semantics(checkpoint)
    actor = quality.world.decoder.construct_actor(
        checkpoint, semantics, warm_start_path=warm_start_path, source_checkpoint_path=source_checkpoint_path
    )
    validate_checkpoint(checkpoint, actor=actor, lineage=checkpoint["lineage"])
    actor.load_training_artifact(checkpoint["actor"])
    if sha256_file(path) != digest:
        raise ValueError("world-projection checkpoint changed during read")
    identity = dict(
        kind="native23_world_projection_pytorch_cpu_research_reader_v1",
        checkpoint_path=str(path),
        checkpoint_sha256=digest,
        actor_state_sha256=checkpoint["actor"]["state_sha256"],
        lineage_sha256=checkpoint["lineage_sha256"],
        completed_updates=checkpoint["trainer_state"]["completed_update_count"],
        release_compatibility=semantics["compatibility"],
        world_projection=training_contract(),
        backend="pytorch_cpu_deterministic_mean_not_ONNX",
        export_or_transport_acceptance_claimed=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    return quality.world.decoder.DecoderLoraCPUActor(actor), identity, semantics
