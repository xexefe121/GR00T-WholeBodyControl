"""Strict native23 SIM research reader; no live/export acceptance is added."""

from pathlib import Path

import torch

from gear_sonic.trl.mjlab.native23_world_quality_runner import validate_checkpoint, world_schema_view
from gear_sonic.utils import g1_true23_world_tracking_checkpoint as world
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_world_quality import quality_contract


def validate_semantics(checkpoint):
    result = world.validate_semantics(world_schema_view(checkpoint))
    required = {
        "gear_sonic/utils/g1_true23_world_quality.py",
        "gear_sonic/utils/g1_true23_world_quality_checkpoint.py",
        "gear_sonic/trl/mjlab/native23_world_quality_runner.py",
        "gear_sonic/scripts/train_g1_true23_world_quality.py",
    }
    rows = checkpoint["lineage"]["materials"]["source_files"]["files"]
    if not required.issubset({row["logical_path"] for row in rows}):
        raise ValueError("world-quality executed source closure is incomplete")
    result["world_quality_bonus"] = quality_contract()
    return result


def load_cpu_actor(checkpoint_path, *, warm_start_path, source_checkpoint_path):
    requested = Path(checkpoint_path)
    if requested.is_symlink():
        raise ValueError("world-quality checkpoint may not be a symlink")
    path = requested.resolve(strict=True)
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    semantics = validate_semantics(checkpoint)
    actor = world.decoder.construct_actor(
        checkpoint, semantics, warm_start_path=warm_start_path, source_checkpoint_path=source_checkpoint_path
    )
    validate_checkpoint(checkpoint, actor=actor, lineage=checkpoint["lineage"])
    actor.load_training_artifact(checkpoint["actor"])
    if sha256_file(path) != digest:
        raise ValueError("world-quality checkpoint changed during read")
    identity = dict(
        kind="native23_world_quality_pytorch_cpu_research_reader_v1",
        checkpoint_path=str(path),
        checkpoint_sha256=digest,
        actor_state_sha256=checkpoint["actor"]["state_sha256"],
        lineage_sha256=checkpoint["lineage_sha256"],
        completed_updates=checkpoint["trainer_state"]["completed_update_count"],
        release_compatibility=semantics["compatibility"],
        reward_transform=world.reward_contract(),
        additional_training_failure=world.termination_contract(),
        world_quality_bonus=quality_contract(),
        backend="pytorch_cpu_deterministic_mean_not_ONNX",
        export_or_transport_acceptance_claimed=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    return world.decoder.DecoderLoraCPUActor(actor), identity, semantics
