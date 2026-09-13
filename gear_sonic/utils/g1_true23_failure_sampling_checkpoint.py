"""Strict SIM-only failure-sampling snapshot reader; no live/export acceptance."""

from pathlib import Path

import torch

from gear_sonic.trl.mjlab.native23_failure_sampling_runner import (
    quality_schema_view,
    validate_checkpoint,
    validate_sampler_state,
)
from gear_sonic.utils import g1_true23_world_quality_checkpoint as quality
from gear_sonic.utils.g1_true23_failure_sampling import sampling_contract
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def validate_semantics(checkpoint):
    result = quality.validate_semantics(quality_schema_view(checkpoint))
    validate_sampler_state(checkpoint)
    required = {
        "gear_sonic/utils/g1_true23_failure_sampling.py",
        "gear_sonic/envs/mjlab/sonic_true23_failure_sampling.py",
        "gear_sonic/trl/mjlab/native23_failure_sampling_runner.py",
        "gear_sonic/scripts/train_g1_true23_failure_sampling.py",
        "gear_sonic/utils/g1_true23_failure_sampling_checkpoint.py",
    }
    rows = checkpoint["lineage"]["materials"]["source_files"]["files"]
    if not required.issubset({row["logical_path"] for row in rows}):
        raise ValueError("failure sampling source closure is incomplete")
    result["failure_sampling"] = sampling_contract()
    return result


def load_cpu_actor(checkpoint_path, *, warm_start_path, source_checkpoint_path):
    requested = Path(checkpoint_path)
    if requested.is_symlink():
        raise ValueError("failure sampling snapshot cannot be a symlink")
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
        raise ValueError("failure sampling snapshot changed while reading")
    return (
        quality.world.decoder.DecoderLoraCPUActor(actor),
        dict(
            kind="native23_failure_sampling_cpu_research_reader_v1",
            checkpoint_path=str(path),
            checkpoint_sha256=digest,
            actor_state_sha256=checkpoint["actor"]["state_sha256"],
            lineage_sha256=checkpoint["lineage_sha256"],
            completed_updates=checkpoint["trainer_state"]["completed_update_count"],
            failure_sampling=sampling_contract(),
            backend="pytorch_cpu_deterministic_mean_not_ONNX",
            export_or_transport_acceptance_claimed=False,
            deployment_ready=False,
            hardware_authorized=False,
        ),
        semantics,
    )
