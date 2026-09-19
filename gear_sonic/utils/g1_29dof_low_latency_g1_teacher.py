"""Pinned original G1 encoder plus unchanged original 29-action decoder, CPU only."""

import gc
from pathlib import Path

import torch

from gear_sonic.utils.g1_23dof_contract import (
    LOW_LATENCY_RELEASE_HF_REVISION,
    LOW_LATENCY_RELEASE_SHA256,
    REFERENCE_PROFILE_LOW_LATENCY,
)
from gear_sonic.utils.g1_29dof_low_latency_teacher import (
    _DECODER_PREFIX,
    DECODER_DIMS,
    LowLatency29DoFTeacher,
    _extract_component,
    _load_pinned_legacy_release,
    _numpy_float32_matrix,
    exact_fsq32,
)

G1_ENCODER_PREFIX = "actor_module.encoders.g1.module."
G1_ENCODER_DIMS = (640, 2048, 1024, 512, 512, 64)


class LowLatencyG1Teacher:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        checkpoint, digest, release = _load_pinned_legacy_release(self.checkpoint_path)
        if digest != LOW_LATENCY_RELEASE_SHA256 or release["source_revision"] != LOW_LATENCY_RELEASE_HF_REVISION:
            raise ValueError("G1 teacher requires exact pinned low-latency release")
        if release["reference_profile"] != REFERENCE_PROFILE_LOW_LATENCY:
            raise ValueError("G1 teacher requires low-latency reference profile")
        state = checkpoint["policy_state_dict"]
        self.encoder = _extract_component(
            state,
            prefix=G1_ENCODER_PREFIX,
            dims=G1_ENCODER_DIMS,
            device=torch.device("cpu"),
            context="released G1 encoder",
        )
        self.decoder = _extract_component(
            state,
            prefix=_DECODER_PREFIX,
            dims=DECODER_DIMS,
            device=torch.device("cpu"),
            context="unchanged released29 decoder",
        )
        self.checkpoint_sha256 = digest
        del state, checkpoint
        gc.collect()

    def infer(self, encoder640, history930):
        encoder, _ = _numpy_float32_matrix(encoder640, 640, "encoder640")
        history, _ = _numpy_float32_matrix(history930, 930, "history930")
        if len(encoder) != 1 or len(history) != 1:
            raise ValueError("G1 diagnostic preserves singleton CPU inference shape")
        with torch.inference_mode():
            latent = LowLatency29DoFTeacher._mlp(self.encoder, torch.from_numpy(encoder))
            token = exact_fsq32(latent)
            raw = LowLatency29DoFTeacher._mlp(self.decoder, torch.cat((token, torch.from_numpy(history)), -1))
        if raw.shape != (1, 29) or not torch.isfinite(raw).all():
            raise ValueError("G1 teacher output differs from finite original29 action ABI")
        return raw.numpy()[0].copy(), token.numpy()[0].copy()

    def descriptor(self):
        return dict(
            kind="pinned_low_latency_g1_encoder_original29_decoder_CPU_v1",
            checkpoint_path=str(self.checkpoint_path),
            checkpoint_sha256=self.checkpoint_sha256,
            encoder_prefix=G1_ENCODER_PREFIX,
            encoder_dims=list(G1_ENCODER_DIMS),
            decoder_prefix=_DECODER_PREFIX,
            decoder_dims=list(DECODER_DIMS),
            training_updates=0,
            decoder_modified=False,
            hardware_authorized=False,
            deployment_ready=False,
        )
