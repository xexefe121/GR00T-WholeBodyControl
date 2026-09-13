"""Pinned original29 NORMAL teleop encoder/FSQ/decoder, never a native23 export.

The normal release has seven decoder affines, not low-latency's nine. All29
action rows and measured proprioception slots remain present. Shared tensor
validation/FSQ helpers do not select or relabel a low-latency checkpoint.
"""

from collections.abc import Mapping
import gc
from pathlib import Path

import torch

from gear_sonic.scripts.init_g1_23dof_checkpoint import _load_pinned_legacy_release
from gear_sonic.trl.mjlab.frozen_platform_lora_actor import _tensor_state_sha256
from gear_sonic.utils.g1_23dof_contract import (
    NORMAL_RELEASE_SHA256,
    REFERENCE_PROFILE_NORMAL,
    SOURCE_IL29_JOINT_NAMES,
)
from gear_sonic.utils.g1_29dof_low_latency_teacher import (
    ENCODER_DIMS,
    LowLatency29DoFTeacher,
    _extract_component,
    _numpy_float32_matrix,
    _resolve_device,
    exact_fsq32,
)
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

DECODER_DIMS = (994, 2048, 2048, 1024, 1024, 512, 512, 29)


class ExactNormal29Teacher:
    reference_profile = REFERENCE_PROFILE_NORMAL
    action_joint_names = SOURCE_IL29_JOINT_NAMES

    def __init__(self, checkpoint_path, *, device="cpu"):
        self.device = _resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        if file_sha256(self.checkpoint_path) != NORMAL_RELEASE_SHA256:
            raise ValueError("normal29 refuses any checkpoint other than pinned sonic_release/last.pt")
        checkpoint, digest, release = _load_pinned_legacy_release(self.checkpoint_path)
        if digest != NORMAL_RELEASE_SHA256 or release.get("reference_profile") != self.reference_profile:
            raise ValueError("normal29 release identity/profile mismatch")
        state = checkpoint.get("policy_state_dict")
        if not isinstance(state, Mapping) or not state:
            raise ValueError("normal29 requires the original policy_state_dict")
        self._encoder = _extract_component(
            state,
            prefix="actor_module.encoders.teleop.module.",
            dims=ENCODER_DIMS,
            device=self.device,
            context="normal29 teleop encoder",
        )
        self._decoder = _extract_component(
            state,
            prefix="actor_module.decoders.g1_dyn.module.",
            dims=DECODER_DIMS,
            device=self.device,
            context="normal29 g1_dyn decoder",
        )
        self.checkpoint_sha256 = digest
        self.source_revision = release.get("source_revision")
        self._initial_state_sha256 = self.state_sha256()
        del checkpoint, state
        gc.collect()

    def state_sha256(self):
        return _tensor_state_sha256(
            {
                f"{family}.{i}.{name}": tensor
                for family, layers in (("encoder", self._encoder), ("decoder", self._decoder))
                for i, pair in enumerate(layers)
                for name, tensor in zip(("weight", "bias"), pair, strict=True)
            }
        )

    def infer_with_token(self, semantic267, proprio930):
        semantic, squeezed_e = _numpy_float32_matrix(semantic267, 267, "normal semantic267")
        history, squeezed_h = _numpy_float32_matrix(proprio930, 930, "normal proprio930")
        if not squeezed_e or not squeezed_h:
            raise ValueError("normal29 diagnostic requires one unbatched267/930 observation")
        with torch.inference_mode():
            e = torch.from_numpy(semantic).to(self.device)
            h = torch.from_numpy(history).to(self.device)
            token = exact_fsq32(LowLatency29DoFTeacher._mlp(self._encoder, e))
            raw = LowLatency29DoFTeacher._mlp(self._decoder, torch.cat((token, h), -1))
        if raw.dtype != torch.float32 or raw.shape != (1, 29) or not torch.isfinite(raw).all():
            raise RuntimeError("normal29 decoder did not emit finite float32 physical29 means")
        return raw[0].cpu().numpy().copy(), token[0].cpu().numpy().copy()

    def infer(self, semantic267, proprio930):
        return self.infer_with_token(semantic267, proprio930)[0]

    def descriptor(self):
        if self.state_sha256() != self._initial_state_sha256:
            raise ValueError("normal29 frozen inference tensors changed")
        return dict(
            kind="pinned_original29_normal_teleop_inference_v1",
            controller_semantics="neural_state_feedback_29dof_normal_teleop_policy",
            checkpoint_sha256=self.checkpoint_sha256,
            inference_state_sha256=self._initial_state_sha256,
            source_revision=self.source_revision,
            reference_profile=self.reference_profile,
            semantic_input_dim=267,
            proprio_input_dim=930,
            token_dim=64,
            encoder_dims=list(ENCODER_DIMS),
            decoder_dims=list(DECODER_DIMS),
            quantizer="fsq_tanh_round_even_levels_32_v1",
            activation="SiLU",
            action_joint_names=list(self.action_joint_names),
            output_space="canonical_il29_action",
            action_distribution="deterministic_mean",
            dtype="float32",
            device=str(self.device),
            torch_version=str(torch.__version__),
            retained_action_rows=29,
            missing_joint_zero_mask_applied=False,
            training_updates=0,
            root_feedback_head=False,
            hardware_authorized=False,
            deployment_ready=False,
        )
