"""Original-source root adaptation without changing the released mean decoder.

This is a distinct research actor, not a relabelled full-decoder checkpoint.
Gradients traverse the frozen decoder to reach the root projection. They do
not accumulate on the decoder or encoder, and neither belongs to the optimizer.
"""

import copy

import torch
from torch import nn

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import _tensor_state_sha256
from gear_sonic.trl.mjlab.native23_generalist_actor import Native23GeneralistCore
from gear_sonic.trl.mjlab.native23_original_intent_actor import True23OriginalIntentActorModel
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state

ACTOR_KIND = "g1_native23_frozen_decoder_original_source_root_adapter_v1"
CORE_KIND = "g1_native23_frozen_released_encoder_and_decoder_v1"
TRAINABLE = "root_conditioner_bounded_exploration_and_critic_only"
DECODER_PREFIX = "core.actor_module.decoders.g1_dyn."


class FrozenDecoderCore(Native23GeneralistCore):
    """Adopt an already validated fresh core, preserving all state-dict names."""

    def __init__(self, fresh_core):
        nn.Module.__init__(self)
        if type(fresh_core) is not Native23GeneralistCore:
            raise TypeError("frozen decoder requires a fresh validated native23 core")
        fresh_core.assert_frozen_encoder_unchanged()
        initial = inspect_true23_policy_state(
            {"policy_state_dict": fresh_core.export_policy_state(fresh_core.initial_std)},
            reference_profile=fresh_core.reference_profile,
        )
        if initial != fresh_core._initial_policy_sha256:
            raise ValueError("frozen decoder cannot adopt a trained or modified core")
        # No second checkpoint load or temporary second copy of the large MLP.
        # The enclosing actor immediately replaces its old core with this one.
        self.actor_module = fresh_core.actor_module
        self.codec = fresh_core.codec
        for name in (
            "warm_start_path",
            "source_checkpoint_path",
            "warm_start_metadata",
            "warm_start_initialization_report",
            "warm_start_stage",
            "reference_profile",
            "source_checkpoint_sha256",
            "source_revision",
            "decoder_dims",
            "initial_std",
            "_initial_policy_sha256",
            "_frozen_encoder_sha256",
        ):
            setattr(self, name, copy.deepcopy(getattr(fresh_core, name)))
        self.decoder.requires_grad_(False)
        self._frozen_decoder_sha256 = _tensor_state_sha256(self.decoder.state_dict())
        self.assert_frozen_encoder_unchanged()

    def assert_frozen_encoder_unchanged(self):
        # Override the full-decoder guard, which correctly requires a trainable
        # decoder for that OTHER actor. Never alter its historical contract.
        if any(p.requires_grad for p in self.encoder.parameters()):
            raise RuntimeError("frozen-base encoder became trainable")
        if any(p.requires_grad for p in self.decoder.parameters()):
            raise RuntimeError("frozen-base decoder became trainable")
        if self.frozen_encoder_sha256() != self._frozen_encoder_sha256:
            raise RuntimeError("frozen-base encoder tensor changed")
        if _tensor_state_sha256(self.decoder.state_dict()) != self._frozen_decoder_sha256:
            raise RuntimeError("frozen-base decoder tensor changed")

    def artifact_contract(self):
        return {
            **super().artifact_contract(),
            "kind": CORE_KIND,
            "trainable_decoder_parameters": 0,
            "frozen_decoder_parameters": sum(p.numel() for p in self.decoder.parameters()),
            "frozen_decoder_sha256": self._frozen_decoder_sha256,
            "decoder_training": "frozen_released_native23_row_mapping",
            "decoder_frozen": True,
        }


class True23FrozenDecoderActorModel(True23OriginalIntentActorModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core = FrozenDecoderCore(self.core)

    def artifact_contract(self):
        return {
            **super().artifact_contract(),
            "kind": ACTOR_KIND,
            "base_core_kind": CORE_KIND,
            "actor_trainable": "root_conditioner_and_bounded_exploration_only",
            "full_decoder_checkpoint_relabelling_allowed": False,
        }

    def parameter_groups(self):
        self.core.assert_frozen_encoder_unchanged()
        groups = {
            "root_conditioner": list(self.root_conditioner.parameters()),
            "exploration": [self.distribution.raw_std],
        }
        if not all(p.requires_grad for values in groups.values() for p in values):
            raise RuntimeError("frozen-decoder adapter or exploration became frozen")
        return groups

    def validate_training_artifact(self, artifact):
        super().validate_training_artifact(artifact)
        decoder = {
            name.removeprefix(DECODER_PREFIX): value
            for name, value in artifact["state_dict"].items()
            if name.startswith(DECODER_PREFIX)
        }
        if _tensor_state_sha256(decoder) != self.core._frozen_decoder_sha256:
            raise ValueError("frozen-decoder artifact changes pinned decoder")

    def validate_base_training_artifact(self, artifact):
        del artifact
        raise ValueError("frozen-decoder actor requires fresh released initialization")

    def initial_conditioner_is_zero(self):
        return not bool(torch.count_nonzero(self.root_conditioner.weight).item())
