"""Versioned root-conditioned native23 actor; released SONIC inputs unchanged.

Root feedback is a separate nine-value input. Its zero-initialized projection
adds to the first decoder preactivation, so installing the branch initially
preserves every base-policy mean, even for nonzero root feedback.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import _tensor_state_sha256
from gear_sonic.trl.mjlab.native23_generalist_actor import True23Native23GeneralistActorModel
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract

ROOT_FEEDBACK_ACTOR_KIND = "g1_native23_full_decoder_root_feedback_frozen_sonic_tokenizer_v1"
ROOT_FEEDBACK_ARCHITECTURE = "zero_initialized_additive_first_preactivation_v1"


def conditioned_decoder_forward(decoder, conditioner, decoder_input, root_feedback):
    """Keep the original first affine and every later decoder layer intact."""
    hidden = decoder.module[0](decoder_input) + conditioner(root_feedback)
    for layer in decoder.module[1:]:
        hidden = layer(hidden)
    return hidden


class True23RootFeedbackActorModel(True23Native23GeneralistActorModel):
    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        *,
        root_feedback_obs_group="root_feedback",
        release_compatibility=None,
        **kwargs,
    ):
        from gear_sonic.utils.g1_true23_release_compatibility import validate_release_compatibility

        self.release_compatibility = (
            None if release_compatibility is None else validate_release_compatibility(release_compatibility)
        )
        self.reference_timing = (self.release_compatibility or {}).get("reference_timing", "causal_history")
        tokenizer = kwargs.get("tokenizer_obs_group", "tokenizer")
        proprioception = kwargs.get("proprioception_obs_group", "policy")
        if tuple(obs_groups.get(obs_set, ())) != (tokenizer, proprioception, root_feedback_obs_group):
            raise ValueError("root-feedback actor requires tokenizer, policy and separate root_feedback groups")
        feedback = obs[root_feedback_obs_group]
        if feedback.shape[-1] != 9 or feedback.shape[:-1] != obs[proprioception].shape[:-1]:
            raise ValueError("root-feedback observation must have matching batch shape and nine values")
        base_groups = {**obs_groups, obs_set: [tokenizer, proprioception]}
        super().__init__(obs, base_groups, obs_set, output_dim, **kwargs)
        self.root_feedback_obs_group = root_feedback_obs_group
        self.root_conditioner = nn.Linear(9, self.core.decoder_dims[1], bias=False)
        nn.init.zeros_(self.root_conditioner.weight)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del hidden_state
        if masks is not None:
            from rsl_rl.utils import unpad_trajectories

            obs = unpad_trajectories(obs, masks)
        semantic = obs[self.tokenizer_obs_group]
        if self.tokenizer_has_encoder_index:
            if not torch.isfinite(semantic[..., :1]).all():
                raise ValueError("root-feedback encoder route contains non-finite values")
            semantic = semantic[..., 1:]
        feedback = obs[self.root_feedback_obs_group]
        proprioception = obs[self.proprioception_obs_group]
        if feedback.shape != (*proprioception.shape[:-1], 9) or not torch.isfinite(feedback).all():
            raise ValueError("root-feedback input must be finite, matching-batch [...,9]")
        if feedback.dtype != proprioception.dtype or feedback.device != proprioception.device:
            raise ValueError("root-feedback dtype/device must match policy observations")
        decoded = torch.cat(
            (self.core.encode(semantic), self.core.codec.encode_proprioception(proprioception)), -1
        )
        mean = conditioned_decoder_forward(self.core.decoder, self.root_conditioner, decoded, feedback)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def artifact_contract(self):
        result = {
            **super().artifact_contract(),
            "kind": ROOT_FEEDBACK_ACTOR_KIND,
            "schema_version": 2,
            "architecture": ROOT_FEEDBACK_ARCHITECTURE,
            "root_feedback_contract": root_feedback_contract(self.reference_timing),
            "decoder_inputs": {"obs_dict": 994, "root_feedback": 9},
            "root_conditioner_shape": [self.core.decoder_dims[1], 9],
            "root_conditioner_bias": False,
            "trainable_root_conditioner_parameters": self.root_conditioner.weight.numel(),
            "zero_step_conditioner_preserves_base_mean": True,
            "legacy_single_input_export_permitted": False,
        }
        if self.release_compatibility is not None:
            result.update(
                kind=(
                    "g1_native23_root_feedback_release_compatible_actor_v2"
                    if self.release_compatibility["kind"] == "native23_causal_release_compatibility_v2"
                    else "g1_native23_root_feedback_release_compatible_actor_v1"
                ),
                release_compatibility=self.release_compatibility,
            )
            if self.reference_timing == "received_source_horizon_200ms_v1":
                result["kind"] = "g1_native23_root_feedback_buffered_source_actor_v3"
        return result

    def parameter_groups(self):
        self.core.assert_frozen_encoder_unchanged()
        if not self.root_conditioner.weight.requires_grad:
            raise RuntimeError("root conditioner became frozen")
        return {
            "decoder": list(self.core.decoder.parameters()),
            "root_conditioner": list(self.root_conditioner.parameters()),
            "exploration": [self.distribution.raw_std],
        }

    def export_true23_policy_state(self):
        raise RuntimeError("root-feedback actor forbids legacy single-input policy export; use versioned pair")

    def validate_base_training_artifact(self, artifact):
        """Validate an old actor without mutating this new actor."""
        if torch.count_nonzero(self.root_conditioner.weight).item():
            raise ValueError("base transfer requires untouched zero root conditioner")
        if not isinstance(artifact, Mapping) or set(artifact) != {"contract", "state_dict", "state_sha256"}:
            raise ValueError("base actor artifact fields mismatch")
        if artifact["contract"] != True23Native23GeneralistActorModel.artifact_contract(self):
            raise ValueError("base actor artifact contract mismatch")
        state = artifact["state_dict"]
        expected = self.state_dict()
        expected_base = set(expected) - {"root_conditioner.weight"}
        if not isinstance(state, Mapping) or set(state) != expected_base:
            raise ValueError("base actor state keys mismatch")
        for name, value in state.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"base actor tensor invalid: {name}")
        if _tensor_state_sha256(state) != artifact["state_sha256"]:
            raise ValueError("base actor state hash mismatch")
        # Reuse the strict root actor loader for all checks before any copy,
        # including hash-pinned encoder validation and a complete state shape.
        combined = {**state, "root_conditioner.weight": expected["root_conditioner.weight"].detach().cpu().clone()}
        upgraded = {
            "contract": self.artifact_contract(),
            "state_dict": combined,
            "state_sha256": _tensor_state_sha256(combined),
        }
        self.validate_training_artifact(upgraded)
        return upgraded

    def load_base_training_artifact(self, artifact):
        """Copy old actor/noise only; trainer binds parent and fresh lineage.

        Does not import optimizer state, advance counters, or copy a trained
        root conditioner. Exact same-version resume uses load_training_artifact.
        """
        self.load_training_artifact(self.validate_base_training_artifact(artifact))


__all__ = ["True23RootFeedbackActorModel", "conditioned_decoder_forward"]
