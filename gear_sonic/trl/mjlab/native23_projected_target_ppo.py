# The PPO update is adapted from RSL-RL 5.0.1 (rsl_rl/algorithms/ppo.py).
# Copyright (c) 2026, ETH Zurich
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Opt-in PPO loss for source means outside the unchanged native target range.

The deterministic actor, Gaussian distribution, likelihood, action codec and
physics are unchanged. An explicit auxiliary loss pulls impossible mean targets
toward their already executed projection, making ordinary exploration useful
near a bound. This is a training objective, not a qualification metric.

The update intentionally supports only the fixed-rate, feedforward, single-GPU
PPO configuration used by the bounded native23 campaign. Its zero-weight path is
tested against the installed upstream PPO update with identical batches/RNG.
"""

from __future__ import annotations

import math

import numpy as np
from rsl_rl.algorithms.ppo import PPO
import torch
from torch import nn

from gear_sonic.utils.g1_23dof_contract import MUJOCO_TO_ISAACLAB_DOF, NATIVE_IL23_ACTION_SCALE
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE,
    SAFE_TARGET_RAW_ACTION_CLIP,
)
from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_SCALE_NATIVE_IL23

PROFILE = "source_target_projection_l2_v1"
WEIGHT = 0.05


def projection_objective_contract(profile):
    if profile == "none":
        return {"name": "none", "weight": 0.0}
    if profile != PROFILE:
        raise ValueError("unknown source projection PPO objective")
    return {
        "name": PROFILE,
        "weight": WEIGHT,
        "objective": "mean_batch(sum_joints(((mean_source_raw-clamp_to_reachable_raw)*source_scale)^2))",
        "units": "radian_squared",
        "scope": "current_PPO_minibatch_policy_means_not_sampled_actions",
        "likelihood_entropy_distribution_and_action_codec_changed": False,
        "source_encoder_or_exported_actor_architecture_changed": False,
        "native_physics_gains_limits_or_acceptance_thresholds_changed": False,
        "exploration_effectiveness_improvement_proven": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def projection_objective_transition(previous, args):
    old = previous.get("ppo_auxiliary_objective", projection_objective_contract("none"))
    if old != projection_objective_contract(old["name"]):
        raise ValueError("parent source-projection objective contract differs")
    new = projection_objective_contract(getattr(args, "ppo_auxiliary_objective", "none"))
    changed = old != new
    if changed and not getattr(args, "allow_ppo_objective_transition", False):
        raise ValueError("PPO objective transition requires explicit --allow-ppo-objective-transition")
    return {
        "old": old,
        "new": new,
        "ppo_loss_changed": changed,
        "same_ppo_objective_resume_claimed": not changed,
        "actor_critic_optimizer_and_learning_rates_preserved": True,
        "new_objective_tracking_quality_qualified": False,
    }


def reachable_source_bounds():
    """Exact inverse of the already selected v2 native physical target envelope."""
    order = np.asarray(MUJOCO_TO_ISAACLAB_DOF)
    native, source = np.asarray(NATIVE_IL23_ACTION_SCALE), np.asarray(SOURCE_SCALE_NATIVE_IL23)
    limits = []
    for capacity in (SAFE_TARGET_NEGATIVE_CAPACITY_HARDWARE, SAFE_TARGET_POSITIVE_CAPACITY_HARDWARE):
        cap = np.asarray(capacity)[order]
        fraction = np.minimum(np.tanh((SAFE_TARGET_RAW_ACTION_CLIP - 1e-3) * native / cap), 1 - 1e-6)
        limits.append(cap * fraction / source)
    return -limits[0], limits[1]


def source_projection_loss(mean):
    if mean.ndim != 2 or mean.shape[-1] != 23 or mean.dtype != torch.float32 or not torch.isfinite(mean).all():
        raise ValueError("projection loss requires finite float32 [batch,23] source-action means")
    lower, upper = (torch.as_tensor(x, device=mean.device, dtype=mean.dtype) for x in reachable_source_bounds())
    scale = torch.as_tensor(SOURCE_SCALE_NATIVE_IL23, device=mean.device, dtype=mean.dtype)
    projected = torch.clamp(mean, lower, upper)
    excess = (mean - projected) * scale
    return excess.square().sum(-1).mean(), (mean < lower) | (mean > upper)


class ProjectedTargetPPO(PPO):
    def __init__(self, *args, mean_projection_weight=WEIGHT, **kwargs):
        if not math.isfinite(mean_projection_weight) or not 0 <= mean_projection_weight <= WEIGHT:
            raise ValueError("source projection weight must be between zero and the versioned 0.05")
        self.mean_projection_weight = float(mean_projection_weight)
        super().__init__(*args, **kwargs)
        self._validate_supported_configuration()
        self.projection_update_count = 0
        self.projection_minibatch_count = 0

    def _validate_supported_configuration(self):
        if (
            self.schedule != "fixed"
            or self.rnd is not None
            or self.symmetry is not None
            or self.is_multi_gpu
            or self.actor.is_recurrent
            or self.critic.is_recurrent
        ):
            raise ValueError(
                "projected-target PPO requires fixed-rate nonrecurrent single-GPU PPO without RND/symmetry"
            )

    def update(self):
        self._validate_supported_configuration()
        totals = dict(
            value=0.0, surrogate=0.0, entropy=0.0, source_projection=0.0, source_mean_projected_fraction=0.0
        )
        count = 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )
            self.actor(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            entropy = self.actor.output_entropy
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * ratio.clamp(
                1 - self.clip_param, 1 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max(
                    (values - batch.returns).square(), (value_clipped - batch.returns).square()
                ).mean()
            else:
                value_loss = (batch.returns - values).square().mean()
            projection_loss, projected = source_projection_loss(self.actor.output_distribution_params[0])
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.mean_projection_weight * projection_loss
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite projected-target PPO loss")
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            for key, value in (
                ("value", value_loss),
                ("surrogate", surrogate_loss),
                ("entropy", entropy.mean()),
                ("source_projection", projection_loss),
                ("source_mean_projected_fraction", projected.float().mean()),
            ):
                totals[key] += value.item()
            count += 1
        if count != self.num_learning_epochs * self.num_mini_batches or count == 0:
            raise ValueError("projected-target PPO minibatch count differs from configured update")
        self.storage.clear()
        self.projection_update_count += 1
        self.projection_minibatch_count += count
        return {key: value / count for key, value in totals.items()}

    def projection_runtime_receipt(self):
        return {
            "kind": "native23_executed_source_target_projection_loss_v1",
            "scope": "current_training_invocation_not_prior_checkpoint_updates",
            "completed_ppo_updates": self.projection_update_count,
            "completed_minibatches": self.projection_minibatch_count,
            "coefficient": self.mean_projection_weight,
            "objective": projection_objective_contract(PROFILE),
            "hardware_authorized": False,
            "deployment_ready": False,
        }
