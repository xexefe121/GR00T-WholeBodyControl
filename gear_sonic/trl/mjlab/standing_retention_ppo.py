# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Pinned single-device feed-forward PPO with a standing-only auxiliary loss.

The used PPO branches follow RSL-RL 5.0.1. Reject unsupported modes instead of
silently dropping their semantics. A zero-weight regression must reproduce
the installed upstream algorithm, including its optimizer and random state.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from types import MethodType

from rsl_rl.algorithms.ppo import PPO
import torch

from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256

UPSTREAM_PPO_SHA256 = "a2d35e7ad7b884c80b7434e7d2ce785a6da1e18d93c96f1179fbd3a208669f8c"


def supported(algorithm):
    if type(algorithm) is not PPO:
        raise ValueError("standing retention requires the pinned ordinary PPO instance")
    if (
        algorithm.actor.is_recurrent
        or algorithm.critic.is_recurrent
        or algorithm.rnd is not None
        or algorithm.symmetry is not None
        or algorithm.is_multi_gpu
        or algorithm.schedule != "fixed"
        or algorithm.desired_kl is not None
    ):
        raise ValueError("standing retention supports only fixed-rate feed-forward single-device PPO")


def optimizer_step(optimizer):
    if not isinstance(optimizer, torch.optim.Adam):
        raise ValueError("standing retention requires Adam step accounting")
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if not optimizer.state:
        return 0
    if set(optimizer.state) != set(parameters):
        raise ValueError("standing retention requires coherent per-parameter optimizer state")
    counts = [float(value["step"].item()) for value in optimizer.state.values()]
    if not counts or any(not math.isfinite(value) or value < 0 or value != int(value) for value in counts):
        raise ValueError("standing retention optimizer has invalid step counters")
    if len(set(counts)) != 1:
        raise ValueError("standing retention optimizer step counters diverged")
    return int(counts[0])


def install(algorithm, anchor, *, weight):
    supported(algorithm)
    upstream = Path(inspect.getfile(PPO)).resolve()
    if file_sha256(upstream) != UPSTREAM_PPO_SHA256:
        raise ValueError("upstream PPO changed; revalidate the zero-weight equivalence before training")
    if isinstance(weight, bool) or not math.isfinite(weight) or not 0 <= weight <= 1000:
        raise ValueError("standing retention weight must be finite within [0, 1000]")
    if weight > 0 and anchor is None:
        raise ValueError("positive standing retention requires a checked output anchor")
    if "update" in vars(algorithm):
        raise ValueError("PPO already has a process-local update override")
    algorithm.standing_anchor = anchor
    algorithm.standing_retention_weight = weight
    algorithm.standing_minibatch_step = optimizer_step(algorithm.optimizer)
    algorithm.update = MethodType(update, algorithm)
    return dict(
        upstream_path=str(upstream),
        upstream_sha256=UPSTREAM_PPO_SHA256,
        joint_loss_before_gradient_clipping=True,
        optimizer_steps_per_minibatch=1,
        separate_retention_optimizer=False,
        exploration_distribution_changed=False,
        hardware_authorized=False,
        deployment_ready=False,
    )


def update(self):
    supported(self)
    if optimizer_step(self.optimizer) != self.standing_minibatch_step:
        raise ValueError("standing retention sampling step differs from actual Adam state")
    means = dict(value=0.0, surrogate=0.0, entropy=0.0, standing_retention=0.0)
    start_step = self.standing_minibatch_step
    generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    for batch in generator:
        original_batch_size = batch.observations.batch_size[0]
        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
        self.actor(
            batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True
        )
        actions_log_prob = self.actor.get_output_log_prob(batch.actions)
        values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
        entropy = self.actor.output_entropy[:original_batch_size]
        ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
        surrogate = -torch.squeeze(batch.advantages) * ratio
        surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        if self.use_clipped_value_loss:
            value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - batch.returns).pow(2)
            value_losses_clipped = (value_clipped - batch.returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (batch.returns - values).pow(2).mean()
        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
        retention_loss = None
        if self.standing_retention_weight > 0:
            retention_loss = self.standing_anchor.loss(self.standing_minibatch_step)
            loss = loss + self.standing_retention_weight * retention_loss
        if not torch.isfinite(loss):
            raise ValueError("joint PPO/standing retention loss became nonfinite")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm, error_if_nonfinite=True)
        self.optimizer.step()
        self.standing_minibatch_step += 1
        means["value"] += value_loss.item()
        means["surrogate"] += surrogate_loss.item()
        means["entropy"] += entropy.mean().item()
        if retention_loss is not None:
            means["standing_retention"] += retention_loss.item()
    expected = self.num_learning_epochs * self.num_mini_batches
    if self.standing_minibatch_step - start_step != expected:
        raise ValueError("standing retention minibatch generator count differs from PPO contract")
    if optimizer_step(self.optimizer) != self.standing_minibatch_step:
        raise ValueError("standing retention optimizer did not advance exactly once per minibatch")
    self.storage.clear()
    return {key: value / expected for key, value in means.items()}
