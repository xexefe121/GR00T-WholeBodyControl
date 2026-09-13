"""On-policy LoRA learning from complete native MuJoCo lifecycle attempts.

Only active SONIC actions enter PPO. Fixed acquisition/return controllers do
not acquire fictitious log probabilities or policy gradients. Return outcome
is an explicit terminal assessment of the preceding active trajectory.
"""

import math

import numpy as np
import torch
from torch import nn


def terminal_episode_rewards(report, arrays):
    """Keep failed/partial controls and never reward an abbreviated full dance."""
    returned = arrays["policy_inference_returned"]
    count, completed = len(returned), report["completed_transitions"]
    if count not in (completed, completed + 1) or count > report["requested_transitions"]:
        raise ValueError("lifecycle action accounting differs from full request")
    if not returned.all():
        raise ValueError("policy inference failure is not a usable training transition")
    failure = report["failure"]
    if failure and failure.get("type") not in (None, "TargetIntersectionError"):
        raise ValueError("unexpected simulator/program error must not become an RL terminal")
    if count == 0:
        raise ValueError("no actual policy actions; acquisition failure is not PPO experience")
    rewards = np.zeros(count, dtype=np.float32)
    terms = []
    for name, scale in (
        ("joint_rmse_rad", 0.35),
        ("relative_body_error_m", 0.2),
        ("pelvis_orientation_error_rad", 0.5),
        ("pelvis_position_error_m", 0.25),
    ):
        values = np.asarray(arrays.get(name, []), dtype=np.float64)
        if values.shape != (completed,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("lifecycle tracking rewards require actual completed-state metrics")
        terms.append(np.exp(-np.square(values / scale)))
    if completed:
        rewards[:completed] = 1 + np.mean(terms, axis=0)
    hold = report["return_hold"]
    requested = hold.get("requested_transitions", 250)
    returned_controls = hold.get("completed_transitions", 0)
    if requested != 250 or not 0 <= returned_controls <= requested:
        raise ValueError("lifecycle return assessment requires the complete five-second request")
    full_motion = (
        failure is None
        and completed == report["requested_transitions"]
        and report["motion_fidelity"]["passed"] is True
    )
    full_return = returned_controls == 250 and hold.get("existing_guard_screen_passed") is True
    # Every incomplete motion has a negative terminal assessment, including
    # an early abort followed by a perfect standing return.
    terminal = (25.0 if full_motion else -50.0) + 10.0 * returned_controls / 250
    if full_motion and full_return:
        terminal += 15.0
    rewards[-1] += terminal
    return rewards, dict(
        actual_policy_actions=count,
        actual_completed_controls=completed,
        partial_or_rejected_terminal_action=count > completed,
        terminal_assessment=terminal,
        full_motion=full_motion,
        full_return=full_return,
        fixed_controller_actions_enter_ppo=False,
        shortened_motion_can_receive_completion_bonus=False,
        hardware_authorized=False,
        deployment_ready=False,
    )


def generalized_advantages(rewards, values, *, gamma=0.99, lam=0.95):
    """One actual terminal episode; never bootstrap across a simulator reset."""
    rewards, values = np.asarray(rewards, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if rewards.ndim != 1 or not len(rewards) or rewards.shape != values.shape:
        raise ValueError("GAE requires one nonempty episode of rewards and values")
    if not np.isfinite(rewards).all() or not np.isfinite(values).all() or not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("invalid GAE values or discount")
    advantages = np.empty_like(rewards)
    carry = 0.0
    for index in reversed(range(len(rewards))):
        next_value = 0.0 if index + 1 == len(values) else values[index + 1]
        carry = rewards[index] + gamma * next_value - values[index] + gamma * lam * carry
        advantages[index] = carry
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


def clipped_ppo_losses(log_prob, old_log_prob, advantages, values, old_values, returns, clip=0.2):
    if (
        advantages.ndim != 1
        or not len(advantages)
        or not 0 < clip < 1
        or any(value.shape != advantages.shape for value in (log_prob, old_log_prob, values, old_values, returns))
    ):
        raise ValueError("PPO losses require matching vectors and a finite clip range")
    ratio = (log_prob - old_log_prob).exp()
    policy = -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages).mean()
    clipped_value = old_values + (values - old_values).clamp(-clip, clip)
    value = torch.maximum((values - returns).square(), (clipped_value - returns).square()).mean()
    return policy, value


class LifecyclePolicy:
    """Exact frozen-platform forward boundary with an optional sampled action."""

    def __init__(self, core, critic, *, guard, seed, stochastic):
        if type(stochastic) is not bool:
            raise ValueError("lifecycle stochastic/deterministic mode must be explicit")
        self.core, self.critic, self.guard = core, critic, guard
        self.device = next(core.parameters()).device
        self.std = core.initial_std.to(self.device).clone()
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.stochastic = stochastic
        self.records = []

    def infer(self, encoder267, history930):
        self.guard()
        if np.shape(encoder267) != (267,) or np.shape(history930) != (930,):
            raise ValueError("lifecycle policy requires unchanged 267/930 input boundaries")
        if encoder267.dtype != np.float32 or history930.dtype != np.float32:
            raise ValueError("lifecycle policy inputs must be actual float32 observations")
        with torch.no_grad():
            semantic = torch.as_tensor(encoder267, device=self.device).unsqueeze(0)
            history = torch.as_tensor(history930, device=self.device).unsqueeze(0)
            encoded = self.core.codec.encode_proprioception(history)
            if not torch.equal(encoded, history):
                raise ValueError("actual lifecycle history has nonzero absent joint slots")
            token = self.core.encode(semantic)
            decoder = torch.cat((token, encoded), dim=-1)
            mean = self.core.codec.decode_action(self.core.decoder(decoder))
            action = mean
            if self.stochastic:
                action = mean + self.std * torch.randn(mean.shape, generator=self.generator, device=self.device)
            value = self.critic(torch.cat((semantic, history), dim=-1)).squeeze()
            log_prob = torch.distributions.Normal(mean, self.std).log_prob(action).sum(-1).squeeze()
            if not all(torch.isfinite(x).all() for x in (mean, action, value, log_prob, decoder)):
                raise ValueError("nonfinite lifecycle inference")
        raw = action[0].cpu().numpy().copy()
        self.records.append(
            dict(
                semantic=encoder267.copy(),
                history=history930.copy(),
                action=raw.copy(),
                mean=mean[0].cpu().numpy().copy(),
                value=float(value),
                log_prob=float(log_prob),
            )
        )
        return raw, decoder[0].cpu().numpy().copy()


def make_critic(device):
    return nn.Sequential(nn.Linear(1197, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1)).to(device)


class LifecyclePPO:
    """Explicit complete-episode PPO; not a relabelled MJLab checkpoint."""

    def __init__(self, core, critic, anchor, *, guard, learning_rate=5e-6, epochs=4, batch_size=128, seed=0):
        if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1e-3:
            raise ValueError("invalid lifecycle learning rate")
        if type(epochs) is not int or type(batch_size) is not int or epochs < 1 or batch_size < 1:
            raise ValueError("invalid lifecycle minibatch configuration")
        self.core, self.critic, self.anchor, self.guard = core, critic, anchor, guard
        self.device = next(core.parameters()).device
        self.actor_parameters = [p for p in core.parameters() if p.requires_grad]
        wanted = {id(p) for name, p in core.named_parameters() if name.endswith(("lora_a", "lora_b"))}
        if not wanted or {id(p) for p in self.actor_parameters} != wanted:
            raise ValueError("lifecycle actor optimizer must contain only decoder LoRA")
        self.parameters = self.actor_parameters + list(critic.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)
        self.std = core.initial_std.to(self.device).clone()
        self.epochs, self.batch_size = epochs, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.update_count = self.minibatch_count = self.active_action_count = 0

    def update(self, episodes):
        if not episodes:
            raise ValueError("no complete lifecycle attempts supplied to PPO")
        fields = {key: [] for key in ("semantic", "history", "action", "value", "log_prob", "advantage", "return")}
        for episode in episodes:
            records, rewards = episode["records"], episode["rewards"]
            if not records or len(records) != len(rewards):
                raise ValueError("lifecycle rewards do not match actual sampled policy actions")
            if episode.get("policy_update_at_collection") != self.update_count:
                raise ValueError("off-policy or stale lifecycle episode")
            advantage, returns = generalized_advantages(rewards, [row["value"] for row in records])
            for key in ("semantic", "history", "action", "value", "log_prob"):
                fields[key].extend(row[key] for row in records)
            fields["advantage"].extend(advantage)
            fields["return"].extend(returns)
        data = {
            key: torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)
            for key, value in fields.items()
        }
        count = len(data["action"])
        if not all(torch.isfinite(value).all() for value in data.values()):
            raise ValueError("nonfinite lifecycle rollout batch")
        advantage = data["advantage"]
        data["advantage"] = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        losses = []
        for _ in range(self.epochs):
            order = torch.randperm(count, generator=self.generator)
            for start in range(0, count, self.batch_size):
                self.guard()
                indices = order[start : start + self.batch_size].to(self.device)
                semantic, history = data["semantic"][indices], data["history"][indices]
                mean = self.core(semantic, history)
                log_prob = torch.distributions.Normal(mean, self.std).log_prob(data["action"][indices]).sum(-1)
                value = self.critic(torch.cat((semantic, history), dim=-1)).squeeze(-1)
                policy_loss, value_loss = clipped_ppo_losses(
                    log_prob,
                    data["log_prob"][indices],
                    data["advantage"][indices],
                    value,
                    data["value"][indices],
                    data["return"][indices],
                )
                retention = self.anchor.loss(self.minibatch_count) if self.anchor is not None else mean.sum() * 0
                loss = policy_loss + value_loss + 10 * retention
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite lifecycle PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_parameters, 1.0, error_if_nonfinite=True)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0, error_if_nonfinite=True)
                self.optimizer.step()
                self.minibatch_count += 1
                losses.append([float(policy_loss.detach()), float(value_loss.detach()), float(retention.detach())])
        self.core.assert_frozen_platform_unchanged()
        if not torch.equal(self.std.cpu(), self.core.initial_std.cpu()):
            raise ValueError("lifecycle PPO changed released action std")
        self.update_count += 1
        self.active_action_count += count
        return dict(
            update=self.update_count,
            minibatches=self.minibatch_count,
            actual_active_actions=count,
            total_active_actions=self.active_action_count,
            mean_losses=dict(
                zip(("policy", "value", "standing_retention"), np.mean(losses, axis=0).tolist(), strict=True)
            ),
        )
