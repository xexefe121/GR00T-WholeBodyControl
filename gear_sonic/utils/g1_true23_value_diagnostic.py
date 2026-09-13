"""Offline value-learning measurements. These artifacts are never policies.

The two critic probes start from identical parameters, normalization buffers and
Adam moments. They fit the same fixed on-policy targets with the same shuffled
minibatches. Only PPO value clipping differs. This measures a local optimization
constraint, not better closed-loop control or generalization.
"""

import copy
import math

import numpy as np
import torch

from gear_sonic.trl.mjlab.frozen_platform_lora_runner import _state_sha256


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def stats(values):
    value = torch.as_tensor(values).detach().cpu().double().flatten()
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError("value diagnostic requires nonempty finite samples")
    quantiles = torch.quantile(value, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0], dtype=torch.float64))
    return dict(zip(("min", "p05", "median", "p95", "max"), quantiles.tolist())) | {
        "mean": value.mean().item(),
        "std": value.std(correction=0).item(),
        "rms": value.square().mean().sqrt().item(),
        "count": value.numel(),
    }


def calibration(values, targets):
    value, target = values.detach().double(), targets.detach().double()
    if value.shape != target.shape or not torch.isfinite(value).all() or not torch.isfinite(target).all():
        raise ValueError("value predictions and targets must be matching finite arrays")
    error = target - value
    variance = target.var(correction=0).item()
    return {
        "predictions": stats(value),
        "targets": stats(target),
        "error": stats(error),
        "mse": error.square().mean().item(),
        "explained_variance": None if variance == 0 else 1 - error.var(correction=0).item() / variance,
    }


def loss_terms(values, old_values, returns, clip_param):
    if not math.isfinite(clip_param) or clip_param <= 0:
        raise ValueError("value clip must be finite and positive")
    if values.shape != old_values.shape or values.shape != returns.shape:
        raise ValueError("value loss arrays must have identical shapes")
    raw = (values - returns).square()
    delta = values - old_values
    clipped = (old_values + delta.clamp(-clip_param, clip_param) - returns).square()
    return raw, torch.maximum(raw, clipped), (delta.abs() > clip_param) & (clipped > raw)


def verify_gae(block, gamma, lam):
    """Independent float64 NumPy recurrence, including stock timeout semantics."""
    arrays = {
        key: block[key].cpu().numpy().astype(np.float64)
        for key in (
            "raw_rewards",
            "rewards",
            "values",
            "returns",
            "dones",
            "timeouts",
            "last_values",
            "advantages",
        )
    }
    expected_rewards = arrays["raw_rewards"] + gamma * arrays["values"] * arrays["timeouts"]
    reward_error = float(np.abs(expected_rewards - arrays["rewards"]).max())
    expected = np.zeros_like(arrays["returns"])
    advantage = np.zeros_like(arrays["last_values"])
    for step in reversed(range(len(expected))):
        next_value = arrays["last_values"] if step == len(expected) - 1 else arrays["values"][step + 1]
        alive = 1 - arrays["dones"][step]
        delta = arrays["rewards"][step] + gamma * alive * next_value - arrays["values"][step]
        advantage = delta + gamma * lam * alive * advantage
        expected[step] = advantage + arrays["values"][step]
    return_error = float(np.abs(expected - arrays["returns"]).max())
    raw_advantage = arrays["returns"] - arrays["values"]
    normalized = (raw_advantage - raw_advantage.mean()) / (raw_advantage.std(ddof=1) + 1e-8)
    advantage_error = float(np.abs(normalized - arrays["advantages"]).max())
    np.testing.assert_allclose(expected_rewards, arrays["rewards"], atol=2e-5, rtol=2e-6)
    np.testing.assert_allclose(expected, arrays["returns"], atol=5e-4, rtol=3e-6)
    np.testing.assert_allclose(normalized, arrays["advantages"], atol=1e-5, rtol=2e-6)
    return dict(
        timeout_reward_max_error=reward_error,
        gae_max_error=return_error,
        normalized_advantage_max_error=advantage_error,
        verified=True,
    )


def critic_adam_state(optimizer):
    """Extract the existing critic-only Adam state without resetting its moments."""
    saved = copy.deepcopy(optimizer.state_dict())
    groups = [group for group in saved["param_groups"] if group.get("name") == "critic"]
    if len(groups) != 1 or type(optimizer) is not torch.optim.Adam:
        raise ValueError("diagnostic requires one explicit critic group in Adam")
    group = groups[0]
    old_ids = list(group["params"])
    group["params"] = list(range(len(old_ids)))
    state = {}
    for new_id, old_id in enumerate(old_ids):
        if old_id in saved["state"]:
            state[new_id] = {
                key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for key, value in saved["state"][old_id].items()
            }
    return {"state": state, "param_groups": [group]}


def fit_probe(
    critic,
    observations,
    old_values,
    returns,
    optimizer_state,
    *,
    use_value_clip,
    clip_param,
    max_grad_norm,
    value_loss_coef,
    indices,
    num_mini_batches,
    num_epochs,
):
    """Clone critic; never mutate the supplied actor, critic, optimizer or samples."""
    model = copy.deepcopy(critic).cpu()
    model.eval()  # Stored rollout-end normalizer stays frozen, as in PPO.update.
    model._forward_pre_hooks.clear()  # The training process precision guard is not serializable.
    if model.distribution is not None or model.is_recurrent:
        raise ValueError("critic diagnostic requires deterministic feedforward MLP")
    initial_sha = _state_sha256(cpu_state(model))
    parameters = list(model.parameters())
    if len(optimizer_state["param_groups"][0]["params"]) != len(parameters):
        raise ValueError("critic Adam ownership differs from cloned critic")
    optimizer = torch.optim.Adam(parameters)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    count = old_values.numel()
    if returns.shape != old_values.shape or count % num_mini_batches or num_epochs < 1:
        raise ValueError("probe requires aligned complete minibatches and positive epochs")
    if indices.dtype != torch.int64 or not torch.equal(indices.sort().values, torch.arange(count)):
        raise ValueError("probe shuffle must cover every captured sample exactly once")
    size = count // num_mini_batches
    with torch.no_grad():
        before = model(observations).detach()
    rows = []
    for epoch in range(num_epochs):
        for minibatch in range(num_mini_batches):
            chosen = indices[minibatch * size : (minibatch + 1) * size]
            obs = {key: value[chosen] for key, value in observations.items()}
            prediction = model(obs)
            raw, clipped, blocked = loss_terms(prediction, old_values[chosen], returns[chosen], clip_param)
            objective = (clipped if use_value_clip else raw).mean() * value_loss_coef
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite critic gradient")
            rows.append(
                {
                    "epoch": epoch,
                    "minibatch": minibatch,
                    "raw_mse": raw.mean().item(),
                    "objective": objective.item(),
                    "flat_clipped_branch_fraction": blocked.float().mean().item(),
                    "outside_value_clip_fraction": ((prediction - old_values[chosen]).abs() > clip_param)
                    .float()
                    .mean()
                    .item(),
                    "gradient_norm_before_clip": norm.item(),
                    "gradient_scale": min(1.0, max_grad_norm / (norm.item() + 1e-6)),
                }
            )
            optimizer.step()
    with torch.no_grad():
        after = model(observations).detach()
    return {
        "use_clipped_value_loss": use_value_clip,
        "initial_critic_sha256": initial_sha,
        "final_critic_sha256": _state_sha256(cpu_state(model)),
        "before": calibration(before, returns),
        "after": calibration(after, returns),
        "prediction_delta": stats(after - before),
        "minibatches": rows,
        "actor_optimized": False,
        "controller_improvement_proven": False,
    }
