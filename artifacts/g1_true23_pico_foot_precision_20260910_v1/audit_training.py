"""Independent stored reward, frozen-weight and sampled-data exposure audit."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import _tensor_state_sha256
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file


def audit(directory):
    output = directory / "arithmetic_audit.json"
    if output.exists():
        raise FileExistsError("PICO arithmetic audit refuses overwrite")
    pins = {}

    def bind(path):
        path = Path(path).resolve(strict=True)
        pins[str(path)] = sha256_file(path)
        return path

    def read(name):
        return json.loads(bind(directory / name).read_text())

    def arrays(name):
        with np.load(bind(directory / name), allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    bind(__file__)
    outcome, request, resolved = read("outcome.json"), read("request.json"), read("resolved_training.json")
    args = request["arguments"]
    assert outcome["completed"] and outcome["simulator_updates"] == 500 + args["updates"]
    assert outcome["additional_updates"] == args["updates"]
    assert request["pico_used_for_training"] and resolved["pico_used_for_training"]
    n, envs = args["updates"] * args["rollout_steps"], args["num_envs"]
    assert outcome["controls_per_env"] == n and outcome["transitions"] == n * envs
    checkpoints = [
        torch.load(
            bind(directory / "checkpoints" / f"foot_precision_model_{i}.pt"), map_location="cpu", weights_only=True
        )
        for i in (500, 500 + args["updates"])
    ]
    start, end = checkpoints
    parent = torch.load(bind(args["parent_checkpoint"]), map_location="cpu", weights_only=True)
    assert sha256_file(Path(args["parent_checkpoint"])) == outcome["parent_checkpoint_sha256"]

    def exact(a, b):
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(a.cpu(), b.cpu())
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                exact(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert type(a) is type(b) and len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                exact(x, y)
        else:
            assert type(a) is type(b) and a == b

    for key in ("actor", "critic_state_dict", "optimizer_state_dict", "trainer_state"):
        exact(parent[key], start[key])
    assert start["header"] != parent["header"] and start["lineage"] != parent["lineage"]
    exact(start["optimizer_state_dict"]["param_groups"], end["optimizer_state_dict"]["param_groups"])
    adam = start["optimizer_state_dict"]["state"]
    assert len(adam) == sum(len(g["params"]) for g in start["optimizer_state_dict"]["param_groups"])
    assert adam and set(adam) == set(end["optimizer_state_dict"]["state"])
    for k, value in adam.items():
        assert value["step"].item() == 4000
        assert end["optimizer_state_dict"]["state"][k]["step"].item() == 4000 + args["updates"] * 8
    assert (
        start["lineage"] == end["lineage"]
        and end["lineage"]["materials"]["resolved_config"]["payload"] == resolved
    )
    assert end["trainer_state"]["env_common_step_counter"] - start["trainer_state"]["env_common_step_counter"] == n
    assert sha256_file(Path(outcome["checkpoint"])) == outcome["checkpoint_sha256"]
    states = [c["actor"]["state_dict"] for c in checkpoints]
    for c, state in zip(checkpoints, states, strict=True):
        assert _tensor_state_sha256(state) == c["actor"]["state_sha256"]
    frozen = [key for key in states[0] if key.startswith("core.")]
    assert frozen and all(torch.equal(states[0][k], states[1][k]) for k in frozen)
    changes = {k: float((states[1][k] - states[0][k]).abs().max()) for k in states[0] if k not in frozen}
    assert all(np.isfinite(v) and v > 0 for v in changes.values())
    assert sum(k.startswith("lora_b.") for k in changes) == 7
    assert sum(k.startswith("lora_a.") for k in changes) == 7
    assert start["critic_state_sha256"] != end["critic_state_sha256"]
    rewards, inputs, failure = (
        arrays(name) for name in ("reward_capture.npz", "sampled_actual_inputs.npz", "world_failure_capture.npz")
    )
    selected = np.array([i for i in range(n) if i < 16 or i % 64 == 0 or i >= n - 16])
    np.testing.assert_array_equal(inputs["control_index"], selected)
    assert inputs["tokenizer"].shape == (len(selected), envs, 268)
    assert inputs["policy"].shape == (len(selected), envs, 930)
    assert inputs["sampled_raw_action23"].shape == (len(selected), envs, 23)
    assert rewards["base_reward"].shape == (n, envs)
    assert all(np.isfinite(v).all() for group in (rewards, inputs, failure) for v in group.values())
    assert rewards["ppo_recorded"].all()
    term, timeout = rewards["terminated"], rewards["timeouts"]
    np.testing.assert_array_equal(rewards["stored_done"], term | timeout)
    errors = {}

    def close(name, actual, expected):
        errors[name] = float(np.max(np.abs(actual - expected)))
        assert errors[name] <= 5e-5, (name, errors[name])

    before, after = rewards["phi_before"], rewards["phi_after_including_reset_states"]
    close("potential_before", before, -np.log1p(rewards["world_cost_parts_before"].sum(-1)))
    close("potential_after", after, -np.log1p(rewards["world_cost_parts_after_including_reset_states"].sum(-1)))
    shaping = np.where(timeout, np.float32(-0.01) * before, np.float32(0.99) * np.where(term, 0, after) - before)
    bonus = np.where(term | timeout, 0, np.float32(2) * np.exp(after))
    close("shaping", rewards["shaping_reward"], shaping)
    close("quality", rewards["world_quality_bonus"], bonus)
    close("base", rewards["base_reward"], rewards["weighted_base_components"].sum(-1))
    close("pre_foot", rewards["pre_foot_precision_returned_reward"], rewards["base_reward"] + shaping + bonus)
    foot = np.where(
        term | timeout,
        np.float32(0),
        np.float32(2)
        / (np.float32(1) + rewards["world_cost_parts_after_including_reset_states"][..., 1] / np.float32(9)),
    )
    close("foot_precision", rewards["foot_precision_bonus"], foot)
    close("returned", rewards["returned_reward"], rewards["base_reward"] + shaping + bonus + foot)
    close(
        "PPO_stored",
        rewards["stored_reward_with_timeout_bootstrap"],
        rewards["returned_reward"] + np.float32(0.99) * rewards["critic_value_before_bootstrap"] * timeout,
    )
    close(
        "world_error",
        failure["error_m"],
        np.linalg.norm(failure["desired_position_w"] - failure["measured_position_w"], axis=-1),
    )
    np.testing.assert_array_equal(failure["failure"], failure["error_m"] > 0.30)
    assert not np.any(failure["failure"] & ~term)
    gradients = read("gradient_capture.json")
    assert len(gradients) == args["updates"] * 8 == outcome["optimizer_steps"]
    assert all(np.isfinite(list(g.values())).all() for g in gradients)
    gradient_max = max(
        max(sum(g[k] ** 2 for k in ("root_conditioner", "decoder_adapters", "exploration")) ** 0.5, g["critic"])
        for g in gradients
    )
    assert gradient_max <= 0.500002
    spans = json.loads(bind(args["spans"]).read_text())["spans"]
    exposure = {}
    covered = np.zeros_like(inputs["reference_q0"], dtype=bool)
    for row in spans:
        mask = (inputs["reference_q0"] >= row["start"]) & (inputs["reference_q0"] < row["start"] + row["length"])
        assert not np.any(covered & mask)
        covered |= mask
        exposure[row["name"]] = dict(
            sampled_env_controls=int(mask.sum()),
            unique_sampled_reference_anchors=int(np.unique(inputs["reference_q0"][mask]).size),
        )
    assert covered.all() and set(exposure) == {"walk002", "walk003", "pico"}
    report = dict(
        kind="pico_foot_precision_training_arithmetic_audit_v1",
        parent_actor_critic_adam_and_counters_bit_exact=True,
        all_adam_steps_advance_from4000=True,
        passed=True,
        completed_updates=500 + args["updates"],
        additional_updates=args["updates"],
        all_transition_rewards_verified=n * envs,
        sampled_input_controls=len(selected),
        sampled_exposure=exposure,
        complete_reference_coverage_or_tracking_claimed=False,
        maximum_reward_arithmetic_errors=errors,
        gradient_norm_max=gradient_max,
        all_seven_lora_a_b_root_exploration_changed=True,
        frozen_base_bit_exact=True,
        actor_parameter_changes=changes,
        inputs=pins,
        deployment_ready=False,
        hardware_authorized=False,
    )
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("inputs", "actor_parameter_changes")}), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    audit(parser.parse_args().directory.resolve(strict=True))
