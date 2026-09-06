"""Standing-only adapter import must not bypass PPO or deployment contracts."""

import copy
from types import SimpleNamespace

import pytest
import torch

from gear_sonic.trl.mjlab.frozen_platform_lora_actor import _tensor_state_sha256
from gear_sonic.utils.g1_true23_standing_initialization import (
    apply_standing_initialization,
    validate_standing_payload,
)


def fixture():
    contract = {"fixture_contract": True}
    state = {"decoder.layers.0.lora_a": torch.ones((1, 2)), "decoder.layers.0.lora_b": torch.ones((1, 1))}
    digest = _tensor_state_sha256(state)
    payload = dict(
        kind="g1_true23_standing_only_lora_adapter_diagnostic_v1",
        adapter_contract=contract,
        adapter_state_dict=state,
        adapter_state_sha256=digest,
        optimizer_steps=500,
        hardware_authorized=False,
        deployment_ready=False,
        promotion_eligible=False,
        ppo_resume_checkpoint=False,
        full_motion_teacher_accepted=False,
    )
    descriptor = dict(frozen_platform_contract=contract, adapter_state_sha256=digest, source_optimizer_steps=500)
    fit = dict(adapter_contract=contract, adapter_state_sha256=digest, steps=500)
    return payload, descriptor, fit


def test_valid_separate_diagnostic_payload():
    payload, _, fit = fixture()
    assert validate_standing_payload(payload, fit) is payload


@pytest.mark.parametrize(
    "flag",
    [
        "hardware_authorized",
        "deployment_ready",
        "promotion_eligible",
        "ppo_resume_checkpoint",
        "full_motion_teacher_accepted",
    ],
)
def test_no_relabel_as_resume_or_qualified_motion(flag):
    payload, _, fit = fixture()
    payload[flag] = True
    with pytest.raises(ValueError, match="must not claim"):
        validate_standing_payload(payload, fit)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_tensor_cannot_be_admitted_even_with_its_matching_hash(bad):
    payload, _, fit = fixture()
    payload["adapter_state_dict"]["decoder.layers.0.lora_a"][0, 0] = bad
    payload["adapter_state_sha256"] = fit["adapter_state_sha256"] = _tensor_state_sha256(
        payload["adapter_state_dict"]
    )
    with pytest.raises(ValueError, match="finite float32"):
        validate_standing_payload(payload, fit)


def test_tampered_adapter_and_other_frozen_platform_rejected():
    payload, _, fit = fixture()
    with pytest.raises(ValueError, match="trainer's frozen platform"):
        validate_standing_payload(payload, fit, expected_contract={"other": True})
    payload["adapter_state_dict"]["decoder.layers.0.lora_a"][0, 0] += 1
    with pytest.raises(ValueError, match="tensor hash"):
        validate_standing_payload(payload, fit)


class Core:
    def __init__(self, contract, state):
        self.contract = contract
        self.state = {key: torch.zeros_like(value) for key, value in state.items()}

    def adapter_contract(self):
        return self.contract

    def lora_state_dict(self):
        return copy.deepcopy(self.state)

    def load_lora_state_dict(self, state, strict):
        assert strict
        self.state = copy.deepcopy(state)

    def assert_frozen_platform_unchanged(self):
        pass

    def merged_true23_policy_sha256(self, std):
        assert std.shape == (23,)
        return _tensor_state_sha256(self.state)


def runner_fixture(payload, descriptor, *, dirty=False, fail=False):
    core = Core(descriptor["frozen_platform_contract"], payload["adapter_state_dict"])
    critic = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(critic.parameters())
    if dirty:
        optimizer.state[next(iter(critic.parameters()))] = {"step": 1}
    actor = SimpleNamespace(core=core, distribution=SimpleNamespace(std_param=torch.ones(23)))

    def boundary():
        if fail:
            raise ValueError("fixture boundary rejected")

    return SimpleNamespace(
        alg=SimpleNamespace(get_policy=lambda: actor, critic=critic, optimizer=optimizer),
        _require_counter_coherence=lambda: 0,
        _assert_boundary=boundary,
        _frozen_lora_runtime={},
    )


def test_fresh_import_changes_adapter_only():
    payload, descriptor, _ = fixture()
    runner = runner_fixture(payload, descriptor)
    before = _tensor_state_sha256(runner.alg.critic.state_dict())
    result = apply_standing_initialization(runner, payload, descriptor)
    assert _tensor_state_sha256(runner.alg.get_policy().core.state) == descriptor["adapter_state_sha256"]
    assert result["merged_true23_policy_sha256"] == descriptor["adapter_state_sha256"]
    assert _tensor_state_sha256(runner.alg.critic.state_dict()) == before
    assert not runner.alg.optimizer.state
    assert runner._require_counter_coherence() == 0


@pytest.mark.parametrize("dirty", [False, True])
def test_failed_or_nonfresh_import_leaves_original_adapter(dirty):
    payload, descriptor, _ = fixture()
    runner = runner_fixture(payload, descriptor, dirty=dirty, fail=not dirty)
    before = _tensor_state_sha256(runner.alg.get_policy().core.state)
    with pytest.raises(ValueError, match="fresh|boundary rejected"):
        apply_standing_initialization(runner, payload, descriptor)
    assert _tensor_state_sha256(runner.alg.get_policy().core.state) == before
    assert not runner._frozen_lora_runtime
