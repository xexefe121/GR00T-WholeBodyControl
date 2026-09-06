import copy

import numpy as np
import pytest
import torch

from gear_sonic.utils.g1_true23_standing_retention import StandingOutputAnchor, batch_indices, training_records


def report():
    names = ["bounded_nominal", "bounded_plus", "bounded_minus", "bounded_holdout"]
    return dict(
        kind="g1_true23_representable_standing_teacher_comparison_v1",
        held_out_episode_separate_from_training=True,
        hardware_authorized=False,
        deployment_ready=False,
        full_motion_teacher_accepted=False,
        records=[
            dict(
                name=name,
                role="standing_train" if index < 3 else "held_out_episode",
                arrays=f"{name}.npz",
                result=dict(standing_only_labels_usable=True, original_request_projected_before_actuation=True),
            )
            for index, name in enumerate(names)
        ],
    )


def test_held_out_episode_is_never_returned_for_training():
    rows = training_records(report())
    assert [row["name"] for row in rows] == ["bounded_nominal", "bounded_plus", "bounded_minus"]


@pytest.mark.parametrize("change", ["promote", "alias", "swap", "failed"])
def test_invalid_training_partition_or_evidence_rejected(change):
    value = report()
    if change == "promote":
        value["full_motion_teacher_accepted"] = True
    elif change == "alias":
        value["records"][3]["arrays"] = value["records"][0]["arrays"]
    elif change == "swap":
        value["records"][0]["role"] = "held_out_episode"
    else:
        value["records"][0]["result"]["standing_only_labels_usable"] = False
    with pytest.raises(ValueError):
        training_records(value)


def test_sampling_resumes_from_step_without_consuming_global_rng():
    before = torch.get_rng_state().clone()
    indices = batch_indices(4000, 128, 1500, 23)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
    torch.testing.assert_close(indices, batch_indices(4000, 128, 1500, 23), rtol=0, atol=0)
    assert not torch.equal(indices, batch_indices(4001, 128, 1500, 23))
    assert int(indices.min()) >= 0 and int(indices.max()) < 1500


@pytest.mark.parametrize("args", [(-1, 2, 10, 1), (0, 11, 10, 1), (0, 2, 10, -1), (True, 2, 10, 1)])
def test_bad_sampling_counters_reject(args):
    with pytest.raises(ValueError):
        batch_indices(*args)


class Codec:
    def validate_padded_proprioception(self, value):
        assert value.shape[-1] == 930

    def encode_proprioception(self, value):
        return value

    def decode_action(self, value):
        return value


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(267, 2)
        self.encoder.requires_grad_(False)
        self.decoder = torch.nn.Linear(932, 23)
        self.codec = Codec()

    def adapter_contract(self):
        return {"fixture": True}

    def lora_state_dict(self):
        return copy.deepcopy(self.decoder.state_dict())

    def load_lora_state_dict(self, state, strict):
        self.decoder.load_state_dict(state, strict=strict)

    def assert_frozen_platform_unchanged(self):
        assert not any(value.requires_grad for value in self.encoder.parameters())

    def encode(self, value):
        return self.encoder(value)


def test_anchor_restores_current_weights_and_only_decoder_gets_gradient():
    torch.manual_seed(1)
    core = Core()
    original = core.lora_state_dict()
    payload = dict(adapter_contract=core.adapter_contract(), adapter_state_dict=copy.deepcopy(original))
    payload["adapter_state_dict"]["bias"] += 0.5
    data = dict(encoder267=np.zeros((5, 267), np.float32), history930=np.zeros((5, 930), np.float32))
    anchor = StandingOutputAnchor(core, data, payload, batch_size=3, seed=1)
    for key, value in original.items():
        torch.testing.assert_close(core.decoder.state_dict()[key], value, rtol=0, atol=0)
    assert not anchor.inputs.requires_grad and not anchor.raw.requires_grad and not anchor.target.requires_grad
    assert anchor.metrics()["target_rmse_rad"] > 0
    anchor.loss(0).backward()
    assert core.decoder.bias.grad.abs().max() > 0
    assert all(value.grad is None for value in core.encoder.parameters())
    core.load_lora_state_dict(payload["adapter_state_dict"], strict=True)
    assert anchor.loss(0).item() == 0


def test_bad_anchor_restores_adapter_transactionally():
    core = Core()
    original = core.lora_state_dict()
    payload = dict(adapter_contract=core.adapter_contract(), adapter_state_dict=copy.deepcopy(original))
    payload["adapter_state_dict"]["bias"] += 0.5
    data = dict(encoder267=np.full((5, 267), np.nan, np.float32), history930=np.zeros((5, 930), np.float32))
    with pytest.raises(ValueError, match="nonfinite"):
        StandingOutputAnchor(core, data, payload, batch_size=3, seed=1)
    for key, value in original.items():
        torch.testing.assert_close(core.decoder.state_dict()[key], value, rtol=0, atol=0)
