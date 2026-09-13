"""CPU inference ABI/masking parity and rejection of old checkpoint kinds."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from gear_sonic.trl.mjlab.native23_original_intent_actor import True23OriginalIntentActorModel
from gear_sonic.trl.mjlab.native23_root_feedback_runner import CHECKPOINT_HEADER
from gear_sonic.utils.g1_true23_original_intent_checkpoint import OriginalIntentCPUActor, validate_semantics


def actor_fixture():
    # Small deterministic layers exercise the SAME production forward methods.
    actor = True23OriginalIntentActorModel.__new__(True23OriginalIntentActorModel)
    nn.Module.__init__(actor)
    actor.tokenizer_obs_group = "tokenizer"
    actor.proprioception_obs_group = "policy"
    actor.root_feedback_obs_group = "root_feedback"
    actor.tokenizer_has_encoder_index = False
    generator = torch.Generator().manual_seed(20260909)
    encoder = nn.Linear(267, 64)
    decoder = nn.Module()
    decoder.module = nn.Sequential(nn.Linear(994, 16), nn.SiLU(), nn.Linear(16, 23))
    actor.root_conditioner = nn.Linear(9, 16, bias=False)
    mask = torch.ones(930)
    mask[torch.arange(5, 930, 29)] = 0
    for layer in (encoder, *decoder.modules(), actor.root_conditioner):
        if isinstance(layer, nn.Linear):
            with torch.no_grad():
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * 0.02)
                if layer.bias is not None:
                    layer.bias.zero_()
    actor.core = SimpleNamespace(
        encode=encoder, decoder=decoder, codec=SimpleNamespace(encode_proprioception=lambda x: x * mask)
    )
    return actor, mask.numpy()


def test_cpu_mean_matches_training_forward_and_keeps_raw_trace():
    actor, mask = actor_fixture()
    reader = OriginalIntentCPUActor(actor)
    rng = np.random.default_rng(23)
    inputs = [rng.normal(size=n).astype(np.float32) for n in (267, 930, 9)]
    before = [x.copy() for x in inputs]
    output, combined = reader.infer(*inputs)
    with torch.inference_mode():
        expected = actor(
            {k: torch.from_numpy(v)[None] for k, v in zip(("tokenizer", "policy", "root_feedback"), inputs)}
        )
    np.testing.assert_array_equal(output, expected.numpy()[0])
    assert output.shape == (23,) and combined.shape == (994,)
    np.testing.assert_array_equal(combined[64:], inputs[1])
    changed = inputs[1].copy()
    changed[mask == 0] += 100
    altered, raw_altered = reader.infer(inputs[0], changed, inputs[2])
    np.testing.assert_array_equal(altered, output)
    assert not np.array_equal(raw_altered, combined)
    for a, b in zip(inputs, before):
        np.testing.assert_array_equal(a, b)
    no_feedback, _ = reader.infer(inputs[0], inputs[1], np.zeros(9, np.float32))
    assert not np.array_equal(no_feedback, output)


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("damage", ("shape", "dtype", "nan", "list"))
def test_cpu_rejects_invalid_input(index, damage):
    actor, _ = actor_fixture()
    values = [np.zeros(n, np.float32) for n in (267, 930, 9)]
    if damage == "shape":
        values[index] = values[index][None]
    elif damage == "dtype":
        values[index] = values[index].astype(np.float64)
    elif damage == "nan":
        values[index][0] = np.nan
    else:
        values[index] = values[index].tolist()
    with pytest.raises(ValueError, match="finite float32"):
        OriginalIntentCPUActor(actor).infer(*values)


@pytest.mark.parametrize("value", (None, [], {}, {"header": {}}))
def test_reader_rejects_unknown_header(value):
    with pytest.raises(ValueError, match="research checkpoint header"):
        validate_semantics(value)


@pytest.mark.parametrize(
    "kind",
    (
        None,
        "g1_native23_root_feedback_buffered_source_actor_v3",
        "g1_native23_root_feedback_release_compatible_actor_v1",
    ),
)
def test_reader_rejects_old_actor_without_loading_files(kind):
    with pytest.raises(ValueError, match="old/relabelled"):
        validate_semantics({"header": CHECKPOINT_HEADER, "actor": {"contract": {"kind": kind}}})


def test_reader_rejects_non_original_actor():
    with pytest.raises(TypeError, match="original-intent actor"):
        OriginalIntentCPUActor(nn.Linear(2, 2))
