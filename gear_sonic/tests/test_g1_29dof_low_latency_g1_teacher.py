"""Small fixtures check the G1 encoder boundary without loading trainer state."""

import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from gear_sonic.utils import g1_29dof_low_latency_g1_teacher as runtime


@pytest.fixture
def release(monkeypatch, tmp_path):
    generator = torch.Generator().manual_seed(640)
    state = {}
    for prefix, dims in ((runtime.G1_ENCODER_PREFIX, (640, 7, 64)), (runtime._DECODER_PREFIX, (994, 11, 29))):
        for i, (n, m) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            state[f"{prefix}{2 * i}.weight"] = torch.randn(m, n, generator=generator) * 0.03
            state[f"{prefix}{2 * i}.bias"] = torch.randn(m, generator=generator) * 0.03
    monkeypatch.setattr(runtime, "G1_ENCODER_DIMS", (640, 7, 64))
    monkeypatch.setattr(runtime, "DECODER_DIMS", (994, 11, 29))
    metadata = dict(
        source_revision=runtime.LOW_LATENCY_RELEASE_HF_REVISION,
        reference_profile=runtime.REFERENCE_PROFILE_LOW_LATENCY,
    )
    payload = [{"policy_state_dict": state}, runtime.LOW_LATENCY_RELEASE_SHA256, metadata]
    monkeypatch.setattr(runtime, "_load_pinned_legacy_release", lambda path: payload)
    path = tmp_path / "synthetic.pt"
    path.touch()
    return path, payload


def test_exact_singleton_encoder_fsq_shared_decoder(release):
    path, payload = release
    teacher = runtime.LowLatencyG1Teacher(path)
    rng = np.random.default_rng(17)
    encoder = rng.normal(size=640).astype(np.float32)
    history = rng.normal(size=930).astype(np.float32)
    state = payload[0]["policy_state_dict"]

    def network(prefix, x):
        x = F.silu(F.linear(x, state[prefix + "0.weight"], state[prefix + "0.bias"]))
        return F.linear(x, state[prefix + "2.weight"], state[prefix + "2.bias"])

    latent = network(runtime.G1_ENCODER_PREFIX, torch.from_numpy(encoder)[None])
    half = 31 * 1.001 / 2
    token = torch.round(torch.tanh(latent + math.atanh(0.5 / half)) * half - 0.5) / 16
    raw = network(runtime._DECODER_PREFIX, torch.cat((token, torch.from_numpy(history)[None]), -1))
    output, actual_token = teacher.infer(encoder, history)
    np.testing.assert_array_equal(output, raw.numpy()[0])
    np.testing.assert_array_equal(actual_token, token.numpy()[0])
    np.testing.assert_array_equal(teacher.infer(encoder, history)[0], output)
    assert output.dtype == actual_token.dtype == np.float32
    assert teacher.descriptor()["training_updates"] == 0
    assert teacher.descriptor()["decoder_modified"] is False
    assert teacher.descriptor()["deployment_ready"] is False


@pytest.mark.parametrize("bad", ["batch", "width", "dtype", "nan", "history"])
def test_invalid_inference_rejected(release, bad):
    teacher = runtime.LowLatencyG1Teacher(release[0])
    encoder, history = np.zeros(640, np.float32), np.zeros(930, np.float32)
    if bad == "batch":
        encoder = np.zeros((2, 640), np.float32)
    elif bad == "width":
        encoder = encoder[:-1]
    elif bad == "dtype":
        encoder = encoder.astype(float)
    elif bad == "nan":
        encoder[0] = np.nan
    else:
        history = history[:-1]
    with pytest.raises(ValueError):
        teacher.infer(encoder, history)


@pytest.mark.parametrize("bad", ["digest", "revision", "profile", "keys", "shape", "finite"])
def test_invalid_release_rejected(release, bad):
    path, payload = release
    key = runtime.G1_ENCODER_PREFIX + "0.weight"
    if bad == "digest":
        payload[1] = "0" * 64
    elif bad in {"revision", "profile"}:
        payload[2]["source_revision" if bad == "revision" else "reference_profile"] = "wrong"
    elif bad == "keys":
        del payload[0]["policy_state_dict"][key]
    elif bad == "shape":
        payload[0]["policy_state_dict"][key] = torch.zeros(1)
    else:
        payload[0]["policy_state_dict"][key][0, 0] = float("nan")
    with pytest.raises(ValueError):
        runtime.LowLatencyG1Teacher(path)
