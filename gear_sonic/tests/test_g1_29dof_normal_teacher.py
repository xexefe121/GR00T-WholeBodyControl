"""Tiny synthetic interface tests; real source-weight evidence is a separate preflight."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.utils import g1_29dof_normal_teacher as module
from gear_sonic.utils.g1_23dof_contract import NORMAL_RELEASE_SHA256, REFERENCE_PROFILE_NORMAL
from gear_sonic.utils.g1_29dof_normal_lifecycle import run_case


@pytest.fixture
def toy_teacher(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.pt"
    path.write_bytes(b"explicit synthetic fixture, never deserialized")
    rng = torch.Generator().manual_seed(20260910)
    state = {}
    for prefix, dims in (("encoders.teleop", (267, 7, 64)), ("decoders.g1_dyn", (994, 11, 29))):
        for i, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            state[f"actor_module.{prefix}.module.{2 * i}.weight"] = torch.randn(right, left, generator=rng) * 0.01
            state[f"actor_module.{prefix}.module.{2 * i}.bias"] = torch.randn(right, generator=rng) * 0.01
    monkeypatch.setattr(module, "ENCODER_DIMS", (267, 7, 64))
    monkeypatch.setattr(module, "DECODER_DIMS", (994, 11, 29))
    monkeypatch.setattr(module, "file_sha256", lambda path: NORMAL_RELEASE_SHA256)
    monkeypatch.setattr(
        module,
        "_load_pinned_legacy_release",
        lambda path: (
            {"policy_state_dict": state},
            NORMAL_RELEASE_SHA256,
            {"reference_profile": REFERENCE_PROFILE_NORMAL, "source_revision": None},
        ),
    )
    return module.ExactNormal29Teacher(path)


def test_wrong_bytes_rejected_before_legacy_pickle_loader(tmp_path, monkeypatch):
    path = tmp_path / "untrusted.pt"
    path.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(module, "_load_pinned_legacy_release", lambda path: pytest.fail("unsafe loader called"))
    with pytest.raises(ValueError, match="other than pinned"):
        module.ExactNormal29Teacher(path)


def test_normal_actor_keeps29_outputs_and_proprioceptive_missing23_axes(toy_teacher):
    e, h = np.zeros(267, np.float32), np.zeros(930, np.float32)
    raw, token = toy_teacher.infer_with_token(e, h)
    assert raw.shape == (29,) and token.shape == (64,) and raw.dtype == np.float32
    np.testing.assert_array_equal(raw, toy_teacher.infer(e, h))
    h[30 + 5] = 1.0  # Physical29 waist-roll feedback; cannot be masked as absent.
    assert not np.array_equal(raw, toy_teacher.infer(e, h))
    assert toy_teacher.descriptor()["retained_action_rows"] == 29
    assert toy_teacher.descriptor()["missing_joint_zero_mask_applied"] is False
    assert toy_teacher.descriptor()["reference_profile"] == REFERENCE_PROFILE_NORMAL


@pytest.mark.parametrize(
    "encoder,history",
    [
        (np.zeros(267, np.float64), np.zeros(930, np.float32)),
        (np.zeros(266, np.float32), np.zeros(930, np.float32)),
        (np.zeros((1, 267), np.float32), np.zeros((1, 930), np.float32)),
        (np.full(267, np.nan, np.float32), np.zeros(930, np.float32)),
    ],
)
def test_invalid_or_batched_input_refused(toy_teacher, encoder, history):
    with pytest.raises(ValueError):
        toy_teacher.infer(encoder, history)


def test_frozen_tensor_mutation_detected(toy_teacher):
    toy_teacher._decoder[-1][1][0] += 1
    with pytest.raises(ValueError, match="tensors changed"):
        toy_teacher.descriptor()


def test_normal_lifecycle_refuses_low_latency_before_physics():
    with pytest.raises(ValueError, match="different source policy"):
        run_case(None, None, None, SimpleNamespace(reference_profile="low_latency"), None, None, None)


def test_normal_lifecycle_refuses_native23_model_before_physics():
    native = SimpleNamespace(nq=30, nv=29, nu=23, opt=SimpleNamespace(timestep=0.002))
    with pytest.raises(ValueError, match="original29 model"):
        run_case(
            native,
            None,
            None,
            SimpleNamespace(reference_profile=REFERENCE_PROFILE_NORMAL),
            np.zeros((22, 36)),
            [],
            np.zeros((22, 21), np.float32),
        )
