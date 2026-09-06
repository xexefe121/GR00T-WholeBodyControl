from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gear_sonic.utils.g1_true23_native_model_actuation import (
    NativeModelActuationProfile,
    native_model_pd_numpy,
    native_model_pd_torch,
)


@pytest.fixture
def profile():
    return NativeModelActuationProfile.from_sim_config(
        Path(__file__).resolve().parents[1] / "config/sim_validation/g1_23dof_mujoco_sim2sim.json"
    )


def test_nominal_uses_full_effort_not_quarter_or_target_projection(profile):
    q = np.zeros((1, 23))
    target = np.asarray(profile.effort)[None] * 0.8 / np.asarray(profile.kp)
    requested, applied, invalid, cost = native_model_pd_numpy(target, q, q, profile)
    np.testing.assert_allclose(applied, requested)
    assert not invalid.any() and cost[0] == 0
    assert not profile.contract()["quarter_effort_projection"]
    assert not profile.contract()["hardware_authorized"]


def test_saturation_is_motor_effort_not_target_rewrite(profile):
    target = np.full((2, 23), 100.0)
    before = target.copy()
    requested, applied, invalid, cost = native_model_pd_numpy(
        target, np.zeros_like(target), np.zeros_like(target), profile
    )
    np.testing.assert_array_equal(target, before)
    np.testing.assert_allclose(applied, np.broadcast_to(profile.effort, target.shape))
    assert not invalid.any() and (cost > 0).all() and (requested > applied).all()


def test_numpy_torch_identical_including_invalid_rows(profile):
    rng = np.random.default_rng(17)
    values = [rng.normal(size=(20, 23)) for _ in range(3)]
    values[0][3, 1] = np.nan
    expected = native_model_pd_numpy(*values, profile)
    actual = native_model_pd_torch(*(torch.tensor(v) for v in values), profile)
    for a, b in zip(expected, actual, strict=True):
        np.testing.assert_allclose(a, b.numpy(), rtol=1e-12, atol=1e-12, equal_nan=True)
    assert expected[2][3] and not expected[1][3].any()


@pytest.mark.parametrize(
    "field,value", [("kp", (0.0,) * 23), ("effort", (-1.0,) * 23), ("kd", (float("nan"),) * 23), ("decimation", 5)]
)
def test_profile_rejects_invalid_mechanics(profile, field, value):
    with pytest.raises(ValueError):
        replace(profile, **{field: value})


def test_shape_mismatch_rejected(profile):
    with pytest.raises(ValueError):
        native_model_pd_numpy(np.zeros(29), np.zeros(23), np.zeros(23), profile)
