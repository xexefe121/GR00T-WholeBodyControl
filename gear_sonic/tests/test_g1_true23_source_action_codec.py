from pathlib import Path

import numpy as np
import pytest
import torch

from gear_sonic.scripts.audit_g1_true23_source_action_codec import audit_source_oracle, compile_source_oracle
from gear_sonic.scripts.diagnose_g1_true23_release_semantics import ReferenceAblation
from gear_sonic.utils.g1_23dof_contract import NATIVE_IL23_TO_CANONICAL_IL29
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_INNER_LOWER_HARDWARE,
    SAFE_TARGET_INNER_UPPER_HARDWARE,
    safe_target_transform_numpy,
    safe_target_transform_torch,
)
from gear_sonic.utils.g1_true23_source_action_codec import (
    SOURCE_ACTION_CONVENTION,
    source_action_history_numpy,
    source_action_history_torch,
    source_scaled_precompensation,
    source_scaled_precompensation_torch,
)


def test_against_compiled_original_cpp_not_shared_native_constants():
    assets = Path(__file__).resolve().parents[3] / "GR00T-WholeBodyControl"
    if not (assets / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp").exists():
        pytest.skip("original SONIC source fixture unavailable")
    oracle = compile_source_oracle(assets)
    audit = audit_source_oracle(oracle)
    assert audit["pass_source_action_equations"]
    assert 1.568 < audit["legacy_hip_pitch_displacement_ratio"] < 1.569
    # Historical Python diagnostic had 0.907 here; original C++ is twice that.
    assert 1.8144 < oracle["kd_hardware29"][4] < 1.8145
    assert oracle["kd_hardware29"][4] == oracle["kd_hardware29"][5]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_projection_and_history_with_unchanged_physical_bounds(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    raw = np.random.default_rng(2855).uniform(-9.999, 9.999, (1024, 23)).astype(np.float32)
    inverse, projection = source_scaled_precompensation_torch(torch.as_tensor(raw, device=device))
    expected = [source_scaled_precompensation(row) for row in raw]
    np.testing.assert_allclose(inverse.cpu(), np.stack([v[0] for v in expected]), atol=1e-6, rtol=0)
    np.testing.assert_allclose(projection.cpu(), np.stack([v[1] for v in expected]), atol=1e-6, rtol=0)
    safe, targets = safe_target_transform_torch(inverse)
    cpu = [safe_target_transform_numpy(v[0]) for v in expected]
    np.testing.assert_allclose(targets.cpu(), np.stack([v[1] for v in cpu]), atol=1e-6, rtol=0)
    assert inverse.abs().max() < 10
    assert np.all(targets.cpu().numpy() >= np.asarray(SAFE_TARGET_INNER_LOWER_HARDWARE) - 5e-7)
    assert np.all(targets.cpu().numpy() <= np.asarray(SAFE_TARGET_INNER_UPPER_HARDWARE) + 5e-7)
    history = np.random.default_rng(18).normal(size=(16, 930)).astype(np.float32)
    actual = source_action_history_torch(torch.as_tensor(history, device=device))
    expected_history = np.stack([source_action_history_numpy(row) for row in history])
    np.testing.assert_array_equal(actual.cpu(), expected_history)
    assert torch.isfinite(safe).all()


def test_history_modifies_only_retained_previous_action_slots():
    history = np.arange(930, dtype=np.float32)
    before = history.copy()
    converted = source_action_history_numpy(history)
    changed = np.zeros(930, bool)
    changed[610 + np.arange(10)[:, None] * 29 + np.asarray(NATIVE_IL23_TO_CANONICAL_IL29)] = True
    np.testing.assert_array_equal(converted[~changed], history[~changed])
    np.testing.assert_array_equal(history, before)
    assert converted[610] > history[610] * 1.5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 10.0, -10.0])
def test_invalid_actions_rejected_before_physics(value):
    raw = np.zeros(23, np.float32)
    raw[4] = value
    with pytest.raises(ValueError):
        source_scaled_precompensation(raw)
    with pytest.raises(ValueError):
        source_scaled_precompensation_torch(torch.from_numpy(raw[None]))


def test_legacy_ablation_history_unchanged_and_v2_explicit():
    history = np.ones(930, np.float32)
    legacy = ReferenceAblation({}, "causal_past", action_convention="released_bounded_linear")
    updated = ReferenceAblation({}, "causal_past", action_convention=SOURCE_ACTION_CONVENTION)
    np.testing.assert_array_equal(legacy.transform_history(history), history)
    np.testing.assert_array_equal(updated.transform_history(history), source_action_history_numpy(history))
    assert legacy.contract()["decoder994_history_unchanged"]
    assert not updated.contract()["decoder994_history_unchanged"]
    assert updated.contract()["source_action_codec"]["kind"] == SOURCE_ACTION_CONVENTION
