from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.utils.g1_23dof_safe_target_transform import (
    safe_target_transform_numpy,
    safe_target_transform_torch,
)
from gear_sonic.utils.g1_true23_release_action_diagnostic import (
    bounded_linear_precompensation,
    bounded_linear_precompensation_torch,
)
from gear_sonic.utils.g1_true23_release_compatibility import (
    release_compatibility_contract,
    validate_release_compatibility,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_training_matches_cpu_diagnostic_and_history(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    raw = np.random.default_rng(8923).uniform(-9.99, 9.99, (1024, 23)).astype(np.float32)
    raw[0] = 0
    raw[1] = np.linspace(-0.1, 0.1, 23)
    inverse, projected = bounded_linear_precompensation_torch(torch.as_tensor(raw, device=device))
    expected = [bounded_linear_precompensation(row) for row in raw]
    np.testing.assert_allclose(inverse.cpu(), np.stack([v[0] for v in expected]), atol=1e-6, rtol=0)
    np.testing.assert_allclose(projected.cpu(), np.stack([v[1] for v in expected]), atol=1e-6, rtol=0)
    safe, target = safe_target_transform_torch(inverse)
    cpu = [safe_target_transform_numpy(v[0]) for v in expected]
    np.testing.assert_allclose(safe.cpu(), np.stack([v[0] for v in cpu]), atol=2e-6, rtol=0)
    np.testing.assert_allclose(target.cpu(), np.stack([v[1] for v in cpu]), atol=1e-6, rtol=0)
    assert inverse.abs().max() < 10


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 10.0, -10.0])
def test_invalid_training_action_cannot_enter_physics(value):
    raw = torch.zeros(4, 23)
    raw[2, 3] = value
    with pytest.raises(ValueError, match="bound"):
        bounded_linear_precompensation_torch(raw)


def test_contract_tampering_rejected():
    contract = release_compatibility_contract("a" * 64)
    assert validate_release_compatibility(contract) == contract
    with pytest.raises(ValueError):
        validate_release_compatibility({**contract, "action_convention": "native_tanh"})


def test_training_reference_selects_q9_without_changing_native_motion():
    from gear_sonic.envs.mjlab.sonic_true23_release_compatible import release_vr_orientation, release_vr_position
    from gear_sonic.utils.g1_true23_virtual_source_reference import virtual_source_vr_terms

    source = Path(__file__).resolve().parents[3] / "GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml"
    if not source.exists():
        pytest.skip("original29 reference fixture unavailable")
    q = torch.zeros(30, 23)
    q[:, 16] = torch.linspace(0, 1, 30)
    before = q.clone()
    command = SimpleNamespace(motion=SimpleNamespace(joint_pos=q), time_steps=torch.tensor([9, 18]))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command))
    actual = torch.cat((release_vr_position(env, str(source)), release_vr_orientation(env, str(source))), -1)
    expected = virtual_source_vr_terms({"joint_pos": q.numpy()}, source)[[9, 18]]
    np.testing.assert_array_equal(actual, expected)
    torch.testing.assert_close(q, before, rtol=0, atol=0)
    command.time_steps = torch.tensor([11, 20])
    assert not torch.equal(actual[:, :9], release_vr_position(env, str(source)))


def test_legacy_parent_cannot_be_relabelled_as_corrected_training():
    from gear_sonic.scripts.train_g1_true23_root_feedback import validate_bounds
    from gear_sonic.tests.test_train_g1_true23_root_feedback import arguments

    with pytest.raises(ValueError, match="fresh initialization"):
        validate_bounds(
            arguments(
                "regression", "--release-source-geometry", "source.xml", "--initialize-actor-from", "legacy.pt"
            )
        )


def test_legacy_cpu_runtime_rejects_new_semantics_before_loading_models(tmp_path):
    import json

    from gear_sonic.utils.g1_true23_root_feedback_benchmark import load_root_feedback_pair

    path = tmp_path / "pair.json"
    path.write_text(json.dumps({"release_compatibility": release_compatibility_contract("a" * 64)}))
    with pytest.raises(ValueError, match="executing runtime"):
        load_root_feedback_pair(path)


def test_new_runtime_rejects_legacy_manifest_before_loading_models(tmp_path):
    from gear_sonic.utils.g1_true23_root_feedback_benchmark import load_root_feedback_pair

    path = tmp_path / "pair.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="executing runtime"):
        load_root_feedback_pair(path, expected_release_compatibility=release_compatibility_contract("a" * 64))


def test_source_scale_version_is_distinct_and_legacy_digest_unchanged():
    from gear_sonic.utils.g1_true23_source_action_codec import SOURCE_ACTION_CONVENTION

    old = release_compatibility_contract("386b1bb9ea5b69ccd6fd0283a73ffea1ee052df95564e23a780125fbcbe2c645")
    assert old["contract_sha256"] == "80297f575f66381e57f45085bc7d14e1005bcef50870a78f3b6e38964bc69e13"
    new = release_compatibility_contract(old["source_geometry_sha256"], SOURCE_ACTION_CONVENTION)
    assert validate_release_compatibility(new) == new
    assert new["kind"] == "native23_causal_release_compatibility_v2"
    assert new["contract_sha256"] != old["contract_sha256"]
    broken = {**new, "source_action_codec": {**new["source_action_codec"], "deployment_ready": True}}
    with pytest.raises(ValueError):
        validate_release_compatibility(broken)


def test_source_history_observation_preserves_physical_action_buffer():
    from gear_sonic.envs.mjlab.sonic_true23_release_compatible import source_previous_action
    from gear_sonic.utils.g1_23dof_contract import NATIVE_IL23_TO_CANONICAL_IL29
    from gear_sonic.utils.g1_true23_source_action_codec import (
        SOURCE_ACTION_CONVENTION,
        source_action_history_numpy,
    )

    safe = torch.linspace(-1, 1, 23).reshape(1, 23)
    before = safe.clone()
    action = SimpleNamespace(
        cfg=SimpleNamespace(action_convention=SOURCE_ACTION_CONVENTION), safe_native_action=safe
    )
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda _: action))
    actual = source_previous_action(env)
    history = np.zeros(930, np.float32)
    history[610 + np.asarray(NATIVE_IL23_TO_CANONICAL_IL29)] = safe[0].numpy()
    np.testing.assert_array_equal(actual[0], source_action_history_numpy(history)[610:639])
    torch.testing.assert_close(safe, before, atol=0, rtol=0)
    action.cfg.action_convention = "released_bounded_linear"
    with pytest.raises(ValueError, match="matching"):
        source_previous_action(env)


def test_source_scaled_training_requires_explicit_geometry():
    from gear_sonic.scripts.train_g1_true23_root_feedback import validate_bounds
    from gear_sonic.tests.test_train_g1_true23_root_feedback import arguments

    with pytest.raises(ValueError, match="explicit release source geometry"):
        validate_bounds(arguments("smoke", "--release-action-convention", "released29_scale_bounded_linear_v2"))
