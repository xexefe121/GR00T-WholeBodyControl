"""Root feedback CPU ABI, causal feature parity, and real integration tests."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_generalist_benchmark import run_reference_diagnostic
from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract, root_feedback_torch
from gear_sonic.utils.g1_true23_root_feedback_benchmark import (
    ForcePulse,
    RootFeedbackRuntimeAdapter,
    load_root_feedback_pair,
    run_root_feedback_diagnostic,
)


class ZeroPolicy:
    def infer(self, encoder, history, feedback=None):
        return np.zeros(23, dtype=np.float32), np.r_[np.zeros(64, dtype=np.float32), history]


@pytest.fixture
def local_case():
    root = Path(__file__).resolve().parents[2]
    assets = root.parent / "GR00T-WholeBodyControl"
    motion = (
        root / "artifacts/g1_true23_frozen_lora/interior_effort_20260906_v1/stationary/stationary_reference.npz"
    )
    if not motion.is_file() or not (assets / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml").is_file():
        pytest.skip("pinned local native23 model/motion unavailable")
    return dict(root=root, asset_root=assets, motion_path=motion)


def test_no_force_adapter_preserves_exact_existing_default_physics(local_case):
    original, old = run_reference_diagnostic(**local_case, policy=ZeroPolicy(), maximum_controls=3)
    result, new = run_root_feedback_diagnostic(**local_case, policy=ZeroPolicy(), maximum_controls=3)
    assert result["failure"] is None and original["failure"] is None
    for key in old:
        np.testing.assert_array_equal(new[key], old[key])
    assert result["initial_state_and_history_sha256"] == original["initial_state_and_history_sha256"]
    assert "runtime_adapter" not in original
    assert not result["tracking"]["full_source_motion_completed"]
    assert not result["runtime_adapter"]["hardware_state_estimation_validated"]
    assert result["state_pose_writes_after_reset"] == 0


def test_root_runtime_uses_current_yaw_q10_proof_and_matches_torch():
    adapter = RootFeedbackRuntimeAdapter()
    angle = 0.7
    state = np.r_[[3.0, -2.0, 0.8], [np.cos(angle / 2), 0, 0, np.sin(angle / 2)], np.zeros(23)]
    velocity = np.r_[[0.2, -0.1, 0.03], np.zeros(26)]
    desired = np.array([3.02, -1.98, 0.81], dtype=np.float32)
    previous = np.array([3.0, -2.0, 0.8], dtype=np.float32)
    original_state = state.copy()
    adapter.infer(
        ZeroPolicy(),
        np.zeros(267, dtype=np.float32),
        np.zeros(930, dtype=np.float32),
        control_index=0,
        desired_position_w=desired,
        previous_desired_position_w=previous,
        measured_qpos=state,
        measured_qvel=velocity,
    )
    arrays = adapter.arrays()
    expected = root_feedback_torch(
        *[
            torch.from_numpy(value)
            for value in (
                desired,
                state[:3].astype(np.float32),
                (desired - previous) / np.float32(0.02),
                velocity[:3].astype(np.float32),
                state[3:7].astype(np.float32),
            )
        ]
    ).numpy()
    np.testing.assert_allclose(arrays["root_feedback9"][0], expected, rtol=0, atol=2e-7)
    np.testing.assert_array_equal(state, original_state)
    state[:2] += [8, -3]
    adapter.infer(
        ZeroPolicy(),
        np.zeros(267, dtype=np.float32),
        np.zeros(930, dtype=np.float32),
        control_index=1,
        desired_position_w=desired,
        previous_desired_position_w=previous,
        measured_qpos=state,
        measured_qvel=velocity,
    )
    assert np.linalg.norm(adapter.arrays()["root_feedback9"][1, :3] - expected[:3]) > 8
    np.testing.assert_array_equal(adapter.arrays()["root_feedback9"][1, 3:], expected[3:])


def test_real_external_force_changes_dynamics_without_pose_rewrites(local_case):
    pulse = ForcePulse(20, 5, (10.0, 0.0, 0.0))
    _, nominal = run_root_feedback_diagnostic(**local_case, policy=ZeroPolicy(), maximum_controls=8)
    report, perturbed = run_root_feedback_diagnostic(
        **local_case, policy=ZeroPolicy(), maximum_controls=8, pulses=(pulse,)
    )
    assert report["failure"] is None
    np.testing.assert_array_equal(nominal["qpos"][:3], perturbed["qpos"][:3])
    assert not np.array_equal(nominal["qpos"][3:], perturbed["qpos"][3:])
    np.testing.assert_array_equal(perturbed["physics_pre_qpos"][1:], perturbed["physics_post_qpos"][:-1])
    np.testing.assert_array_equal(perturbed["physics_pre_qvel"][1:], perturbed["physics_post_qvel"][:-1])
    assert np.count_nonzero(np.linalg.norm(perturbed["physics_external_force_world_n"], axis=1)) == 5
    row = report["root_response"]["pulses"][0]
    np.testing.assert_allclose(row["applied_impulse_world_ns"], [0.1, 0, 0], rtol=0, atol=1e-12)
    assert row["scheduled_force_fully_integrated"]
    assert not row["recovery_window_fully_observed"]
    assert not row["perturbation_recovery_qualified"]
    assert not report["tracking"]["full_source_motion_completed"]
    assert not np.array_equal(perturbed["root_feedback9"][3:], nominal["root_feedback9"][3:])


@pytest.mark.parametrize(
    "args",
    [
        (-1, 5, (1, 0, 0)),
        (0, True, (1, 0, 0)),
        (0, 251, (1, 0, 0)),
        (0, 5, (0, 0, 0)),
        (0, 5, (101, 0, 0)),
        (0, 5, (np.nan, 0, 0)),
    ],
)
def test_force_schedule_is_bounded(args):
    with pytest.raises(ValueError):
        ForcePulse(*args)


def test_overlapping_forces_rejected():
    with pytest.raises(ValueError, match="overlap"):
        RootFeedbackRuntimeAdapter((ForcePulse(0, 10, (1, 0, 0)), ForcePulse(9, 3, (1, 0, 0))))


@pytest.fixture
def fake_pair(tmp_path, monkeypatch):
    import onnxruntime
    from gear_sonic.envs.mjlab.sonic_true23_causal_history import causal_history_profile_contract

    manifest = dict(
        schema_version=2,
        kind="g1_native23_root_feedback_diagnostic_pair",
        diagnostic_only=True,
        semantic_profile=causal_history_profile_contract(),
        root_feedback_contract=root_feedback_contract(),
        source=dict(checkpoint_sha256="a" * 64, actor_state_sha256="b" * 64),
        deployment_ready=False,
        promotion_eligible=False,
        hardware_authorized=False,
        active_motor_control_authorized=False,
        completed_motion_qualification=False,
        physical_root_state_estimator_qualified=False,
    )
    metadata = {}
    for key in ("encoder", "decoder"):
        path = tmp_path / (key + ".onnx")
        path.write_bytes(key.encode())
        manifest[key] = dict(filename=path.name, sha256=sha256_file(path), parity=dict(parity_max_abs_error=0))
        metadata[key] = dict(
            source_checkpoint_sha256="a" * 64,
            actor_state_sha256="b" * 64,
            root_feedback_contract_sha256=manifest["root_feedback_contract"]["contract_sha256"],
            semantic_contract_sha256=manifest["semantic_profile"]["contract_sha256"],
            artifact_role="native23_root_feedback_diagnostic_" + key,
            hardware_authorized="false",
            deployment_ready="false",
        )
    manifest["encoder"].update(
        input_name="teleop_obs", input_shape=[1, 267], output_name="token", output_shape=[1, 64]
    )
    manifest["decoder"].update(
        inputs=[dict(name="obs_dict", shape=[1, 994]), dict(name="root_feedback", shape=[1, 9])],
        output_name="action",
        output_shape=[1, 23],
    )

    def node(name, width):
        return SimpleNamespace(name=name, shape=[1, width], type="tensor(float)")

    class Session:
        def __init__(self, path, **_kwargs):
            self.key = Path(path).stem

        def get_inputs(self):
            return (
                [node("teleop_obs", 267)]
                if self.key == "encoder"
                else [node("obs_dict", 994), node("root_feedback", 9)]
            )

        def get_outputs(self):
            return [node("token", 64)] if self.key == "encoder" else [node("action", 23)]

        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map=metadata[self.key])

        def run(self, _outputs, inputs):
            if self.key == "encoder":
                return [np.zeros((1, 64), dtype=np.float32)]
            output = np.zeros((1, 23), dtype=np.float32)
            output[:, :9] = inputs["root_feedback"]
            return [output]

    monkeypatch.setattr(onnxruntime, "InferenceSession", Session)
    path = tmp_path / "root_feedback.diagnostic.json"

    def write(value):
        path.write_text(json.dumps(value))
        return path

    return manifest, metadata, write


def test_new_pair_passes_separate_feedback_without_modifying_994(fake_pair):
    manifest, _, write = fake_pair
    policy, identity = load_root_feedback_pair(write(manifest))
    history = np.arange(930, dtype=np.float32)
    feedback = np.arange(9, dtype=np.float32)
    raw, combined = policy.infer(np.zeros(267, dtype=np.float32), history, feedback)
    np.testing.assert_array_equal(combined[64:], history)
    np.testing.assert_array_equal(raw[:9], feedback)
    assert identity["root_feedback_contract"] == root_feedback_contract()


@pytest.mark.parametrize(
    "kind",
    [
        "old_kind",
        "old_schema",
        "feedback_contract",
        "feedback_metadata",
        "old_decoder",
        "encoder_parity",
        "hash",
        "authority",
        "physical_estimator",
    ],
)
def test_root_pair_fail_closed(fake_pair, kind):
    original, metadata, write = fake_pair
    manifest = copy.deepcopy(original)
    if kind == "old_kind":
        manifest["kind"] = "g1_native23_generalist_diagnostic_pair"
    elif kind == "old_schema":
        manifest["schema_version"] = 1
    elif kind == "feedback_contract":
        manifest["root_feedback_contract"]["dimension"] = 8
    elif kind == "feedback_metadata":
        metadata["decoder"]["root_feedback_contract_sha256"] = "f" * 64
    elif kind == "old_decoder":
        manifest["decoder"]["inputs"] = [dict(name="obs_dict", shape=[1, 994])]
    elif kind == "encoder_parity":
        manifest["encoder"]["parity"]["parity_max_abs_error"] = 1e-6
    elif kind == "hash":
        manifest["decoder"]["sha256"] = "f" * 64
    elif kind == "physical_estimator":
        manifest["physical_root_state_estimator_qualified"] = True
    else:
        manifest["hardware_authorized"] = True
    with pytest.raises(ValueError):
        load_root_feedback_pair(write(manifest))


def test_actual_root0_exporter_manifest_loads_exact_version2_pair():
    """Integration against the real exporter artifact, not a guessed fixture schema."""
    import onnxruntime as ort

    path = (
        Path(__file__).resolve().parents[2]
        / "artifacts/g1_true23_generalist/root_feedback_smoke_20260907_v1/export_initial"
        / "root_feedback.diagnostic.json"
    )
    if not path.is_file():
        pytest.skip("actual root0 diagnostic export not installed in this checkout")
    assert sha256_file(path) == "5fbe2f3f8b6b3fd29eb9258c6a3a6062e4faa8a868a301d5e18511691f7dc569"
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    policy, identity = load_root_feedback_pair(path, session_options=options)
    assert identity["encoder_sha256"] == "e334d6e3d272d764b144cffce7d35f985df653b46ad85bfb18386de1abe207f1"
    assert identity["decoder_sha256"] == "da7abe98adc6345c59ac3a703ae47c53eb8b67d0d7bc05ecd872e7de8c5f0fdc"
    raw, combined = policy.infer(np.zeros(267, np.float32), np.zeros(930, np.float32), np.zeros(9, np.float32))
    assert raw.shape == (23,) and combined.shape == (994,)
    assert np.isfinite(raw).all() and np.isfinite(combined).all()
