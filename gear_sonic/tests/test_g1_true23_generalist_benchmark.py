"""Offline benchmark accounting and actual native-model integration tests."""

from pathlib import Path
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_generalist_benchmark import (
    run_reference_diagnostic,
    observation_phase_contract,
    compare_original29_source,
    load_generalist_pair,
    summarize_tracking,
    task_points,
    validate_lifecycle_checkpoint,
)


def test_perfect_prefix_never_counts_as_full_motion():
    result = summarize_tracking(np.zeros((2, 5)), completed=2, requested=2, available=100, failure=None)
    assert not result["full_source_motion_completed"]
    assert not result["provisional_reference_landmark_screen_passed"]


def test_full_completion_stays_distinct_from_fidelity_and_lifecycle():
    result = summarize_tracking(np.full((100, 5), 0.3), completed=100, requested=100, available=100, failure=None)
    assert result["full_source_motion_completed"]
    assert not result["provisional_reference_landmark_screen_passed"]
    assert not result["lifecycle_qualified"]
    assert not result["generalization_qualified"]
    assert not result["simulator_qualified"]


def test_perfect_full_tracking_does_not_qualify_deployment():
    result = summarize_tracking(np.zeros((100, 5)), completed=100, requested=100, available=100, failure=None)
    assert result["provisional_reference_landmark_screen_passed"]
    assert not result["deployment_ready"]
    assert not result["original_29dof_source_fidelity_measured"]


@pytest.mark.parametrize("error", [np.nan, np.inf, -0.01])
def test_invalid_tracking_not_silently_accepted(error):
    with pytest.raises(ValueError, match="finite"):
        summarize_tracking(np.full((1, 5), error), completed=1, requested=1, available=1, failure=None)


def test_recorded_failure_prevents_even_last_frame_success():
    result = summarize_tracking(np.zeros((1, 5)), completed=1, requested=1, available=1, failure={"type": "Fall"})
    assert not result["full_source_motion_completed"]


def test_missing_observation_fails():
    with pytest.raises(ValueError, match="every"):
        summarize_tracking(np.zeros((1, 5)), completed=2, requested=2, available=2, failure=None)


def test_landmarks_share_exact_offsets():
    positions = np.zeros((24, 3))
    quaternions = np.tile([1.0, 0, 0, 0], (24, 1))
    np.testing.assert_allclose(
        task_points(positions, quaternions),
        [[0, 0, 0], [0, 0, 0], [0.18, -0.025, 0], [0.18, 0.025, 0], [0, 0, 0.35]],
    )
    with pytest.raises(ValueError, match="native23"):
        task_points(np.zeros((30, 3)), quaternions)


def test_mjlab_checkpoint_cannot_be_misread_as_cpu_lifecycle():
    with pytest.raises(ValueError, match="CPU lifecycle"):
        validate_lifecycle_checkpoint({"kind": "g1_true23_frozen_sonic_lora_training_resume"})


class ZeroPolicy:
    def infer(self, encoder, history):
        return np.zeros(23, dtype=np.float32), np.concatenate((np.zeros(64, dtype=np.float32), history))


@pytest.fixture
def local_case():
    root = Path(__file__).resolve().parents[2]
    assets = root.parent / "GR00T-WholeBodyControl"
    motion = (
        root / "artifacts/g1_true23_frozen_lora/interior_effort_20260906_v1/stationary/stationary_reference.npz"
    )
    if not motion.is_file() or not (assets / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml").is_file():
        pytest.skip("local pinned native model/motion assets unavailable")
    return dict(root=root, asset_root=assets, motion_path=motion)


def test_actual_nominal_native_physics_and_prefix_accounting(local_case):
    report, arrays = run_reference_diagnostic(**local_case, policy=ZeroPolicy(), maximum_controls=3)
    assert report["failure"] is None
    assert report["completed_physics_steps"] == 30
    assert report["completed_controls"] == 3
    assert report["available_controls"] == 500
    assert report["state_pose_writes_after_reset"] == 0
    assert not report["tracking"]["full_source_motion_completed"]
    assert not report["quarter_effort_target_projection"]
    assert report["original_comparison_reproduced_scope"] == "reference_reset_boundary_only"
    assert not report["historical_original_frontend_reproduced"]
    assert report["observation_phase_contract"] == observation_phase_contract()
    np.testing.assert_array_equal(arrays["physics_pre_qpos"][1:], arrays["physics_post_qpos"][:-1])
    np.testing.assert_array_equal(arrays["physics_pre_qvel"][1:], arrays["physics_post_qvel"][:-1])
    np.testing.assert_array_equal(arrays["applied_torque23"], arrays["engine_actuator_force23"])
    np.testing.assert_allclose(np.diff(arrays["physics_time"], axis=1), 0.002, atol=1e-10)
    np.testing.assert_array_equal(arrays["decoder994"][:, 64:], arrays["history930"])
    np.testing.assert_array_equal(
        arrays["history930"][:, 27:30], arrays["physics_pre_qvel"][::10, 3:6].astype(np.float32)
    )
    np.testing.assert_array_equal(
        arrays["torque_saturated23"],
        np.abs(arrays["requested_torque23"]) > np.asarray(report["effort_limit_hardware_nm"]),
    )


def test_nominal_and_historical_are_explicit_different_gain_profiles(local_case):
    native, _ = run_reference_diagnostic(**local_case, policy=ZeroPolicy(), maximum_controls=1)
    historical, _ = run_reference_diagnostic(
        **local_case, policy=ZeroPolicy(), maximum_controls=1, profile="historical_released_gains"
    )
    assert native["initial_state_and_history_sha256"] == historical["initial_state_and_history_sha256"]
    assert native["compiled_model_sha256"] == historical["compiled_model_sha256"]
    assert native["kp_hardware"][0] < historical["kp_hardware"][0]
    assert native["kd_hardware"][4] > historical["kd_hardware"][4]
    assert native["observation_phase_contract"] == historical["observation_phase_contract"]
    assert historical["actuator_profile_scope"] == "gain_arrays_only_not_historical_observation_frontend"
    assert not historical["historical_original_frontend_reproduced"]


def test_observation_phase_contract_keeps_first10_evidence_bounded():
    contract = observation_phase_contract()
    assert contract["current_cvel_phase"] == "same_post_integration_qpos_qvel_instant"
    assert contract["refresh_operations"] == ["mj_kinematics", "mj_comPos", "mj_comVel"]
    assert not contract["extra_integration_or_contact_solve"]
    assert not contract["qacc_warmstart_modified"]
    assert not contract["historical_original_frontend_reproduced"]
    assert not contract["historical_first10_evidence"]["full_horizon_historical_reproduction_proven"]


def test_actor_failure_preserves_empty_evidence_not_success(local_case):
    class BrokenPolicy:
        def infer(self, encoder, history):
            raise RuntimeError("intentional inference failure")

    report, arrays = run_reference_diagnostic(**local_case, policy=BrokenPolicy(), maximum_controls=1)
    assert report["completed_controls"] == 0
    assert report["failure"]["message"] == "intentional inference failure"
    assert arrays["applied_torque23"].shape == (0, 23)
    assert not report["tracking"]["full_source_motion_completed"]


@pytest.fixture
def fake_pair(tmp_path, monkeypatch):
    from gear_sonic.envs.mjlab.sonic_true23_causal_history import causal_history_profile_contract
    from gear_sonic.utils import g1_true23_sonic_library_replay as library

    manifest = dict(
        schema_version=1,
        kind="g1_native23_generalist_diagnostic_pair",
        diagnostic_only=True,
        semantic_profile=causal_history_profile_contract(),
        source=dict(checkpoint_sha256="a" * 64, actor_state_sha256="b" * 64),
        deployment_ready=False,
        promotion_eligible=False,
        hardware_authorized=False,
        active_motor_control_authorized=False,
        completed_motion_qualification=False,
    )
    for key, sizes in (("encoder", (267, 64)), ("decoder", (994, 23))):
        manifest[key] = dict(
            filename=key + ".diagnostic.onnx",
            sha256="c" * 64,
            input_name="input",
            output_name="output",
            input_shape=[1, sizes[0]],
            output_shape=[1, sizes[1]],
            parity=dict(parity_max_abs_error=0),
        )
    metadata = {}
    for key in ("encoder", "decoder"):
        metadata[key] = dict(
            source_checkpoint_sha256="a" * 64,
            actor_state_sha256="b" * 64,
            semantic_contract_sha256=manifest["semantic_profile"]["contract_sha256"],
            artifact_role="native23_generalist_diagnostic_" + key,
            hardware_authorized="false",
            deployment_ready="false",
        )

    def session(key):
        return SimpleNamespace(
            get_modelmeta=lambda: SimpleNamespace(custom_metadata_map=metadata[key]),
            get_inputs=lambda: [SimpleNamespace(name="input")],
            get_outputs=lambda: [SimpleNamespace(name="output")],
        )

    monkeypatch.setattr(
        library,
        "ExactHashSonicPolicy",
        lambda *args, **kwargs: SimpleNamespace(encoder=session("encoder"), decoder=session("decoder")),
    )
    path = tmp_path / "generalist.diagnostic.json"

    def write(value):
        path.write_text(json.dumps(value))
        return path

    return manifest, metadata, write


def test_matched_candidate_metadata_accepted(fake_pair):
    manifest, metadata, write = fake_pair
    _, identity = load_generalist_pair(write(manifest))
    assert identity["source"]["checkpoint_sha256"] == metadata["encoder"]["source_checkpoint_sha256"]


def test_mismatched_encoder_decoder_checkpoint_rejected(fake_pair):
    manifest, metadata, write = fake_pair
    metadata["encoder"]["source_checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="do not share"):
        load_generalist_pair(write(manifest))


@pytest.mark.parametrize("failure", ["semantics", "authorization", "encoder_parity", "path", "shape"])
def test_candidate_manifest_fail_closed(fake_pair, failure):
    original, _, write = fake_pair
    manifest = copy.deepcopy(original)
    if failure == "semantics":
        manifest["semantic_profile"]["contract_sha256"] = "f" * 64
    elif failure == "authorization":
        manifest["hardware_authorized"] = True
    elif failure == "encoder_parity":
        manifest["encoder"]["parity"]["parity_max_abs_error"] = 0.000001
    elif failure == "path":
        manifest["encoder"]["filename"] = "../encoder.onnx"
    else:
        manifest["decoder"]["output_shape"] = [1, 29]
    with pytest.raises(ValueError):
        load_generalist_pair(write(manifest))


def test_source_comparison_keeps_planner_policy_and_retarget_separate(local_case):
    root = local_case["root"]
    folder = root / "artifacts/g1_true23_frozen_lora/original29_neutral_hand_frame_fit_20260906_v1"
    trace_folder = root / "artifacts/g1_true23_frozen_lora/original29_recorded_baseline_20260906_v1"
    if not (folder / "happy_dance.native23.npz").is_file():
        pytest.skip("source-comparison assets unavailable")
    options = {**local_case, "motion_path": folder / "happy_dance.native23.npz"}
    report, arrays = run_reference_diagnostic(**options, policy=ZeroPolicy(), maximum_controls=1)
    result = compare_original29_source(
        report,
        arrays,
        source_trace=trace_folder / "happy_dance.cpp_parameters_and_float32_targets.npz",
        source_model=trace_folder / "original29.mjb",
        retarget_lineage=folder / "happy_dance.report.json",
        asset_root=local_case["asset_root"],
    )
    assert result["measured"]
    assert len(result["comparisons"]) == 4
    assert not result["whole_source_duration_compared"]
    assert not result["evaluated_source29_dynamics"]
    assert all(row["samples"] == 1 for row in result["comparisons"].values())
    assert "executed_native23_vs_original29_planned_choreography" in result["comparisons"]
    assert "adapted_native23_reference_vs_original29_recorded_policy" in result["comparisons"]


def test_reference_diagnostic_preserves_current_library_integration_with_same_policy(local_case, monkeypatch):
    from gear_sonic.utils import g1_true23_sonic_library_replay as library

    policy = ZeroPolicy()
    # Existing runner's constructor contract is bypassed only to inject the
    # same deterministic test actor into both actual native CPU integrations.
    monkeypatch.setattr(library, "ExactHashSonicPolicy", lambda *args, **kwargs: policy)
    _, old_arrays = library.run_library_motion_replay(
        repository_root=local_case["asset_root"], motion_path=local_case["motion_path"], maximum_steps=3
    )
    _, new_arrays = run_reference_diagnostic(**local_case, policy=policy, maximum_controls=3)
    np.testing.assert_array_equal(old_arrays["qpos"], new_arrays["qpos"].astype(np.float32))
