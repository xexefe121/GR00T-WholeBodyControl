from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.scripts import record_g1_true23_saved_teleop_diagnostic as diagnostic


@pytest.mark.parametrize("magnitude", [0.1, 9.9])
def test_source_action_ablation_matches_existing_codec_and_records_projection(magnitude):
    source = np.full(23, magnitude, dtype=np.float32)
    history = np.ones(930, dtype=np.float32)
    before = history.copy()
    encoder = np.zeros(267, dtype=np.float32)

    def infer(actual_encoder, actual_history):
        np.testing.assert_array_equal(actual_encoder, encoder)
        np.testing.assert_array_equal(actual_history, diagnostic.source_action_history_numpy(history))
        return source.copy(), np.zeros(994, dtype=np.float32)

    wrapped = diagnostic.SourceActionDiagnosticPolicy(SimpleNamespace(infer=infer))
    raw, decoder = wrapped.infer(encoder, history)
    expected, projection = diagnostic.source_scaled_precompensation(source)
    np.testing.assert_array_equal(raw, expected)
    np.testing.assert_array_equal(history, before)
    assert decoder.shape == (994,)
    assert np.max(np.abs(raw)) < 10
    arrays = wrapped.trace_arrays()
    np.testing.assert_array_equal(arrays["source_raw_action23_native"], source[None])
    np.testing.assert_array_equal(arrays["source_projection_delta_rad23_native"], projection[None])
    report = wrapped.evidence()
    assert report["policy_queries_recorded"] == 1
    assert report["projected_control_count"] == int(magnitude > 1)
    assert report["training_runtime_semantics_match_claimed"] is False
    assert report["hardware_authorized"] is False


def test_source_action_ablation_rejects_invalid_request_and_keeps_empty_trace():
    wrapped = diagnostic.SourceActionDiagnosticPolicy(
        SimpleNamespace(infer=lambda *args: (np.full(23, np.inf, np.float32), np.zeros(994)))
    )
    with pytest.raises(RuntimeError, match="source action diagnostic rejected"):
        wrapped.infer(np.zeros(267, np.float32), np.zeros(930, np.float32))
    assert wrapped.trace_arrays()["source_raw_action23_native"].shape == (0, 23)
    assert wrapped.evidence()["maximum_absolute_projection_rad"] == 0


def test_legacy_profile_requires_explicit_unpaired_diagnostic_flag():
    arguments = [
        "--repository-root",
        "/assets",
        "--decoder-report",
        "/decoder.json",
        "--candidate-summary",
        "/candidate.json",
        "--packets",
        "/packets.json",
        "--output-directory",
        "/output",
    ]
    with pytest.raises(SystemExit):
        diagnostic.parse_args(arguments)
    args = diagnostic.parse_args([*arguments, "--legacy-unpaired-diagnostic"])
    assert args.legacy_unpaired_diagnostic
    with pytest.raises(SystemExit):
        diagnostic.parse_args([*arguments, "--encoder-report", "/encoder.json"])


@pytest.mark.parametrize("original", [False, True])
def test_paired_controller_uses_both_bound_models_and_unchanged_guards(monkeypatch, original):
    pair = dict(
        encoder=dict(path="/encoder.onnx", sha256="a" * 64),
        decoder=dict(path="/decoder.onnx", sha256="b" * 64, report_path="/decoder.json", report_sha256="c" * 64),
        source=dict(update_count=25),
    )
    args = SimpleNamespace(
        legacy_unpaired_diagnostic=False,
        repository_root=Path("/assets"),
        encoder_report=Path("/encoder.json"),
        decoder_report=Path("/decoder.json"),
        original_native23_v14_diagnostic=original,
    )
    monkeypatch.setattr(diagnostic, "load_diagnostic_pair", lambda *args: pair)
    captured = {}

    def policy(**kwargs):
        captured["policy"] = kwargs
        return "validated_pair_policy"

    def controller(**kwargs):
        captured["controller"] = kwargs
        return SimpleNamespace(use_released_retained_gains=lambda: captured.update(released_gains=True))

    monkeypatch.setattr(diagnostic, "ExactHashSonicPolicy", policy)
    monkeypatch.setattr(diagnostic, "SupervisedCleanTrue23MujocoController", controller)
    monkeypatch.setattr(diagnostic, "UnitreeZeroVelocityFallbackPolicy", lambda path: "unchanged_fallback")
    _, identity = diagnostic.build_recording_controller(args)
    assert captured["policy"]["expected_encoder_sha256"] == (
        diagnostic.ORIGINAL_WALK_ENCODER_SHA256 if original else "a" * 64
    )
    assert captured["policy"]["expected_decoder_sha256"] == (
        diagnostic.ORIGINAL_WALK_DECODER_SHA256 if original else "b" * 64
    )
    assert captured["controller"]["minimum_base_height_m"] == 0.30
    assert captured["controller"]["maximum_base_tilt_rad"] == 1.0
    assert captured["controller"]["fallback_tilt_trigger_rad"] == 0.50
    assert captured["controller"]["fallback_policy"] == "unchanged_fallback"
    assert captured["released_gains"] is True
    assert identity["diagnostic_pair"] == (None if original else pair)
    assert identity["original_native23_v14_diagnostic"] is original
    assert identity["legacy_unpaired_diagnostic"] is False
    if original:
        assert captured["policy"]["decoder_path"] == Path("/assets") / diagnostic.ORIGINAL_WALK_DECODER


def test_original_baseline_is_explicit_and_cannot_mix_or_replace_paired_arguments():
    base = ["--repository-root", "/assets", "--packets", "/packets.json", "--output-directory", "/output"]
    args = diagnostic.parse_args([*base, "--original-native23-v14-diagnostic"])
    assert args.original_native23_v14_diagnostic
    for extra in (
        [],
        ["--encoder-report", "/encoder.json"],
        ["--original-native23-v14-diagnostic", "--decoder-report", "/decoder.json"],
        ["--original-native23-v14-diagnostic", "--encoder-report", "/encoder.json"],
        ["--original-native23-v14-diagnostic", "--candidate-summary", "/summary.json"],
    ):
        with pytest.raises(SystemExit):
            diagnostic.parse_args([*base, *extra])


def test_source_orientation_probe_removes_only_initial_yaw_and_preserves_turn():
    packets = [
        dict(
            reference_anchor_quaternion_xyzw=Rotation.from_euler("z", yaw).as_quat().tolist(),
            unchanged_joint_values=[0.1, 0.2],
            control_monotonic_ns=100 + i * 20,
        )
        for i, yaw in enumerate([0.7, 1.2])
    ]
    original_packets = deepcopy(packets)

    def original(packet):
        result = deepcopy(packet)
        result["reference_anchor_quaternion_xyzw"] = [0, 0, 0, 1]
        return result

    controller = SimpleNamespace(retarget_pico_reference_packet=original)
    diagnostic.preserve_calibrated_source_orientation(controller, packets[0])
    for packet, yaw in zip(packets, [0.0, 0.5]):
        result = controller.retarget_pico_reference_packet(packet)
        actual = Rotation.from_quat(result.pop("reference_anchor_quaternion_xyzw"))
        np.testing.assert_allclose(actual.as_matrix(), Rotation.from_euler("z", yaw).as_matrix(), atol=1e-14)
        expected = deepcopy(packet)
        expected.pop("reference_anchor_quaternion_xyzw")
        assert result == expected
    assert packets == original_packets


def test_virtual_source_changes_only_vr_terms_and_rejects_changed_reference(monkeypatch):
    packets = [
        dict(control_source_frame_index=10 + i, q_ref23_native=(np.arange(23) * 0.01 + i).tolist())
        for i in range(2)
    ]
    untouched = deepcopy(packets)
    monkeypatch.setattr(
        diagnostic, "validate_reference_terms", lambda p: dict(control_index=p["control_source_frame_index"])
    )

    def geometry(motion, path):
        expected = np.asarray([p["q_ref23_native"] for p in packets])[:, diagnostic.NATIVE_TO_MJ]
        np.testing.assert_array_equal(motion["joint_pos"], expected)
        return np.asarray([np.arange(21) + i for i in range(2)], dtype=np.float32)

    monkeypatch.setattr(diagnostic, "virtual_source_vr_terms", geometry)
    controller = SimpleNamespace(retarget_pico_reference_packet=deepcopy)
    diagnostic.preserve_virtual_source_geometry(controller, packets, Path("/source.xml"))
    for i, packet in enumerate(packets):
        result = controller.retarget_pico_reference_packet(packet)
        assert result.pop("vr_3point_local_target") == (np.arange(9) + i).tolist()
        assert result.pop("vr_3point_local_orn_target") == (np.arange(9, 21) + i).tolist()
        assert result == untouched[i]
    assert packets == untouched
    packets[0]["q_ref23_native"][0] += 0.001
    with pytest.raises(ValueError, match="reference changed"):
        controller.retarget_pico_reference_packet(packets[0])


@pytest.mark.parametrize("failure", [False, True, "preintegration"])
def test_records_success_or_failed_integrated_state_without_retry(monkeypatch, failure):
    qpos = np.zeros(30)
    qpos[2:4] = [0.75, 1.0]
    controller = SimpleNamespace(
        data=SimpleNamespace(qpos=qpos, qvel=np.zeros(29), time=0.0),
        completed=0,
        fallback_active=False,
        fallback_trigger=None,
        fallback_transition=None,
    )
    monkeypatch.setattr(diagnostic, "initialize_live_controller", lambda *args: None)

    def step(controller, packet):
        if failure == "preintegration" and controller.completed == 1:
            raise RuntimeError("student emitted nonfinite or clipped raw action")
        controller.completed += 1
        controller.data.time += 0.02
        if failure and controller.completed == 2:
            controller.data.qpos[2] = 0.4
            controller.fallback_active = True
            controller.fallback_trigger = "base_tilt"
            controller.fallback_transition = 1
            raise RuntimeError("fallback physical gate failed")

    monkeypatch.setattr(diagnostic, "step_live_packet", step)
    packets = [dict(control_source_frame_index=i + 10) for i in range(4)]
    arrays, report = diagnostic.record(controller, packets)
    assert report["passed"] is (not failure)
    assert report["successful_controls"] == (1 if failure else 4)
    assert controller.completed == (1 if failure == "preintegration" else 2 if failure else 4)
    assert len(arrays["qpos"]) == controller.completed + 1
    assert report["minimum_base_height_m"] == (0.4 if failure is True else 0.75)
    assert report["failed_control_source_frame_index"] == (11 if failure else None)
    expected_integrated = False if failure == "preintegration" else True if failure else None
    assert report["failed_attempt_integrated"] is expected_integrated
    assert report["attempted_controls"] == (2 if failure else 4)
    np.testing.assert_allclose(arrays["simulation_time"], np.arange(controller.completed + 1) * 0.02)
