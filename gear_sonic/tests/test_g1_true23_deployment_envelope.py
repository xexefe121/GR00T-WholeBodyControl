"""Numerical contracts for the robot-free deployment envelope diagnosis."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.scripts.evaluate_g1_true23_deployment_envelope import (
    DEFAULT_Q,
    apply_target_envelope,
    load_measured_initial_state,
    measured_ground_contact_height,
    read_cpp_array,
    simulate_sampled_posture_hold,
)
from gear_sonic.utils.g1_23dof_contract import HARDWARE_JOINT_IDS


@pytest.mark.parametrize(
    "options, expected",
    [
        ([], "paired export reports required"),
        (["--encoder-report", "missing.json"], "must be supplied together"),
        (["--allow-unpaired-diagnostic", "--residual-manifest", "missing.json"], "choose exactly one"),
        (["--allow-unpaired-diagnostic", "--project-transition-effort"], "requires --transition-balance-model"),
    ],
)
def test_cli_rejects_unknown_pair_or_ignored_projection_before_loading_assets(
    monkeypatch, capsys, options, expected
):
    import sys

    from gear_sonic.scripts.evaluate_g1_true23_deployment_envelope import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--asset-root",
            "missing",
            "--motion",
            "missing.npz",
            "--decoder",
            "missing.onnx",
            "--expected-decoder-sha256",
            "a" * 64,
            "--output-dir",
            "missing-output",
            *options,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert expected in capsys.readouterr().err


@pytest.fixture
def measured_payload():
    return {
        "kind": "g1_true23_motor_health_readonly_v1",
        "read_only": True,
        "robot_commands_published": False,
        "invalid_samples": 0,
        "valid_crc_samples": 3,
        "advancing_samples": 2,
        "mode_machine": 4,
        "imu_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "motors": [
            {
                "compact_index": index,
                "motor_slot": slot,
                "mode": 1,
                "motorstate": 0,
                "q_rad": float(DEFAULT_Q[index]),
                "dq_rad_s": 0.0,
                "tau_est_nm": 0.0,
            }
            for index, slot in enumerate(HARDWARE_JOINT_IDS)
        ],
    }


def test_measured_state_binds_readonly_source_and_native23_order(tmp_path, measured_payload):
    path = tmp_path / "health.json"
    path.write_text(json.dumps(measured_payload), encoding="utf-8")
    state = load_measured_initial_state(path)
    np.testing.assert_array_equal(state["q"], DEFAULT_Q)
    np.testing.assert_array_equal(state["dq"], np.zeros(23))
    assert state["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "change",
    [
        {"read_only": False},
        {"robot_commands_published": True},
        {"invalid_samples": 1},
        {"invalid_samples": False},
        {"valid_crc_samples": "3"},
        {"advancing_samples": 0},
        {"mode_machine": 0},
        {"motors": []},
        {"imu_quaternion_wxyz": [2.0, 0.0, 0.0, 0.0]},
        {"imu_quaternion_wxyz": [True, False, False, False]},
        {"imu_quaternion_wxyz": [float("nan")] * 4},
    ],
)
def test_invalid_measured_evidence_rejected(tmp_path, measured_payload, change):
    measured_payload.update(change)
    path = tmp_path / "health.json"
    path.write_text(json.dumps(measured_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="measured"):
        load_measured_initial_state(path)


@pytest.mark.parametrize(
    "change",
    [
        {"motor_slot": 13},
        {"compact_index": True},
        {"mode": 0},
        {"motorstate": 1 << 30},
        {"q_rad": float("nan")},
        {"dq_rad_s": True},
        {"tau_est_nm": "0.0"},
    ],
)
def test_invalid_measured_motor_rejected(tmp_path, measured_payload, change):
    measured_payload["motors"][13].update(change)
    path = tmp_path / "health.json"
    path.write_text(json.dumps(measured_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="measured motor"):
        load_measured_initial_state(path)


@pytest.mark.parametrize("duration", [-0.1, 2.1, 0.001, float("nan"), float("inf")])
def test_invalid_hold_duration_rejected_before_physics(duration):
    controller = SimpleNamespace(physics=SimpleNamespace(timestep_s=0.002))
    with pytest.raises(ValueError, match="hold duration"):
        simulate_sampled_posture_hold(controller, kp=np.ones(23), kd=np.ones(23), duration_s=duration)


def test_sampled_hold_uses_initial_pose_and_tracks_guard():
    data = SimpleNamespace(
        qpos=np.r_[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, DEFAULT_Q], qvel=np.zeros(29), ctrl=np.zeros(23)
    )
    initial = copy.deepcopy(data)

    def physics_step(model, state):
        state.qpos[10] += 0.001

    controller = SimpleNamespace(
        physics=SimpleNamespace(timestep_s=0.002, decimation=10, effort=np.ones(23) * 0.01),
        data=data,
        model=None,
        module=SimpleNamespace(mj_step=physics_step),
        history=[np.zeros(93) for _ in range(10)],
        _policy_frame=lambda: np.zeros(93),
    )
    report, trajectory = simulate_sampled_posture_hold(controller, kp=np.ones(23), kd=np.ones(23), duration_s=0.02)
    np.testing.assert_array_equal(trajectory[0], initial.qpos)
    assert trajectory.shape == (11, 30)
    assert report["completed_physics_steps"] == 10
    assert report["maximum_knee_flexion_delta_rad"] == pytest.approx(0.01)
    assert report["first_tenth_effort_guard_violation"] is not None
    assert not report["mode_transfer_simulated"]
    assert data.ctrl[3] == pytest.approx(-0.009)


def test_measured_contact_uses_collision_geometry_not_ankle_origin():
    import mujoco

    from gear_sonic.scripts.evaluate_g1_true23_deployment_envelope import MODEL

    root = Path(__file__).resolve().parents[2]
    model_path = root.parent / "GR00T-WholeBodyControl" / MODEL
    if not model_path.is_file():
        pytest.skip("local native23 mesh assets unavailable")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    controller = SimpleNamespace(model=model, module=mujoco)
    height, report = measured_ground_contact_height(controller, DEFAULT_Q, [1.0, 0.0, 0.0, 0.0])
    assert 0.2 < height < 0.95
    assert min(report["foot_sphere_clearances_m"]) == pytest.approx(0.0, abs=1e-12)
    assert len(report["foot_sphere_clearances_m"]) == 8
    assert not report["root_position_measured"] and not report["gantry_forces_modeled"]


def test_target_scaling_is_default_relative_and_slew_is_per_physics_step():
    full = DEFAULT_Q + np.linspace(-1.0, 1.0, 23)
    last = DEFAULT_Q.copy()
    target, requested = apply_target_envelope(
        full, last, fraction=0.6, joint_scale=np.ones(23), slew_rate=5.0, dt=0.002
    )
    np.testing.assert_allclose(requested, DEFAULT_Q + 0.6 * np.linspace(-1.0, 1.0, 23))
    assert np.max(np.abs(target - last)) == pytest.approx(0.01)
    target2, _ = apply_target_envelope(
        full, target, fraction=0.6, joint_scale=np.ones(23), slew_rate=5.0, dt=0.002
    )
    assert np.max(np.abs(target2 - last)) == pytest.approx(0.02)


def test_full_authority_without_slew_reproduces_simulator_target():
    full = DEFAULT_Q + np.linspace(-0.1, 0.1, 23)
    target, _ = apply_target_envelope(
        full, DEFAULT_Q, fraction=1.0, joint_scale=np.ones(23), slew_rate=None, dt=0.002
    )
    np.testing.assert_array_equal(target, full)


def test_cpp_profile_reader_rejects_missing_or_wrong_width():
    with pytest.raises(ValueError, match="missing"):
        read_cpp_array("", "kStageOneKp")
    with pytest.raises(ValueError, match="invalid"):
        read_cpp_array("kStageOneKp = {1., 2.};", "kStageOneKp")


@pytest.mark.parametrize(
    "fraction,slew,dt",
    [(0.0, 5.0, 0.002), (1.1, 5.0, 0.002), (0.6, 0.0, 0.002), (0.6, -1.0, 0.002), (0.6, 5.0, 0.0)],
)
def test_invalid_envelope_rejected(fraction, slew, dt):
    with pytest.raises(ValueError):
        apply_target_envelope(
            DEFAULT_Q, DEFAULT_Q, fraction=fraction, joint_scale=np.ones(23), slew_rate=slew, dt=dt
        )


def test_unlimited_reference_cell_matches_existing_simulator(monkeypatch):
    """Run actual physics, when the local hash-pinned asset bundle is present."""
    from gear_sonic.scripts.evaluate_g1_true23_deployment_envelope import ENCODER, _policy, run_case
    from gear_sonic.utils import g1_true23_sonic_library_replay as library

    root = Path(__file__).resolve().parents[2]
    assets = root.parent / "GR00T-WholeBodyControl"
    motion_path = (
        assets / "artifacts/g1_true23/sonic_library_true23_happy_physical_reference_v1/happy_dance.true23.npz"
    )
    decoder = (
        root / "artifacts/g1_true23_frozen_lora/original_sonic_happy_residual_v1/candidate.plus_0p002.decoder.onnx"
    )
    if not all(path.is_file() for path in (motion_path, decoder, assets / ENCODER)):
        pytest.skip("local pinned simulation assets unavailable")
    decoder_hash = "44d1fb2701f1e65460f1c2c23f676bce4f1d4a44b3b112798dc5034af37946b8"
    policy = _policy(assets / ENCODER, decoder, decoder_hash)
    monkeypatch.setattr(library, "ExactHashSonicPolicy", lambda *args, **kwargs: policy)
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    original, original_arrays = library.run_library_motion_replay(
        repository_root=assets,
        motion_path=motion_path,
        maximum_steps=535,
        decoder_path=decoder,
        expected_decoder_sha256=decoder_hash,
        gain_profile="released_retained",
    )
    candidate, candidate_arrays = run_case(
        root=root,
        asset_root=assets,
        policy=policy,
        motion=motion,
        kp=library.RELEASED_RETAINED_KP,
        kd=library.RELEASED_RETAINED_KD,
        fraction=1.0,
        joint_scale=np.ones(23),
        ankle_effort=35.0,
        slew_rate=None,
        initial_state="reference",
        maximum_steps=535,
    )
    assert candidate["completed_transitions"] == original["completed_transitions"]
    assert candidate["completed_transitions"] >= 30
    np.testing.assert_array_equal(candidate_arrays["qpos"].astype(np.float32), original_arrays["qpos"])


def test_active_projection_failure_preserves_partial_terminal_state(monkeypatch):
    from gear_sonic.scripts import evaluate_g1_true23_deployment_envelope as envelope

    root = Path(__file__).resolve().parents[2]
    assets = root.parent / "GR00T-WholeBodyControl"
    motion_path = (
        assets / "artifacts/g1_true23/sonic_library_true23_happy_physical_reference_v1/happy_dance.true23.npz"
    )
    if not motion_path.is_file():
        pytest.skip("local motion asset unavailable")
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {name: archive[name].copy() for name in archive.files}
    calls = []

    def fail_second_substep(requested, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("empty effort/position/slew target intersection")
        return requested.copy()

    monkeypatch.setattr(envelope, "effort_feasible_target", fail_second_substep)
    policy = SimpleNamespace(infer=lambda *_: (np.zeros(23, dtype=np.float32), None))
    report, arrays = envelope.run_case(
        root=root,
        asset_root=assets,
        policy=policy,
        motion=motion,
        kp=np.ones(23),
        kd=np.ones(23),
        fraction=1.0,
        joint_scale=np.ones(23),
        ankle_effort=35.0,
        slew_rate=5.0,
        initial_state="reference",
        maximum_steps=1,
        project_active_effort=True,
    )
    assert len(calls) == 2 and report["completed_transitions"] == 0
    assert report["completed_active_physics_steps"] == report["active_partial_transition_substeps"] == 1
    assert report["active_elapsed_simulation_s"] == 0.002
    assert not report["motion_fidelity"]["passed"]
    assert arrays["qpos"].shape == (1, 30)
    assert not np.array_equal(arrays["qpos"][-1], arrays["terminal_active_qpos"])
