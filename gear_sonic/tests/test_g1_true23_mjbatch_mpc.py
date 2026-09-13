"""Native geometry/tangent-state witnesses for the optional offline planner."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("mjbatch")
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker, load_native_bundle, motion_states


@pytest.mark.parametrize("fault", ["clock_reset", "engine_warning"])
def test_physical_engine_fault_stops_at_actual_substep(monkeypatch, tmp_path, fault):
    from types import SimpleNamespace

    from gear_sonic.scripts.evaluate_g1_true23_mjbatch_mpc import run

    original_step = mujoco.mj_step

    def inject_physical_fault(model, data):
        original_step(model, data)
        if fault == "clock_reset":
            data.time = 0.0
        else:
            data.warning.number[0] = 1

    monkeypatch.setattr(mujoco, "mj_step", inject_physical_fault)
    root = Path(__file__).resolve().parents[2]
    result = run(SimpleNamespace(
        bundle=root / "artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1",
        clip="walk002", probe="standing", standing_seconds=0.02, source_seconds=3,
        horizon=2, commit=1, iterations=1, threads=1, feedback_clip=0.1,
        checkpoint_controls=100, motion_override=None, target_seed=None,
        ankle_limit_margin=None, ankle_limit_weight=None, all_joint_limit_margin=None,
        all_joint_limit_weight=None, relative_foot_weight=0, fd_epsilon=1e-6,
        output=tmp_path / fault,
    ))
    assert result["failure"]["kind"] == "engine_warning_or_clock_reset"
    assert result["physics_steps"] == 1 and result["completed_controls"] == 1
    assert result["failure"]["substep"] == 1 and not result["probe_completed"]
    with np.load(tmp_path / fault / "trace.npz") as trace:
        np.testing.assert_array_equal(trace["physics_substeps"], [1])
        assert trace["physics_qpos"].shape == (2, 30)


@pytest.fixture(scope="module")
def tracker():
    root = Path(__file__).resolve().parents[2]
    bundle = root / "artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1"
    native, contract, motion, _, _ = load_native_bundle(bundle, "walk002")
    servo = position_servo_copy(native, contract["kp"], contract["kd"], contract["native_effort"])
    return Native23Tracker(servo, contract, motion, horizon=2, threads=1)


def test_native_tangent_integrate_and_difference_match_mujoco(tracker):
    rng = np.random.default_rng(53)
    qpos = np.tile(tracker.states[10, :30], (12, 1))
    delta = rng.normal(size=(12, 29)) * 0.05
    actual = tracker.integrate(qpos, delta)
    expected = qpos.copy()
    for row in range(len(qpos)):
        mujoco.mj_integratePos(tracker.model, expected[row], delta[row], 1.0)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    old = np.column_stack((qpos, rng.normal(size=(12, 29))))
    new = np.column_stack((actual, rng.normal(size=(12, 29))))
    expected_delta = np.empty((12, 58))
    for row in range(len(qpos)):
        mujoco.mj_differentiatePos(tracker.model, expected_delta[row, :29], 1.0, old[row, :30], new[row, :30])
    expected_delta[:, 29:] = new[:, 30:] - old[:, 30:]
    np.testing.assert_allclose(tracker.difference(old, new), expected_delta, atol=1e-14, rtol=0)


def test_cost_probe_uses_preintegration_kinematics_and_state(tracker):
    states = tracker.states[10:13].copy()
    states[:, 7] += [0.01, -0.02, 0.03]
    states[:, 36] = [0.4, -0.5, 0.6]
    targets = tracker.target_reference(np.arange(2))
    tracker.linearize(states, targets)
    observed = tracker.feat[:, 0].copy()
    expected = tracker.features(states[:-1])
    np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(observed[:, 42:], states[:-1])


def test_reference_frame_count_and_height_are_unmodified(tracker):
    assert len(tracker.reference) == len(tracker.motion["joint_pos"])
    np.testing.assert_array_equal(tracker.states[:, :3], tracker.motion["body_pos_w"][:, 0])
    np.testing.assert_array_equal(tracker.states[:, 7:30], tracker.motion["joint_pos"])
    expected = motion_states(tracker.motion)
    np.testing.assert_array_equal(tracker.reference[:, 42:], expected)
    assert tracker.sub == 10 and tracker.model.opt.timestep == 0.002


def test_reference_body_poses_match_native_forward_kinematics(tracker):
    # A physical mapping witness: complete articulated poses, including walking,
    # must describe this 23-joint model rather than merely have matching shapes.
    data = mujoco.MjData(tracker.model)
    for frame in (10, 350, 400, 500, 800, 1000):
        data.qpos[:] = tracker.states[frame, :30]
        mujoco.mj_kinematics(tracker.model, data)
        np.testing.assert_allclose(data.xpos[1:], tracker.motion["body_pos_w"][frame], atol=1e-13, rtol=0)
        dots = np.abs(np.sum(data.xquat[1:] * tracker.motion["body_quat_w"][frame], axis=-1))
        np.testing.assert_allclose(dots, 1.0, atol=1e-13, rtol=0)


def test_measured_joint_limits_and_speed_increase_cost(tracker):
    reference = tracker.reference[10:11].copy()
    base = np.sum(tracker.residual(0, reference) ** 2)
    damaged = reference.copy()
    damaged[0, 42 + 7 + 5] = tracker.lo[5] - 0.03
    damaged[0, 42 + 36 + 5] = tracker.contract["native_velocity"][5] * 1.1
    assert np.sum(tracker.residual(0, damaged) ** 2) > base + 100


def test_cached_initial_rollout_cannot_start_from_another_physical_state(tracker):
    from gear_sonic.utils.g1_true23_mjbatch_ilqr_core import ilqr

    states = tracker.states[10:13].copy()
    states[0, 0] += 0.001
    with pytest.raises(ValueError, match="actual planning state"):
        ilqr(tracker, tracker.states[10], tracker.target_reference(np.arange(2)), initial_rollout=(states, 0.0))


def test_ankle_only_margin_has_independent_expected_cost_and_derivative(tracker):
    guarded = Native23Tracker(
        tracker.model,
        tracker.contract,
        tracker.motion,
        horizon=2,
        threads=1,
        ankle_limit_margin=0.05,
        ankle_limit_weight=2000,
    )
    feature = tracker.reference[10:11].copy()
    feature[0, 42 + 7 + 5] = tracker.lo[5] + 0.02
    feature[0, 42 + 7 + 11] = tracker.hi[11] - 0.025
    feature[0, 42 + 7] = tracker.lo[0] + 0.005

    def increment(value):
        return float(np.sum(guarded.residual(0, value) ** 2) - np.sum(tracker.residual(0, value) ** 2))

    # Both ankles are inside native ranges but within the declared new margin.
    # The hip lies in its old margin; its cost must cancel exactly.
    assert increment(feature) == pytest.approx(2000 * (0.03**2 + 0.025**2), abs=1e-11)
    direction = np.zeros_like(feature)
    direction[0, 42 + 7 + 5], direction[0, 42 + 7 + 11] = 0.3, -0.4
    h = 1e-4
    plus, minus, center = (
        increment(feature + h * direction),
        increment(feature - h * direction),
        increment(feature),
    )
    assert (plus - minus) / (2 * h) == pytest.approx(-76.0, abs=1e-7)
    assert (plus - 2 * center + minus) / h**2 == pytest.approx(1000.0, abs=1e-4)
    other = np.delete(np.arange(23), [5, 11])
    np.testing.assert_array_equal(guarded.limit_margins[other], 0.01)
    np.testing.assert_array_equal(guarded.limit_weights[other], 200.0)
    np.testing.assert_array_equal(guarded.model.actuator_gainprm, tracker.model.actuator_gainprm)
    np.testing.assert_array_equal(guarded.model.actuator_biasprm, tracker.model.actuator_biasprm)
    np.testing.assert_array_equal(guarded.model.actuator_forcerange, tracker.model.actuator_forcerange)


def test_all_joint_margin_has_independent_expected_cost_gradient_and_curvature(tracker):
    guarded = Native23Tracker(
        tracker.model,
        tracker.contract,
        tracker.motion,
        horizon=2,
        threads=1,
        all_joint_limit_margin=0.05,
        all_joint_limit_weight=2000,
    )
    feature = tracker.reference[10:11].copy()
    lower = np.arange(23) % 2 == 0
    distance = np.where(lower, 0.02, 0.025)
    feature[0, 49:72] = np.where(lower, tracker.lo + distance, tracker.hi - distance)
    # Every joint lies outside the old margin. Independent quadratic penalty
    # is w*(margin-distance)^2, whether approaching its lower or upper bound.
    gap = 0.05 - distance
    inward = np.linspace(0.1, 0.5, 23)
    direction = np.zeros_like(feature)
    direction[0, 49:72] = np.where(lower, inward, -inward)

    def increment(value):
        return float(np.sum(guarded.residual(0, value) ** 2) - np.sum(tracker.residual(0, value) ** 2))

    h = 1e-3
    center = increment(feature)
    plus, minus = increment(feature + h * direction), increment(feature - h * direction)
    assert center == pytest.approx(2000 * np.sum(gap**2), abs=1e-10)
    assert (plus - minus) / (2 * h) == pytest.approx(-4000 * gap @ inward, abs=1e-6)
    assert (plus - 2 * center + minus) / h**2 == pytest.approx(4000 * inward @ inward, abs=1e-3)
    assert guarded.all_joint_limit_override["native_joint_indices"] == list(range(23))
    assert guarded.ankle_limit_override is None
    np.testing.assert_array_equal(guarded.model.actuator_gainprm, tracker.model.actuator_gainprm)
    np.testing.assert_array_equal(guarded.model.actuator_biasprm, tracker.model.actuator_biasprm)
    np.testing.assert_array_equal(guarded.model.actuator_forcerange, tracker.model.actuator_forcerange)


def test_relative_foot_added_gradient_matches_independent_native_body_jacobians(tracker):
    weighted = Native23Tracker(
        tracker.model, tracker.contract, tracker.motion, horizon=2, threads=1, relative_foot_weight=400
    )
    states = tracker.states[410:413].copy()
    states[:, 7 + 3] += 0.05
    states[:, 0] += 0.07
    tracker.window(410)
    weighted.window(410)
    targets = tracker.target_reference(np.arange(2))
    try:
        for current in (tracker, weighted):
            current.linearize(states, targets)
        old_lx = tracker.expand(states, targets)[0]
        new_lx = weighted.expand(states, targets)[0]
        data = mujoco.MjData(tracker.model)
        pelvis = tracker.model.body("pelvis").id
        feet = [tracker.model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
        root_jacobian, foot_jacobian = np.zeros((3, 29)), np.zeros((3, 29))
        for knot, state in enumerate(states):
            data.qpos[:] = state[:30]
            mujoco.mj_kinematics(tracker.model, data)
            mujoco.mj_comPos(tracker.model, data)
            mujoco.mj_jacBody(tracker.model, data, root_jacobian, None, pelvis)
            expected = np.zeros(58)
            reference = tracker.motion["body_pos_w"][410 + knot]
            for foot in feet:
                mujoco.mj_jacBody(tracker.model, data, foot_jacobian, None, foot)
                error = (data.xpos[foot] - data.xpos[pelvis]) - (reference[foot - 1] - reference[pelvis - 1])
                expected[:29] += 800 * (foot_jacobian - root_jacobian).T @ error
            np.testing.assert_allclose(new_lx[knot] - old_lx[knot], expected, atol=3e-5, rtol=1e-4)
            np.testing.assert_allclose((new_lx[knot] - old_lx[knot])[:3], 0, atol=1e-9, rtol=0)
    finally:
        tracker.window(10)


def test_default_limit_cost_and_derivatives_match_frozen_native323_planner(tracker):
    import importlib.util

    frozen = Path(
        "/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_full_v1/"
        "walk002_native323_h30_clip1_v1/g1_true23_mjbatch_mpc_snapshot.py"
    )
    if not frozen.exists():
        pytest.skip("frozen native323 experiment fixture is unavailable")
    spec = importlib.util.spec_from_file_location("frozen_native23_cost", frozen)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    core_spec = importlib.util.spec_from_file_location(
        "frozen_native23_dynamics", frozen.with_name("g1_true23_mjbatch_ilqr_core_snapshot.py")
    )
    old_core = importlib.util.module_from_spec(core_spec)
    core_spec.loader.exec_module(old_core)
    old = module.Native23Tracker(tracker.model, tracker.contract, tracker.motion, horizon=2, threads=1)
    try:
        for control in (0, 300, 400, 600):
            tracker.window(control + 10)
            old.window(control + 10)
            states = tracker.states[control + 10 : control + 13].copy()
            targets = tracker.target_reference(np.arange(2))
            for current in (old, tracker):
                current.linearize(states, targets)
            legacy_dynamics = old_core.Planner.linearize(tracker, states, targets)
            current_dynamics = tracker.linearize(states, targets)
            for before, after in zip(legacy_dynamics, current_dynamics):
                np.testing.assert_array_equal(before, after)
            for before, after in zip(old.expand(states, targets), tracker.expand(states, targets)):
                np.testing.assert_array_equal(before, after)
            np.testing.assert_array_equal(
                old.residual(0, old.features(states)), tracker.residual(0, tracker.features(states))
            )
    finally:
        tracker.window(10)


def test_finite_difference_epsilon_is_local_and_consistent_in_dynamics_and_cost(tracker):
    import importlib.util

    import gear_sonic.utils.g1_true23_mjbatch_ilqr_core as current_core

    directory = Path(
        "/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_full_v1/walk002_native323_h30_clip1_v1"
    )
    if not directory.exists():
        pytest.skip("frozen native323 experiment fixture is unavailable")
    modules = {}
    for name in ("g1_true23_mjbatch_ilqr_core", "g1_true23_mjbatch_mpc"):
        spec = importlib.util.spec_from_file_location("coarse_frozen_" + name, directory / (name + "_snapshot.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.EPS = 1e-3
        modules[name] = module
    legacy = modules["g1_true23_mjbatch_mpc"].Native23Tracker(
        tracker.model, tracker.contract, tracker.motion, horizon=2, threads=1
    )
    coarse = Native23Tracker(
        tracker.model, tracker.contract, tracker.motion, horizon=2, threads=1, fd_epsilon=1e-3
    )
    states = tracker.states[410:413].copy()
    legacy.window(410)
    coarse.window(410)
    targets = coarse.target_reference(np.arange(2))
    expected = modules["g1_true23_mjbatch_ilqr_core"].Planner.linearize(legacy, states, targets)
    actual = coarse.linearize(states, targets)
    for before, after in zip(expected, actual):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(legacy.expand(states, targets), coarse.expand(states, targets)):
        np.testing.assert_array_equal(before, after)
    assert coarse.fd_epsilon == 1e-3
    assert tracker.fd_epsilon == current_core.EPS == 1e-6


def test_portable_retarget_false_fk_receipt_cannot_replace_measurement(tmp_path):
    import json
    import shutil

    from gear_sonic.utils.g1_true23_mjbatch_mpc import load_motion_override, sha256

    pack = Path("/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_inputs_v1/walk002")
    if not pack.exists():
        pytest.skip("portable simulation experiment fixture is unavailable")
    bundle = Path(__file__).resolve().parents[2] / "artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1"
    native, contract, original, timeline, manifest = load_native_bundle(bundle, "walk002")
    for name in ("reference.npz", "portable_receipt.json", "report.json"):
        shutil.copyfile(pack / name, tmp_path / name)
    path = tmp_path / "reference.npz"
    with np.load(path, allow_pickle=False) as archive:
        damaged = {name: archive[name].copy() for name in archive.files}
    # Keep native bounds, finite samples, derivative contract and receipt hash
    # valid while breaking the claimed articulated geometry at one real frame.
    damaged["joint_pos"][100, 0] += 1e-5
    damaged["joint_vel"] = np.gradient(damaged["joint_pos"], 0.02, axis=0)
    np.savez_compressed(path, **damaged)
    receipt = json.loads((tmp_path / "portable_receipt.json").read_text())
    receipt["reference_sha256"] = sha256(path)
    (tmp_path / "portable_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="FK position mismatch"):
        load_motion_override(path, bundle, "walk002", native, contract, original, timeline, manifest)


def test_declared_floor_transform_cannot_hide_joint_change(tracker, tmp_path):
    import json

    from gear_sonic.utils.g1_true23_mjbatch_mpc import sha256, validate_floor_transform

    before = {name: value.copy() if name == "fps" else value[:3].copy() for name, value in tracker.motion.items()}
    candidate = {name: value.copy() for name, value in before.items()}
    candidate["body_pos_w"][:, :, 2] += 0.1
    data = mujoco.MjData(tracker.model)
    data.qpos[:] = tracker.states[0, :30]
    mujoco.mj_kinematics(tracker.model, data)
    clearance = []
    for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        body = tracker.model.body(name).id
        spheres = [
            i
            for i in range(tracker.model.ngeom)
            if tracker.model.geom_bodyid[i] == body
            and tracker.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
            and tracker.model.geom_contype[i] != 0
        ]
        clearance.append(float(np.min(data.geom_xpos[spheres, 2] - tracker.model.geom_size[spheres, 0])))
    clearance = np.tile(clearance, (3, 1))
    np.savez_compressed(tmp_path / "before_floor_reference.npz", **before)
    (tmp_path / "floor_transform_receipt.json").write_text(
        json.dumps(
            dict(
                input_reference_file="before_floor_reference.npz",
                input_reference_sha256=sha256(tmp_path / "before_floor_reference.npz"),
            )
        )
    )
    np.savez_compressed(
        tmp_path / "frame_lift.npz",
        frame_lift_m=np.full(3, 0.1),
        raw_required_lift_m=np.maximum(0, -clearance.min(axis=1)),
        before_foot_clearance_m=clearance,
        after_foot_clearance_m=clearance + 0.1,
    )
    metadata = dict(
        kind="causal_upward_whole_pose_translation",
        physical_floor_changed=False,
        root_relative_intent_preserved=True,
        joints_unchanged=True,
        source_timing_changed=False,
        transform_receipt_file="floor_transform_receipt.json",
        transform_receipt_sha256=sha256(tmp_path / "floor_transform_receipt.json"),
        frame_lift_file="frame_lift.npz",
        frame_lift_sha256=sha256(tmp_path / "frame_lift.npz"),
        lift_min_m=0.1,
        lift_max_m=0.1,
    )
    result = validate_floor_transform(tmp_path, metadata, tracker.model, candidate)
    assert result["independent_numeric_whole_pose_and_native_clearance_validation"] is True
    with pytest.raises(ValueError, match="output reference hash mismatch"):
        validate_floor_transform(tmp_path, metadata, tracker.model, candidate, reference_sha256="wrong-output")
    candidate["joint_pos"][1, 3] += 0.01
    with pytest.raises(AssertionError, match="floor transform changed joint_pos"):
        validate_floor_transform(tmp_path, metadata, tracker.model, candidate)


def test_checkpoint_interrupted_write_preserves_previous_complete_archive(tmp_path, monkeypatch):
    from gear_sonic.scripts.evaluate_g1_true23_mjbatch_mpc import atomic_trace

    path = tmp_path / "trace.partial.npz"
    atomic_trace(path, {"qpos": np.arange(60).reshape(2, 30)}, checkpoint_metadata=np.asarray("generation1"))
    previous = path.read_bytes()
    save = np.savez_compressed

    def interrupted(handle, **_):
        handle.write(b"incomplete archive")
        raise OSError("simulated disk-write interruption")

    monkeypatch.setattr(np, "savez_compressed", interrupted)
    with pytest.raises(OSError, match="simulated disk-write interruption"):
        atomic_trace(path, {"qpos": np.zeros((3, 30))})
    assert path.read_bytes() == previous
    with np.load(path, allow_pickle=False) as saved:
        assert saved["checkpoint_metadata"].item() == "generation1"
        np.testing.assert_array_equal(saved["qpos"], np.arange(60).reshape(2, 30))
    monkeypatch.setattr(np, "savez_compressed", save)
    atomic_trace(path, {"qpos": np.zeros((3, 30))}, checkpoint_metadata=np.asarray("generation2"))
    with np.load(path, allow_pickle=False) as saved:
        assert saved["checkpoint_metadata"].item() == "generation2"
        assert saved["qpos"].shape == (3, 30)
