"""Behavioral safety witnesses for opt-in native23 MPC feasibility handling."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

pytest.importorskip("mjbatch")
from gear_sonic.utils import g1_true23_mjbatch_ilqr_core as core
from gear_sonic.utils.g1_true23_mjbatch_model import (
    Native23Feasibility,
    position_servo_copy,
    preview_native_control,
)
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker, load_native_bundle


@pytest.fixture(scope="module")
def native_case():
    bundle = Path(__file__).resolve().parents[2] / "artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1"
    return load_native_bundle(bundle, "walk002")[:3]


def tracker_for(case, hard=True):
    model, contract, motion = case
    servo = position_servo_copy(model, contract["kp"], contract["kd"], contract["native_effort"])
    return Native23Tracker(servo, contract, motion, horizon=2, threads=1, hard_feasibility=hard)


def test_default_off_solver_and_rollout_are_bit_exact_to_frozen_run(native_case):
    path = Path("/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_full_v1/"
                "pico_v4_native323_freshseed_allmargin_5iter_full_v1/final_frozen/"
                "g1_true23_mjbatch_ilqr_core_snapshot.py")
    if not path.exists():
        pytest.skip("frozen native323 fixture unavailable")
    spec = importlib.util.spec_from_file_location("old_feasibility_default", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    before, after = tracker_for(native_case, False), tracker_for(native_case, False)
    state = after.states[10]
    targets = after.target_reference(np.arange(2))
    for expected, actual in zip(old.ilqr(before, state, targets.copy(), iters=2),
                                core.ilqr(after, state, targets.copy(), iters=2)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("damage", ["position", "speed", "nan", "target_nan"])
def test_invalid_seed_never_reaches_derivatives_or_infinite_acceptance(native_case, monkeypatch, damage):
    tracker = tracker_for(native_case)
    state = tracker.states[10].copy()
    targets = tracker.target_reference(np.arange(2))
    if damage == "position":
        state[7] = tracker.hi[0] + 2e-6
    elif damage == "speed":
        state[36] = tracker.contract["native_velocity"][0] * (1 + 1e-10)
    elif damage == "nan":
        state[7] = np.nan
    else:
        targets[0, 0] = np.nan
    monkeypatch.setattr(tracker, "linearize", lambda *_: pytest.fail("infeasible seed reached derivative pass"))
    with pytest.raises(core.NoFeasiblePlan) as captured:
        core.ilqr(tracker, state, targets)
    assert captured.value.diagnostics["status"] == "no_feasible_seed"
    assert not any(captured.value.diagnostics["initial_rollout"]["feasible"])


def test_hard_and_unchecked_quiet_rollout_match_bit_exactly(native_case):
    hard, old = tracker_for(native_case), tracker_for(native_case, False)
    targets = hard.target_reference(np.arange(2))
    for expected, actual in zip(old.rollout(old.states[10], targets), hard.rollout(hard.states[10], targets)):
        np.testing.assert_array_equal(actual, expected)
    assert all(hard.last_rollout_feasibility["feasible"])


@pytest.mark.parametrize("fault", ["joint_range", "engine_warning", "clock_reset_or_nonfinite"])
def test_bad_intermediate_batch_lane_is_latched_without_rejecting_other_lanes(native_case, monkeypatch, fault):
    tracker = tracker_for(native_case)
    original = type(tracker.line).step
    calls = 0

    def inject(batch, *args, **kwargs):
        nonlocal calls
        original(batch, *args, **kwargs)
        if batch is not tracker.line:
            return
        calls += 1
        if calls == 3:
            if fault == "joint_range":
                batch.bind("qpos")[0, 7] = tracker.hi[0] + 2e-6
            elif fault == "engine_warning":
                batch.bind("warning")[0, 0, 1] = 1
            else:
                batch.bind("time")[0] = 0

    monkeypatch.setattr(type(tracker.line), "step", inject)
    _, _, costs = tracker.rollout(tracker.states[10], tracker.target_reference(np.arange(2)))
    evidence = tracker.last_rollout_feasibility
    assert np.isinf(costs[0]) and np.isfinite(costs[1:]).all()
    assert evidence["feasible"] == [False] + [True] * 8
    assert evidence["first_violation"][0]["control"] == 0
    assert evidence["first_violation"][0]["substep"] == 3
    assert fault in evidence["first_violation"][0]["reasons"]


def test_rejected_sweeps_preserve_last_accepted_incumbent_and_gain(monkeypatch):
    # Deliberately exercise accept, NaN/all-inf rejection, and finite worse cost.
    # Expected incumbent is prescribed independently of the optimizer's formulas.
    class FakePlanner:
        T, nq, nv, nx, nu = 1, 1, 1, 2, 1
        feasibility = object()
        lo, hi = np.array([-1.]), np.array([1.])
        last_rollout_feasibility = {"feasible": [True] * 9}
        call = 0

        def rollout(self, state, targets, gains=None):
            self.call += 1
            xs = np.zeros((2, 9, 2))
            us = np.zeros((1, 9, 1))
            totals = np.full(9, 10.0)
            if self.call == 2:
                xs[1, :, 0], us[0, :, 0], totals[:] = 0.25, 0.3, 9.0
            elif self.call == 3:
                totals[:] = np.inf
                totals[0] = np.nan
            elif self.call > 3:
                totals[:] = 11
            return xs, us, totals

        def linearize(self, *_):
            return np.zeros((1, 2, 2)), np.zeros((1, 2, 1))

        def expand(self, *_):
            return np.zeros((2, 2)), np.zeros((2, 2, 2)), np.zeros((1, 1)), np.zeros((1, 1, 1))

    gain = 0

    def sweep(*_):
        nonlocal gain
        gain += 1
        return np.zeros((1, 1)), np.full((1, 1, 2), gain, float)

    monkeypatch.setattr(core, "backward", sweep)
    state, target, K, cost = core.ilqr(FakePlanner(), np.zeros(2), np.zeros((1, 1)), iters=3)
    np.testing.assert_array_equal(state[1], [.25, 0])
    np.testing.assert_array_equal(target, [[.3]])
    np.testing.assert_array_equal(K, 1)
    assert cost == 9


def test_imminent_guard_preserves_full_physical_state_and_matches_independent_pd(native_case):
    native, contract, _ = native_case
    tracker = tracker_for(native_case)
    actual = mujoco.MjData(native)
    actual.qpos[:], actual.qvel[:] = tracker.states[10, :30], tracker.states[10, 30:]
    mujoco.mj_forward(native, actual)
    actual.qacc_warmstart[:] = np.linspace(-.001, .001, native.nv)
    before = copy.deepcopy(actual)
    target = tracker.target_reference(np.arange(2))[0]
    result, states, forces = preview_native_control(native, actual, target, contract, tracker.feasibility)
    assert result["feasible"] and result["checked_physics_steps"] == 10
    for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "time", "warning"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(before, name))
    for index in range(10):
        before.ctrl[:] = np.clip(np.asarray(contract["kp"]) * (target - before.qpos[7:])
                                 - np.asarray(contract["kd"]) * before.qvel[6:],
                                 -np.asarray(contract["native_effort"]), contract["native_effort"])
        mujoco.mj_step(native, before)
        np.testing.assert_array_equal(states[index + 1], np.r_[before.qpos, before.qvel])
        np.testing.assert_array_equal(forces[index], before.qfrc_actuator[6:])


@pytest.mark.parametrize("kind", ["joint_range", "joint_speed", "engine_warning", "clock_reset_or_nonfinite"])
def test_imminent_guard_latches_first_actual_substep_fault(native_case, monkeypatch, kind):
    native, contract, _ = native_case
    tracker = tracker_for(native_case)
    actual = mujoco.MjData(native)
    actual.qpos[:], actual.qvel[:] = tracker.states[10, :30], tracker.states[10, 30:]
    mujoco.mj_forward(native, actual)
    step = mujoco.mj_step
    calls = 0

    def fault(model, data):
        nonlocal calls
        step(model, data)
        calls += 1
        if calls == 3:
            if kind == "joint_range":
                data.qpos[7] = native.jnt_range[1, 1] + 2e-6
            elif kind == "joint_speed":
                data.qvel[6] = contract["native_velocity"][0] * (1 + 1e-10)
            elif kind == "engine_warning":
                data.warning.number[0] = 1
            else:
                data.time = 0

    monkeypatch.setattr(mujoco, "mj_step", fault)
    result, _, _ = preview_native_control(
        native, actual, tracker.target_reference(0), contract, tracker.feasibility
    )
    assert not result["feasible"] and result["first_violation"]["substep"] == 3
    assert kind in result["first_violation"]["reasons"] and calls == 3
    assert actual.time == 0 and not np.any(actual.warning.number)


def test_no_feasible_seed_produces_honest_empty_trace_and_no_physical_control(native_case, monkeypatch, tmp_path):
    from gear_sonic.scripts.evaluate_g1_true23_mjbatch_mpc import run

    original = Native23Feasibility.assess

    def reject(self, *args, **kwargs):
        invalid, reasons, metrics = original(self, *args, **kwargs)
        invalid[:] = True
        reasons["injected_no_feasible_seed"] = np.ones_like(invalid)
        return invalid, reasons, metrics

    monkeypatch.setattr(Native23Feasibility, "assess", reject)
    monkeypatch.setattr(mujoco, "mj_step", lambda *_: pytest.fail("rejected seed executed physical control"))
    bundle = Path(__file__).resolve().parents[2] / "artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1"
    result = run(SimpleNamespace(
        bundle=bundle, clip="walk002", probe="standing", standing_seconds=.02, source_seconds=3,
        horizon=2, commit=1, iterations=1, threads=1, feedback_clip=.1, checkpoint_controls=100,
        motion_override=None, target_seed=None, ankle_limit_margin=None, ankle_limit_weight=None,
        all_joint_limit_margin=None, all_joint_limit_weight=None, relative_foot_weight=0,
        fd_epsilon=1e-6, output=tmp_path / "rejected", hard_feasibility=True,
    ))
    assert result["failure"]["kind"] == "no_feasible_seed"
    assert result["physics_steps"] == result["completed_controls"] == 0
    assert not result["probe_completed"] and not result["feasibility_rejection_is_success"]
    with np.load(tmp_path / "rejected/trace.npz") as trace:
        assert trace["target"].shape == trace["physics_torque"].shape == (0, 23)
        assert trace["planned_state"].shape == (0, 59)
        assert trace["feedback_gain"].shape == (0, 23, 58)


def test_exact_field_shapes_quaternion_and_unassisted_torque_contract(native_case):
    model, contract, _ = native_case
    tracker = tracker_for(native_case)
    state = tracker.states[10].copy()
    with pytest.raises(ValueError, match="force shape"):
        tracker.feasibility.assess(state[:30], state[30:], force=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="warning shape"):
        tracker.feasibility.assess(state[:30], state[30:], warning=np.zeros((1, 1)))
    damaged = state.copy()
    damaged[3:7] = 0
    invalid, reasons, _ = tracker.feasibility.assess(damaged[:30], damaged[30:])
    assert invalid[0] and reasons["invalid_root_quaternion"][0]
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = state[:30], state[30:]
    mujoco.mj_forward(model, data)
    target = tracker.target_reference(0)
    data.xfrc_applied[1, 2] = 1
    with pytest.raises(ValueError, match="external"):
        preview_native_control(model, data, target, contract, tracker.feasibility)
    data.xfrc_applied[:] = 0
    with pytest.raises(ValueError, match="unit-gain torque"):
        preview_native_control(tracker.model, data, target, contract, tracker.feasibility)
    data.qpos[7] = tracker.hi[0] + 1e-4
    result, states, forces = preview_native_control(model, data, target, contract, tracker.feasibility)
    assert not result["feasible"] and states.shape == (1, 59) and forces.shape == (0, 23)


@pytest.mark.parametrize("fault", ["nan_time", "nan_force", "nan_height"])
def test_actual_guard_catches_postpreview_nonfinite_fault_and_preserves_report(tmp_path, monkeypatch, fault):
    import json

    from gear_sonic.scripts.evaluate_g1_true23_mjbatch_mpc import run

    original = mujoco.mj_step
    calls = 0

    def inject(model, data):
        nonlocal calls
        original(model, data)
        calls += 1
        # First ten calls validate the private future. The eleventh is the
        # actual first substep; simulate a fault that arose after prediction.
        if calls == 11:
            if fault == "nan_time":
                data.time = np.nan
            elif fault == "nan_force":
                data.qfrc_actuator[6] = np.nan
            else:
                data.qpos[2] = np.nan

    monkeypatch.setattr(mujoco, "mj_step", inject)
    bundle = Path(__file__).resolve().parents[2] / "artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1"
    result = run(SimpleNamespace(
        bundle=bundle, clip="walk002", probe="standing", standing_seconds=.02, source_seconds=3,
        horizon=2, commit=1, iterations=1, threads=1, feedback_clip=.1, checkpoint_controls=100,
        motion_override=None, target_seed=None, ankle_limit_margin=None, ankle_limit_weight=None,
        all_joint_limit_margin=None, all_joint_limit_weight=None, relative_foot_weight=0,
        fd_epsilon=1e-6, output=tmp_path / fault, hard_feasibility=True,
    ))
    assert result["failure"]["kind"] == "hard_physical_feasibility"
    assert result["physics_steps"] == 1 and result["completed_controls"] == 1
    saved = (tmp_path / fault / "report.json").read_text()
    assert "NaN" not in saved and "Infinity" not in saved
    assert json.loads(saved)["failure"]["substep"] == 1
    with np.load(tmp_path / fault / "trace.npz") as trace:
        assert trace["physics_actuator_force"].shape == (1, 23)
        assert trace["physics_time"].shape == (2,)
        np.testing.assert_array_equal(trace["physics_expected_time"], [0, .002])
        assert trace["physics_warning_number"].shape == trace["physics_warning_lastinfo"].shape == (2, 8)
        if fault == "nan_time":
            assert np.isnan(trace["physics_time"][-1])
        elif fault == "nan_force":
            assert np.isnan(trace["physics_actuator_force"][-1, 0])
        else:
            assert np.isnan(trace["physics_qpos"][-1, 2])


def test_independent_accumulated_clock_avoids_false_long_pico_failure_and_detects_drift(native_case):
    tracker = tracker_for(native_case)
    state = tracker.states[10]
    expected = observed = 0.
    for _ in range(65300):
        expected += .002
        observed += .002
    # Full PICO genuinely exceeds the old tight multiplication-roundoff bound.
    assert abs(observed - 65300 * .002) > 1e-10
    invalid, _, _ = tracker.feasibility.assess(state[:30], state[30:], time=np.array([observed]),
                                               expected_time=expected)
    assert not invalid[0]
    for damaged in (observed - .002, observed + .002, observed + 2e-10, np.nan):
        invalid, reasons, _ = tracker.feasibility.assess(state[:30], state[30:], time=np.array([damaged]),
                                                        expected_time=expected)
        assert invalid[0] and reasons["clock_reset_or_nonfinite"][0]


@pytest.mark.parametrize("fault", ["none", "mismatched_forecast", "nonfinite_time"])
def test_bounded_substep_saves_raw_failure_before_returning(native_case, monkeypatch, fault):
    from gear_sonic.scripts.continue_g1_true23_mpc_hard_feasibility import record_native_substep

    native, contract, _ = native_case
    tracker = tracker_for(native_case)
    data = mujoco.MjData(native)
    data.qpos[:], data.qvel[:] = tracker.states[10, :30], tracker.states[10, 30:]
    data.time = 75.4
    mujoco.mj_forward(native, data)
    target = tracker.target_reference(0)
    _, states, forces = preview_native_control(native, data, target, contract, tracker.feasibility)
    predicted = states[1].copy()
    if fault == "mismatched_forecast":
        predicted[0] += 1e-6
    if fault == "nonfinite_time":
        original = mujoco.mj_step

        def inject(model, private):
            original(model, private)
            private.time = np.nan

        monkeypatch.setattr(mujoco, "mj_step", inject)
    trace = {name: [] for name in (
        "physics_qpos", "physics_qvel", "physics_torque", "physics_actuator_force", "physics_time",
        "physics_expected_time", "physics_warning_number", "physics_warning_lastinfo",
    )}
    expected, failure = record_native_substep(
        native, data, target, contract, tracker.feasibility, trace, 75.4, predicted, forces[0]
    )
    assert expected == 75.4 + .002
    assert all(len(values) == 1 for values in trace.values())
    np.testing.assert_array_equal(trace["physics_qpos"][0], data.qpos)
    assert trace["physics_expected_time"] == [expected]
    if fault == "none":
        assert failure is None
    elif fault == "mismatched_forecast":
        assert failure["kind"] == "private_forecast_mismatch"
    else:
        assert failure["kind"] == "actual_physics_infeasible"
        assert np.isnan(trace["physics_time"][0])


def test_restoration_zero_feedback_is_scoped_and_default_guidance_is_unchanged(native_case):
    from gear_sonic.utils.g1_true23_mjbatch_restoration import Native23RestorationTracker

    native, contract, motion = native_case
    servo = position_servo_copy(native, contract["kp"], contract["kd"], contract["native_effort"])
    seed = np.repeat(motion["joint_pos"][10:11], 30, axis=0)
    restorer = Native23RestorationTracker(servo, contract, motion, seed, threads=1)
    restorer.window(10)
    actual = restorer.states[10]
    nominal, _, _ = restorer.rollout(actual, seed)
    nominal = nominal[:, 0].copy()
    nominal[:, 0] += .01
    k, K = np.zeros((30, 23)), np.zeros((30, 23, 58))
    k[:, 13] = .004
    K[:, 13, 0] = 2.
    preserved_K = K.copy()
    gains = (nominal, k, K)
    # The pre-existing base rollout remains the default's numerical referee.
    expected = Native23Tracker.rollout(restorer, actual, seed, gains)
    for before, after in zip(expected, restorer.rollout(actual, seed, gains)):
        np.testing.assert_array_equal(before, after)
    restorer.zero_rollout_feedback = True
    _, targets, _ = restorer.rollout(actual, seed, gains)
    # Without feedback, every control is the known feedforward target, independent of state drift.
    expected_targets = np.clip(seed[:, None, :] + core.ALPHAS[None, :, None] * k[:, None, :],
                               restorer.lo, restorer.hi)
    np.testing.assert_array_equal(targets, expected_targets)
    np.testing.assert_array_equal(K, preserved_K)

    assert not np.array_equal(expected[1][0], targets[0])
    # An identically named instance attribute cannot change the main tracker or caller's K.
    main = tracker_for(native_case)
    main_state, main_seed = main.states[10], main.target_reference(np.arange(2))
    main_gains = (nominal[:3], k[:2], K[:2])
    baseline = main.rollout(main_state, main_seed, main_gains)
    main.zero_rollout_feedback = True
    for before, after in zip(baseline, main.rollout(main_state, main_seed, main_gains)):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(K, preserved_K)


@pytest.mark.parametrize("control,expected_mode,attempt_count", [(3770, "guided", 1), (3805, "zero_feedback", 2)])
def test_shared_restoration_reproduces_certified_bounded_targets(control, expected_mode, attempt_count):
    from gear_sonic.utils.g1_true23_mjbatch_mpc import load_motion_override
    from gear_sonic.utils.g1_true23_mjbatch_restoration import restore_feasible_seed

    root = Path("/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/")
    reference = Path("/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910/"
                     "mjbatch_intent_floor_inputs_v1/pico/reference.npz")
    directory = root / "hard_feasibility_zero_feedback_retry_3740_v1"
    fixture = directory / ("restoration_%05d_guided.npz" % control)
    if not fixture.exists():
        pytest.skip("frozen bounded restoration witness unavailable")
    bundle = Path(__file__).resolve().parents[2] / "artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1"
    native, contract, original, timeline, manifest = load_native_bundle(bundle, "pico")
    motion, _ = load_motion_override(reference, bundle, "pico", native, contract, original, timeline, manifest)
    with np.load(fixture, allow_pickle=False) as archive:
        initial, spec, warm = (archive[k].copy() for k in
                               ("initial_integration", "integration_state_spec", "original_warm_targets"))
    spec = int(spec)
    data = mujoco.MjData(native)
    mujoco.mj_setState(native, data, initial, spec)
    mujoco.mj_forward(native, data)
    mujoco.mj_setState(native, data, initial, spec)
    servo = position_servo_copy(native, contract["kp"], contract["kd"], contract["native_effort"])
    tracker = Native23Tracker(servo, contract, motion, horizon=30, threads=8, hard_feasibility=True,
                              all_joint_limit_margin=.05, all_joint_limit_weight=2000, relative_foot_weight=400)
    tracker.window(control + 10)
    saved_warm, proposals = warm.copy(), []

    def save(mode, targets):
        proposals.append((mode, targets.copy()))
        return {}

    best, report, _ = restore_feasible_seed(
        tracker, native, data, contract, motion, warm, control + 10, save_proposal=save
    )
    assert report["accepted"] and len(report["attempts"]) == attempt_count
    assert proposals[-1][0] == expected_mode
    expected_path = directory / ("restoration_%05d_%s.npz" % (control, expected_mode))
    with np.load(expected_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(best[3], archive["targets"])
    np.testing.assert_array_equal(best[3], proposals[-1][1])
    states, _, costs = tracker.rollout(np.r_[data.qpos, data.qvel], best[3])
    np.testing.assert_array_equal(best[2], states[:, 0])
    assert best[0] == costs[0]
    restored = np.empty_like(initial)
    mujoco.mj_getState(native, data, restored, spec)
    np.testing.assert_array_equal(restored, initial)
    np.testing.assert_array_equal(warm, saved_warm)
