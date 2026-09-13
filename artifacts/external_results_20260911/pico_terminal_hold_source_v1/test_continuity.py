"""Strict fail-closed boundary tests using saved arrays; no native stepping/inference."""
import ast
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils import pico_terminal_continuity as module

HERE = Path(__file__).resolve().parent


def test_original_controller_body_unchanged():
    original = (HERE.parent / "pico_control_lm_integration_v1/repo/gear_sonic/scripts/evaluate_g1_true23_mjbatch_mpc.py").read_text()
    adapter = (HERE / "repo/gear_sonic/scripts/evaluate_g1_true23_mjbatch_mpc.py").read_text()
    start, stop = "        planner.window(completed + 10)", "        if args.checkpoint_controls and completed"
    original = original[original.index(start):original.index(stop)]
    adapter = adapter[adapter.index(start):adapter.index(stop)]
    adapter = adapter.replace("frame = min(completed + 11, len(reference_states) - 1)", "frame = completed + 11")
    adapter = adapter.replace("source_first_state_frame=min(completed + 10, len(reference_states) - 1)", "source_first_state_frame=completed + 10")
    adapter = adapter.replace("                global_control=completed,\n", "")
    assert adapter == original


def test_helper_cannot_run_physics_or_policy():
    tree = ast.parse((HERE / "repo/gear_sonic/utils/pico_terminal_continuity.py").read_text())
    forbidden = {"mj_step", "mj_step1", "mj_step2", "ilqr", "propose", "rollout", "mj_setState"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            assert name not in forbidden


@pytest.mark.parametrize("change", ["one_ulp", "signed_zero", "shape", "dtype"])
def test_bitexact_rejects_changes(change):
    original = np.asarray([0., 1.])
    changed = original.copy()
    if change == "one_ulp":
        changed[1] = np.nextafter(1., 2.)
    elif change == "signed_zero":
        changed[0] = -0.
    elif change == "shape":
        changed = changed.reshape(1, 2)
    else:
        changed = changed.astype(np.float32)
    with pytest.raises(ValueError):
        module.exact(changed, original, change)


@lru_cache(maxsize=1)
def saved_tail():
    with np.load(HERE.parent / "pico_full_control_lm_v1/trace.npz") as archive:
        return {key: module.expected_slice(key, archive[key], 6500, 6530).copy() for key in archive.files}


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    obj = module.Continuity.__new__(module.Continuity)
    obj.args = SimpleNamespace(output=tmp_path)
    obj.native = object()
    obj.boundary_verified = False
    obj.initial_integration = np.zeros(291)
    obj.initial_history = {}
    obj.current_record_start = 6500
    obj.tail = {key: value.copy() for key, value in saved_tail().items()}
    with np.load(HERE.parent / "pico_full_endpoint_independent_v1/endpoint.npz") as archive:
        obj.endpoint = {key: archive[key].copy() for key in archive.files}
    data = SimpleNamespace(**{key: obj.endpoint[key].copy() for key in ("qpos", "qvel", "qacc_warmstart", "ctrl")},
                           time=float(obj.endpoint["time"]),
                           warning=SimpleNamespace(number=obj.endpoint["warning_counts"].copy(),
                                                   lastinfo=obj.endpoint["warning_lastinfo"].copy()))
    # Boundary action check uses the same contract algebra, independently of any live policy.
    contract_path = Path("/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json")
    contract = {key: np.asarray(value) for key, value in json.loads(contract_path.read_text()).items()}
    action = ((obj.tail["target"][-1] - contract["default_q"]) * contract["kp"] / (.25 * contract["training_effort"])).astype(np.float32)
    fresh = SimpleNamespace(previous_action=action, recorded_controls=6530, contract=contract,
                            actual_action_max_abs=0., actual_action_components_outside_five=0,
                            history=SimpleNamespace(data={"fixture": np.zeros(1)}))
    trace = {key: value.copy() for key, value in obj.tail.items()}
    trace["global_control"] = np.arange(6500, 6530)
    monkeypatch.setattr(module, "snapshot", lambda *args: obj.endpoint["final_integration"].copy())
    return obj, trace, data, fresh


def test_boundary_preserves_live_state_and_only_resets_buffers(boundary):
    obj, trace, data, fresh = boundary
    qpos, action = data.qpos.copy(), fresh.previous_action.copy()
    result = obj.verify_boundary(trace, [], data, fresh, np.zeros((30, 23)), float(trace["physics_expected_time"][-1]))
    assert obj.boundary_verified
    assert len(result["qpos"]) == 1 and len(result["target"]) == 0
    module.exact(data.qpos, qpos, "unmodified live state")
    module.exact(fresh.previous_action, action, "unmodified live history")
    assert json.loads((obj.args.output / "boundary_verification.json").read_text())["independent_endpoint_all291_bitexact"]


def test_empty_hold_failure_keeps_native_trace_shapes(boundary):
    obj, trace, data, fresh = boundary
    expected_time = float(trace["physics_expected_time"][-1])
    result = obj.verify_boundary(trace, [], data, fresh, np.zeros((30, 23)), expected_time)
    obj.save(obj.args.output / "empty_hold.npz", result, data, fresh, np.zeros((30, 23)), expected_time)
    with np.load(obj.args.output / "empty_hold.npz") as archive:
        assert archive["target"].shape == (0, 23)
        assert archive["feedback_gain"].shape == (0, 23, 58)
        assert archive["fresh_seed_measured_history"].shape == (0, 300)
        assert archive["physics_torque"].shape == (0, 23)
        assert archive["qpos"].shape == (1, 30)
        assert archive["initial_integration"].shape == (291,)


@pytest.mark.parametrize("field", ["target", "physics_qpos", "physics_qvel", "physics_torque",
                                    "physics_actuator_force", "physics_time", "physics_expected_time",
                                    "physics_warning_number", "physics_warning_lastinfo",
                                    "feedback_gain", "fresh_seed_measured_history", "global_control"])
def test_boundary_rejects_one_saved_sample_change(boundary, field):
    obj, trace, data, fresh = boundary
    trace[field].flat[0] += 1
    with pytest.raises(ValueError):
        obj.verify_boundary(trace, [], data, fresh, np.zeros((30, 23)), float(obj.tail["physics_expected_time"][-1]))
    assert not obj.boundary_verified
    receipt = json.loads((obj.args.output / "boundary_verification.json").read_text())
    assert not receipt["verified"] and receipt["extension_controls_executed"] == 0
    assert (obj.args.output / "reexecuted_tail.npz").exists()


@pytest.mark.parametrize("field", ["qacc_warmstart", "ctrl", "time", "warning_counts", "integration"])
def test_boundary_rejects_endpoint_change(boundary, monkeypatch, field):
    obj, trace, data, fresh = boundary
    if field == "integration":
        changed = obj.endpoint["final_integration"].copy()
        changed[290] += 1
        monkeypatch.setattr(module, "snapshot", lambda *args: changed.copy())
    elif field == "time":
        data.time += .002
    elif field == "warning_counts":
        data.warning.number[0] += 1
    else:
        getattr(data, field)[0] += 1
    with pytest.raises(ValueError):
        obj.verify_boundary(trace, [], data, fresh, np.zeros((30, 23)), float(trace["physics_expected_time"][-1]))
    assert not obj.boundary_verified
