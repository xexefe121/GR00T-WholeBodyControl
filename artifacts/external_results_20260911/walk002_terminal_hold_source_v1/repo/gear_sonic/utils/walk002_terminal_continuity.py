"""Artifact-only checkpoint/tail certification for one separate same-MPC hold.

This module cannot select targets or run policy, optimizer, or dynamics. The
copied evaluator alone executes the unchanged controller and native PD loop.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np

START, BOUNDARY, EXTENSION = 1400, 1417, 250
TRACE_SHA = "UNBOUND_MAIN_TRACE"
ENDPOINT_SHA = "UNBOUND_ROOT_ENDPOINT"
STATE_FIELDS = {"qpos", "qvel"}
PHYSICS_STATE_FIELDS = {
    "physics_qpos", "physics_qvel", "physics_time", "physics_expected_time",
    "physics_warning_number", "physics_warning_lastinfo",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, arrays):
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def exact(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"{name}: shape/dtype mismatch {actual.shape}/{actual.dtype} vs {expected.shape}/{expected.dtype}")
    if actual.tobytes(order="C") != expected.tobytes(order="C"):
        raise ValueError(f"{name}: bitexact comparison failed")


def snapshot(native, data):
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(native, spec))
    mujoco.mj_getState(native, data, state, spec)
    return state


def history_snapshot(fresh):
    return dict(previous_action=fresh.previous_action.copy(),
                recorded_controls=np.asarray(fresh.recorded_controls),
                actual_action_max_abs=np.asarray(fresh.actual_action_max_abs),
                actual_action_components_outside_five=np.asarray(fresh.actual_action_components_outside_five),
                **{"history_" + key: value.copy() for key, value in fresh.history.data.items()})


def history_flat(fresh):
    return np.concatenate([fresh.history.data[key].reshape(-1) for key in sorted(fresh.history.data)])


def expected_slice(name, values, start, stop):
    if name in STATE_FIELDS:
        return values[start:stop + 1]
    if name in PHYSICS_STATE_FIELDS:
        return values[start * 10:stop * 10 + 1]
    if name.startswith("physics_") and name != "physics_substeps":
        return values[start * 10:stop * 10]
    return values[start:stop]


class Continuity:
    def __init__(self, args, native, fresh, timeline):
        self.args, self.native, self.fresh = args, native, fresh
        self.boundary_verified = False
        self.started = time.perf_counter()
        self.hold_started = None
        self.source = args.preceding_run
        self.endpoint_path = args.preceding_endpoint
        required = dict(clip="walk002", probe="full-lifecycle", horizon=30, commit=5, iterations=5,
                        threads=8, fd_epsilon=1e-6, feedback_clip=.1,
                        all_joint_limit_margin=.05, all_joint_limit_weight=2000., relative_foot_weight=400.,
                        hard_feasibility=True, restoration=True, restoration_control_lm_retry=True)
        for key, value in required.items():
            if getattr(args, key) != value:
                raise ValueError("hold requires unchanged canonical configuration: " + key)
        if fresh is None or args.target_seed is None or args.motion_override is None:
            raise ValueError("hold requires both original seeds and the original v4 reference")
        if timeline["total_requested_controls"] != BOUNDARY:
            raise ValueError("wrong canonical lifecycle")
        if (native.nq, native.nv, native.nu, native.nbody, native.na, native.nmocap, native.nuserdata) != (30, 29, 23, 25, 0, 0, 0):
            raise ValueError("unexpected native integration topology")
        if mujoco.__version__ != "3.2.3" or native.opt.timestep != .002:
            raise ValueError("wrong native engine or clock")
        self.inputs = [self.source / name for name in ("trace.npz", "trace.partial.npz", "report.json", "request.json")]
        self.inputs += [self.endpoint_path, self.endpoint_path.with_name("report.json"), Path(__file__)]
        self.hashes = {str(path): sha(path) for path in self.inputs}
        if self.hashes[str(self.source / "trace.npz")] != TRACE_SHA or self.hashes[str(self.endpoint_path)] != ENDPOINT_SHA:
            raise ValueError("wrong immutable canonical trace or independent endpoint")
        report = json.loads((self.source / "report.json").read_text())
        endpoint_report = json.loads(self.endpoint_path.with_name("report.json").read_text())
        if report["completed_controls"] != BOUNDARY or report["failure"] is not None:
            raise ValueError("preceding full run is not physically complete")
        if not (endpoint_report["all_recorded_samples_bitexact"] and endpoint_report["completed_controls"] == BOUNDARY
                and endpoint_report["compared_physics_steps"] == BOUNDARY * 10
                and endpoint_report["original_trace_sha256"] == TRACE_SHA
                and endpoint_report["endpoint_sha256"] == ENDPOINT_SHA):
            raise ValueError("independent endpoint receipt mismatch")
        with np.load(self.endpoint_path, allow_pickle=False) as archive:
            self.endpoint = {key: archive[key].copy() for key in archive.files}
        with np.load(self.source / "trace.partial.npz", allow_pickle=False) as archive:
            partial = {key: archive[key].copy() for key in archive.files}
        self.metadata = json.loads(str(partial.pop("checkpoint_metadata")))
        if (self.metadata["completed_controls"] != START or self.metadata["failure"] is not None
                or self.metadata["request_sha256"] != self.hashes[str(self.source / "request.json")]):
            raise ValueError("wrong1400 checkpoint or request binding")
        self.warm = partial.pop("checkpoint_warm_targets")
        self.qacc = partial.pop("checkpoint_qacc_warmstart")
        self.ctrl = partial.pop("checkpoint_ctrl")
        if self.warm.shape != (30, 23) or self.metadata["plans"][-1]["control"] != START - 5:
            raise ValueError("checkpoint warm plan is not1395/H30")
        if self.metadata["plans"][-1]["controls_executed"] != 5:
            raise ValueError("checkpoint is not at a five-control commitment boundary")
        self.tail = {}
        self.history_checks = 0
        with np.load(self.source / "trace.npz", allow_pickle=False) as archive:
            for key in archive.files:
                values = archive[key]
                exact(partial[key], expected_slice(key, values, 0, START), "canonical checkpoint prefix " + key)
                self.tail[key] = expected_slice(key, values, START, BOUNDARY).copy()
            # Sensor/action-only reconstruction; record_control runs no inference or dynamics.
            qpos, qvel, targets = archive["qpos"], archive["qvel"], archive["target"]
            actions, histories = archive["fresh_seed_previous_action"], archive["fresh_seed_measured_history"]
            for control in range(START):
                exact(fresh.previous_action, actions[control], f"history previous action{control}")
                exact(history_flat(fresh), histories[control], f"history terms{control}")
                fresh.record_control(control, qpos[control], qvel[control], targets[control])
                self.history_checks += 1
            exact(fresh.previous_action, actions[START], "checkpoint previous action1400")
            exact(history_flat(fresh), histories[START], "checkpoint named history1400")
        self.initial_qpos, self.initial_qvel = partial["qpos"][-1], partial["qvel"][-1]
        self.initial_time = float(partial["physics_time"][-1])
        self.expected_time = float(partial["physics_expected_time"][-1])
        self.initial_warnings = partial["physics_warning_number"][-1]
        self.initial_lastinfo = partial["physics_warning_lastinfo"][-1]
        if np.any(self.initial_warnings) or np.any(self.initial_lastinfo):
            raise ValueError("nonzero initial warning ledger")
        self.initial_history = history_snapshot(fresh)
        self.initial_integration = None
        self.current_record_start = START

    def initialize(self):
        data = mujoco.MjData(self.native)
        data.qpos[:], data.qvel[:] = self.initial_qpos, self.initial_qvel
        data.time = self.initial_time
        data.ctrl[:], data.qacc_warmstart[:] = self.ctrl, self.qacc
        mujoco.mj_forward(self.native, data)
        # mj_forward computes derived values; restore saved integration inputs exactly.
        data.ctrl[:], data.qacc_warmstart[:] = self.ctrl, self.qacc
        data.warning.number[:], data.warning.lastinfo[:] = self.initial_warnings, self.initial_lastinfo
        exact(data.qpos, self.initial_qpos, "initial qpos")
        exact(data.qvel, self.initial_qvel, "initial qvel")
        if data.time != self.initial_time or np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            raise ValueError("initial clock or external forces mismatch")
        self.initial_integration = snapshot(self.native, data)
        if self.initial_integration.shape != (291,):
            raise ValueError("native integration must contain291 values")
        atomic(self.args.output / "checkpoint1400_restored.npz", dict(
            initial_integration=self.initial_integration, integration_state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),
            expected_time=self.expected_time, warm_targets=self.warm,
            warning_counts=data.warning.number.copy(), warning_lastinfo=data.warning.lastinfo.copy(),
            **self.initial_history))
        return data

    def request_fields(self):
        return dict(kind="separate_same_mpc_terminal_hold_after_exact_tail_reexecution",
                    requested_controls=EXTENSION, source_requested_controls=0,
                    global_control_start=BOUNDARY, global_control_stop=BOUNDARY + EXTENSION,
                    reexecution_control_start=START, reexecution_control_stop=BOUNDARY,
                    physical_initialization="checkpoint1400 integration/history/last full1395 warm plan",
                    initial_qpos=self.initial_qpos.tolist(), initial_qvel=self.initial_qvel.tolist(),
                    continuity_input_hashes=self.hashes,
                    extension_gate="all saved tail fields bitexact plus root independent full291 endpoint; compare only",
                    held_reference="unchanged last original and v4 reference sample via existing clamped planner/BFM indices",
                    endpoint_injected=False, physical_state_rewrites_after_initialization=0,
                    preceding_original_trace_sha256=TRACE_SHA, preceding_endpoint_sha256=ENDPOINT_SHA)

    def tail_plan_summary(self, plans):
        result = copy.deepcopy(plans)
        if not result or result[-1]["control"] != 1415 or result[-1]["controls_executed"] != 2:
            raise ValueError("tail must end after two actual controls of plan1415")
        result[-1]["prospective_commit_controls"] = result[-1]["controls_committed"]
        result[-1]["controls_committed"] = 2
        return result

    def pending_commit(self, plans, planned_states, planned_targets, gains, local, count, completed):
        # These are the live outputs of the existing1415 solve, never a new solve at1417.
        if (completed, local, count) != (1417, 2, 5):
            raise ValueError("terminal boundary must preserve plan1415 locals2..4")
        if not plans or plans[-1]["control"] != 1415 or plans[-1]["controls_executed"] != 2:
            raise ValueError("wrong pending plan or executed count")
        if (np.asarray(planned_states).shape, np.asarray(planned_targets).shape, np.asarray(gains).shape) != ((31, 59), (30, 23), (30, 23, 58)):
            raise ValueError("pending full H30 plan schema mismatch")
        atomic(self.args.output / "pending_plan1415.npz", dict(
            planned_states=planned_states.copy(), planned_targets=planned_targets.copy(), gains=gains.copy(),
            origin_plan_control=np.asarray(1415), boundary_control=np.asarray(1417),
            pending_local_first=np.asarray(local), pending_controls=np.asarray(count - local),
            execution_started=np.asarray(False)))
        result = copy.deepcopy(plans[-1])
        result.update(control=1417, origin_plan_control=1415, pending_local_first=2,
                      origin_planned_commit_controls=5, controls_committed=3, controls_executed=0,
                      imminent_control_checks=[], source_first_state_frame=1427,
                      origin_solve_ms=result["solve_ms"], solve_ms=0.,
                      solve_reused_across_terminal_boundary=True)
        return result

    def verify_boundary(self, trace, plans, data, fresh, warm, expected_time):
        arrays = {key: np.asarray(value) for key, value in trace.items()}
        # Persist actual reproduced tail before checking; mismatches remain reviewable.
        atomic(self.args.output / "reexecuted_tail.npz", dict(
            **arrays, initial_integration=self.initial_integration,
            final_integration=snapshot(self.native, data),
            integration_state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION)))
        (self.args.output / "reexecuted_tail_plans.json").write_text(json.dumps(finite_json(self.tail_plan_summary(plans)), indent=2, allow_nan=False))
        checks = {}
        try:
            if set(arrays) != set(self.tail) | {"global_control"}:
                raise ValueError("tail trace field inventory mismatch")
            for key, expected in self.tail.items():
                exact(arrays[key], expected, "reexecuted tail " + key)
                checks[key] = True
            exact(arrays["global_control"], np.arange(START, BOUNDARY), "global control clock")
            state = snapshot(self.native, data)
            if int(self.endpoint["integration_state_spec"]) != int(mujoco.mjtState.mjSTATE_INTEGRATION):
                raise ValueError("endpoint integration spec mismatch")
            exact(state, self.endpoint["final_integration"], "all291 independent endpoint values")
            for key in ("qpos", "qvel", "qacc_warmstart", "ctrl"):
                exact(getattr(data, key), self.endpoint[key], "endpoint " + key)
            if data.time != float(self.endpoint["time"]):
                raise ValueError("endpoint time mismatch")
            exact(data.warning.number, self.endpoint["warning_counts"], "endpoint warnings")
            exact(data.warning.lastinfo, self.endpoint["warning_lastinfo"], "endpoint warning lastinfo")
            if expected_time != float(self.tail["physics_expected_time"][-1]) or fresh.recorded_controls != BOUNDARY:
                raise ValueError("independent clock or history global count mismatch")
            expected_action = ((self.tail["target"][-1] - fresh.contract["default_q"]) * fresh.contract["kp"]
                               / (.25 * fresh.contract["training_effort"])).astype(np.float32)
            exact(fresh.previous_action, expected_action, "boundary previous actual action")
        except Exception as error:
            (self.args.output / "boundary_verification.json").write_text(json.dumps(dict(
                verified=False, error=str(error), completed_checks=checks, extension_controls_executed=0), indent=2))
            raise
        self.initial_integration = state.copy()
        self.initial_history = history_snapshot(fresh)
        self.current_record_start = BOUNDARY
        atomic(self.args.output / "boundary1417.npz", dict(
            initial_integration=state, integration_state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),
            expected_time=expected_time, warm_targets=warm.copy(),
            warning_counts=data.warning.number.copy(), warning_lastinfo=data.warning.lastinfo.copy(),
            **self.initial_history))
        self.boundary_verified = True
        self.hold_started = time.perf_counter()
        (self.args.output / "boundary_verification.json").write_text(json.dumps(dict(
            verified=True, compared_controls=17, compared_physics_steps=170, all_saved_fields_bitexact=checks,
            independent_endpoint_all291_bitexact=True, endpoint_compared_only=True,
            preceding_trace_sha256=TRACE_SHA, endpoint_sha256=ENDPOINT_SHA,
            reexecuted_tail_sha256=sha(self.args.output / "reexecuted_tail.npz"),
            boundary_sha256=sha(self.args.output / "boundary1417.npz"),
            physical_state_rewrites_at_boundary=0), indent=2))
        # Reset recording buffers only. Actual MjData, planner, seed history, warm plan and clock stay live.
        result = {key: [] for key in trace}
        for key in STATE_FIELDS | PHYSICS_STATE_FIELDS:
            result[key] = [arrays[key][-1].copy()]
        return result

    def save(self, path, trace, data, fresh, warm, expected_time, metadata=None):
        arrays = {key: np.asarray(value) for key, value in trace.items()}
        # Preserve canonical schema even when the first held control is rejected.
        trailing_shapes = dict(
            qpos=(30,), qvel=(29,), target=(23,), planned_target=(23,), planned_state=(59,),
            feedback_gain=(23, 58), feedback_correction_raw=(23,), feedback_correction_applied=(23,),
            joint_error=(23,), root_error=(3,), body_position_error=(6,),
            physics_qpos=(30,), physics_qvel=(29,), physics_requested_torque=(23,), physics_torque=(23,),
            physics_actuator_force=(23,), physics_warning_number=(8,), physics_warning_lastinfo=(8,),
            fresh_seed_previous_action=(23,), fresh_seed_measured_history=(300,),
        )
        for key, shape in trailing_shapes.items():
            if key in arrays and not arrays[key].size:
                arrays[key] = arrays[key].reshape((0, *shape))
        arrays.update(initial_integration=self.initial_integration,
                      final_integration=snapshot(self.native, data),
                      integration_state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION),
                      final_expected_time=np.asarray(expected_time), checkpoint_warm_targets=warm,
                      checkpoint_qacc_warmstart=data.qacc_warmstart.copy(), checkpoint_ctrl=data.ctrl.copy(),
                      **{"initial_" + key: value for key, value in self.initial_history.items()},
                      **{"final_" + key: value for key, value in history_snapshot(fresh).items()})
        if metadata is not None:
            arrays["checkpoint_metadata"] = np.asarray(json.dumps(finite_json(metadata), allow_nan=False))
        atomic(path, arrays)

    def checkpoint(self, trace, plans, data, fresh, warm, expected_time, completed, failure):
        metadata = dict(kind="incomplete_separate_same_mpc_hold", completed_controls=max(0, completed - BOUNDARY),
                        completed_global_controls=completed, requested_controls=EXTENSION,
                        boundary_verified=self.boundary_verified, physics_steps=len(trace["physics_torque"]),
                        simulation_time=float(data.time), failure=failure,
                        is_final_result=False, request_sha256=sha(self.args.output / "request.json"), plans=plans)
        self.save(self.args.output / "trace.partial.npz", trace, data, fresh, warm, expected_time, metadata)

    def finalize(self, trace, plans, data, fresh, warm, expected_time, completed, failure, restoration_events):
        name = "trace.npz" if self.boundary_verified else "reexecuted_tail_failed.npz"
        self.save(self.args.output / name, trace, data, fresh, warm, expected_time)
        (self.args.output / "plans.json").write_text(json.dumps(finite_json(plans), indent=2, allow_nan=False))
        count = len(trace["target"]) if self.boundary_verified else 0
        if not self.boundary_verified and failure is None:
            failure = dict(kind="tail_boundary_never_verified")
        result = dict(kind="separate_same_mpc_terminal_hold", completed_controls=count, requested_controls=EXTENSION,
                      global_control_start=BOUNDARY, completed_global_controls=completed,
                      probe_completed=self.boundary_verified and count == EXTENSION and failure is None,
                      failure=failure, boundary_verified=self.boundary_verified, full_source_completed=False,
                      physics_steps=len(trace["physics_torque"]) if self.boundary_verified else 0,
                      reexecution_controls=17 if self.boundary_verified else len(trace["target"]),
                      elapsed_wall_seconds=time.perf_counter() - self.started,
                      hold_wall_seconds=time.perf_counter() - self.hold_started if self.hold_started else None,
                      simulation_time=float(data.time), warning_counts=data.warning.number.tolist(),
                      range_excess_max=max(trace["range_excess"]) if count else None,
                      velocity_ratio_max=max(trace["velocity_ratio"]) if count else None,
                      effort_ratio_max=max(trace["effort_ratio"]) if count else None,
                      restoration_events=restoration_events, restoration_triggers=len(restoration_events),
                      physical_state_rewrites_after_initialization=0, root_assistance_forces=0,
                      held_reference="last original and v4 sample via unchanged clamped planner/BFM indices",
                      controller="same H30/5/commit5 MPC, guided/K0/thirdLM and original seeds",
                      quiet_qualified=False, independent_native_replay_required=True,
                      hardware_authorized=False, deployment_ready=False, timing_qualified=False,
                      preceding_trace_sha256=TRACE_SHA, preceding_endpoint_sha256=ENDPOINT_SHA,
                      request_sha256=sha(self.args.output / "request.json"), trace_sha256=sha(self.args.output / name),
                      plans_sha256=sha(self.args.output / "plans.json"))
        (self.args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps(result), flush=True)
        return result
