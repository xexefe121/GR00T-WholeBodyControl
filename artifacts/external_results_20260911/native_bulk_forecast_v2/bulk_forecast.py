"""Private bulk forecast: full compute and first-failure prefix are explicit.

Not API-compatible with early-stop forecasts. Traces contain ALL computed steps,
including any steps after first strict failure. No policy/controller integration.
All ndarray views expire on the next call; one instance is nonconcurrent.
"""
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import threading
from types import SimpleNamespace

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PARENT = BASE / 'phase_student_preallocated_forecast_v2/native_forecast.py'
assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == '81af01274f55c703850cb3ec3801bb41912e3f3786440af5099b769b7bd28824'
spec = importlib.util.spec_from_file_location('frozen_native_forecast_parent', PARENT)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class BulkForecast(parent.NativeForecast):
    def __init__(self, model, contract, maximum_controls=5):
        assert 1 <= maximum_controls <= 5
        super().__init__(model, contract, maximum_controls)
        receipt = json.loads((HERE / 'build.json').read_text())
        for name, expected in receipt['inputs'].items():
            assert digest(name) == expected, name
        assert digest(HERE / 'native_rollout.so') == receipt['output_sha256']
        self.library = ctypes.CDLL(str(HERE / 'native_rollout.so'))
        self.library.sonic_native_version.restype = ctypes.c_int
        self.library.sonic_step_address.restype = ctypes.c_void_p
        assert self.library.sonic_native_version() == mujoco.mj_version() == 323
        loaded = ctypes.CDLL(receipt['native_library'])
        assert self.library.sonic_step_address() == ctypes.cast(loaded.mj_step, ctypes.c_void_p).value
        # NumPy dtype/contiguity and shape preconditions stay at the boundary.
        double_array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        int_array = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        self.loop = self.library.sonic_private_rollout
        self.loop.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [double_array]*4 + [ctypes.c_int] + [double_array]*5 + [int_array]*2
        self.loop.restype = ctypes.c_int
        self._busy = False
        self._call_lock = threading.Lock()
        self.owned_target = np.empty(23, dtype=np.float64)
        self._proxy_force = np.empty(29)
        self._proxy = SimpleNamespace(qpos=None, qvel=None, time=0., qfrc_actuator=self._proxy_force,
                                     warning=SimpleNamespace(number=None, lastinfo=None))

    def predict(self, source, target, horizon_controls):
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError('BulkForecast is nonconcurrent')
        self._busy = True
        try:
            return self._predict_locked(source, target, horizon_controls)
        finally:
            self._busy = False
            self._call_lock.release()

    def _predict_locked(self, source, target, horizon_controls):
        assert type(horizon_controls) is int and 0 < horizon_controls <= self.max_controls
        target = np.asarray(target)
        assert target.shape == (23,) and target.dtype == np.float64 and target.flags.c_contiguous
        # Copy before validation or any private state/trace mutation. A view into
        # an expiring prior trace must not become a moving command in C++.
        np.copyto(self.owned_target, target)
        target = self.owned_target
        assert np.isfinite(target).all()
        if np.any(target < self.limits[:, 0]) or np.any(target > self.limits[:, 1]):
            raise ValueError('target must already be within native bounds')
        if np.any(source.qfrc_applied) or np.any(source.xfrc_applied):
            raise ValueError('external forces are forbidden')
        if np.shares_memory(source.qpos, self.private.qpos) or np.shares_memory(source.qvel, self.private.qvel):
            raise ValueError('private data aliases caller')
        mujoco.mj_getState(self.model, source, self.before, self.spec)
        np.copyto(self.warning, source.warning.number)
        np.copyto(self.lastinfo, source.warning.lastinfo)
        d = self.private
        try:
            mujoco.mj_setState(self.model, d, self.before, self.spec)
            d.warning.number[:] = self.warning
            d.warning.lastinfo[:] = self.lastinfo
            mujoco.mj_forward(self.model, d)
            if not np.array_equal(d.warning.number, self.warning) or not np.array_equal(d.warning.lastinfo, self.lastinfo):
                raise ValueError('private initialization produced a warning')
            mujoco.mj_setState(self.model, d, self.before, self.spec)
            d.warning.number[:] = self.warning
            d.warning.lastinfo[:] = self.lastinfo
            mujoco.mj_getState(self.model, d, self.roundtrip, self.spec)
            if not np.array_equal(self.roundtrip, self.before):
                raise ValueError('private integration restoration mismatch')
            start = expected = float(source.time)
            maximum = dict(joint_excess_rad=0., speed_ratio=0., effort_ratio=0., tilt_rad=0.,
                           clock_error_seconds=0., ideal_clock_difference_seconds=0.)
            minimum = float(d.qpos[2])
            first = self._assess(0, expected, start, maximum)
            self.q[0], self.v[0], self.t[0] = d.qpos, d.qvel, d.time
            self.warnings[0], self.infos[0] = d.warning.number, d.warning.lastinfo
            steps = 0
            tick = time.perf_counter_ns()
            if first is None:
                steps = self.loop(self.model._address, d._address, target, self.kp, self.kd, self.effort,
                                  horizon_controls*10, self.q, self.v, self.t, self.torque, self.force,
                                  self.warnings, self.infos)
                if not 1 <= steps <= horizon_controls*10:
                    raise RuntimeError('native ABI rejected model or horizon')
            rollout_ms = (time.perf_counter_ns() - tick)/1e6
            prefix_maximum = dict(maximum) if first is not None else None
            prefix_minimum = minimum if first is not None else None
            tick = time.perf_counter_ns()
            # Reuse the unchanged scalar NumPy predicates through a read-only
            # array proxy. This never writes private or caller integration state.
            try:
                self.private = self._proxy
                for step in range(1, steps+1):
                    p = self._proxy
                    p.qpos, p.qvel, p.time = self.q[step], self.v[step], self.t[step]
                    p.qfrc_actuator[6:] = self.force[step-1]
                    p.warning.number, p.warning.lastinfo = self.warnings[step], self.infos[step]
                    expected += .002
                    failure = self._assess(step, expected, start, maximum)
                    minimum = min(minimum, float(p.qpos[2]))
                    if first is None and failure is not None:
                        first = failure
                        prefix_maximum, prefix_minimum = dict(maximum), minimum
            finally:
                self.private = d
            assessment_ms = (time.perf_counter_ns() - tick)/1e6
            checked_prefix = first['physics_step'] if first is not None else steps
            report = dict(feasible=first is None and steps == horizon_controls*10,
                          first_failure=first, physics_steps=steps, computed_steps=steps,
                          assessed_steps=steps, checked_prefix_steps=checked_prefix,
                          computed_steps_after_first_failure=steps-checked_prefix,
                          requested_physics_steps=horizon_controls*10,
                          diagnostic_continued_after_failure=steps>checked_prefix,
                          strict_early_stop=False, maximum=maximum, minimum_root_height_m=minimum,
                          prefix_maximum=maximum if prefix_maximum is None else prefix_maximum,
                          prefix_minimum_root_height_m=minimum if prefix_minimum is None else prefix_minimum,
                          initial_time=start, final_time=float(d.time), final_warning_counts=d.warning.number.tolist(),
                          original_data_unchanged=True, native_rollout_ms=rollout_ms, assessment_ms=assessment_ms)
            trace = dict(physics_qpos=self.q[:steps+1], physics_qvel=self.v[:steps+1], physics_time=self.t[:steps+1],
                         physics_torque=self.torque[:steps], physics_actuator_force=self.force[:steps],
                         warning_counts=self.warnings[:steps+1], warning_lastinfo=self.infos[:steps+1])
            self.forecasts += 1
            return report, trace
        finally:
            self.private = d
            mujoco.mj_getState(self.model, source, self.after, self.spec)
            if not np.array_equal(self.after, self.before):
                raise ValueError('caller integration state changed')
            if not np.array_equal(source.warning.number, self.warning) or not np.array_equal(source.warning.lastinfo, self.lastinfo):
                raise ValueError('caller warnings changed')
