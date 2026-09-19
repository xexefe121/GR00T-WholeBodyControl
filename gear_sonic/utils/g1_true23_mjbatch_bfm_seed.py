"""Offline native23 BFM rollout seed from measured controller history.

Only private MuJoCo data are stepped. This module has no robot/DDS interface.
The caller retains authority over actual targets, scoring and physical execution.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_bfm_seed_observations import (
    BFMHistory, _quaternion_matrix, reference_features, state_and_terms,
)

WEIGHTS_SHA = "28a2d82a2975c37b8a1533f2e3b0224d27f8a0813283640ee7eeee348fe42c2e"
GRAPH_SHAS = {"actor": "7d9dc682dc38c6972c5023bb6bd069c8d9d28a99fa33a871906befaf9a1cc64e",
              "backward": "2fe7cf1a0f76c101703fd53a855f2281763d47fef45c0f1455789bb04c21c3b7"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _arrays_sha(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = np.ascontiguousarray(value)
        digest.update(json.dumps([key, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class BFMSeedRolloutError(RuntimeError):
    """An invalid optional private rollout; other MPC seeds remain available."""
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = {} if diagnostics is None else diagnostics


class Native23BFMRolloutSeed:
    """Stateful measured history, with stateless counterfactual proposals.

    Call record_control once per real control before stepping the real plant,
    using the pre-control state and the selected, actually applied target.
    propose copies that history and never changes it or the real plant state.
    """
    def __init__(self, native_model, contract, original_motion, onnx_directory,
                 *, dependency_directory=None, threads=1):
        if (native_model.nq, native_model.nv, native_model.nu) != (30, 29, 23):
            raise ValueError("BFM seed requires native23 topology")
        if native_model.opt.timestep != .002 or native_model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
            raise ValueError("BFM seed requires native500Hz Euler physics")
        if (np.any(native_model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED)
                or np.any(native_model.actuator_gainprm[:, 0] != 1.)
                or np.any(native_model.actuator_biastype != mujoco.mjtBias.mjBIAS_NONE)
                or np.any(native_model.actuator_dyntype != mujoco.mjtDyn.mjDYN_NONE)):
            raise ValueError("BFM seed requires native unit-gain torque actuators with no bias or activation dynamics")
        if type(threads) is not int or threads < 1:
            raise ValueError("positive inference thread count required")
        if dependency_directory is not None:
            dependency_directory = str(Path(dependency_directory).resolve())
            if dependency_directory not in sys.path:
                sys.path.insert(0, dependency_directory)
        ort = importlib.import_module("onnxruntime")
        if ort.__version__ != "1.23.2":
            raise ValueError("validated BFM seed runtime is ONNX Runtime1.23.2")
        folder = Path(onnx_directory)
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest.get("weights_sha256") != WEIGHTS_SHA:
            raise ValueError("BFM seed weights identity mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sessions = {}
        for name, digest in GRAPH_SHAS.items():
            path = folder / (name + ".onnx")
            if _sha(path) != digest or manifest.get(name + "_sha256") != digest:
                raise ValueError("BFM seed graph identity mismatch: " + name)
            self.sessions[name] = ort.InferenceSession(str(path), sess_options=options,
                                                       providers=["CPUExecutionProvider"])
        self.model = native_model
        self.contract = {key: np.asarray(contract[key], dtype=np.float64).copy()
                         for key in ("default_q", "kp", "kd", "training_effort", "native_effort", "native_velocity")}
        self.contract["body_names"] = tuple(contract["body_names"])
        for key in ("default_q", "kp", "kd", "training_effort", "native_effort", "native_velocity"):
            if self.contract[key].shape != (23,) or not np.isfinite(self.contract[key]).all():
                raise ValueError("invalid native23 BFM contract: " + key)
        if any(np.any(self.contract[key] <= 0) for key in ("kp", "training_effort", "native_effort", "native_velocity")):
            raise ValueError("BFM seed gains and limits must be positive")
        self.lo, self.hi = native_model.jnt_range[1:].T.copy()
        self.motion = {key: np.asarray(value).copy() for key, value in original_motion.items()}
        if float(np.asarray(self.motion["fps"]).ravel()[0]) != 50:
            raise ValueError("BFM seed requires original50Hz goal clock")
        self.state, self.privileged = reference_features(self.motion, self.contract)
        for value in self.motion.values(): value.setflags(write=False)
        self.history = BFMHistory()
        self.previous_action = np.zeros(23, dtype=np.float32)
        self.recorded_controls = 0
        self.actual_action_max_abs = 0.
        self.actual_action_components_outside_five = 0
        runtime_dir = Path(ort.__file__).parent
        runtime_files = list((runtime_dir / "capi").glob("*pybind11_state*.so")) + list((runtime_dir / "capi").glob("*pybind11_state*.pyd"))
        self._identity = dict(kind="offline_native23_measured_history_BFM_rollout_seed",
            weights_sha256=WEIGHTS_SHA, graph_sha256=GRAPH_SHAS.copy(),
            graph_manifest_sha256=_sha(folder / "manifest.json"),
            original_goal_arrays_sha256=_arrays_sha(self.motion),
            contract_arrays_sha256=_arrays_sha({k:v for k,v in self.contract.items() if k!="body_names"}),
            native_model_parameters_sha256=_arrays_sha({key:getattr(native_model,key) for key in
                ("body_mass","body_inertia","body_pos","body_quat","jnt_range","jnt_pos","jnt_axis","dof_armature","dof_damping","dof_frictionloss","geom_type","geom_size","geom_pos","geom_quat","geom_friction","geom_contype","geom_conaffinity","actuator_forcerange","actuator_forcelimited","actuator_ctrlrange","actuator_ctrllimited","actuator_gear","actuator_gaintype","actuator_gainprm","actuator_biastype","actuator_biasprm","actuator_dyntype","actuator_dynprm","actuator_trntype","actuator_trnid")}),
            runtime_version=ort.__version__, runtime_binary_sha256={str(p):_sha(p) for p in runtime_files},
            mujoco=mujoco.__version__, numpy=np.__version__, threads=threads,
            module_sha256=_sha(__file__), observation_module_sha256=_sha(Path(__file__).with_name("g1_true23_bfm_seed_observations.py")),
            original_goal_horizon=8, position_gain=1., yaw_gain=2.,
            conservative_original_derivative_future_pose_frames=1,
            historical_action="normalized actually applied MPC target, no clipping of action history",
            local_future_action="BFM raw actor*5 before native target clipping",
            terminal_goal="hold last known original goal beyond archive, available-window latent average",
            real_history_updated_only_by_record_control=True, private_physics_hz=500,
            private_rollout_warning_or_clock_reset="typed rejection; caller retains other seed candidates",
            root_assistance=False, robot_or_DDS_interface=False, deployment_qualified=False)

    def identity(self):
        return json.loads(json.dumps(self._identity))

    @staticmethod
    def _state(qpos, qvel):
        qpos, qvel = np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64)
        if qpos.shape != (30,) or qvel.shape != (29,) or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("BFM seed requires finite actual qpos30/qvel29")
        return qpos, qvel

    def _terms(self, qpos, qvel, action):
        return state_and_terms(qpos[7:], qvel[6:], qpos[3:7], qvel[3:6], action, self.contract["default_q"])

    def record_control(self, control, pre_qpos, pre_qvel, applied_target):
        if control != self.recorded_controls:
            raise ValueError("BFM measured-history clock mismatch")
        qpos, qvel = self._state(pre_qpos, pre_qvel)
        target = np.asarray(applied_target, dtype=np.float64)
        if target.shape != (23,) or not np.isfinite(target).all() or np.any(target < self.lo-1e-12) or np.any(target > self.hi+1e-12):
            raise ValueError("BFM history requires the selected actually clipped native target")
        _, terms = self._terms(qpos, qvel, self.previous_action)
        self.history.before_update(terms)
        c = self.contract
        self.previous_action = ((target-c["default_q"])*c["kp"]/(.25*c["training_effort"])).astype(np.float32)
        self.actual_action_max_abs = max(self.actual_action_max_abs, float(np.max(np.abs(self.previous_action))))
        self.actual_action_components_outside_five += int(np.sum(np.abs(self.previous_action)>5))
        self.recorded_controls += 1

    def _goal(self, frame, qpos):
        frame = min(frame, len(self.state)-1)
        stop = min(frame+8, len(self.state))
        states = self.state[frame:stop].copy(); priv = self.privileged[frame:stop].copy()
        refrot = _quaternion_matrix(self.motion["body_quat_w"][frame,0]); actrot = _quaternion_matrix(qpos[3:7])
        ry = np.arctan2(refrot[1,0],refrot[0,0]); ay = np.arctan2(actrot[1,0],actrot[0,0])
        dy = np.arctan2(np.sin(ry-ay),np.cos(ry-ay))
        omega = np.array([0.,0.,np.clip(2*dy,-.8,.8)])
        delta = self.motion["body_pos_w"][frame,0]-qpos[:3]; delta[2] = 0
        delta *= min(1.,.6/max(np.linalg.norm(delta),1e-8))
        heading_actual = np.array([[np.cos(ay),-np.sin(ay),0],[np.sin(ay),np.cos(ay),0],[0,0,1.]])
        for t,index in enumerate(range(frame,stop)):
            rot = _quaternion_matrix(self.motion["body_quat_w"][index,0]); yaw = np.arctan2(rot[1,0],rot[0,0])
            heading = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
            positions = np.vstack((np.zeros(3),priv[t,1:73].reshape(24,3)))
            velocity = self.motion["body_lin_vel_w"][index,0]
            local_delta = heading_actual.T@(velocity+delta)-heading.T@velocity
            priv[t,223:298] += (local_delta+np.cross(omega,positions)).reshape(-1)
            priv[t,298:373] += np.tile(omega,25)
            states[t,-3:] += omega
        latent = self.sessions["backward"].run(None,dict(state=np.ascontiguousarray(states,dtype=np.float32),
                                                         privileged=np.ascontiguousarray(priv,dtype=np.float32)))[0].mean(0,keepdims=True)
        return 16*latent/np.maximum(np.linalg.norm(latent,axis=-1,keepdims=True),1e-12)

    @staticmethod
    def _validate_private_physics(data, expected_time, control, physics_step):
        diagnostics = dict(control=int(control), private_physics_step=int(physics_step),
                           expected_time=float(expected_time), actual_time=float(data.time),
                           warning_counts=data.warning.number.tolist())
        if np.any(data.warning.number):
            raise BFMSeedRolloutError("MuJoCo warning in private BFM seed rollout", diagnostics)
        if not np.isfinite(data.time) or abs(data.time-expected_time) > 1e-12:
            raise BFMSeedRolloutError("private BFM seed physics clock reset or mismatch", diagnostics)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise BFMSeedRolloutError("nonfinite private BFM seed rollout", diagnostics)

    def propose(self, control, qpos, qvel, horizon=30):
        if control != self.recorded_controls or type(horizon) is not int or horizon < 1:
            raise ValueError("BFM proposal control/horizon mismatch")
        qpos, qvel = self._state(qpos,qvel)
        local_history = BFMHistory()
        for key in local_history.data: local_history.data[key][:] = self.history.data[key]
        action = self.previous_action.copy()
        data = mujoco.MjData(self.model); data.qpos[:],data.qvel[:] = qpos,qvel
        mujoco.mj_forward(self.model,data)
        self._validate_private_physics(data, 0., control, 0)
        c=self.contract; targets=[]; peak_range=peak_velocity=peak_effort=peak_tilt=0.
        minimum_height=float(data.qpos[2]); inference=[]; started=time.perf_counter()
        for local in range(horizon):
            tick=time.perf_counter()
            state,terms=self._terms(data.qpos,data.qvel,action)
            history=local_history.before_update(terms)
            z=self._goal(control+11+local,data.qpos)
            inputs=dict(state=state[None],last_action=action[None],history=history[None],z=z)
            action=self.sessions["actor"].run(None,{k:np.ascontiguousarray(v,dtype=np.float32) for k,v in inputs.items()})[0][0]*5
            if not np.isfinite(action).all():
                raise BFMSeedRolloutError("nonfinite BFM counterfactual action", dict(control=int(control), local_control=local))
            target=np.clip(c["default_q"]+action*.25*c["training_effort"]/c["kp"],self.lo,self.hi)
            targets.append(target.copy()); inference.append((time.perf_counter()-tick)*1000)
            for substep in range(10):
                data.ctrl[:]=np.clip(c["kp"]*(target-data.qpos[7:])-c["kd"]*data.qvel[6:],-c["native_effort"],c["native_effort"])
                mujoco.mj_step(self.model,data)
                physics_step = local*10+substep+1
                self._validate_private_physics(data, physics_step*.002, control, physics_step)
                peak_range=max(peak_range,float(np.maximum(self.lo-data.qpos[7:],data.qpos[7:]-self.hi).max()))
                peak_velocity=max(peak_velocity,float(np.max(np.abs(data.qvel[6:])/c["native_velocity"])))
                peak_effort=max(peak_effort,float(np.max(np.abs(data.qfrc_actuator[6:])/c["native_effort"])))
                peak_tilt=max(peak_tilt,float(np.arccos(np.clip(1-2*np.sum(data.qpos[4:6]**2),-1,1))))
                minimum_height=min(minimum_height,float(data.qpos[2]))
        diagnostics=dict(control=int(control),horizon=horizon,elapsed_ms=(time.perf_counter()-started)*1000,
            inference_ms_p50_p95_max=np.percentile(inference,[50,95,100]).tolist(),
            recorded_actual_controls=self.recorded_controls,actual_history_action_max_abs=self.actual_action_max_abs,
            actual_history_action_components_outside_five=self.actual_action_components_outside_five,
            current_history_action_max_abs=float(np.max(np.abs(self.history.data["actions"]))),
            private_range_excess_max_rad=peak_range,private_velocity_ratio_max=peak_velocity,private_effort_ratio_max=peak_effort,
            private_root_height_min_m=minimum_height,private_root_tilt_max_rad=peak_tilt,
            private_engine_warning_counts=data.warning.number.tolist(),
            terminal_goal_hold_controls=sum(control+11+t>=len(self.state) for t in range(horizon)),
            conservative_raw_pose_support_seconds=(horizon+8)*.02,
            actual_history_mutated_by_proposal=False,real_physics_state_mutated=False,
            real_controller_or_teacher_qualified=False)
        return np.asarray(targets),diagnostics
