"""SIM-only hypothetical source29 dynamics, separate from physical native23.

Only root/retained23 measurements are assimilated. Six absent joint states
persist in a separate MuJoCo model; they are explicitly predictions, never
physical measurements. No robot transport, shared-state mutation or native
actuator command exists in this module. A hypothesis, not a deployed observer.
"""

from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.scripts import simulate_g1_sonic_library_motions as original
from gear_sonic.utils.g1_23dof_contract import SOURCE_IL29_EXCLUDED_INDICES, SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters, file_sha256
from gear_sonic.utils.g1_true23_original29_reference import SOURCE_JOINT_NAMES
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

KEEP_HW = np.asarray(SOURCE_MJ29_KEEP_INDICES)
MISSING_IL = np.asarray(SOURCE_IL29_EXCLUDED_INDICES)
MISSING_HW = np.asarray(original.ISAAC_TO_MUJOCO_INDEX)[MISSING_IL]
MODEL_FILE_SHA256 = "40bd428f7a2e090a3bb2f85f61e965be179b6ab91b63d6f1a827733cf6a0cd96"


class VirtualSourceModel:
    def __init__(self, model_path, parameters, *, source_effort29):
        if not isinstance(parameters, CppParameters):
            raise ValueError("virtual source requires captured, validated source parameters")
        self.path = Path(model_path).resolve(strict=True)
        if file_sha256(self.path) != MODEL_FILE_SHA256:
            raise ValueError("virtual source model differs from its source29 research pin")
        self.model = mujoco.MjModel.from_binary_path(str(self.path))
        model = self.model
        if (model.nq, model.nv, model.nu, model.njnt) != (36, 35, 29, 30) or model.opt.timestep != 0.002:
            raise ValueError("internal source model requires29 scalar joints and2ms physics")
        if tuple(model.joint(i).name for i in range(1, 30)) != tuple(SOURCE_JOINT_NAMES):
            raise ValueError("virtual source joint order differs")
        if not np.array_equal(model.actuator_trnid[:, 0], np.arange(1, 30)):
            raise ValueError("virtual source requires canonical direct transmissions")
        self.parameters = parameters
        self.effort = np.array(source_effort29, dtype=np.float64, copy=True)
        if self.effort.shape != (29,) or not np.isfinite(self.effort).all() or np.any(self.effort <= 0):
            raise ValueError("virtual source requires explicit positive source-only effort29")
        self.data = mujoco.MjData(model)
        self.data.qpos[7:] = parameters.default_angles
        self.completed = 0
        self.failed = False
        self.initialized = False
        self.internal_physics_steps = 0
        self.maximum_virtual_joint_excess = 0.0
        self.maximum_internal_all29_joint_excess = 0.0
        self._model_hash = compiled_model_sha256(model)

    def internal_state12(self):
        return np.r_[
            self.data.qpos[7 + MISSING_HW] - self.parameters.default_angles[MISSING_HW],
            self.data.qvel[6 + MISSING_HW],
        ].astype(np.float32)

    def advance(self, measured_qpos23, measured_qvel23, source_raw29):
        if self.failed:
            raise RuntimeError("failed virtual source cannot restart implicitly")
        for value, size in ((measured_qpos23, 30), (measured_qvel23, 29), (source_raw29, 29)):
            if not isinstance(value, np.ndarray) or value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError("virtual source requires finite native23 state and full29 source proposal")
        if abs(np.linalg.norm(measured_qpos23[3:7]) - 1) > 1e-5:
            raise ValueError("virtual source requires normalized measured root orientation")
        data, model, p = self.data, self.model, self.parameters
        # Writes are confined to this separate prediction model, never native23.
        data.qpos[:7] = measured_qpos23[:7]
        data.qvel[:6] = measured_qvel23[:6]
        data.qpos[7 + KEEP_HW] = measured_qpos23[7:]
        data.qvel[6 + KEEP_HW] = measured_qvel23[6:]
        if not self.initialized:
            mujoco.mj_forward(model, data)
            self.initialized = True
        with np.errstate(over="raise", invalid="raise"):
            # Source-only finite C++ affine formula, not a relaxed native target.
            raw = source_raw29.astype(np.float32).astype(np.float64)
            target = (
                (p.default_angles + raw[original.MUJOCO_TO_ISAAC_INDEX] * p.action_scale)
                .astype(np.float32)
                .astype(np.float64)
            )
        if not np.isfinite(target).all():
            self.failed = True
            raise RuntimeError("virtual source target became nonfinite")
        for _ in range(10):
            data.ctrl[:] = np.clip(
                p.kps * (target - data.qpos[7:]) - p.kds * data.qvel[6:], -self.effort, self.effort
            )
            mujoco.mj_step(model, data)
            self.internal_physics_steps += 1
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                self.failed = True
                raise RuntimeError("virtual source prediction became nonfinite")
            q = data.qpos[7:]
            limits = model.jnt_range[1:]
            excess = np.maximum(np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0)
            self.maximum_virtual_joint_excess = max(
                self.maximum_virtual_joint_excess, float(excess[MISSING_HW].max())
            )
            self.maximum_internal_all29_joint_excess = max(
                self.maximum_internal_all29_joint_excess, float(excess.max())
            )
        self.completed += 1
        return self.internal_state12()

    def descriptor(self):
        if compiled_model_sha256(self.model) != self._model_hash:
            raise ValueError("virtual source physical model changed during prediction")
        return dict(
            kind="internal_hypothetical_source29_mujoco_prediction_v1",
            source_model_sha256=MODEL_FILE_SHA256,
            source_compiled_sha256=self._model_hash,
            missing_source_il_indices=MISSING_IL.tolist(),
            missing_source_hardware_indices=MISSING_HW.tolist(),
            retained_native23_hardware_map=KEEP_HW.tolist(),
            source_only_effort29=self.effort.tolist(),
            completed_predictions=self.completed,
            internal_source_physics_steps=self.internal_physics_steps,
            maximum_predicted_missing_joint_range_excess_rad=self.maximum_virtual_joint_excess,
            maximum_internal_all29_joint_range_excess_rad=self.maximum_internal_all29_joint_excess,
            source_only_prediction_not_a_native_plant=True,
            missing_physical_measurements_fabricated=False,
            native_model_or_state_written=False,
            failed=self.failed,
            hardware_authorized=False,
            deployment_ready=False,
        )
