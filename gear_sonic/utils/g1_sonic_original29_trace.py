"""Record legacy and explicitly parameter-corrected original29 simulations.

The legacy simulator uses private globals with a delegated step recorder.
An additive variant applies captured C++ gains/defaults/scales and the final
float32 motor-target cast. It retains legacy reference/observation algorithms
and torque clipping; this is NOT full C++ deployment or firmware equivalence.
No shared module, policy, gain or model is patched.
Both pre-command and post-control states are retained, with every physics
substep and ONNX input/output. This is source-baseline evidence, not a native23
teacher, hardware model or hardware authorization.
"""

from __future__ import annotations

from collections import deque
from types import FunctionType

import mujoco
import numpy as np

from gear_sonic.scripts import simulate_g1_sonic_library_motions as stock
from gear_sonic.utils.g1_sonic_cpp_parameters import CppParameters
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256


class _RecordedSession:
    def __init__(self, session):
        self.session = session
        self.inputs, self.outputs = [], []

    def run(self, names, feeds):
        if list(feeds) != ["obs_dict"]:
            raise ValueError("original29 trace requires the exact ONNX observation boundary")
        self.inputs.append(np.array(feeds["obs_dict"], copy=True))
        outputs = self.session.run(names, feeds)
        if len(outputs) != 1:
            raise ValueError("original29 trace requires exactly one ONNX output")
        self.outputs.append(np.array(outputs[0], copy=True))
        return outputs


class _RecordedMujoco:
    def __init__(self, model):
        self.model = model
        self.pre_qpos, self.post_qpos, self.pre_qvel, self.post_qvel = [], [], [], []
        self.commands, self.actual_forces = [], []

    def __getattr__(self, name):
        return getattr(mujoco, name)

    def mj_step(self, model, data):
        if model is not self.model:
            raise ValueError("trace cannot silently switch the simulation model")
        self.pre_qpos.append(data.qpos.copy())
        self.pre_qvel.append(data.qvel.copy())
        self.commands.append(data.ctrl.copy())
        mujoco.mj_step(model, data)
        self.post_qpos.append(data.qpos.copy())
        self.post_qvel.append(data.qvel.copy())
        self.actual_forces.append(data.qfrc_actuator[6:].copy())


def _simulate_parameter_variant(model, encoder, decoder, raw, recorder, parameters, diagnostics):
    """Legacy observation/playback semantics with explicit C++ command arithmetic.

    Initial joints stay at the legacy float32 standing command, not at source
    motion joints. Full C++ state acquisition, history startup, playback and
    firmware behavior are not executed by this offline comparison.
    """
    _, quaternions, joints, velocity = stock._resample_reference(raw)
    data = mujoco.MjData(model)
    data.qpos[:7] = raw[0, :7]
    data.qpos[7:] = parameters.default_angles.astype(np.float32)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    history = deque((stock._zero_history_frame() for _ in range(9)), maxlen=10)
    last_action = np.zeros(29, dtype=float)
    heading = stock._quat_multiply(
        stock._heading_quaternion(data.qpos[3:7]),
        stock._quat_conjugate(stock._heading_quaternion(quaternions[0])),
    )
    poses, actions, targets = [], [], []
    maximum_velocity, maximum_torque_ratio = 0.0, 0.0
    minimum_height, failure = float("inf"), None
    for frame in range(len(joints)):
        state = stock._current_history_frame(data, last_action)
        # C++ subtracts its double default, not the old Python float32 default.
        state["joint_position"] = (data.qpos[7:] - parameters.default_angles)[stock.ISAAC_TO_MUJOCO_INDEX]
        history.append(state)
        observation = stock._encoder_observation(quaternions, joints, velocity, frame, data.qpos[3:7], heading)
        token = encoder.run(["encoded_tokens"], {"obs_dict": observation[None]})[0]
        decoder_input = np.concatenate((token.reshape(-1), stock._policy_observation(history))).astype(np.float32)
        action = np.asarray(decoder.run(["action"], {"obs_dict": decoder_input[None]})[0], dtype=float).reshape(-1)
        if action.shape != (29,) or not np.isfinite(action).all() or np.max(np.abs(action)) >= 10:
            failure = "invalid_or_out_of_bounds_action"
            break
        target = parameters.target(action)
        for _ in range(10):
            torque = parameters.kps * (target - data.qpos[7:]) - parameters.kds * data.qvel[6:]
            data.ctrl[:] = np.clip(torque, -parameters.effort, parameters.effort)
            maximum_torque_ratio = max(maximum_torque_ratio, float(np.max(np.abs(data.ctrl) / parameters.effort)))
            recorder.mj_step(model, data)
        poses.append(data.qpos.copy())
        actions.append(action.copy())
        targets.append(target.copy())
        last_action = action
        maximum_velocity = max(maximum_velocity, float(np.max(np.abs(data.qvel[6:]))))
        minimum_height = min(minimum_height, float(data.qpos[2]))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.qpos[2] < 0.08:
            failure = "invalid_or_low_state"
            break
    if not poses:
        raise ValueError(f"parameter variant completed no physics transitions: {failure}")
    poses, actions, targets = map(np.asarray, (poses, actions, targets))
    diagnostics.update(raw_actions=actions, target_positions_hardware29=targets)
    return poses, {
        "passed": len(poses) == len(joints) and failure is None,
        "requested_control_steps": len(joints),
        "completed_control_steps": len(poses),
        "physics_steps_per_control": 10,
        "minimum_root_height_m": minimum_height,
        "maximum_absolute_joint_velocity_rad_s": maximum_velocity,
        "maximum_torque_limit_ratio": maximum_torque_ratio,
        "joint_reference_rmse_rad": float(
            np.sqrt(np.mean((poses[:, 7:] - joints[: len(poses), stock.MUJOCO_TO_ISAAC_INDEX]) ** 2))
        ),
        "maximum_absolute_raw_action": float(np.max(np.abs(actions))),
        "maximum_absolute_target_position_rad": float(np.max(np.abs(targets))),
        "horizontal_displacement_m": float(np.linalg.norm(poses[-1, :2] - poses[0, :2])),
        "failure": failure,
    }


def trace_original29_motion(model, encoder, decoder, planner_qpos, *, parameters=None):
    """Trace either the exact legacy loop or an explicitly captured-parameter variant."""
    if parameters is not None and not isinstance(parameters, CppParameters):
        raise ValueError("the parameter variant requires validated CppParameters")
    raw = np.array(planner_qpos, dtype=float, copy=True)
    if (
        (model.nq, model.nv, model.nu) != (36, 35, 29)
        or raw.ndim != 2
        or raw.shape[1] != 36
        or len(raw) < 2
        or not np.isfinite(raw).all()
        or not np.allclose(np.linalg.norm(raw[:, 3:7], axis=1), 1, atol=1e-6, rtol=0)
        or not np.isclose(model.opt.timestep, 0.002, atol=1e-12, rtol=0)
    ):
        raise ValueError("original29 trace needs exact ABI, finite unit-quaternion planner poses and 2-ms physics")
    model_hash = compiled_model_sha256(model)
    recorder = _RecordedMujoco(model)
    recorded_encoder, recorded_decoder = _RecordedSession(encoder), _RecordedSession(decoder)
    original = stock.simulate_motion
    traced = FunctionType(
        original.__code__,
        {**original.__globals__, "mujoco": recorder},
        name=original.__name__,
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    traced.__kwdefaults__ = original.__kwdefaults__
    diagnostics = {}
    if parameters is None:
        poses, metrics = traced(model, recorded_encoder, recorded_decoder, raw, diagnostic_arrays=diagnostics)
    else:
        poses, metrics = _simulate_parameter_variant(
            model, recorded_encoder, recorded_decoder, raw, recorder, parameters, diagnostics
        )
    if compiled_model_sha256(model) != model_hash:
        raise ValueError("stock simulation changed its compiled model")
    count = len(poses)
    substeps = metrics["physics_steps_per_control"]
    if count < 1 or len(recorder.pre_qpos) != count * substeps or substeps != 10:
        raise ValueError("trace did not retain every original29 physics substep")
    physics_pre = np.asarray(recorder.pre_qpos)
    physics_post = np.asarray(recorder.post_qpos)
    pre_qpos, post_qpos = physics_pre[::substeps], physics_post[substeps - 1 :: substeps]
    np.testing.assert_array_equal(post_qpos, poses)
    np.testing.assert_array_equal(pre_qpos[1:], post_qpos[:-1])
    np.testing.assert_array_equal(physics_pre[1:], physics_post[:-1])
    root, quaternion, joint_isaac, velocity_isaac = stock._resample_reference(raw)
    planned = np.column_stack((root, quaternion, joint_isaac[:, stock.MUJOCO_TO_ISAAC_INDEX]))
    arrays = {
        "control_dt": np.array([stock.CONTROL_DT]),
        "physics_dt": np.array([model.opt.timestep]),
        "command_time_s": np.arange(count) * stock.CONTROL_DT,
        "post_control_time_s": (np.arange(count) + 1) * stock.CONTROL_DT,
        "planned_qpos50": planned,
        "planned_joint_velocity_hardware29": velocity_isaac[:, stock.MUJOCO_TO_ISAAC_INDEX],
        "pre_qpos": pre_qpos.copy(),
        "qpos": post_qpos.copy(),
        "pre_qvel": np.asarray(recorder.pre_qvel)[::substeps].copy(),
        "post_qvel": np.asarray(recorder.post_qvel)[substeps - 1 :: substeps].copy(),
        "physics_pre_qpos": physics_pre,
        "physics_post_qpos": physics_post,
        "physics_pre_qvel": np.asarray(recorder.pre_qvel),
        "physics_post_qvel": np.asarray(recorder.post_qvel),
        "physics_requested_torque_hardware29": np.asarray(recorder.commands),
        "physics_actual_actuated_generalized_force": np.asarray(recorder.actual_forces),
        "encoder_inputs": np.concatenate(recorded_encoder.inputs),
        "encoder_outputs": np.concatenate(recorded_encoder.outputs),
        "decoder_inputs": np.concatenate(recorded_decoder.inputs),
        "decoder_outputs": np.concatenate(recorded_decoder.outputs),
        **diagnostics,
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite original29 trace is not a usable motion source")
    # A rejected action can add an ONNX call without a completed control step.
    if len(recorded_encoder.inputs) not in (count, count + 1) or len(recorded_decoder.inputs) != len(
        recorded_encoder.inputs
    ):
        raise ValueError("ONNX/control trace lengths disagree")
    details = {
        "stock_metrics": metrics,
        "compiled_model_sha256": model_hash,
        "frames_requested": len(planned),
        "frames_completed": count,
        "physics_steps_recorded": len(physics_pre),
        "onnx_calls_recorded": len(recorded_encoder.inputs),
        "complete_original_clip": count == len(planned) and metrics["passed"],
        "profile": "legacy_python" if parameters is None else "cpp_parameters_and_float32_targets",
        "stock_simulator_bytecode_executed_without_control_law_changes": parameters is None,
        "cpp_header_parameters_and_float32_command_targets": parameters is not None,
        "complete_cpp_deployment_equivalence_proven": False,
        "firmware_torque_clipping_equivalence_proven": False,
        "legacy_simulated_nominal_effort_clipping_retained": True,
        "shared_modules_or_models_modified": False,
        "qpos_is_post_control_state": True,
        "pre_qpos_is_same_command_time_state": True,
        "final_post_control_state_retained": True,
        "planner_root_xyz_is_not_an_encoder_input": True,
        "native23_joint_velocity_and_effort_limits_applied_to_stock29": False,
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    return arrays, details
