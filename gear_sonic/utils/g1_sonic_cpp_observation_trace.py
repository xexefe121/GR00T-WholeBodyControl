"""Original29 diagnostic with actual captured C++ observation gatherers.

Reuse the parameter loop's bytecode with a private observation provider; never
patch shared modules. Retain legacy reference resampling and nominal torque
clipping. This is neither the live/receding-horizon planner nor full C++/DDS
execution. Record actual MuJoCo time and warning counters, not only synthetic
sampling labels, so engine resets/warnings cannot become successful teachers.
"""

from __future__ import annotations

from types import FunctionType, SimpleNamespace

import mujoco
import numpy as np

from gear_sonic.scripts import simulate_g1_sonic_library_motions as legacy
from gear_sonic.utils import g1_sonic_original29_trace as original
from gear_sonic.utils.g1_sonic_cpp_observations import CppObservations

PROFILE = "cpp_captured_observations_and_parameters_v1"


def same_state_observation_comparison(planner_qpos, baseline, encoder, decoder, *, capture):
    """Replay old recorded *states*, never new actions, through the C++ boundary.

    Keep the final rejected inference, if present. This isolates observation
    changes without confusing them with closed-loop trajectory divergence.
    It is not a second physics rollout and cannot qualify the old trajectory.
    """
    raw = np.asarray(planner_qpos, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 36 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError("same-state comparison requires the complete original reference")
    root, quaternion, joint, velocity = legacy._resample_reference(raw)
    count, calls = len(baseline["pre_qpos"]), len(baseline["encoder_inputs"])
    shapes = {
        "pre_qpos": (count, 36),
        "qpos": (count, 36),
        "pre_qvel": (count, 35),
        "post_qvel": (count, 35),
        "raw_actions": (count, 29),
        "encoder_inputs": (calls, 1762),
        "encoder_outputs": (calls, 64),
        "decoder_inputs": (calls, 994),
        "decoder_outputs": (calls, 29),
    }
    if count < 1 or calls not in (count, count + 1) or calls > len(joint):
        raise ValueError("same-state comparison cannot trim or invent inference calls")
    for key, shape in shapes.items():
        if baseline[key].shape != shape or not np.isfinite(baseline[key]).all():
            raise ValueError(f"invalid recorded baseline boundary: {key}")
    planned = np.column_stack((root, quaternion, joint[:, legacy.MUJOCO_TO_ISAAC_INDEX]))
    if (
        not np.array_equal(planned, baseline["planned_qpos50"])
        or not np.array_equal(baseline["pre_qpos"][1:], baseline["qpos"][:-1])
        or not np.array_equal(baseline["pre_qvel"][1:], baseline["post_qvel"][:-1])
        or not np.array_equal(baseline["raw_actions"], baseline["decoder_outputs"][:count])
        or not np.array_equal(baseline["encoder_outputs"], baseline["decoder_inputs"][:, :64])
    ):
        raise ValueError("recorded reference, state, action or token lineage changed")
    gathered, tokens, decoder_inputs, actions = [], [], [], []
    with CppObservations(capture, quaternion, joint, velocity, raw[0, 3:7]) as observer:
        for frame in range(calls):
            pose = baseline["pre_qpos"][frame] if frame < count else baseline["qpos"][-1]
            speed = baseline["pre_qvel"][frame] if frame < count else baseline["post_qvel"][-1]
            previous_action = np.zeros(29) if frame == 0 else baseline["raw_actions"][frame - 1]
            observer.append(pose, speed, previous_action)
            observation, history = observer.gather(frame)
            token = np.asarray(encoder.run(["encoded_tokens"], {"obs_dict": observation[None]})[0])
            if token.shape != (1, 64) or not np.isfinite(token).all():
                raise ValueError("same-state encoder output drift")
            decoder_input = np.concatenate((token[0], history)).astype(np.float32)
            action = np.asarray(decoder.run(["action"], {"obs_dict": decoder_input[None]})[0])
            if action.shape != (1, 29) or not np.isfinite(action).all():
                raise ValueError("same-state decoder output drift")
            gathered.append(observation)
            tokens.append(token[0])
            decoder_inputs.append(decoder_input)
            actions.append(action[0])
    arrays = dict(
        encoder_inputs=np.asarray(gathered),
        encoder_outputs=np.asarray(tokens),
        decoder_inputs=np.asarray(decoder_inputs),
        decoder_outputs=np.asarray(actions),
    )

    def difference(new, old):
        delta = np.abs(new.astype(float) - old.astype(float))
        per_frame = delta.max(axis=1)
        return {
            "maximum_absolute_difference": float(delta.max()),
            "maximum_initial_nine_calls": float(per_frame[:9].max()),
            "maximum_after_nine_calls": float(per_frame[9:].max()) if calls > 9 else None,
            "elements_numerically_changed": int(np.count_nonzero(delta)),
            "calls_numerically_changed": int(np.count_nonzero(per_frame)),
            "worst_call": int(np.argmax(per_frame)),
        }

    segments = {
        "angular_velocity": (0, 30),
        "joint_position": (30, 320),
        "joint_velocity": (320, 610),
        "previous_action": (610, 900),
        "gravity": (900, 930),
    }
    return arrays, {
        "kind": "g1_sonic_cpp_same_recorded_state_observation_comparison_v1",
        "completed_baseline_control_steps": count,
        "inference_calls_compared": calls,
        "rejected_final_baseline_inference_retained": calls == count + 1,
        "differences": {key: difference(value, baseline[key]) for key, value in arrays.items()},
        "history_segment_differences": {
            name: difference(
                arrays["decoder_inputs"][:, 64 + start : 64 + end],
                baseline["decoder_inputs"][:, 64 + start : 64 + end],
            )
            for name, (start, end) in segments.items()
        },
        "counterfactual_actions_at_or_above_legacy_bound": int(
            np.sum(np.max(np.abs(arrays["decoder_outputs"]), axis=1) >= 10)
        ),
        "counterfactual_actions_executed": False,
        "closed_loop_comparison": False,
        "teacher_accepted": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def audit_engine_trace(arrays):
    if not np.array_equal(arrays["physics_dt"], [0.002]):
        raise ValueError("engine audit requires the unchanged 2-ms physics period")
    before, after = arrays["physics_engine_pre_time_s"], arrays["physics_engine_post_time_s"]
    dt = float(arrays["physics_dt"][0])
    warnings = arrays["physics_engine_warning_counts"]
    if before.ndim != 1 or after.shape != before.shape or len(before) < 1 or warnings.shape != (len(before), 8):
        raise ValueError("engine audit needs every time sample and all eight warning counters")
    finite = bool(np.isfinite(before).all() and np.isfinite(after).all() and np.isfinite(warnings).all())
    clock = bool(
        finite
        and before[0] == 0
        and np.array_equal(before[1:], after[:-1])
        and np.allclose(after - before, dt, atol=1e-9, rtol=0)
        and np.allclose(after, (np.arange(len(after)) + 1) * dt, atol=1e-9, rtol=0)
    )
    warnings_valid = bool(np.isfinite(warnings).all() and np.all(warnings >= 0) and not np.any(warnings))
    return {
        "passed": clock and warnings_valid,
        "physics_steps_checked": len(before),
        "actual_engine_time_continuous": clock,
        "actual_final_engine_time_s": float(after[-1]) if np.isfinite(after[-1]) else None,
        "expected_final_engine_time_s": len(after) * dt,
        "all_runtime_warning_counters_zero": warnings_valid,
        "maximum_warning_counts": warnings.max(axis=0).tolist(),
        "warning_names": [mujoco.mjtWarning(index).name for index in range(8)],
        "clock_tolerance_s": 1e-9,
        "original_loop_not_silently_restarted_or_rewritten": True,
    }


def private_function(function, **replacements):
    result = FunctionType(
        function.__code__,
        {**function.__globals__, **replacements},
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def trace_cpp_observation_motion(model, encoder, decoder, planner_qpos, *, parameters, capture):
    raw = np.array(planner_qpos, dtype=float, copy=True)
    if (
        raw.ndim != 2
        or raw.shape[1] != 36
        or len(raw) < 2
        or not np.isfinite(raw).all()
        or not np.allclose(np.linalg.norm(raw[:, 3:7], axis=1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("requires the complete finite original29 source")
    _, quaternion, joint, velocity = legacy._resample_reference(raw)
    recorders = []

    class EngineRecorder(original._RecordedMujoco):
        def __init__(self, active_model):
            super().__init__(active_model)
            self.times_before, self.times_after, self.warnings, self.warning_info = [], [], [], []
            recorders.append(self)

        def mj_step(self, active_model, data):
            self.times_before.append(float(data.time))
            super().mj_step(active_model, data)
            self.times_after.append(float(data.time))
            self.warnings.append([int(item.number) for item in data.warning])
            self.warning_info.append([int(item.lastinfo) for item in data.warning])

    with CppObservations(capture, quaternion, joint, velocity, raw[0, 3:7]) as observer:
        latest_history = None

        def current_state(data, last_action):
            observer.append(data.qpos, data.qvel, last_action)
            # The original loop retains its Python bookkeeping, but policy
            # inputs below come from the actual C++ StateLogger/gatherers.
            return legacy._current_history_frame(data, last_action)

        def encoder_state(ref_quaternion, ref_joint, ref_velocity, frame, robot_quaternion, heading):
            nonlocal latest_history
            result, latest_history = observer.gather(frame, play=True)
            return result

        def policy_state(history):
            if latest_history is None:
                raise ValueError("C++ observation order changed")
            return latest_history.copy()

        stock = SimpleNamespace(**vars(legacy))
        stock._current_history_frame = current_state
        stock._encoder_observation = encoder_state
        stock._policy_observation = policy_state
        loop = private_function(original._simulate_parameter_variant, stock=stock)
        traced = private_function(
            original.trace_original29_motion, _simulate_parameter_variant=loop, _RecordedMujoco=EngineRecorder
        )
        arrays, details = traced(model, encoder, decoder, raw, parameters=parameters)
    if len(recorders) != 1:
        raise ValueError("the trace must use one uninterrupted engine instance")
    recorder = recorders[0]
    arrays.update(
        physics_engine_pre_time_s=np.asarray(recorder.times_before),
        physics_engine_post_time_s=np.asarray(recorder.times_after),
        physics_engine_warning_counts=np.asarray(recorder.warnings, dtype=np.int64),
        physics_engine_warning_lastinfo=np.asarray(recorder.warning_info, dtype=np.int64),
    )
    engine = audit_engine_trace(arrays)
    details.update(
        profile=PROFILE,
        complete_original_clip=bool(details["complete_original_clip"] and engine["passed"]),
        actual_engine_audit=engine,
        actual_cpp_observation_gatherers_executed=True,
        cpp_observation_witness_sha256=capture["binary_sha256"],
        parameter_loop_bytecode_reused_with_private_observation_provider=True,
        sensor_float32_precision_boundary_simulated=True,
        legacy_reference_resampling_and_nominal_effort_clipping_retained=True,
        live_receding_horizon_planner_executed=False,
        padding_gravity_is_legacy_cpp_semantics_not_a_valid_sensor_measurement=True,
        complete_cpp_deployment_equivalence_proven=False,
    )
    return arrays, details
