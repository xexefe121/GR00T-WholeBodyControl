"""Versioned root-conditioned CPU policy diagnostics, never hardware control.

The existing nominal native23 physics referee remains the sole integrator.
This adapter receives copied observations and returns policy outputs plus
scheduled external pelvis forces; it cannot rewrite live robot state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_generalist_benchmark import FLAGS, run_reference_diagnostic


@dataclass(frozen=True)
class ForcePulse:
    """Bounded simulator-only force, indexed in 2 ms physics substeps."""

    start_substep: int
    duration_substeps: int
    force_world_n: tuple[float, float, float]

    def __post_init__(self):
        if type(self.start_substep) is not int or self.start_substep < 0:
            raise ValueError("force start must be a nonnegative integer substep")
        if type(self.duration_substeps) is not int or not 1 <= self.duration_substeps <= 250:
            raise ValueError("force duration must be 1..250 substeps")
        values = np.asarray(self.force_world_n)
        if (
            values.shape != (3,)
            or values.dtype.kind not in "fi"
            or not np.isfinite(values).all()
            or not 0 < np.linalg.norm(values) <= 100
        ):
            raise ValueError("force must be finite, nonzero, and at most 100 N")

    @property
    def stop_substep(self):
        return self.start_substep + self.duration_substeps


class RootFeedbackRuntimeAdapter:
    """Current-q10 measured feedback; desired velocity uses only q9/q10 proof."""

    def __init__(self, pulses=()):
        self.pulses = tuple(pulses)
        if len(self.pulses) > 8 or any(not isinstance(pulse, ForcePulse) for pulse in self.pulses):
            raise ValueError("root diagnostic allows at most eight declared force pulses")
        ordered = sorted(self.pulses, key=lambda pulse: pulse.start_substep)
        if any(left.stop_substep > right.start_substep for left, right in zip(ordered, ordered[1:])):
            raise ValueError("external force pulses may not overlap")
        self.records = []

    def infer(
        self,
        policy,
        encoder,
        history,
        *,
        control_index,
        desired_position_w,
        previous_desired_position_w,
        measured_qpos,
        measured_qvel,
    ):
        from gear_sonic.utils.g1_true23_root_feedback import root_feedback_numpy

        if measured_qpos.shape != (30,) or measured_qvel.shape != (29,):
            raise ValueError("root adapter requires copied native23 simulator state")
        desired = np.asarray(desired_position_w, dtype=np.float32)
        previous = np.asarray(previous_desired_position_w, dtype=np.float32)
        # Identical float32 arithmetic to the training proof-frame frontend.
        desired_velocity = (desired - previous) / np.float32(0.02)
        measured = measured_qpos[:3].astype(np.float32)
        quaternion = measured_qpos[3:7].astype(np.float32)
        measured_velocity = measured_qvel[:3].astype(np.float32)
        feedback = root_feedback_numpy(desired, measured, desired_velocity, measured_velocity, quaternion)
        if feedback.shape != (9,) or feedback.dtype != np.float32 or not np.isfinite(feedback).all():
            raise ValueError("root feature helper violated the finite float32 feedback9 ABI")
        self.records.append(
            dict(
                control_index=control_index,
                root_feedback9=feedback.copy(),
                desired_root_position_w=desired.copy(),
                previous_desired_root_position_w=previous.copy(),
                desired_root_velocity_w=desired_velocity.copy(),
                measured_root_position_w=measured.copy(),
                measured_root_velocity_w=measured_velocity.copy(),
                measured_root_quaternion_wxyz=quaternion.copy(),
            )
        )
        return policy.infer(encoder, history, feedback.copy())

    def external_force_world(self, physics_substep):
        result = np.zeros(3, dtype=np.float64)
        for pulse in self.pulses:
            if pulse.start_substep <= physics_substep < pulse.stop_substep:
                result[:] = pulse.force_world_n
        return result

    def contract(self):
        from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract

        return dict(
            kind="g1_true23_root_feedback_cpu_runtime_v1",
            root_feedback_contract=root_feedback_contract(),
            desired_position_frame="q10_current_causal_proof",
            desired_velocity="float32(root_q10-root_q9)/float32(0.02)",
            npz_body_linear_velocity_or_future_q11_consumed=False,
            measured_state_frame="post_integration_current_q10",
            measured_state_source="privileged_mujoco_free_joint_world_pose_and_linear_velocity",
            state_estimator_deployment_dependency=[
                "time-synchronized base position and orientation in a persistent world/start frame",
                "world-frame base linear velocity estimate with tested latency/drift/dropout handling",
                "source-to-robot world/start-frame registration and reset semantics",
            ],
            hardware_state_estimation_validated=False,
            adapter_receives_simulator_state_copies_only=True,
            state_rewrite_or_fallback_authority=False,
            forces=[asdict(pulse) for pulse in self.pulses],
            perturbation_application="external world-frame force on pelvis body1 at each scheduled 2ms substep",
            perturbation_is_root_actuator_or_pose_rewrite=False,
            **FLAGS,
        )

    def arrays(self):
        dimensions = {
            "root_feedback9": 9,
            "desired_root_position_w": 3,
            "previous_desired_root_position_w": 3,
            "desired_root_velocity_w": 3,
            "measured_root_position_w": 3,
            "measured_root_velocity_w": 3,
            "measured_root_quaternion_wxyz": 4,
        }
        return {
            key: np.asarray([row[key] for row in self.records], dtype=np.float32).reshape(-1, width)
            for key, width in dimensions.items()
        }


def load_root_feedback_pair(manifest_path, *, session_options=None, expected_release_compatibility=None):
    """Require the new two-input decoder ABI; never relabel an old 994-only pair."""
    import onnxruntime as ort

    from gear_sonic.utils.g1_true23_buffered_reference import reference_profile_contract
    from gear_sonic.utils.g1_true23_root_feedback import root_feedback_contract

    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    compatibility = manifest.get("release_compatibility")
    timing = (compatibility or {}).get("reference_timing", "causal_history")
    if compatibility != expected_release_compatibility:
        raise ValueError("root pair release compatibility differs from executing runtime")
    if compatibility is not None:
        from gear_sonic.utils.g1_true23_release_compatibility import validate_release_compatibility

        validate_release_compatibility(compatibility)
        if manifest.get("actor_contract", {}).get("release_compatibility") != compatibility:
            raise ValueError("root actor and runtime release compatibility differ")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("kind") != "g1_native23_root_feedback_diagnostic_pair"
        or manifest.get("diagnostic_only") is not True
        or manifest.get("semantic_profile") != reference_profile_contract(timing)
        or manifest.get("root_feedback_contract") != root_feedback_contract(timing)
        or any(
            manifest.get(flag) is not False
            for flag in (
                "deployment_ready",
                "promotion_eligible",
                "hardware_authorized",
                "active_motor_control_authorized",
                "completed_motion_qualification",
                "physical_root_state_estimator_qualified",
            )
        )
    ):
        raise ValueError("root pair lacks exact versioned diagnostic semantics/authorization contract")
    source = manifest.get("source", {})
    if (
        {
            name: manifest["encoder"].get(name)
            for name in ("input_name", "input_shape", "output_name", "output_shape")
        }
        != dict(input_name="teleop_obs", input_shape=[1, 267], output_name="token", output_shape=[1, 64])
        or manifest["decoder"].get("inputs")
        != [dict(name="obs_dict", shape=[1, 994]), dict(name="root_feedback", shape=[1, 9])]
        or manifest["decoder"].get("output_name") != "action"
        or manifest["decoder"].get("output_shape") != [1, 23]
    ):
        raise ValueError("root manifest requires exact encoder267 and two-input994+9 declaration")
    for key in ("checkpoint_sha256", "actor_state_sha256"):
        digest = source.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("root pair lacks source identity")
    paths, sessions = {}, {}
    for key in ("encoder", "decoder"):
        part = manifest[key]
        name = part.get("filename")
        if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("root pair component must name a sibling file")
        paths[key] = manifest_path.parent / name
        if sha256_file(paths[key]) != part.get("sha256"):
            raise ValueError("root pair component SHA256 mismatch")
        sessions[key] = ort.InferenceSession(
            str(paths[key]), sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        expected = {
            **(
                {"release_compatibility_sha256": compatibility["contract_sha256"]}
                if compatibility is not None
                else {}
            ),
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "actor_state_sha256": source["actor_state_sha256"],
            "root_feedback_contract_sha256": manifest["root_feedback_contract"]["contract_sha256"],
            "semantic_contract_sha256": manifest["semantic_profile"]["contract_sha256"],
            "artifact_role": "native23_root_feedback_diagnostic_" + key,
            "hardware_authorized": "false",
            "deployment_ready": "false",
        }
        metadata = sessions[key].get_modelmeta().custom_metadata_map
        if any(metadata.get(name) != value for name, value in expected.items()):
            raise ValueError("root ONNX components do not share source/feedback semantics")
    encoder, decoder = sessions["encoder"], sessions["decoder"]
    if (
        [(value.name, value.shape, value.type) for value in encoder.get_inputs()]
        != [("teleop_obs", [1, 267], "tensor(float)")]
        or [(value.name, value.shape, value.type) for value in encoder.get_outputs()]
        != [("token", [1, 64], "tensor(float)")]
        or [(value.name, value.shape, value.type) for value in decoder.get_inputs()]
        != [("obs_dict", [1, 994], "tensor(float)"), ("root_feedback", [1, 9], "tensor(float)")]
        or [(value.name, value.shape, value.type) for value in decoder.get_outputs()]
        != [("action", [1, 23], "tensor(float)")]
    ):
        raise ValueError("root ONNX requires encoder267 and decoder two-input994+9 ABI")
    if manifest["encoder"].get("parity", {}).get("parity_max_abs_error") != 0:
        raise ValueError("root encoder must have exact frozen token parity evidence")

    class Policy:
        def infer(self, encoder267, history930, root_feedback9):
            for value, shape in ((encoder267, (267,)), (history930, (930,)), (root_feedback9, (9,))):
                if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError("root ONNX inputs must match finite float32 267/930/9 ABI")
            token = encoder.run(["token"], {"teleop_obs": encoder267[None]})[0][0]
            combined = np.concatenate((token, history930)).astype(np.float32)
            raw = decoder.run(["action"], {"obs_dict": combined[None], "root_feedback": root_feedback9[None]})[0][
                0
            ]
            return raw.copy(), combined

    return Policy(), dict(
        manifest_path=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        source=source,
        encoder_sha256=manifest["encoder"]["sha256"],
        decoder_sha256=manifest["decoder"]["sha256"],
        component_paths={key: str(path) for key, path in paths.items()},
        root_feedback_contract=manifest["root_feedback_contract"],
        release_compatibility=compatibility,
        diagnostic_only=True,
    )


def summarize_root_perturbations(adapter, arrays, *, completed_controls, failure):
    """Report response without turning a prefix or position-only screen into a pass."""
    records = adapter.arrays()
    count = min(completed_controls, len(records["root_feedback9"]))
    root_error = np.linalg.norm(
        records["desired_root_position_w"][:count] - records["measured_root_position_w"][:count], axis=1
    )
    velocity_error = np.linalg.norm(
        records["desired_root_velocity_w"][:count] - records["measured_root_velocity_w"][:count], axis=1
    )
    forces = arrays.get("physics_external_force_world_n", np.empty((0, 3))).reshape(-1, 3)
    rows = []
    for pulse in adapter.pulses:
        recovery_start = (pulse.stop_substep + 9) // 10
        recovery_stop = recovery_start + 100  # Explicit two-second response window.
        available_stop = min(count, recovery_stop)
        sampled = root_error[recovery_start:available_stop]
        sampled_velocity = velocity_error[recovery_start:available_stop]
        applied_count = max(0, min(len(forces), pulse.stop_substep) - pulse.start_substep)
        rows.append(
            dict(
                pulse=asdict(pulse),
                expected_impulse_world_ns=(
                    np.asarray(pulse.force_world_n) * pulse.duration_substeps * 0.002
                ).tolist(),
                applied_impulse_world_ns=(
                    forces[pulse.start_substep : pulse.stop_substep].sum(axis=0) * 0.002
                ).tolist(),
                scheduled_force_fully_integrated=applied_count == pulse.duration_substeps,
                recovery_control_start=recovery_start,
                recovery_control_stop=recovery_stop,
                recovery_window_fully_observed=count >= recovery_stop,
                recovery_root_error_max_m=float(sampled.max()) if len(sampled) else None,
                recovery_root_error_final_m=float(sampled[-1]) if len(sampled) else None,
                recovery_velocity_error_final_m_s=float(sampled_velocity[-1]) if len(sampled_velocity) else None,
                perturbation_recovery_qualified=False,
            )
        )
    return dict(
        pulses=rows,
        current_root_position_error_p95_m=float(np.percentile(root_error, 95)) if count else None,
        current_root_velocity_error_p95_m_s=float(np.percentile(velocity_error, 95)) if count else None,
        closed_loop_feedback_observations=count,
        control_failure=failure,
        dynamics_or_contact_qualification=False,
        **FLAGS,
    )


def run_root_feedback_diagnostic(*, root, asset_root, motion_path, policy, pulses=(), maximum_controls=None):
    adapter = RootFeedbackRuntimeAdapter(pulses)
    report, arrays = run_reference_diagnostic(
        root=root,
        asset_root=asset_root,
        motion_path=motion_path,
        policy=policy,
        profile="native_model",
        maximum_controls=maximum_controls,
        runtime_adapter=adapter,
    )
    arrays.update(adapter.arrays())
    report["kind"] = "g1_true23_root_feedback_reference_diagnostic_v1"
    report["root_response"] = summarize_root_perturbations(
        adapter, arrays, completed_controls=report["completed_controls"], failure=report["failure"]
    )
    return report, arrays
