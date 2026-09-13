"""Offline SONIC v1.1 inference and explicit heading-reference native23 adapter.

This is a different frozen release, not a relabelled old normal/low-latency
checkpoint. Its timing is the normal step5 profile but its orientation and full
encoder ABI differ. No motor ratings, range preview, or hardware entry points
are modified. The adapter is diagnostic until full physical qualification.
"""

import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from gear_sonic.teleop.buffered_source_horizon import _relative_orientation_6d
from gear_sonic.utils.g1_23dof_contract import (
    NATIVE_IL23_TO_CANONICAL_IL29,
    REFERENCE_PROFILE_NORMAL,
    SOURCE_IL29_EXCLUDED_INDICES,
)
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_released_core_comparison import ReleasedCoreAdapter
from gear_sonic.utils.g1_true23_source_action_codec import source_scaled_precompensation

MODEL_SHA256 = {
    "model_encoder.onnx": "fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae",
    "model_decoder.onnx": "34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509",
    "observation_config.yaml": "4a67713b310932e50aca81f19188c8d76013148e98b15c8b5bbea995f12e59f0",
}
SEMANTIC_SOURCE_INDICES = np.r_[np.arange(650, 911), np.arange(644, 650)]
SEMANTIC_SOURCE_INDICES.setflags(write=False)


def finite_vector(value, width, name):
    result = np.asarray(value)
    if result.shape != (width,) or result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError(f"{name} requires finite float32 vector{width}")
    return result


def heading_orientation_6d(measured_wxyz, reference_wxyz):
    """Upstream mode1: inverse robot yaw times full reference orientation.

    Heading follows calc_heading_d's rotated +X axis, including its treatment
    of nearly-unit floating-point inputs. Reference matrix normalization is
    inherited from the existing quaternion-to-6D primitive. Result is row-wise
    first two columns, never Euler angles or reference-heading normalization.
    """
    measured = np.asarray(measured_wxyz, dtype=np.float64)
    reference = np.asarray(reference_wxyz, dtype=np.float64)
    for value in (measured, reference):
        if value.shape != (4,) or not np.isfinite(value).all() or abs(np.linalg.norm(value) - 1) > 1e-4:
            raise ValueError("v1.1 heading requires finite normalized WXYZ quaternions")
    w, x, y, z = measured
    yaw = math.atan2(2.0 * z * w + 2.0 * y * x, (2.0 * w * w - 1.0) + 2.0 * x * x)
    heading = np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=np.float64)
    return _relative_orientation_6d(heading, reference)


def semantic_from_window(window, measured_wxyz):
    return np.concatenate(
        (
            window.lower_body240,
            window.virtual_vr21,
            heading_orientation_6d(measured_wxyz, window.anchor_root_quaternion_wxyz),
        )
    ).astype(np.float32)


def pack_v11_encoder(semantic267):
    semantic = finite_vector(semantic267, 267, "v1.1 heading semantic")
    packed = np.zeros((1, 1751), dtype=np.float32)
    packed[0, 0] = 1.0
    packed[0, SEMANTIC_SOURCE_INDICES] = semantic
    return packed


def validate_native_history(history930):
    history = finite_vector(history930, 930, "v1.1 native history")
    for start in (30, 320, 610):
        block = history[start : start + 290].reshape(10, 29)
        if np.any(block[:, SOURCE_IL29_EXCLUDED_INDICES]):
            raise ValueError("v1.1 native history requires zero absent joint slots")
    return history


class SonicV11OnnxTeacher:
    """Pinned official v1.1 full29 graph, CPU float32 deterministic mean."""

    reference_profile = REFERENCE_PROFILE_NORMAL

    def __init__(self, model_directory):
        self.directory = Path(model_directory).resolve(strict=True)
        self.files = {name: self.directory / name for name in MODEL_SHA256}
        self._verify_files()
        config = yaml.safe_load(self.files["observation_config.yaml"].read_text())
        teleop = next(mode for mode in config["encoder"]["encoder_modes"] if mode["name"] == "teleop")
        if teleop["mode_id"] != 1 or teleop["required_observations"][-1] != "motion_anchor_orientation_heading":
            raise ValueError("v1.1 requires its matching heading-normalized observation config")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(
            str(self.files["model_encoder.onnx"]), options, providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            str(self.files["model_decoder.onnx"]), options, providers=["CPUExecutionProvider"]
        )
        for session, input_shape, output_shape in (
            (self.encoder, [1, 1751], [1, 64]),
            (self.decoder, [1, 994], [1, 29]),
        ):
            if (
                len(session.get_inputs()) != 1
                or session.get_inputs()[0].name != "obs_dict"
                or session.get_inputs()[0].shape != input_shape
            ):
                raise ValueError("v1.1 released input ABI mismatch")
            if len(session.get_outputs()) != 1 or session.get_outputs()[0].shape != output_shape:
                raise ValueError("v1.1 released output ABI mismatch")

    def _verify_files(self):
        for name, path in self.files.items():
            if file_sha256(path) != MODEL_SHA256[name]:
                raise ValueError("v1.1 model bundle identity mismatch: " + name)

    def infer_with_token(self, heading_semantic267, history930):
        history = finite_vector(history930, 930, "v1.1 history")
        token = self.encoder.run(None, {"obs_dict": pack_v11_encoder(heading_semantic267)})[0][0]
        finite_vector(token, 64, "v1.1 token")
        decoder = np.concatenate((token, history))[None]
        raw = self.decoder.run(None, {"obs_dict": decoder})[0][0]
        finite_vector(raw, 29, "v1.1 raw action")
        return raw.copy(), token.copy()

    def descriptor(self):
        self._verify_files()
        return dict(
            kind="official_sonic_v11_full29_onnx_offline_v1",
            files_sha256=MODEL_SHA256.copy(),
            reference_profile=self.reference_profile,
            orientation="reference_root_relative_to_robot_heading_only",
            full_encoder_dim=1751,
            semantic_dim=267,
            history_dim=930,
            token_dim=64,
            source_dof=29,
            source_policy_changed=False,
            decoder_affines=9,
            source_graph_packing_indices=SEMANTIC_SOURCE_INDICES.tolist(),
            inference_provider="CPUExecutionProvider",
            onnxruntime_version=ort.__version__,
            training_updates=0,
            hardware_authorized=False,
            deployment_ready=False,
        )


class SonicV11NativePolicy:
    def __init__(self, model_directory):
        self.teacher = SonicV11OnnxTeacher(model_directory)
        self.profile = REFERENCE_PROFILE_NORMAL
        self.outputs29 = []

    def infer(self, heading_semantic267, history930):
        validate_native_history(history930)
        raw29, token = self.teacher.infer_with_token(heading_semantic267, history930)
        self.outputs29.append(raw29.copy())
        return raw29[np.asarray(NATIVE_IL23_TO_CANONICAL_IL29)].copy(), np.concatenate((token, history930))

    def identity(self):
        return dict(
            kind="official_sonic_v11_native23_row_selection_offline_v1",
            teacher=self.teacher.descriptor(),
            reference_profile=self.profile,
            absent_joint_feedback="fixed_zero",
            selected_action_rows=list(NATIVE_IL23_TO_CANONICAL_IL29),
            training_updates=0,
            hardware_authorized=False,
            deployment_ready=False,
        )


class SonicV11NativeAdapter(ReleasedCoreAdapter):
    """New heading observation; inherit all existing source/preview settings."""

    def infer(self, policy, encoder, history, *, control_index, **state):
        del encoder
        if (
            not isinstance(policy, SonicV11NativePolicy)
            or policy.profile != self.profile
            or control_index != self.next_control
        ):
            raise ValueError("v1.1 policy profile or sequential control mismatch")
        window = self._push(self.emission_start + control_index)
        self.next_control += 1
        np.testing.assert_array_equal(
            window.anchor_root_position_w, state["previous_desired_position_w"].astype(np.float32)
        )
        np.testing.assert_array_equal(window.next_root_position_w, state["desired_position_w"].astype(np.float32))
        actual = semantic_from_window(window, state["measured_qpos"][3:7])
        raw, decoder = policy.infer(actual, history)
        row = dict(
            encoder267=actual.copy(),
            history930=history.copy(),
            decoder994=decoder.copy(),
            released_raw23=raw.copy(),
            measured_qpos=state["measured_qpos"].copy(),
            measured_qvel=state["measured_qvel"].copy(),
            timestamps=np.asarray(
                [
                    window.emission_source_timestamp_s,
                    window.encoder_anchor_timestamp_s,
                    window.root_setpoint_timestamp_s,
                ],
                dtype=np.float64,
            ),
        )
        self.attempts.append(row)
        inverse, projection = source_scaled_precompensation(raw)
        row.update(inverse23=inverse.copy(), projection23=projection.copy())
        safe = self.preview.filter(inverse, state["measured_qpos"], state["measured_qvel"])
        row["accepted23"] = safe.copy()
        return safe, decoder

    def contract(self):
        result = super().contract()
        result["kind"] = "native23_offline_sonic_v11_heading_adapter_v1"
        result["relative_orientation"] = "received_q0_root_relative_to_current_robot_heading_only"
        result["native23_preview_gains_and_limits_unchanged"] = True
        return result
