"""Versioned root feedback, separate from SONIC's frozen semantic interface.

Inputs share one fixed world frame. Desired position is the received q10
reference; desired velocity uses only q9 -> q10. Simulator root state is not
evidence that a physical robot has a qualified world-state estimator.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def root_feedback_contract(reference_timing="causal_history"):
    if reference_timing not in ("causal_history", "received_source_horizon_200ms_v1"):
        raise ValueError("unknown root feedback reference timing")
    body = {
        "kind": "g1_native23_root_feedback9_v1",
        "dimension": 9,
        "components": [
            "desired_minus_measured_position_xyz",
            "desired_linear_velocity_xyz",
            "measured_linear_velocity_xyz",
        ],
        "component_units": ["m", "m_per_s", "m_per_s"],
        "coordinate_frame": "current_measured_pelvis_yaw_frame",
        "quaternion_order": "wxyz",
        "desired_position_frame": "q10_current_received_proof",
        "desired_velocity_definition": "(root_q10-root_q9)/0.02",
        "measured_state_frame": "q10_current_control_boundary",
        "source_sample_period_s": 0.02,
        "sonic_semantic_branch_reference_frame": "q9_unchanged",
        "future_samples_consumed": 0,
        "source_world_frame_reanchored_each_step": False,
        "input_clipping": False,
        "output_dtype": "float32",
        "separate_from_existing_994_decoder_input": True,
        "physical_world_state_estimator_required": True,
        "simulator_state_is_physical_estimator_qualification": False,
    }
    if reference_timing == "received_source_horizon_200ms_v1":
        body.update(
            kind="g1_native23_buffered_source_root_feedback9_v2",
            desired_position_frame="received_horizon_q1_age_180ms",
            desired_velocity_definition="(received_root_q1-received_root_q0)/0.02",
            measured_state_frame="current_control_boundary_not_delayed",
            sonic_semantic_branch_reference_frame="received_horizon_q0_age_200ms",
            reference_timing=reference_timing,
        )
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**body, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def _numpy_inputs(values):
    arrays = []
    for value in values:
        raw = np.asarray(value)
        if raw.dtype.kind not in "fi" or not np.isfinite(raw).all():
            raise ValueError("root feedback requires finite numeric inputs")
        arrays.append(raw.astype(np.float32))
    shape = arrays[0].shape
    if not shape or shape[-1] != 3 or any(value.shape != shape for value in arrays[:4]):
        raise ValueError("root position/velocity arrays require identical [...,3] shapes")
    quaternion = arrays[4]
    if quaternion.shape != (*shape[:-1], 4):
        raise ValueError("root quaternion requires matching [...,4] WXYZ shape")
    if not np.isfinite(np.concatenate([value.reshape(-1) for value in arrays])).all():
        raise ValueError("root feedback overflows float32")
    if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1.0, atol=1e-4, rtol=0):
        raise ValueError("root quaternion must be normalized WXYZ")
    return arrays


def root_feedback_numpy(
    desired_position_w, measured_position_w, desired_velocity_w, measured_velocity_w, measured_quaternion_wxyz
):
    desired, measured, desired_velocity, measured_velocity, quaternion = _numpy_inputs(
        (
            desired_position_w,
            measured_position_w,
            desired_velocity_w,
            measured_velocity_w,
            measured_quaternion_wxyz,
        )
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cosine, sine = np.cos(yaw), np.sin(yaw)

    def rotate(vector):
        return np.stack(
            (
                cosine * vector[..., 0] + sine * vector[..., 1],
                -sine * vector[..., 0] + cosine * vector[..., 1],
                vector[..., 2],
            ),
            axis=-1,
        )

    result = np.concatenate(
        tuple(rotate(vector) for vector in (desired - measured, desired_velocity, measured_velocity)), axis=-1
    )
    if not np.isfinite(result).all():
        raise ValueError("root feedback produced nonfinite values")
    return result.astype(np.float32, copy=False)


def root_feedback_torch(
    desired_position_w, measured_position_w, desired_velocity_w, measured_velocity_w, measured_quaternion_wxyz
):
    import torch

    values = (
        desired_position_w,
        measured_position_w,
        desired_velocity_w,
        measured_velocity_w,
        measured_quaternion_wxyz,
    )
    if any(
        not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.float64)
        for value in values
    ):
        raise ValueError("root feedback requires floating torch tensors")
    if any(value.device != values[0].device or not torch.isfinite(value).all() for value in values):
        raise ValueError("root feedback requires finite tensors on one device")
    desired, measured, desired_velocity, measured_velocity, quaternion = (value.float() for value in values)
    shape = desired.shape
    if not shape or shape[-1] != 3 or any(value.shape != shape for value in values[:4]):
        raise ValueError("root position/velocity arrays require identical [...,3] shapes")
    if quaternion.shape != (*shape[:-1], 4):
        raise ValueError("root quaternion requires matching [...,4] WXYZ shape")
    if not torch.all(torch.abs(torch.linalg.vector_norm(quaternion, dim=-1) - 1) <= 1e-4):
        raise ValueError("root quaternion must be normalized WXYZ")
    w, x, y, z = quaternion.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cosine, sine = torch.cos(yaw), torch.sin(yaw)

    def rotate(vector):
        return torch.stack(
            (
                cosine * vector[..., 0] + sine * vector[..., 1],
                -sine * vector[..., 0] + cosine * vector[..., 1],
                vector[..., 2],
            ),
            dim=-1,
        )

    result = torch.cat(
        tuple(rotate(vector) for vector in (desired - measured, desired_velocity, measured_velocity)), dim=-1
    )
    if not torch.isfinite(result).all():
        raise ValueError("root feedback produced nonfinite values")
    return result
