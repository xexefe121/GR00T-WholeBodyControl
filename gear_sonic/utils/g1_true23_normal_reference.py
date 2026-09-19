"""Explicit normal-core training reference; never reinterpret a 200 ms policy."""

import numpy as np

from gear_sonic.teleop.normal_source_horizon import POSITION_INDICES, normal_horizon_contract
from gear_sonic.utils.g1_true23_buffered_reference import continued_standing_source
from gear_sonic.utils.g1_true23_generalist_corpus import canonical_digest

NORMAL_TIMING = "received_normal_source_horizon_920ms_v1"


def normal_reference_contract():
    payload = dict(
        kind="native23_normal_received_training_reference_v1",
        reference_timing=NORMAL_TIMING,
        received_buffer=normal_horizon_contract(),
        root_setpoint_age_s=0.90,
        scored_reference_phase="unchanged_post_control_q2",
        generated_standing_tail_samples=46,
        source_frames_retimed_or_reanchored=False,
        missing_proprioception="physical_absence_zero_never_reference_motion",
        old_200ms_checkpoint_relabelling=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    return {**payload, "contract_sha256": canonical_digest(payload)}


def normal_root_feedback_contract():
    payload = dict(
        kind="native23_normal_received_root_feedback9_v1",
        reference_timing=NORMAL_TIMING,
        dimension=9,
        components=["desired_minus_measured_position_xyz", "desired_velocity_xyz", "measured_velocity_xyz"],
        component_units=["m", "m_per_s", "m_per_s"],
        coordinate_frame="current_measured_pelvis_yaw_frame",
        desired_position="received_q1_age_900ms",
        desired_velocity="float32(received_root_q1-received_root_q0)/float32(0.02)",
        measured_state="current_control_boundary_not_delayed",
        semantic_anchor="received_q0_age_920ms",
        separate_from_existing_994_decoder_input=True,
        input_clipping=False,
        physical_world_state_estimator_required=True,
        simulator_state_is_physical_estimator_qualification=False,
        deployment_ready=False,
        hardware_authorized=False,
    )
    return {**payload, "contract_sha256": canonical_digest(payload)}


def normal_standing_source(motion, virtual_vr=None):
    """Add exactly46 declared synthetic standing samples, to one isolated clip."""
    source = continued_standing_source(motion, virtual_vr)
    return {key: np.concatenate((value, np.repeat(value[-1:], 36, axis=0))) for key, value in source.items()}


def normal_lower_horizon_cache(motion):
    source = normal_standing_source(motion)
    count = len(motion["joint_pos"])
    indices = np.arange(count)[:, None] + POSITION_INDICES[None]
    selected = source["joint_pos"][indices, :12]
    velocity = (source["joint_pos"][indices + 1, :12] - selected) / np.float32(0.02)
    result = np.concatenate((selected.reshape(count, 120), velocity.reshape(count, 120)), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("normal training source horizon overflowed")
    return result
