"""Explicit delayed-source semantics; old causal policy contracts stay unchanged.

Simulation's scripted standing source continues transmitting ten extra samples
after the scored lifecycle. These are generated inputs, not captured dance data
or extra scored controls. The live received-sample buffer never pads an EOF.
"""

import numpy as np

from gear_sonic.teleop.buffered_source_horizon import buffered_horizon_contract
from gear_sonic.utils.g1_true23_generalist_corpus import canonical_digest

CAUSAL_TIMING = "causal_history"
BUFFERED_TIMING = "received_source_horizon_200ms_v1"
REFERENCE_TIMINGS = (CAUSAL_TIMING, BUFFERED_TIMING)
SCRIPTED_TAIL_FRAMES = 10


def reference_profile_contract(timing=CAUSAL_TIMING):
    if timing == CAUSAL_TIMING:
        from gear_sonic.envs.mjlab.sonic_true23_causal_history import causal_history_profile_contract

        return causal_history_profile_contract()
    if timing != BUFFERED_TIMING:
        raise ValueError("unknown reference timing")
    payload = dict(
        schema="g1_true23_received_source_horizon_profile_v1",
        profile=BUFFERED_TIMING,
        encoder_input_dim=267,
        buffer=buffered_horizon_contract(),
        lower_body_order="mujoco_hardware_left6_then_right6",
        position_offsets_from_anchor_s=[i * 0.02 for i in range(10)],
        velocity_definition="forward_difference_received_anchor_i_to_i_plus_1_over_0p02s",
        reference_anchor_age_s=0.2,
        root_setpoint_age_s=0.18,
        measured_state="current_control_boundary_not_buffered",
        future_samples_relative_to_emission=False,
        released_profile_relabel_permitted=False,
        retraining_required=True,
        simulation_source_tail=dict(
            generated_standing_samples=SCRIPTED_TAIL_FRAMES,
            captured_source=False,
            scored_lifecycle_frames_or_targets_changed=False,
            purpose="keep_simulated_source_transmitting_during_final_buffered_controls",
        ),
        deployment_ready=False,
        hardware_authorized=False,
    )
    return {**payload, "contract_sha256": canonical_digest(payload)}


def continued_standing_source(motion, virtual_vr=None):
    """Copy a complete scripted lifecycle plus explicit new terminal inputs.

    Only an already constant standing endpoint may extend. This helper belongs to
    simulation input generation, never the received-source buffer or live transport.
    """
    fields = ("joint_pos", "body_pos_w", "body_quat_w")
    count = len(motion["joint_pos"])
    if count < 11:
        raise ValueError("buffered lifecycle requires at least eleven source frames")
    for name in fields:
        value = np.asarray(motion[name])
        if len(value) != count or not np.isfinite(value).all():
            raise ValueError("invalid complete lifecycle source")
        if not np.array_equal(value[-10:], np.repeat(value[-1:], 10, axis=0)):
            raise ValueError("buffered simulation requires explicit constant terminal standing")
    for name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
        if np.any(motion[name][-10:]):
            raise ValueError("terminal standing source must have zero velocities")
    source = {
        "joint_pos": np.asarray(motion["joint_pos"], np.float32),
        "root_position_w": np.asarray(motion["body_pos_w"][:, 0], np.float32),
        "root_quaternion_wxyz": np.asarray(motion["body_quat_w"][:, 0], np.float32),
    }
    if virtual_vr is not None:
        vr = np.asarray(virtual_vr)
        if vr.shape != (count, 21) or vr.dtype != np.float32 or not np.isfinite(vr).all():
            raise ValueError("source VR requires exact finite float32 [frames,21]")
        if not np.array_equal(vr[-10:], np.repeat(vr[-1:], 10, axis=0)):
            raise ValueError("terminal source VR must be constant")
        source["virtual_vr21"] = vr
    return {
        key: np.concatenate((value, np.repeat(value[-1:], SCRIPTED_TAIL_FRAMES, axis=0)))
        for key, value in source.items()
    }


def lower_horizon_cache(motion):
    source = continued_standing_source(motion)
    count = len(motion["joint_pos"])
    indexes = np.arange(count)[:, None] + np.arange(11)[None]
    joints = source["joint_pos"][indexes, :12]
    return np.concatenate(
        (
            joints[:, :-1].reshape(count, 120),
            ((joints[:, 1:] - joints[:, :-1]) / np.float32(0.02)).reshape(count, 120),
        ),
        axis=-1,
    )
