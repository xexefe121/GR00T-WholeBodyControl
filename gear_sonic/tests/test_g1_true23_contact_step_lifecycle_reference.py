from copy import deepcopy

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_contact_step_lifecycle_reference import POLICY_KIND, validate_step_preservation
from gear_sonic.utils.g1_true23_contact_step_transition import PROFILE
from gear_sonic.utils.g1_true23_root_feedback_campaign import validate_campaign_evaluation


def sample():
    cursor, phases = 0, []
    for name, count in (
        ("initial_standing", 50),
        ("acquisition_ramp", 100),
        ("source_motion", 12),
        ("return_ramp", 100),
        ("returned_standing", 50),
        ("standing_proof_margin", 50),
    ):
        phases.append(
            dict(
                name=name,
                requested_controls=count,
                control_start=cursor,
                control_stop=cursor + count,
                frame_start=11 + cursor,
                frame_stop=11 + cursor + count,
            )
        )
        cursor += count
    frames = 11 + cursor
    quat = np.zeros((frames, 24, 4))
    quat[..., 0] = 1
    motion = dict(
        fps=np.array([50.0]),
        joint_pos=np.tile(np.arange(frames)[:, None] * 0.0001, (1, 23)),
        joint_vel=np.zeros((frames, 23)),
        body_pos_w=np.zeros((frames, 24, 3)),
        body_quat_w=quat,
        body_lin_vel_w=np.zeros((frames, 24, 3)),
        body_ang_vel_w=np.zeros((frames, 24, 3)),
    )
    timeline = dict(
        phases=phases,
        prehistory_frames=11,
        total_frames=frames,
        total_requested_controls=cursor,
        source_frames=12,
        source_frame_indices=list(range(12)),
        source_timing_scale=1.0,
        source_start_frame=161,
        source_stop_frame_exclusive=173,
        configured_standing_qpos=[0.0] * 30,
        returned_standing_qpos=[0.0] * 30,
        return_target="fixed_planned_terminal_xy_and_heading",
        source_motion_sha256="a" * 64,
        generated_transition_profile=PROFILE,
    )
    return motion, timeline


def test_preservation_check_accepts_unchanged_source_standing_and_phases():
    motion, timeline = sample()
    validate_step_preservation(motion, timeline, deepcopy(motion), deepcopy(timeline))


@pytest.mark.parametrize(
    "mutation", ["source", "standing", "warmup", "phase_overlap", "retime", "profile", "span", "fps"]
)
def test_step_reference_mutations_fail_closed(mutation):
    original, before = sample()
    motion, timeline = deepcopy(original), deepcopy(before)
    if mutation == "source":
        motion["joint_pos"][161, 0] += 0.1
    elif mutation == "standing":
        motion["joint_vel"][-1, 0] += 0.1
    elif mutation == "warmup":
        motion["joint_pos"][10, 0] += 0.1
    elif mutation == "phase_overlap":
        timeline["phases"][2]["control_start"] -= 1
    elif mutation == "retime":
        timeline["source_timing_scale"] = 2.0
    elif mutation == "profile":
        timeline["generated_transition_profile"] = "sliding_ramp"
    elif mutation == "span":
        timeline["source_start_frame"] += 1
    else:
        motion["fps"] = np.array([25.0])
    with pytest.raises((ValueError, AssertionError)):
        validate_step_preservation(original, before, motion, timeline)


def test_step_policy_report_cannot_silently_become_old_training_evidence():
    with pytest.raises(ValueError, match="CPU lifecycle campaign"):
        validate_campaign_evaluation(
            {"kind": POLICY_KIND}, checkpoint_sha256="x", actor_sha256="y", lineage_sha256="z", updates=1200
        )
