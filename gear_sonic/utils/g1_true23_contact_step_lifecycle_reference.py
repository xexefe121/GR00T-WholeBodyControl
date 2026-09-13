"""Independent loader for SIM-only contact-step lifecycle references."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from gear_sonic.scripts.prepare_g1_true23_contact_step_lifecycle import (
    REPORT_KIND,
    audit_step_geometry,
    motion_from_poses,
)
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_contact_step_transition import PROFILE
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256, motion_qpos
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion

POLICY_KIND = "g1_true23_contact_step_lifecycle_policy_diagnostic_v1"


def validate_step_preservation(original, old_timeline, motion, timeline):
    count = validate_library_motion(motion)
    if (
        timeline.get("generated_transition_profile") != PROFILE
        or timeline.get("prehistory_frames") != 11
        or timeline.get("total_frames") != count
        or timeline.get("total_requested_controls") != count - 11
        or len(timeline.get("phases", [])) != len(old_timeline["phases"])
        or set(motion) != set(original)
    ):
        raise ValueError("contact-step timeline/physical channels differ from declared profile")
    for key in (
        "source_frames",
        "source_frame_indices",
        "source_timing_scale",
        "configured_standing_qpos",
        "returned_standing_qpos",
        "return_target",
        "source_motion_sha256",
    ):
        if timeline.get(key) != old_timeline.get(key):
            raise ValueError(f"contact-step changed source or standing contract: {key}")
    for key in motion:
        np.testing.assert_array_equal(
            motion[key] if key == "fps" else motion[key][:11],
            original[key] if key == "fps" else original[key][:11],
        )
    cursor = 0
    for phase, old in zip(timeline["phases"], old_timeline["phases"], strict=True):
        controls = phase["requested_controls"]
        if (
            phase["name"] != old["name"]
            or type(controls) is not int
            or not 1 <= controls <= 100000
            or phase["control_start"] != cursor
            or phase["control_stop"] != cursor + controls
            or phase["frame_start"] != cursor + 11
            or phase["frame_stop"] != cursor + 11 + controls
        ):
            raise ValueError("contact-step phase coverage changed or overlaps")
        if phase["name"] in ("acquisition_ramp", "return_ramp"):
            if not 100 <= controls <= 1000:
                raise ValueError("generated step duration is outside the declared diagnostic bounds")
        else:
            if controls != old["requested_controls"]:
                raise ValueError("contact-step shortened original source or standing")
            for key in motion:
                if key != "fps":
                    np.testing.assert_array_equal(
                        motion[key][phase["frame_start"] : phase["frame_stop"]],
                        original[key][old["frame_start"] : old["frame_stop"]],
                    )
        if phase["name"] == "source_motion" and (
            timeline["source_start_frame"] != phase["frame_start"]
            or timeline["source_stop_frame_exclusive"] != phase["frame_stop"]
        ):
            raise ValueError("contact-step source span changed")
        cursor += controls
    if cursor != count - 11:
        raise ValueError("contact-step truncated final standing proof")


def load_contact_step_reference(report_path, original, old_timeline, model, source_sha256):
    report_path = Path(report_path).resolve(strict=True)
    report = json.loads(report_path.read_text())
    if (
        report.get("kind") != REPORT_KIND
        or report.get("training_reference_accepted") is not False
        or report.get("hardware_authorized") is not False
        or report.get("deployment_ready") is not False
        or report.get("compiled_physics_model_sha256") != compiled_model_sha256(model)
    ):
        raise ValueError("contact-step reference is not an unchanged-model SIM diagnostic")
    timeline = deepcopy(report["timeline"])
    expected = {**old_timeline, "source_motion_sha256": source_sha256}
    path = Path(timeline["timeline_path"]).resolve(strict=True)
    if path.parent != report_path.parent or sha256_file(path) != timeline["timeline_sha256"]:
        raise ValueError("contact-step reference byte binding changed")
    bindings = {str(report_path): sha256_file(report_path), str(path): sha256_file(path)}
    for name, digest in report["inputs"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"contact-step input changed: {name}")
        bindings[name] = digest
    with np.load(path, allow_pickle=False) as archive:
        motion = {key: archive[key].copy() for key in archive.files}
    # Preserve the actual hash-bound normal baseline, including its previously
    # regenerated standing velocity bytes. Rebuilding an unrepaired baseline
    # can differ at 1e-16 in those velocities, despite identical physical poses.
    parents = [Path(name) for name in report["inputs"] if Path(name).name == "report.json"]
    if len(parents) != 1:
        raise ValueError("contact-step requires one explicitly bound normal baseline")
    parent = json.loads(parents[0].read_text())
    if parent.get("kind") != "g1_true23_root_feedback_single_policy_lifecycle_campaign_v1":
        raise ValueError("contact-step parent is not a normal lifecycle")
    parent_timeline = parent["timeline"]
    parent_path = Path(parent_timeline["timeline_path"]).resolve(strict=True)
    if (
        parent_timeline["phases"] != expected["phases"]
        or parent_timeline["source_motion_sha256"] != source_sha256
        or bindings.get(str(parent_path)) != parent_timeline["timeline_sha256"]
    ):
        raise ValueError("contact-step parent source, phase coverage or reference binding changed")
    with np.load(parent_path, allow_pickle=False) as archive:
        parent_motion = {key: archive[key].copy() for key in archive.files}
    for phase in expected["phases"]:
        if phase["name"] in ("acquisition_ramp", "return_ramp"):
            continue
        span = slice(phase["frame_start"], phase["frame_stop"])
        fields = (
            tuple(key for key in original if key != "fps")
            if phase["name"] == "source_motion"
            else ("joint_pos", "body_pos_w", "body_quat_w")
        )
        for key in fields:
            np.testing.assert_array_equal(parent_motion[key][span], original[key][span])
    validate_step_preservation(parent_motion, parent_timeline, motion, timeline)
    # Independently regenerate every FK channel; derivative checks apply to
    # generated samples only, since original source/standing channels stay exact.
    rebuilt = motion_from_poses(model, motion_qpos(model, motion))
    np.testing.assert_allclose(motion["body_pos_w"], rebuilt["body_pos_w"], atol=2e-5, rtol=0)
    dot = np.abs(np.sum(motion["body_quat_w"] * rebuilt["body_quat_w"], axis=-1))
    np.testing.assert_allclose(dot, 1.0, atol=2e-5, rtol=0)
    for phase in timeline["phases"]:
        if phase["name"] not in ("acquisition_ramp", "return_ramp"):
            continue
        span = slice(phase["frame_start"], phase["frame_stop"])
        for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
            np.testing.assert_allclose(motion[key][span], rebuilt[key][span], atol=1e-4, rtol=1e-5)
    geometry = audit_step_geometry(model, motion, timeline)
    if not geometry["provisional_geometry_screen_passed"]:
        raise ValueError("independent contact-step bounds/collision/floor audit failed")
    historical = timeline.pop("generated_reference_repair", None)
    if historical is not None:
        timeline["pre_step_generated_reference_repair"] = historical
    timeline.update(
        contact_step_reference_report_sha256=bindings[str(report_path)],
        contact_step_independent_geometry_audit=geometry,
        contact_step_conditional_support_failures={
            name: value["frames_with_no_support_solution"] for name, value in report["reference_support"].items()
        },
        contact_step_dynamic_feasibility_qualified=False,
        eligible_parent_for_training_continuation=False,
    )
    return motion, timeline, bindings
