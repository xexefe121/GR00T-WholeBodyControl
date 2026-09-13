"""Explicit retarget-native/original29 correspondence for residual training.

Native lifecycle channels change to independently exported intent IK references.
Original29 source poses/tasks/VR21 and every lifecycle span retain their bytes.
This module never calls the old unmodified-projection pairing assertion for a
retargeted reference, and never relabels that assertion as having passed.
"""

import copy
import json
import os
from pathlib import Path
import shutil

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_curriculum import MOTION_KEYS


KIND = "native23_intent_retarget_original29_training_bank_v2"
RESIDUAL_KIND = "native23_frozen_bfm_intent_retarget_residual_v2"
TRAINING_CLIPS = ("walk002", "walk003", "pico")


def platform_path(value):
    value = str(value).replace("\\", "/")
    if os.name != "nt" and len(value) > 2 and value[1:3] == ":/":
        return Path("/mnt/" + value[0].lower() + "/" + value[3:])
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 7:
        return Path(value[5].upper() + ":/" + value[7:])
    return Path(value)


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def check_motion(motion, model, frames, *, verify_fk=True):
    if np.asarray(motion["fps"]).shape != (1,) or float(motion["fps"][0]) != 50.:
        raise ValueError("intent retarget must retain the 50-Hz grid")
    shapes = dict(joint_pos=(frames, 23), joint_vel=(frames, 23), body_pos_w=(frames, 24, 3),
                  body_quat_w=(frames, 24, 4), body_lin_vel_w=(frames, 24, 3), body_ang_vel_w=(frames, 24, 3))
    for key in MOTION_KEYS:
        value = np.asarray(motion[key])
        if value.shape != shapes[key] or not np.isfinite(value).all():
            raise ValueError("invalid retarget native channel: " + key)
    joints = motion["joint_pos"]
    if np.any(joints < model.jnt_range[1:, 0] - 1e-8) or np.any(joints > model.jnt_range[1:, 1] + 1e-8):
        raise ValueError("retarget native joint reference exceeds physical ranges")
    if not np.allclose(np.linalg.norm(motion["body_quat_w"], axis=-1), 1., atol=1e-6, rtol=0):
        raise ValueError("retarget body quaternions must remain normalized")
    max_position, max_quaternion = 0., 0.
    if verify_fk:
        data = mujoco.MjData(model)
        for frame in range(frames):
            data.qpos[:] = np.r_[motion["body_pos_w"][frame, 0], motion["body_quat_w"][frame, 0], joints[frame]]
            mujoco.mj_kinematics(model, data)
            max_position = max(max_position, float(np.max(np.abs(data.xpos[1:] - motion["body_pos_w"][frame]))))
            # Quaternion sign is immaterial to FK; no saved channel is rewritten.
            difference = 1 - np.abs(np.sum(data.xquat[1:] * motion["body_quat_w"][frame], axis=-1))
            max_quaternion = max(max_quaternion, float(np.max(np.abs(difference))))
        if max_position > 2e-6 or max_quaternion > 2e-6:
            raise ValueError(f"retarget native FK channel mismatch: {max_position},{max_quaternion}")
    return dict(frames=frames, body_position_fk_max_abs_m=max_position,
                body_quaternion_fk_max_abs_one_minus_dot=max_quaternion,
                joint_ranges_passed=True, dynamics_qualified=False)


def prepare_bank(previous_bank_report, retarget_root, output):
    from gear_sonic.envs.mjlab.sonic_true23_original_intent import make_spec

    previous_bank_report, retarget_root, output = map(platform_path, (previous_bank_report, retarget_root, output))
    if output.exists():
        raise FileExistsError("choose a new retarget training bank folder")
    prior = json.loads(previous_bank_report.read_text())
    inputs = {str(previous_bank_report.resolve()): sha256_file(previous_bank_report)}

    def bind(value, expected=None):
        path = platform_path(value).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError("retarget bank input changed: " + str(path))
        inputs[str(path)] = digest
        return path

    spec_path, spans_path = (bind(prior["files"][name]) for name in ("spec", "spans"))
    old_spec, spans = json.loads(spec_path.read_text()), json.loads(spans_path.read_text())
    old_files = {name: bind(value["path"], value["sha256"]) for name, value in old_spec["files"].items()}
    for name in ("spec", "spans", "motion"):
        original = str(prior["files"][name])
        if original in prior.get("outputs", {}):
            bind(original, prior["outputs"][original])
    if ([row["name"] for row in spans["spans"]] != list(TRAINING_CLIPS)
        or prior["source_frames"] != 7266 or prior["lifecycle_frames"] != 9549):
        raise ValueError("unexpected previous bank recording order or frame ownership")
    original = archive(old_files["original_reference"])
    model = mujoco.MjModel.from_xml_path(str(old_files["native_model"]))
    members, checks = [], []
    for row in spans["spans"]:
        clip = row["name"]
        report_path = bind(retarget_root / clip / "report.json")
        report = json.loads(report_path.read_text())
        native_path = bind(retarget_root / clip / "reference.npz", report["reference_sha256"])
        if (report.get("clip") != clip or report.get("frames") != row["length"]
            or report.get("full_original_timeline") is not True or report.get("source_timing_scale") != 1.
            or report.get("retarget_geometry_changed") is not True):
            raise ValueError("retarget lifecycle/timing contract differs for " + clip)
        candidates = [name for name in report["source_provenance"]
                      if Path(name.replace("\\", "/")).name in ("original_reference.npz", "original29.npz")]
        if len(candidates) != 1:
            raise ValueError("retarget export must identify exactly one immutable original29 archive")
        source_path = bind(candidates[0], report["source_provenance"][candidates[0]])
        source = archive(source_path)
        section = slice(row["start"], row["start"] + row["length"])
        for key, value in original.items():
            if key not in source or not np.array_equal(value[section], source[key]):
                raise ValueError(f"original29 source channel or frame correspondence differs: {clip}/{key}")
        native = archive(native_path)
        checks.append(dict(clip=clip, original29_arrays_bit_exact=True, **check_motion(native, model, row["length"])))
        members.append(native)
    combined = {key: np.concatenate([part[key] for part in members], axis=0) for key in MOTION_KEYS}
    combined["fps"] = np.array([50.])
    output.mkdir(parents=True)
    native_path, original_path = output / "native_reference.npz", output / "original_reference.npz"
    with native_path.open("xb") as stream:
        np.savez_compressed(stream, **combined)
    shutil.copyfile(old_files["original_reference"], original_path)
    new_spans = output / "lifecycle.spans.json"
    shutil.copyfile(spans_path, new_spans)
    spec = make_spec(native_motion=native_path, original_reference=original_path,
                     source_model=old_files["source_model"], native_model=old_files["native_model"])
    new_spec = output / "original_intent.spec.json"
    with new_spec.open("x") as stream:
        json.dump(spec, stream, indent=2, allow_nan=False)
    report = dict(kind=KIND, training_recording_ids=prior["training_recording_ids"],
                  evaluation_only_recording_ids=prior["evaluation_only_recording_ids"],
                  source_frames=prior["source_frames"], lifecycle_frames=prior["lifecycle_frames"],
                  original29_archive_bit_exact=sha256_file(original_path)==sha256_file(old_files["original_reference"]),
                  lifecycle_spans_bit_exact=sha256_file(new_spans)==sha256_file(spans_path),
                  original29_task_and_vr21_sources_unchanged=True, retarget_native_lifecycles_bit_exact=True,
                  native_root_and_retained_joints_changed=True, source_timing_scale=1., source_frame_mapping="identity within each existing lifecycle span",
                  physical_state_transition_between_recordings=False, checks=checks, inputs=inputs,
                  files=dict(spec=str(new_spec.resolve()), spans=str(new_spans.resolve()), motion=str(native_path.resolve()),
                             original_reference=str(original_path.resolve())),
                  outputs={str(path.resolve()): sha256_file(path) for path in (new_spec,new_spans,native_path,original_path)},
                  dynamics_qualified=False, hardware_authorized=False)
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def validate_bank(path):
    path = platform_path(path)
    bank = json.loads(path.read_text())
    if (bank.get("kind") != KIND or bank.get("source_timing_scale") != 1.
        or bank.get("original29_archive_bit_exact") is not True or bank.get("lifecycle_spans_bit_exact") is not True
        or bank.get("original29_task_and_vr21_sources_unchanged") is not True
        or bank.get("retarget_native_lifecycles_bit_exact") is not True):
        raise ValueError("invalid intent retarget bank correspondence")
    for group in ("inputs", "outputs"):
        for value, expected in bank[group].items():
            if sha256_file(platform_path(value)) != expected:
                raise ValueError("retarget bank bound artifact changed: " + value)
    return bank


def install_reference_loader(bank_path):
    """Install a v2-only loader in this process; no source files are modified.

    The existing task/VR21 reward functions and cache consume the same original29
    fields. Only their old unretargeted correspondence loader is replaced with
    the explicitly bound retarget lifecycle correspondence.
    """
    from gear_sonic.envs.mjlab import sonic_true23_original_intent as intent
    from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks
    from gear_sonic.utils.g1_true23_original29_reference import build_original29_reference

    bank = validate_bank(bank_path)
    expected_spec = json.loads(platform_path(bank["files"]["spec"]).read_text())
    original_loader = intent.load_reference

    def load_reference(spec):
        if spec != expected_spec:
            return original_loader(spec)
        intent.validate_spec(spec)
        recorded = archive(spec["files"]["original_reference"]["path"])
        native = archive(spec["files"]["native_motion"]["path"])
        source = mujoco.MjModel.from_xml_path(spec["files"]["source_model"]["path"])
        model = mujoco.MjModel.from_xml_path(spec["files"]["native_model"]["path"])
        reference = build_original29_reference(source, recorded["source_qpos29"])
        for name, value in reference.arrays().items():
            if name not in recorded or not np.array_equal(value, recorded[name]):
                raise ValueError("retarget bank original29 reconstruction changed: " + name)
        check_motion(native, model, len(reference.source_qpos29), verify_fk=False)
        tasks, hand_frame = neutral_wrist_hand_tasks(source, model)
        tasks = tuple(next(task for task in tasks if task.name == name) for name in intent.TASK_NAMES)
        pair = dict(kind=KIND, bank_sha256=sha256_file(platform_path(bank_path)),
                    frames=len(reference.source_qpos29), source_speed_factor=1.,
                    frame_correspondence="existing complete lifecycle identity indices",
                    original29_tasks_and_vr21_unchanged=True, root_and_retained_joints_preserved=False,
                    native_reference_retargeted=True, dynamics_qualified=False)
        return dict(reference=reference, native=native, tasks=tasks, pair=pair, hand_frame=hand_frame,
                    body_names=tuple(model.body(i).name for i in range(1, model.nbody)))

    intent.load_reference = load_reference
    return dict(kind=KIND, bank_sha256=sha256_file(platform_path(bank_path)),
                loader="explicit process-local retarget correspondence", original_loader_preserved_for_other_specs=True)
