"""Prepare lossless original29 intent beside a matched native23 public replay.

Reads a pinned public recording and a complete existing native23 lifecycle.
Recomputes original source FK directly; no original29 policy rollout is needed.
No dynamics, training, robot transport, or existing checkpoint relabelling.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from gear_sonic.scripts.prepare_g1_true23_twist2_replay import load_pinned_recording
from gear_sonic.scripts.record_g1_sonic_public29_baseline import lifecycle29, source29_poses, stock
from gear_sonic.utils.g1_23dof_contract import SOURCE_MJ29_KEEP_INDICES
from gear_sonic.utils.g1_true23_buffered_reference import continued_standing_source
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks
from gear_sonic.utils.g1_true23_original29_reference import (
    FLAGS,
    build_original29_reference,
    reference_contract,
    task_targets_from_received_vr,
    verify_unmodified_native_pair,
)
from gear_sonic.utils.g1_true23_virtual_source_reference import virtual_source_vr_terms


def _rotation_error(first, second):
    dot = np.abs(np.sum(first * second, axis=-1))
    dot /= np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    return 2 * np.arccos(np.clip(dot, 0, 1))


def native_task_points(model, qpos, tasks):
    data = mujoco.MjData(model)
    position, quaternion = [], []
    for pose in qpos:
        data.qpos[:] = pose
        mujoco.mj_fwdPosition(model, data)
        position.append(
            [
                data.xpos[model.body(task.target_body).id]
                + stock._quat_rotate(data.xquat[model.body(task.target_body).id], np.asarray(task.target_point))
                for task in tasks
            ]
        )
        quaternion.append([data.xquat[model.body(task.target_body).id].copy() for task in tasks])
    return np.asarray(position), np.asarray(quaternion)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recording", "source-model", "native-model", "native-evaluation-directory", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError("original reference preparation refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"input SHA256 mismatch: {path}")
        inputs[str(path)] = digest
        return path

    def archive(path, expected=None):
        with np.load(bind(path, expected), allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    recording = load_pinned_recording(bind(args.recording))
    report = json.loads(bind(args.native_evaluation_directory / "report.json").read_text())
    timeline = report["timeline"]
    motion = archive(timeline["timeline_path"], timeline["timeline_sha256"])
    standing = np.zeros(36)
    native_standing = np.asarray(timeline["configured_standing_qpos"])
    if native_standing.shape != (30,):
        raise ValueError("requires a native23 configured standing lifecycle")
    standing[:7] = native_standing[:7]
    standing[7 + np.asarray(SOURCE_MJ29_KEEP_INDICES)] = native_standing[7:]
    source_poses = source29_poses(recording)
    poses, phases = lifecycle29(source_poses, standing)
    assert phases == [
        {key: row[key] for key in ("name", "control_start", "control_stop")} for row in timeline["phases"]
    ]
    source_model = mujoco.MjModel.from_xml_path(str(bind(args.source_model)))
    reference = build_original29_reference(source_model, poses)
    pairing = verify_unmodified_native_pair(reference, motion)
    received = continued_standing_source(motion, reference.virtual_vr21)
    evaluated_source = archive(args.native_evaluation_directory / "nominal.received_source.npz")
    if evaluated_source.keys() != received.keys():
        raise ValueError("evaluated received-source layout differs from prepared original intent")
    for key in received:
        np.testing.assert_array_equal(received[key], evaluated_source[key])
    world, quat = task_targets_from_received_vr(
        received["root_position_w"][: len(poses)],
        received["root_quaternion_wxyz"][: len(poses)],
        reference.virtual_vr21,
    )
    position_error = float(np.max(np.abs(world - reference.source_task_position_w)))
    orientation_error = float(np.max(_rotation_error(quat, reference.source_task_quaternion_wxyz)))
    if position_error > 2e-6 or orientation_error > 2e-6:
        raise ValueError("serialized intent and original world task targets diverge")
    # Measure the previously used zero-absent input with the same source geometry;
    # this is not a body-origin-versus-hand-offset comparison.
    old_vr = virtual_source_vr_terms(motion, args.source_model)
    old_world, old_quat = task_targets_from_received_vr(poses[:, :3], poses[:, 3:7], old_vr)
    native_model = mujoco.MjModel.from_xml_path(str(bind(args.native_model)))
    all_tasks, hand_convention = neutral_wrist_hand_tasks(source_model, native_model)
    tasks = tuple(
        next(task for task in all_tasks if task.name == name) for name in ("left_hand", "right_hand", "head_proxy")
    )
    native_poses = np.c_[motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0], motion["joint_pos"]]
    native_world, native_quat = native_task_points(native_model, native_poses, tasks)
    phase = next(row for row in phases if row["name"] == "source_motion")
    section = slice(11 + phase["control_start"], 11 + phase["control_stop"])
    native_gap = np.linalg.norm(native_world - reference.source_task_position_w, axis=2)
    old_gap = np.linalg.norm(old_world - reference.source_task_position_w, axis=2)
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    bind(__file__)
    payload = dict(
        kind="g1_true23_paired_original29_source_intent_bundle_v1",
        reference_contract=reference_contract(inputs[str(args.source_model.resolve())]),
        original_recording_sha256=inputs[str(args.recording.resolve())],
        pairing=pairing,
        phases=phases,
        complete_source_frames=len(source_poses),
        total_lifecycle_frames=len(poses),
        received_tail_generated_standing_samples=len(received["joint_pos"]) - len(poses),
        previous_evaluated_received_inputs_bit_exact=True,
        task_targets_received_reconstruction_max_position_component_error_m=position_error,
        task_targets_received_reconstruction_max_orientation_error_rad=orientation_error,
        source_task_order=["left_hand", "right_hand", "head"],
        old_zero_absent_intent_position_p95_m=np.percentile(old_gap[section], 95, axis=0).tolist(),
        old_zero_absent_intent_orientation_p95_rad=np.percentile(
            _rotation_error(old_quat, reference.source_task_quaternion_wxyz)[section], 95, axis=0
        ).tolist(),
        native_planned_pose_original_task_position_p95_m=np.percentile(native_gap[section], 95, axis=0).tolist(),
        native_planned_pose_original_task_orientation_p95_rad=np.percentile(
            _rotation_error(native_quat, reference.source_task_quaternion_wxyz)[section], 95, axis=0
        ).tolist(),
        native_proxy_tasks=[asdict(task) for task in tasks],
        neutral_hand_frame_derivation=hand_convention,
        native_joint_posture_error_at_its_own_reference_is_zero=True,
        original_task_match_does_not_follow_from_zero_native_posture_error=True,
        task_proxy_is_not_physical_contact_or_mesh_correspondence=True,
        new_training_rewards_or_controller_installed=False,
        existing_checkpoint_source_convention_unchanged=True,
        no_physics_integration_or_hardware_connection=True,
        inputs=inputs,
        **FLAGS,
    )
    output.mkdir(parents=True, exist_ok=False)
    pack, source_path = output / "original_reference.npz", output / "received_source.npz"
    with pack.open("xb") as stream:
        np.savez_compressed(
            stream, **reference.arrays(), fps=np.array([50.0]), source_time_s=np.arange(len(poses)) * 0.02
        )
    with source_path.open("xb") as stream:
        np.savez_compressed(stream, **received)
    payload.update(
        reference_path=str(pack),
        reference_sha256=sha256_file(pack),
        received_path=str(source_path),
        received_sha256=sha256_file(source_path),
    )
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"input changed during reference preparation: {path}")
    with (output / "report.json").open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "complete_source_frames",
                    "total_lifecycle_frames",
                    "previous_evaluated_received_inputs_bit_exact",
                    "old_zero_absent_intent_position_p95_m",
                    "old_zero_absent_intent_orientation_p95_rad",
                    "native_planned_pose_original_task_position_p95_m",
                    "native_planned_pose_original_task_orientation_p95_rad",
                    "reference_path",
                    "reference_sha256",
                    "hardware_authorized",
                    "deployment_ready",
                )
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
