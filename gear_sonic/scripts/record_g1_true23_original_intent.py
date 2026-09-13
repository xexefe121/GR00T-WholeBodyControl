"""Full standing/source/return CPU trial of the distinct original-intent actor.

Existing physical referee and acceptance metrics are unchanged. Original hand
and head points are additionally measured at both held q1 and referee q2; the
former reports the training reward phase, never replaces the latter's gates.
"""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from gear_sonic.teleop.buffered_source_simulation import BufferedSourceSimulationAdapter
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS, run_reference_diagnostic
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_generalist_lifecycle import assess_lifecycle_diagnostic
from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks
from gear_sonic.utils.g1_true23_original29_reference import (
    build_original29_reference,
    verify_unmodified_native_pair,
)
from gear_sonic.utils.g1_true23_original_intent_checkpoint import ROOT, load_cpu_actor
from gear_sonic.utils.g1_true23_root_feedback_campaign import validate_buffered_replay_arrays

FLAGS = dict(
    hardware_authorized=False, deployment_ready=False, simulator_qualified=False, candidate_promoted=False
)


class OriginalIntentAdapter(BufferedSourceSimulationAdapter):
    def contract(self):
        return dict(
            **super().contract(),
            source_vr="original29_all_axes_preserved",
            actual_actor_reference_contract="native23_original_source_intent_release_compatibility_v4",
            original_task_metric_phases=["held_received_q1", "unchanged_referee_post_control_q2"],
            legacy_transport_or_export_acceptance_claimed=False,
        )


def task_metrics(model, geometry, reference, motion, trace, phase):
    tasks, convention = neutral_wrist_hand_tasks(geometry, model)
    names = ("left_hand", "right_hand", "head_proxy")
    tasks = [next(t for t in tasks if t.name == n) for n in names]
    data = mujoco.MjData(model)
    points, quats = [], []
    for pose in trace["qpos"][1:]:
        data.qpos[:] = pose
        mujoco.mj_fwdPosition(model, data)
        positions, orientations = [], []
        for task in tasks:
            body = model.body(task.target_body).id
            positions.append(data.xpos[body] + data.xmat[body].reshape(3, 3) @ np.asarray(task.target_point))
            orientations.append(data.xquat[body].copy())
        points.append(positions)
        quats.append(orientations)
    points, quats = np.asarray(points), np.asarray(quats)
    n = len(points)
    start, stop = phase["control_start"], min(n, phase["control_stop"])
    report = dict(
        source_controls_completed=max(0, stop - start),
        order=names,
        neutral_proxy_convention=convention,
        proxy_is_not_physical_contact_registration=True,
        thresholds_relaxed=False,
    )
    if stop <= start:
        return {**report, "measured": False}
    for name, offset in (("held_received_q1", 10), ("unchanged_referee_post_control_q2", 11)):
        wanted_p = reference.source_task_position_w[offset : offset + n]
        wanted_q = reference.source_task_quaternion_wxyz[offset : offset + n]
        angle = (
            (
                Rotation.from_quat(quats.reshape(-1, 4)[:, [1, 2, 3, 0]]).inv()
                * Rotation.from_quat(wanted_q.reshape(-1, 4)[:, [1, 2, 3, 0]])
            )
            .magnitude()
            .reshape(n, 3)
        )
        relative = (points - trace["qpos"][1:, None, :3]) - (
            wanted_p - reference.source_qpos29[offset : offset + n, None, :3]
        )
        section = slice(start, stop)
        report[name] = dict(
            original_task_world_position_p95_m=np.percentile(
                np.linalg.norm(points - wanted_p, axis=-1)[section], 95, axis=0
            ).tolist(),
            original_task_pelvis_centered_world_axes_position_p95_m=np.percentile(
                np.linalg.norm(relative, axis=-1)[section], 95, axis=0
            ).tolist(),
            original_task_world_orientation_p95_rad=np.percentile(angle[section], 95, axis=0).tolist(),
            root_world_position_p95_m=float(
                np.percentile(
                    np.linalg.norm(
                        trace["qpos"][1:, :3] - reference.source_qpos29[offset : offset + n, :3], axis=-1
                    )[section],
                    95,
                )
            ),
            leg_joint_rmse_rad=float(
                np.sqrt(
                    np.mean(
                        (trace["qpos"][1:, 7:19] - motion["joint_pos"][offset : offset + n, :12])[section] ** 2
                    )
                )
            ),
        )
    return dict(**report, measured=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "warm-start",
        "source-checkpoint",
        "asset-root",
        "baseline-directory",
        "original-reference",
        "output-directory",
        "experiment",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("original-intent CPU trial refuses overwrite")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            raise ValueError(f"trial input changed: {path}")
        inputs[str(path)] = digest
        return path

    def read(path, expected=None):
        with np.load(bind(path, expected), allow_pickle=False) as archive:
            return {k: archive[k].copy() for k in archive.files}

    prior = json.loads(bind(args.baseline_directory / "report.json").read_text())
    timeline = prior["timeline"]
    motion_path = bind(timeline["timeline_path"], timeline["timeline_sha256"])
    motion = read(motion_path)
    saved = read(args.original_reference)
    geometry_path = bind(args.asset_root / "gear_sonic/data/robots/g1/g1_29dof.xml")
    native_path = bind(args.asset_root / MODEL)
    bind(ROOT / PHYSICS)
    geometry = mujoco.MjModel.from_xml_path(str(geometry_path))
    model = mujoco.MjModel.from_xml_path(str(native_path))
    reference = build_original29_reference(geometry, saved["source_qpos29"])
    for key, value in reference.arrays().items():
        np.testing.assert_array_equal(saved[key], value)
    pair = verify_unmodified_native_pair(reference, motion)
    torch.set_num_threads(1)
    policy, identity, semantics = load_cpu_actor(
        bind(args.checkpoint),
        warm_start_path=bind(args.warm_start),
        source_checkpoint_path=bind(args.source_checkpoint),
    )
    if identity["release_compatibility"]["source_geometry_sha256"] != sha256_file(geometry_path) or identity[
        "release_compatibility"
    ]["native_geometry_sha256"] != sha256_file(native_path):
        raise ValueError("CPU plant/source geometry differs from trained contract")
    inputs.update(semantics["reverified_repository_sources"])
    adapter = OriginalIntentAdapter(motion, reference.virtual_vr21)
    old_received = read(args.baseline_directory / "nominal.received_source.npz")
    for key, value in adapter.source.items():
        np.testing.assert_array_equal(value, old_received[key])
    bind(args.experiment)
    bind(__file__)
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "started.json").open("x") as stream:
        json.dump(
            dict(inputs=inputs, identity=identity, full_lifecycle_requested=True, **FLAGS),
            stream,
            indent=2,
            allow_nan=False,
        )
    result, trace = run_reference_diagnostic(
        root=ROOT, asset_root=args.asset_root, motion_path=motion_path, policy=policy, runtime_adapter=adapter
    )
    trace.update(adapter.root_adapter.arrays())
    trace.update(
        actual_policy_encoder267=np.asarray(adapter.actual_encoder_inputs),
        released_model_raw23=np.asarray(adapter.model_outputs),
        bounded_linear_projection_delta_rad=np.asarray(adapter.projection_delta_rad).reshape(-1, 23),
        source_emission_anchor_setpoint_timestamps_s=np.asarray(adapter.timestamps),
    )
    input_audit = validate_buffered_replay_arrays(adapter.source, trace)
    lifecycle = assess_lifecycle_diagnostic(timeline, result, trace)
    phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    metrics = task_metrics(model, geometry, reference, motion, trace, phase)
    for key in (
        "compiled_model_sha256",
        "physics_config_sha256",
        "initial_state_and_history_sha256",
        "kp_hardware",
        "kd_hardware",
        "effort_limit_hardware_nm",
        "requested_controls",
    ):
        if result[key] != prior["records"][0]["result"][key]:
            raise ValueError(f"existing referee setup changed: {key}")
    trace_path, source_path = output / "nominal.npz", output / "nominal.received_source.npz"
    for path, arrays in ((trace_path, trace), (source_path, adapter.source)):
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"trial input changed during execution: {path}")
    row = dict(
        case="nominal",
        policy_identity=identity,
        result=result,
        lifecycle=lifecycle,
        trace_path=str(trace_path),
        trace_sha256=sha256_file(trace_path),
    )
    report = dict(
        kind="native23_original_intent_full_lifecycle_cpu_v1",
        records=[row],
        timeline=timeline,
        original_task_metrics=metrics,
        inputs=inputs,
        pairing=pair,
        received_path=str(source_path),
        received_sha256=sha256_file(source_path),
        input_audit=input_audit,
        root_measurement_is_privileged_simulator_state=True,
        no_motion_retiming_or_trimming=True,
        **FLAGS,
    )
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            dict(
                completed=result["completed_controls"],
                requested=result["requested_controls"],
                failure=result["failure"],
                metrics=metrics,
                **FLAGS,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
