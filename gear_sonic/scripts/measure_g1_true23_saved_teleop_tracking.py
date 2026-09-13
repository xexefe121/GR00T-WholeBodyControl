"""Measure saved full-body replay fidelity without stepping or changing a policy.

Compare measured dynamics to the imported native23 reference, not an original29
physics baseline. Align the arbitrary source XY origin and initial yaw once.
Never align height, fit a time lag, or realign each frame. Fallback and failed
prefixes remain explicit; a stability pass is not a tracking qualification.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.measure_g1_true23_original_source_tracking import native_points, point_statistics
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_task_space_retarget import DEFAULT_TASKS
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import sha256_file
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_motion_reference_packet_bundle import build_motion_reference_packet_bundle
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def source_indices(trace, packets, report, source_frame_count):
    """Post-step source time is initial q10 plus elapsed 20-ms control ticks.

    Packet zero carries anchor q9/control q10. The initial robot state is q10;
    its first integrated state is therefore compared to source q11, not q10.
    """
    poses = np.asarray(trace["qpos"])
    times = np.asarray(trace["simulation_time"])
    controls = len(poses) - 1
    indexes = np.asarray([p["control_source_frame_index"] for p in packets])
    anchors = np.asarray([p["pico_anchor_source_frame_index"] for p in packets])
    if (
        poses.shape != (controls + 1, 30)
        or not np.isfinite(poses).all()
        or not np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1, rtol=0, atol=1e-6)
        or not 1 <= controls <= len(packets)
        or report["requested_controls"] != len(packets)
        or report["recorded_post_attempt_states"] != controls
        or report["controller_completed_controls"] != controls
        or times.shape != (controls + 1,)
        or not np.allclose(times, np.arange(controls + 1) * 0.02, rtol=0, atol=1e-8)
        or indexes.dtype.kind not in "iu"
        or not np.array_equal(indexes, np.arange(10, 10 + len(packets)))
        or not np.array_equal(anchors, indexes - 1)
        or 10 + controls >= source_frame_count
    ):
        raise ValueError("tracking requires finite, contiguous, unshifted 50-Hz recorded states")
    transition = report["fallback_first_transition"]
    if report["fallback_active"]:
        if type(transition) is not int or not 0 <= transition < controls:
            raise ValueError("fallback transition does not identify a recorded control")
    elif transition is not None:
        raise ValueError("unlatched fallback has a transition")
    successful = report["successful_controls"]
    # Historical reports only covered post-integration physical failures. A
    # policy can also reject before integration; that attempt has no new state.
    failed_integrated = report.get("failed_attempt_integrated", report["failure"] is not None)
    if (
        (report["failure"] is None and failed_integrated not in (None, False))
        or (report["failure"] is not None and type(failed_integrated) is not bool)
        or successful != controls - int(bool(failed_integrated))
        or report.get("attempted_controls", successful + int(report["failure"] is not None))
        != successful + int(report["failure"] is not None)
    ):
        raise ValueError("recorded failure/success count mismatch")
    if report["passed"] is not (
        report["failure"] is None and controls == len(packets) and not report["fallback_active"]
    ):
        raise ValueError("recorded stability outcome mismatch")
    return np.arange(10, 11 + controls), controls if transition is None else transition


def yaw(poses):
    matrix = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
    return np.arctan2(matrix[:, 1, 0], matrix[:, 0, 0])


def align_reference_once(reference, initial_actual):
    """Only the first pose chooses fixed yaw and XY translation; preserve Z."""
    reference = np.asarray(reference, dtype=np.float64)
    initial_actual = np.asarray(initial_actual, dtype=np.float64)
    angle = float(yaw(initial_actual[None])[0] - yaw(reference[:1])[0])
    rotation = Rotation.from_euler("z", angle)
    aligned = reference.copy()
    aligned[:, :3] = rotation.apply(reference[:, :3])
    translation = initial_actual[:3] - aligned[0, :3]
    translation[2] = 0.0
    aligned[:, :3] += translation
    aligned[:, 2] = reference[:, 2]
    aligned[:, 3:7] = (rotation * Rotation.from_quat(reference[:, [4, 5, 6, 3]])).as_quat()[:, [3, 0, 1, 2]]
    return aligned, dict(yaw_radians=angle, translation_xyz_m=translation.tolist(), height_aligned=False)


def statistics(values):
    values = np.asarray(values)
    return dict(mean=float(values.mean()), p95=float(np.percentile(values, 95)), maximum=float(values.max()))


def tracking_statistics(actual, reference, points, reference_points):
    if not len(actual):
        return None
    joint_error = actual[:, 7:] - reference[:, 7:]
    rotations = Rotation.from_quat(actual[:, [4, 5, 6, 3]])
    desired = Rotation.from_quat(reference[:, [4, 5, 6, 3]])
    yaw_difference = yaw(actual) - yaw(reference)
    return dict(
        samples=len(actual),
        base_position_error_m=statistics(np.linalg.norm(actual[:, :3] - reference[:, :3], axis=1)),
        base_height_abs_error_m=statistics(np.abs(actual[:, 2] - reference[:, 2])),
        base_orientation_error_rad=statistics((rotations.inv() * desired).magnitude()),
        base_yaw_abs_error_rad=statistics(np.abs(np.arctan2(np.sin(yaw_difference), np.cos(yaw_difference)))),
        joint_rmse_rad={
            label: float(np.sqrt(np.mean(joint_error[:, selected] ** 2)))
            for label, selected in (
                ("all23", slice(None)),
                ("legs12", slice(0, 12)),
                ("waist_yaw", slice(12, 13)),
                ("arms10", slice(13, 23)),
            )
        },
        task_position=point_statistics(
            points, reference_points, actual[:, :3], reference[:, :3], [task.name for task in DEFAULT_TASKS]
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-report", "recording-report", "asset-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite tracking evidence")
    source = json.loads(args.source_report.read_text())
    recorded = json.loads(args.recording_report.read_text())
    if (
        source["kind"] != "g1_true23_public_twist2_replay_import_v1"
        or recorded["kind"] != "g1_true23_saved_fullbody_recorded_diagnostic_v1"
        or source["speed_factor"] != 1.0
        or source["added_hold_frames"] != 0
        or source["output_fps"] != 50
        or recorded["recorded_simulation_only"] is not True
        or recorded["hardware_authorized"] is not False
    ):
        raise ValueError("requires unchanged public-motion import and recorded SIM diagnostic")
    model_path = args.asset_root / MODEL
    bindings = {}
    for path, expected in (
        (Path(source["source_path"]), source["source_sha256"]),
        (Path(source["motion_path"]), source["motion_sha256"]),
        (Path(source["packet_path"]), source["packet_sha256"]),
        (Path(recorded["packets"]), recorded["packets_sha256"]),
        (Path(recorded["trace_path"]), recorded["trace_sha256"]),
        (model_path, recorded["model_sha256"]),
        (model_path, source["physical_model_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"tracking evidence hash mismatch: {path}")
        bindings[str(path.resolve())] = expected
    if source["packet_sha256"] != recorded["packets_sha256"]:
        raise ValueError("recording and imported packet identities differ")
    for path in (args.source_report, args.recording_report, Path(__file__)):
        bindings[str(path.resolve())] = sha256_file(path)
    with np.load(source["motion_path"], allow_pickle=False) as archive:
        motion = dict(archive)
    packets = json.loads(Path(source["packet_path"]).read_text())
    if packets != build_motion_reference_packet_bundle(motion, source_motion_sha256=source["motion_sha256"]):
        raise ValueError("packets do not reproduce from the unchanged imported motion")
    packets = packets["robot_independent_reference_packets"]
    with np.load(recorded["trace_path"], allow_pickle=False) as archive:
        trace = dict(archive)
    source_poses = np.column_stack((motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0], motion["joint_pos"]))
    indexes, sonic_count = source_indices(trace, packets, recorded, len(source_poses))
    actual = trace["qpos"]
    reference, calibration = align_reference_once(source_poses[indexes], actual[0])
    _, model, _ = prepare_true23_model(model_path, Path(__file__).resolve().parents[2] / PHYSICS)
    if (model.nq, model.nv, model.nu) != (30, 29, 23) or tuple(
        model.joint(i).name for i in range(1, model.njnt)
    ) != tuple(HARDWARE_23_JOINT_NAMES):
        raise ValueError("expected exact native23 FK model/joint order")
    points, reference_points = native_points(model, actual), native_points(model, reference)
    segments = {}
    for name, stop in (
        ("all_recorded_including_fallback_and_failure", len(actual) - 1),
        ("sonic_only", sonic_count),
    ):
        span = slice(1, stop + 1)
        segments[name] = tracking_statistics(actual[span], reference[span], points[span], reference_points[span])
    result = dict(
        kind="g1_true23_saved_teleop_tracking_diagnostic_v1",
        input_bindings=bindings,
        compiled_fk_model_sha256=compiled_model_sha256(model),
        tasks=[asdict(task) for task in DEFAULT_TASKS],
        calibration=calibration,
        initial_source_frame=int(indexes[0]),
        first_postcontrol_source_frame=int(indexes[1]),
        last_postcontrol_source_frame=int(indexes[-1]),
        source_time_alignment="initial_control_q10_plus_measured_elapsed_time_no_lag_fit",
        initial_height_error_m=float(actual[0, 2] - reference[0, 2]),
        reference_total_frames=len(source_poses),
        requested_controls=recorded["requested_controls"],
        recorded_controls=len(actual) - 1,
        sonic_controls=sonic_count,
        fallback_controls=len(actual) - 1 - sonic_count,
        sonic_fraction_of_requested=sonic_count / recorded["requested_controls"],
        complete_sonic_replay=recorded["passed"],
        recording_failure=recorded["failure"],
        decoder_sha256=recorded["decoder_sha256"],
        reference_orientation=recorded.get("reference_orientation", "not_recorded_in_historical_report"),
        reference_geometry=recorded.get("reference_geometry", "native23_forward_kinematics"),
        action_convention=recorded.get("action_convention", "native_tanh"),
        metrics=segments,
        reference="unchanged_imported_native23_FK_not_original29_task_space_fidelity",
        lag_fitted=False,
        per_frame_world_alignment=False,
        physics_stepped=False,
        tracking_fidelity_qualified=False,
        hardware_authorized=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("complete_sonic_replay", "sonic_controls", "recorded_controls", "metrics")
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
