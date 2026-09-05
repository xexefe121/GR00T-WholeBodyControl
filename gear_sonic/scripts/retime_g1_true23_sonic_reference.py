"""Offline path-preserving slowdown of an existing native23 SONIC reference.

All joint/root path samples remain in the source phase domain. This changes
tempo and recomputes native23 FK/velocities; it does not establish dynamic
feasibility, original-tempo parity, contacts, or hardware readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from gear_sonic.scripts.retarget_g1_29dof_to_23dof_task_space import _interpolate_time_scaled_trajectory
from gear_sonic.utils.g1_23dof_task_space_retarget import build_mjlab_motion_arrays
from gear_sonic.utils.g1_true23_sonic_library_replay import validate_library_motion


def retime_reference(motion, model, scale):
    count = validate_library_motion(motion)
    if not np.isfinite(scale) or not 1 <= scale <= 4:
        raise ValueError("diagnostic time scale must be finite and within 1..4")
    if scale == 1:
        return {key: value.copy() for key, value in motion.items()}, np.arange(count, dtype=np.float64), 1.0
    position, quaternion, joints, actual = _interpolate_time_scaled_trajectory(
        motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0], motion["joint_pos"], scale
    )
    # The shared FK builder only consumes these four trajectory fields. No
    # fabricated IK expert diagnostics, contacts or optimizer success fields.
    trajectory = SimpleNamespace(
        root_pos_w=position, root_quat_wxyz=quaternion, joint_pos_hardware=joints, fps=50.0
    )
    arrays = build_mjlab_motion_arrays(model, trajectory)
    phase = np.linspace(0, count - 1, len(joints))
    validate_library_motion(arrays)
    return arrays, phase, actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--time-scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("retiming outputs must be new")
    with np.load(args.motion, allow_pickle=False) as archive:
        motion = {key: archive[key].copy() for key in archive.files}
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    arrays, phase, actual = retime_reference(motion, model, args.time_scale)
    source_frames = len(motion["joint_pos"])
    exact_source_indices = np.flatnonzero(np.abs(phase - np.round(phase)) <= 1e-10)
    matched_source = np.round(phase[exact_source_indices]).astype(int)
    report = {
        "kind": "g1_true23_path_preserving_retiming_diagnostic_v1",
        "requested_time_scale": args.time_scale,
        "actual_time_scale": actual,
        "source_frame_count": source_frames,
        "retimed_frame_count": len(phase),
        "source_duration_s": (source_frames - 1) / 50,
        "retimed_duration_s": (len(phase) - 1) / 50,
        "original_samples_present_at_exact_phases": len(np.unique(matched_source)),
        "maximum_original_joint_sample_error_rad": float(
            np.max(np.abs(arrays["joint_pos"][exact_source_indices] - motion["joint_pos"][matched_source]))
        ),
        "source_phase_range": [float(phase[0]), float(phase[-1])],
        "source_phase_definition": "linspace(0, source_frame_count-1, retimed_frame_count)",
        "reference_phase_at_first_control_frame": float(phase[10]),
        "velocities_recomputed_from_retimed_path": actual != 1,
        "foot_contacts_or_com_optimized": False,
        "dynamic_feasibility_proven": False,
        "original_tempo_parity": actual == 1,
        "hardware_authorized": False,
        "deployment_ready": False,
        "sources": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                args.motion,
                args.model,
                Path(__file__),
                Path(__file__).with_name("retarget_g1_29dof_to_23dof_task_space.py"),
                Path(__file__).parents[1] / "utils/g1_23dof_task_space_retarget.py",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in report.items() if key != "sources"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
