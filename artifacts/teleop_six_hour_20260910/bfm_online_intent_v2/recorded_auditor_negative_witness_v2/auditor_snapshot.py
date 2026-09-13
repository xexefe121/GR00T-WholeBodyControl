"""Independent immutable recorded-source gate audit; never an online pass."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from artifacts.teleop_six_hour_20260910.inspect_bfm_tracking import inspect
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import ROOT, load_case_motion
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory, output):
    if output.exists():
        raise ValueError("qualification outputs are immutable")
    report = json.loads((directory / "report.json").read_text())
    intent = inspect(directory, output_name=None)
    motion, timeline, motion_path = load_case_motion(report["clip"], report.get("motion_override"))
    _, native, physics = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    with np.load(directory / "trace.npz", allow_pickle=False) as archive:
        required = ("qpos", "qvel", "joint_error", "physics_qpos", "physics_qvel", "physics_torque")
        arrays = {key: archive[key].copy() for key in required}
        substeps = archive["physics_substeps"].copy() if "physics_substeps" in archive else None
        source_frame = archive["source_frame"].copy() if "source_frame" in archive else None
    count = len(arrays["qpos"]) - 1
    shapes = dict(qpos=(count + 1, 30), qvel=(count + 1, 29), joint_error=(count, 23))
    for key, value in arrays.items():
        if not np.isfinite(value).all() or (key in shapes and value.shape != shapes[key]):
            raise ValueError("nonfinite or invalid physical trace: " + key)
    nphysics = len(arrays["physics_torque"])
    if (arrays["physics_torque"].shape != (nphysics, 23)
            or arrays["physics_qpos"].shape != (nphysics + 1, 30)
            or arrays["physics_qvel"].shape != (nphysics + 1, 29)):
        raise ValueError("physical trace dimensions differ")
    if substeps is None:
        if nphysics != count * 10:
            raise ValueError("trace without substep ledger contains partial or missing controls")
        substeps = np.full(count, 10, dtype=int)
    if substeps.shape != (count,) or np.any(substeps != substeps.astype(int)) or np.any(substeps < 1) or np.any(substeps > 10) or int(substeps.sum()) != nphysics:
        raise ValueError("invalid physical substep ledger")
    if np.any(substeps[:-1] != 10):
        raise ValueError("nonterminal incomplete control")
    if source_frame is not None and not np.array_equal(source_frame, np.arange(count) + 11):
        raise ValueError("original source frames removed or repeated")
    if count > len(motion["joint_pos"]) - 11:
        raise ValueError("trace extends past source lifecycle")
    np.testing.assert_allclose(arrays["joint_error"], arrays["qpos"][1:, 7:] - motion["joint_pos"][11:11+count], atol=1e-10, rtol=0)
    indices = np.r_[0, np.cumsum(substeps)]
    np.testing.assert_array_equal(arrays["qpos"], arrays["physics_qpos"][indices])
    np.testing.assert_array_equal(arrays["qvel"], arrays["physics_qvel"][indices])
    limits = native.jnt_range[1:]
    q = arrays["physics_qpos"][:, 7:]
    excess = float(np.maximum(0, np.maximum(limits[:, 0] - q, q - limits[:, 1])).max())
    velocity = np.asarray(json.loads((ROOT / PHYSICS).read_text())["physics"]["velocity_limit_hardware_radps"])
    speed = float(np.max(np.abs(arrays["physics_qvel"][:, 6:]) / velocity))
    effort = float(np.max(np.abs(arrays["physics_torque"]) / physics.effort))
    root_poses = arrays["physics_qpos"][:, :7]
    tilt = np.arccos(np.clip(1 - 2 * np.sum(root_poses[:, 4:6] ** 2, axis=1), -1, 1))
    no_fall = bool(np.min(root_poses[:, 2]) >= .25 and np.max(tilt) <= 1.2)
    clock = report.get("simulated_seconds")
    warnings = report.get("engine_warning_counts")
    warnings_clear = (isinstance(warnings, list) and len(warnings) == 8
                      and all(type(value) is int and value == 0 for value in warnings))
    start, stop = phase["control_start"], min(count, phase["control_stop"])
    has_source = stop > start
    legs = float(np.sqrt(np.mean(arrays["joint_error"][start:stop, :12] ** 2))) if has_source else None
    full_source = count >= phase["control_stop"] and np.all(substeps[:phase["control_stop"]] == 10)
    full_lifecycle = count == len(motion["joint_pos"]) - 11 and np.all(substeps == 10)
    gates = dict(full_original_source=bool(full_source), full_lifecycle=bool(full_lifecycle),
                 no_reported_failure=report.get("failure") is None,
                 no_fall_at_any_physics_step=no_fall,
                 no_engine_warnings=warnings_clear,
                 full_physics_clock=clock is not None and abs(float(clock) - nphysics * .002) <= 1e-8,
                 original_root_position=has_source and intent["original_root_world_p95_m"] <= .20,
                 original_heading=has_source and intent["original_root_yaw_abs_p95_deg"] <= 15,
                 both_root_relative_feet=has_source and max(intent["world_axis_relative_foot_p95_m"]) <= .12,
                 all_twelve_legs=has_source and legs <= .15,
                 both_original_relative_hands=has_source and max(intent["original_hand_head_relative_p95_m"][:2]) <= .15,
                 original_relative_head=has_source and intent["original_hand_head_relative_p95_m"][2] <= .10,
                 every_physics_joint_bound=excess <= 1e-6,
                 every_physics_speed=speed <= 1,
                 every_physics_effort=effort <= 1 + 1e-9)
    result = dict(kind="independent_recorded_native23_candidate_gate_audit", audit_revision=2, clip=report["clip"],
                  recorded_source_tracking_pass=bool(all(gates.values())), gates=gates,
                  failed_gates=[key for key, value in gates.items() if not value],
                  completed_controls=count, physics_steps=nphysics, source_controls=max(0, stop-start),
                  source_requested_controls=phase["requested_controls"], leg_rmse_rad=legs,
                  actual_range_excess_rad=excess, actual_velocity_ratio=speed, actual_effort_ratio=effort,
                  engine_warning_counts=report.get("engine_warning_counts"),
                  foot_and_joint_reference="declared native reference; original29 used for hand/head intent; original root/heading unchanged",
                  original_intent=intent, reference=str(motion_path), reference_sha256=sha(motion_path),
                  trace_sha256=sha(directory / "trace.npz"), report_sha256=sha(directory / "report.json"),
                  audit_sha256=sha(__file__), acceptance_sha256=sha(Path(__file__).with_name("SIM_ACCEPTANCE.md")),
                  timing_qualified=False, received_only_stream_qualified=False,
                  sensor_only_controller_qualified=False, full_body_teleoperation_qualified=False,
                  hardware_authorized=False)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k != "original_intent"}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.directory, args.output)
