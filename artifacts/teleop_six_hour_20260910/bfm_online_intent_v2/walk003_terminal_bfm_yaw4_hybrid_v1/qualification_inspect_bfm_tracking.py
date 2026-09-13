"""Independent original-intent and heading metrics from saved physical traces."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import ROOT, DATA, load_case_motion
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, task_points
from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks


def yaw(q):
    x, y, z, w = np.moveaxis(q[..., [1, 2, 3, 0]], -1, 0)
    return np.arctan2(2 * (w*z+x*y), 1-2*(y*y+z*z))


def resolve_motion_override(report, explicit=None):
    """Bind a local reference to an immutable producer receipt from either OS."""
    declared = report.get("motion_override")
    if declared is None:
        if explicit is not None:
            raise ValueError("original-reference trace cannot acquire an undeclared override")
        return None
    if isinstance(declared, dict):
        if explicit is None:
            raise ValueError("producer reference receipt requires an explicit local motion override")
        expected = declared["reference_sha256"]
    elif explicit is None:
        return Path(declared)
    else:
        expected = report.get("reference_sha256")
        if expected is None and Path(declared).is_file():
            expected = hashlib.sha256(Path(declared).read_bytes()).hexdigest()
        if expected is None:
            raise ValueError("cross-platform local reference requires a recorded reference hash")
    path = Path(explicit)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("local reference differs from the producer receipt")
    return path


def inspect(directory, output_name="original_intent_metrics.json", motion_override=None):
    report = json.loads((directory / "report.json").read_text())
    motion, timeline, motion_path = load_case_motion(report["clip"], resolve_motion_override(report, motion_override))
    provenance=json.loads((directory/"provenance.json").read_text()) if (directory/"provenance.json").exists() else {}
    expected=provenance.get(str(motion_path),provenance.get("hashes",{}).get(str(motion_path)))
    if expected is not None and hashlib.sha256(motion_path.read_bytes()).hexdigest()!=expected:
        raise ValueError("motion bytes changed since physical replay")
    native = mujoco.MjModel.from_xml_path(str(ROOT.parent / "GR00T-WholeBodyControl" / MODEL))
    source = mujoco.MjModel.from_xml_path(str(ROOT.parent / "GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml"))
    tasks, convention = neutral_wrist_hand_tasks(source, native)
    tasks = [next(t for t in tasks if t.name == n) for n in ("left_hand", "right_hand", "head_proxy")]
    original_path = DATA / ("pico_freedancing_v1/optical_reference_v2/original29.npz" if report["clip"] == "pico" else
                            f"{report['clip']}/original_source_bundle_v1/original_reference.npz")
    declared_override = report.get("motion_override")
    if isinstance(declared_override, dict) and declared_override.get("original29_sha256") is not None:
        if hashlib.sha256(original_path.read_bytes()).hexdigest() != declared_override["original29_sha256"]:
            raise ValueError("original29 intent differs from producer receipt")
    with np.load(original_path, allow_pickle=False) as z:
        wanted = z["source_task_position_w"].copy()
        wanted_quat = z["source_task_quaternion_wxyz"].copy()
        original_root = z["source_qpos29"][:,:3].copy()
        original_root_quat = z["source_qpos29"][:,3:7].copy()
    with np.load(directory / "trace.npz", allow_pickle=False) as z:
        poses = z["qpos"][1:].copy()
        joint_error = z["joint_error"].copy()
    phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    start, stop = phase["control_start"], min(len(poses), phase["control_stop"])
    if stop <= start:
        return dict(clip=report["clip"], no_source_frames=True)
    actual, orientation, feet = [], [], []
    data = mujoco.MjData(native)
    for pose in poses[start:stop]:
        data.qpos[:] = pose
        mujoco.mj_kinematics(native, data)
        actual.append([data.xpos[native.body(t.target_body).id] +
                       data.xmat[native.body(t.target_body).id].reshape(3,3) @ t.target_point for t in tasks])
        orientation.append([data.xquat[native.body(t.target_body).id].copy() for t in tasks])
        feet.append(task_points(data.xpos[1:], data.xquat[1:])[:2])
    actual, orientation, feet = np.asarray(actual), np.asarray(orientation), np.asarray(feet)
    frame = slice(start + 11, stop + 11)
    root = poses[start:stop, :3]
    target_root = motion["body_pos_w"][frame, 0]
    actual_yaw = yaw(poses[start:stop, 3:7])
    target_yaw = yaw(motion["body_quat_w"][frame, 0])
    yaw_error = np.arctan2(np.sin(actual_yaw-target_yaw), np.cos(actual_yaw-target_yaw))
    angle = (Rotation.from_quat(orientation.reshape(-1,4)[:,[1,2,3,0]]).inv() *
             Rotation.from_quat(wanted_quat[frame].reshape(-1,4)[:,[1,2,3,0]])).magnitude().reshape(-1,3)
    def local(value, heading):
        return Rotation.from_euler("z", -heading).apply(value)
    local_feet = np.stack([local(feet[:,i]-root, actual_yaw) for i in range(2)], 1)
    native_feet = motion["body_pos_w"][frame][:,[6,12]]
    target_feet = np.stack([local(native_feet[:,i]-target_root,target_yaw) for i in range(2)], 1)
    original_yaw_error=np.arctan2(np.sin(actual_yaw-yaw(original_root_quat[frame])),np.cos(actual_yaw-yaw(original_root_quat[frame])))
    result = dict(metric_revision=4, clip=report["clip"], controls=stop-start, requested=phase["requested_controls"],
                  original_hand_head_world_p95_m=np.percentile(np.linalg.norm(actual-wanted[frame],axis=-1),95,axis=0).tolist(),
                  original_hand_head_relative_p95_m=np.percentile(np.linalg.norm((actual-root[:,None])-(wanted[frame]-original_root[frame,None]),axis=-1),95,axis=0).tolist(),
                  retarget_root_compensated_hand_head_p95_m=np.percentile(np.linalg.norm((actual-root[:,None])-(wanted[frame]-target_root[:,None]),axis=-1),95,axis=0).tolist(),
                  original_root_world_p95_m=float(np.percentile(np.linalg.norm(root-original_root[frame],axis=-1),95)),
                  original_root_yaw_abs_p95_deg=float(np.rad2deg(np.percentile(np.abs(original_yaw_error),95))),
                  original_hand_head_orientation_p95_rad=np.percentile(angle,95,axis=0).tolist(),
                  yaw_abs_p95_deg=float(np.rad2deg(np.percentile(np.abs(yaw_error),95))),
                  own_heading_foot_p95_m=np.percentile(np.linalg.norm(local_feet-target_feet,axis=-1),95,axis=0).tolist(),
                  world_axis_relative_foot_p95_m=np.percentile(np.linalg.norm((feet-root[:,None])-(native_feet-target_root[:,None]),axis=-1),95,axis=0).tolist(),
                  joint_rmse_by_hardware_joint=np.sqrt(np.mean(joint_error[start:stop]**2,axis=0)).tolist(),
                  convention=convention, heading_alignment_is_diagnostic_only=True,
                  motion_sha256=hashlib.sha256(motion_path.read_bytes()).hexdigest(),
                  trace_sha256=hashlib.sha256((directory/"trace.npz").read_bytes()).hexdigest(),
                  report_sha256=hashlib.sha256((directory/"report.json").read_bytes()).hexdigest(),
                  inspector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  original29_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
                  physical_limits_passed=report["range_excess_max"]<=1e-6 and report["velocity_ratio_max"]<=1 and report["effort_ratio_max"]<=1+1e-9,
                  full_body_tracking_qualified=False, hardware_authorized=False)
    if output_name is not None:
        (directory / output_name).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("convention","joint_rmse_by_hardware_joint")}))
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("directories",nargs="+",type=Path)
    p.add_argument("--motion-override",type=Path)
    args=p.parse_args()
    for directory in args.directories:
        inspect(directory, motion_override=args.motion_override)
