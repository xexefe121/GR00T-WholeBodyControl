"""Immutable partial-plan inspection; no final or deployment qualification."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.scripts.evaluate_g1_true23_bfmzero import DATA, ROOT, load_case_motion
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL, PHYSICS, task_points
from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model


def run(args):
    payload = args.trace.read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        trace = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(trace.pop("checkpoint_metadata")))
    request_bytes = args.request.read_bytes()
    if metadata["request_sha256"] != hashlib.sha256(request_bytes).hexdigest():
        raise ValueError("checkpoint request hash differs")
    request = json.loads(request_bytes)
    if bool(request.get("motion_override")) != (args.motion_override is not None):
        raise ValueError("checkpoint retarget requires explicit matching local reference")
    motion, timeline, motion_path = load_case_motion(request["clip"], args.motion_override)
    if args.motion_override is not None and request["motion_override"]["reference_sha256"] != hashlib.sha256(motion_path.read_bytes()).hexdigest():
        raise ValueError("checkpoint retarget reference hash differs")
    count = metadata["completed_controls"]
    if trace["qpos"].shape != (count + 1, 30) or not np.array_equal(trace["source_frame"], np.arange(count) + 11):
        raise ValueError("checkpoint source clock differs")
    if len(trace["physics_qpos"]) != int(trace["physics_substeps"].sum()) + 1:
        raise ValueError("checkpoint physics count differs")
    np.testing.assert_allclose(trace["joint_error"], trace["qpos"][1:, 7:] - motion["joint_pos"][11:11 + count], atol=1e-12, rtol=0)
    np.testing.assert_allclose(trace["root_error"], trace["qpos"][1:, :3] - motion["body_pos_w"][11:11 + count, 0], atol=1e-12, rtol=0)
    _, model, physics = prepare_true23_model(ROOT.parent / "GR00T-WholeBodyControl" / MODEL, ROOT / PHYSICS)
    limits = model.jnt_range[1:]
    range_excess = float(np.maximum(0, np.maximum(limits[:, 0] - trace["physics_qpos"][:, 7:], trace["physics_qpos"][:, 7:] - limits[:, 1])).max())
    speed = np.asarray(json.loads((ROOT / PHYSICS).read_text())["physics"]["velocity_limit_hardware_radps"])
    speed_ratio = float(np.max(np.abs(trace["physics_qvel"][:, 6:]) / speed))
    effort_ratio = float(np.max(np.abs(trace["physics_torque"]) / physics.effort))
    phase = next(phase for phase in timeline["phases"] if phase["name"] == "source_motion")
    start, stop = phase["control_start"], min(count, phase["control_stop"])
    original_path = DATA / ("pico_freedancing_v1/optical_reference_v2/original29.npz" if request["clip"] == "pico" else
                           f"{request['clip']}/original_source_bundle_v1/original_reference.npz")
    with np.load(original_path, allow_pickle=False) as archive:
        wanted = archive["source_task_position_w"].copy()
        original_q = archive["source_qpos29"].copy()
    original_model = mujoco.MjModel.from_xml_path(str(ROOT.parent / "GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof.xml"))
    tasks, _ = neutral_wrist_hand_tasks(original_model, model)
    tasks = [next(task for task in tasks if task.name == name) for name in ("left_hand", "right_hand", "head_proxy")]
    data = mujoco.MjData(model)
    errors, feet_errors, yaw_errors = [], [], []
    for control in range(start, stop):
        frame = control + 11
        data.qpos[:] = trace["qpos"][control + 1]
        mujoco.mj_kinematics(model, data)
        points = np.array([data.xpos[model.body(task.target_body).id] + data.xmat[model.body(task.target_body).id].reshape(3, 3) @ task.target_point for task in tasks])
        errors.append(np.linalg.norm(points - data.qpos[:3] - (wanted[frame] - original_q[frame, :3]), axis=-1))
        feet = task_points(data.xpos[1:], data.xquat[1:])[:2]
        feet_errors.append(np.linalg.norm((feet - data.qpos[:3]) - (motion["body_pos_w"][frame, [6, 12]] - motion["body_pos_w"][frame, 0]), axis=-1))
        def yaw(q):
            w, x, y, z = q
            return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        difference = yaw(data.qpos[3:7]) - yaw(original_q[frame, 3:7])
        yaw_errors.append(np.degrees(abs(np.arctan2(np.sin(difference), np.cos(difference)))))
    metrics = None
    if stop > start:
        metrics = dict(source_controls=stop - start, source_requested=phase["requested_controls"],
                       leg_rmse=float(np.sqrt(np.mean(trace["joint_error"][start:stop, :12] ** 2))),
                       arm_rmse=float(np.sqrt(np.mean(trace["joint_error"][start:stop, 13:] ** 2))),
                       root_p95=float(np.percentile(np.linalg.norm(trace["root_error"][start:stop], axis=-1), 95)),
                       original_root_p95=float(np.percentile(np.linalg.norm(trace["qpos"][start+1:stop+1, :3] - original_q[start+11:stop+11, :3], axis=-1), 95)),
                       original_yaw_p95_deg=float(np.percentile(yaw_errors, 95)),
                       original_relative_hand_head_p95_m=np.percentile(errors, 95, axis=0).tolist(),
                       world_axis_relative_foot_p95_m=np.percentile(feet_errors, 95, axis=0).tolist())
    output = dict(kind="independent_incomplete_mpc_checkpoint_inspection", clip=request["clip"],
                  metadata={key: value for key, value in metadata.items() if key != "plans"}, metrics=metrics,
                  range_excess_max=range_excess, velocity_ratio_max=speed_ratio, effort_ratio_max=effort_ratio,
                  strict_physical_limits_pass=range_excess <= 1e-6 and speed_ratio <= 1 and effort_ratio <= 1 + 1e-9,
                  trace_sha256=hashlib.sha256(payload).hexdigest(), request_sha256=metadata["request_sha256"],
                  motion_sha256=hashlib.sha256(motion_path.read_bytes()).hexdigest(),
                  original29_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
                  inspector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_mujoco=request["mujoco"], inspection_mujoco=mujoco.__version__,
                  source_preview_seconds=request["conservative_effective_source_preview_seconds"],
                  motion_override=None if args.motion_override is None else str(motion_path),
                  is_final_result=False, full_body_tracking_qualified=False, hardware_authorized=False)
    if args.output.exists():
        raise ValueError("immutable inspection output already exists")
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-override", type=Path)
    run(parser.parse_args())
