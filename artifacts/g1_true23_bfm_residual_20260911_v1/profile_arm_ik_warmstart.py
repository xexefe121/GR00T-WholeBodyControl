"""Consecutive PICO source block: same objective, cold vs previous-arm seed."""

import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_intent_arm_ik import Native23ArmIK


ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")


def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"))
    solver = Native23ArmIK(model)
    velocity = np.asarray(json.loads((ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json").read_text())["physics"]["velocity_limit_hardware_radps"])[solver.qpos_indices-7]
    timeline = json.loads((DATA / "normal_core_pico_v1/report.json").read_text())["timeline"]
    value = str(timeline["timeline_path"]).replace("\\", "/")
    motion_path = Path("/mnt/c/" + value[3:] if value.startswith("C:/") else value)
    motion = dict(np.load(motion_path, allow_pickle=False))
    original = dict(np.load(DATA / "pico_freedancing_v1/optical_reference_v2/original29.npz", allow_pickle=False))
    phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
    start = phase["control_start"] + 11
    rows = []
    for warm in (False, True):
        previous = None
        positions, orientations, times, evaluations, successes, costs, arm_step = [], [], [], [], [], [], []
        for frame in range(start, start+64):
            posture = np.r_[motion["body_pos_w"][frame, 0], motion["body_quat_w"][frame, 0], motion["joint_pos"][frame]]
            seed = posture.copy()
            if warm and previous is not None:
                seed[solver.qpos_indices] = previous[solver.qpos_indices]
            tick = time.perf_counter()
            fitted = solver.solve(seed, original["source_task_position_w"][frame, :2],
                                  original["source_task_quaternion_wxyz"][frame, :2], posture_qpos=posture)
            times.append((time.perf_counter()-tick)*1000)
            positions.append(fitted.position_errors_m)
            orientations.append(fitted.orientation_errors_rad)
            evaluations.append(fitted.nfev)
            successes.append(fitted.success)
            costs.append(fitted.costs)
            if previous is not None:
                arm_step.append(np.abs(fitted.qpos[solver.qpos_indices]-previous[solver.qpos_indices]))
            previous = fitted.qpos.copy()
        arm_step = np.asarray(arm_step)
        worst = np.unravel_index(np.argmax(arm_step), arm_step.shape)
        ratios = arm_step / (.02 * velocity)
        worst_ratio = np.unravel_index(np.argmax(ratios), ratios.shape)
        rows.append(dict(warm_start=warm, position_p95_m=np.percentile(positions, 95, axis=0).tolist(),
                         position_max_m=np.max(positions, axis=0).tolist(),
                         orientation_p95_rad=np.percentile(orientations, 95, axis=0).tolist(),
                         solve_ms_mean=float(np.mean(times)), solve_ms_p95=float(np.percentile(times,95)),
                         solve_ms_max=float(np.max(times)), nfev_mean=float(np.mean(evaluations)),
                         nfev_max=int(np.max(evaluations)), convergence_fraction=float(np.mean(successes)),
                         mean_cost=float(np.mean(costs)), arm_frame_step_max_rad=float(np.max(arm_step)),
                         max_step_frame=int(start+worst[0]+1), max_step_joint=model.joint(int(solver.qpos_indices[worst[1],worst[2]]-6)).name,
                         target_velocity_ratio_max=float(ratios.max()),
                         max_velocity_ratio_frame=int(start+worst_ratio[0]+1),
                         max_velocity_ratio_joint=model.joint(int(solver.qpos_indices[worst_ratio[1],worst_ratio[2]]-6)).name))
    report = dict(clip="pico", frames=64, reference_frame_start=start, reference_frame_stop_exclusive=start+64,
                  source_time_start_s=0., same_objective=True, posture_regularization="current native source qpos for both runs",
                  root_waist_legs="current reference, unchanged", cases=rows)
    with (Path(__file__).resolve().parent / "arm_ik_warmstart_profile_v2.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
