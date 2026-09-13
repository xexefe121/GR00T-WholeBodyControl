"""Fixed 64-frame source samples; native arms only, unchanged root/legs/waist."""

import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_intent_arm_ik import Native23ArmIK


ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1")


def platform_path(value):
    value = str(value).replace("\\", "/")
    return Path("/mnt/c/" + value[3:] if value.startswith("C:/") else value)


def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"))
    solver = Native23ArmIK(model)
    rows = []
    for clip in ("pico", "walk002", "walk003", "walk008"):
        report_path = DATA / ("normal_core_pico_v1/report.json" if clip == "pico" else
                              f"released_core_comparison_v1/normal/{clip}/report.json")
        timeline = json.loads(report_path.read_text())["timeline"]
        motion = dict(np.load(platform_path(timeline["timeline_path"]), allow_pickle=False))
        phase = next(p for p in timeline["phases"] if p["name"] == "source_motion")
        frames = np.linspace(phase["control_start"]+11, phase["control_stop"]+10, 64).astype(int)
        original_path = DATA / ("pico_freedancing_v1/optical_reference_v2/original29.npz" if clip == "pico" else
                                f"{clip}/original_source_bundle_v1/original_reference.npz")
        original = dict(np.load(original_path, allow_pickle=False))
        before, after, angle, nfev, success = [], [], [], [], []
        tick = time.perf_counter()
        for frame in frames:
            pose = np.r_[motion["body_pos_w"][frame, 0], motion["body_quat_w"][frame, 0], motion["joint_pos"][frame]]
            wanted = original["source_task_position_w"][frame, :2]
            wanted_quat = original["source_task_quaternion_wxyz"][frame, :2]
            before.append(np.linalg.norm(solver.hand_poses(pose)[0] - wanted, axis=1))
            result = solver.solve(pose, wanted, wanted_quat)
            after.append(result.position_errors_m)
            angle.append(result.orientation_errors_rad)
            nfev.append(result.nfev)
            success.append(result.success)
        rows.append(dict(clip=clip, samples=64, baseline_position_p95_m=np.percentile(before, 95, axis=0).tolist(),
                         arms_ik_position_p95_m=np.percentile(after, 95, axis=0).tolist(),
                         arms_ik_position_max_m=np.max(after, axis=0).tolist(),
                         arms_ik_orientation_p95_rad=np.percentile(angle, 95, axis=0).tolist(),
                         nfev_mean=float(np.mean(nfev)), nfev_max=int(np.max(nfev)),
                         convergence_fraction=float(np.mean(success)), elapsed_s=time.perf_counter()-tick))
    result = dict(scope="64 fixed source frames per clip; arm-only reference geometry, not dynamics",
                  position_weight=1., orientation_weight=.03, posture_weight=.002, cases=rows)
    with (Path(__file__).resolve().parent / "arm_ik_source_samples.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
