"""One recorded near-feasible static arm seed; no full retarget or dynamics."""
import argparse
import json
from pathlib import Path

import numpy as np

from gear_sonic.utils.g1_true23_mpc_student import load_inputs,sha256
from gear_sonic.utils.g1_true23_arm_collision_projection import Native23CollisionArmProjector


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    model,contract,_,original,_=load_inputs(args.bundle,"pico")
    with np.load(args.seed/"probe_poses.npz",allow_pickle=False) as a:
        frame=int(a["frames"][0]);pose=a["poses"][0,0].copy();seed=a["poses"][0,2].copy()
    assert frame==3842
    goals=pose[:3]+original["source_task_position_w"][frame,:2]-original["source_qpos29"][frame,:3]
    projector=Native23CollisionArmProjector(model,max_iterations=120,hand_limit=.149)
    result,report=projector.solve(pose,goals,seed_pose=seed)
    report.update(frame=frame,static_feasibility_only=True,speed_qualified=False,
                  seed_pose_sha256=sha256(args.seed/"probe_poses.npz"),full_retarget=False,hardware_authorized=False)
    np.savez_compressed(args.output/"pose.npz",original=pose,seed=seed,result=result)
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--bundle",type=Path,required=True);p.add_argument("--seed",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);run(p.parse_args())
