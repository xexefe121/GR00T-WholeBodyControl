"""Independent actual-contact and strict hand/head/neighbor-speed result check."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mpc_student import load_inputs,sha256
from gear_sonic.utils.g1_true23_intent_arm_ik import HAND_POINTS_LOCAL


def run(args):
    model,contract,_,original,_=load_inputs(args.bundle,"pico")
    source=json.loads((args.probe/"report.json").read_text())
    with np.load(args.probe/"probe_poses.npz",allow_pickle=False) as a:frames=a["frames"].copy();poses=a["poses"].copy()
    with np.load(args.references/"pico/reference.npz",allow_pickle=False) as a:reference=a["joint_pos"].copy()
    data=mujoco.MjData(model);out=[]
    for frame,values in zip(frames,poses):
        row=dict(frame=int(frame),poses={})
        for kind,pose in zip(("original","bounded","static"),values):
            data.qpos[:]=pose;data.qvel[:]=0.;mujoco.mj_forward(model,data)
            points=[]
            for name,offset in (("left_wrist_roll_rubber_hand",HAND_POINTS_LOCAL[0]),("right_wrist_roll_rubber_hand",HAND_POINTS_LOCAL[1]),("torso_link",np.array([0.,0.,.35]))):
                body=model.body(name).id;points.append(data.xpos[body]+data.xmat[body].reshape(3,3)@offset)
            desired=original["source_task_position_w"][frame]-original["source_qpos29"][frame,:3]
            errors=np.linalg.norm(np.asarray(points)-pose[:3]-desired,axis=1)
            contacts=[float(c.dist) for c in data.contact if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2] and c.dist<0]
            step=np.asarray(contract["native_velocity"])[13:]*.8*.02
            row["poses"][kind]=dict(relative_hand_head_errors_m=errors.tolist(),strict_all_task_15cm_pass=bool(np.all(errors<=.15)),
                active_self_contact_count=len(contacts),max_active_self_penetration_m=-min(contacts,default=0.),
                incoming_80pct_speed_ratio=float(np.max(np.abs(pose[20:]-reference[frame-1,13:])/step)),
                outgoing_80pct_speed_ratio=float(np.max(np.abs(reference[frame+1,13:]-pose[20:])/step)))
        out.append(row)
    result=dict(probe_report_sha256=sha256(args.probe/"report.json"),pose_arrays_sha256=sha256(args.probe/"probe_poses.npz"),
                independently_recomputed_native_contacts=True,results=out,
                earlier_tolerance_note="v1 hand_gate_pass used a1e-6m tolerance; this sidecar and v2 use exact<=.15m",
                source_snapshot_note="v2 executed same probe script as v1; later head-metric source edit preserved separately, executed probe snapshot restored from matching v1")
    path=args.probe/"independent_geometry_verification.json"
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--bundle",type=Path,required=True);p.add_argument("--references",type=Path,required=True)
    p.add_argument("--probe",type=Path,required=True);run(p.parse_args())
