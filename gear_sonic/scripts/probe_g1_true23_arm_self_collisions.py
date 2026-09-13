"""Active-contact audit and bounded worst-frame arm projection feasibility."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mpc_student import load_inputs,sha256
from gear_sonic.utils.g1_true23_arm_collision_projection import Native23CollisionArmProjector


def run(args):
    assert mujoco.__version__=="3.2.3";args.output.mkdir(parents=True,exist_ok=False)
    model,contract,_,original,timeline=load_inputs(args.bundle,args.clip)
    path=args.references/args.clip/"reference.npz"
    with np.load(path,allow_pickle=False) as a:motion={k:a[k].copy() for k in a.files}
    projector=Native23CollisionArmProjector(model)
    worst={};all_pairs=set();started=time.perf_counter()
    for frame in range(len(motion["joint_pos"])):
        pose=np.r_[motion["body_pos_w"][frame,0],motion["body_quat_w"][frame,0],motion["joint_pos"][frame]]
        projector.forward(pose,pose[projector.qidx])
        for contact in projector.self_contacts():
            key=tuple(sorted((contact["geom1"],contact["geom2"])))
            all_pairs.add(key)
            if key not in worst or contact["distance_m"]<worst[key]["distance_m"]:
                worst[key]=dict(**contact,frame=frame,source_seconds=(frame-361)*.02,
                                contype=[int(model.geom_contype[g]) for g in key],
                                conaffinity=[int(model.geom_conaffinity[g]) for g in key],
                                geom_type=[int(model.geom_type[g]) for g in key],
                                actual_mj_forward_ncon=int(projector.data.ncon))
    ordered=sorted(worst.values(),key=lambda item:item["distance_m"])
    frames=[]
    for item in ordered:
        if all(abs(item["frame"]-old)>30 for old in frames):frames.append(item["frame"])
        if len(frames)>=args.frames:break
    records=[];poses=[]
    maximum_step=.8*np.asarray(contract["native_velocity"])[13:]*.02
    for frame in frames:
        pose=np.r_[motion["body_pos_w"][frame,0],motion["body_quat_w"][frame,0],motion["joint_pos"][frame]]
        previous=np.r_[motion["body_pos_w"][frame-1,0],motion["body_quat_w"][frame-1,0],motion["joint_pos"][frame-1]]
        goals=pose[:3]+(original["source_task_position_w"][frame,:2]-original["source_qpos29"][frame,:3])
        projector.forward(pose,pose[projector.qidx]);before=projector.self_contacts();hands,_=projector.hands()
        before_error=np.linalg.norm(hands-goals,axis=1)
        torso=model.body("torso_link").id
        head=projector.data.xpos[torso]+projector.data.xmat[torso].reshape(3,3)@np.array([0.,0.,.35])
        head_goal=pose[:3]+original["source_task_position_w"][frame,2]-original["source_qpos29"][frame,:3]
        original_relative_head_error=float(np.linalg.norm(head-head_goal))
        derivative=[]
        for item in before[:4]:
            a,b=item["geom1"],item["geom2"]
            projector.forward(pose,pose[projector.qidx]);distance,analytic=projector.distance_gradient(a,b)
            finite=np.empty(10)
            for axis in range(10):
                q=pose[projector.qidx].copy();q[axis]+=1e-6;projector.forward(pose,q)
                plus=projector.distance_gradient(a,b)[0]
                q[axis]-=2e-6;projector.forward(pose,q);minus=projector.distance_gradient(a,b)[0]
                finite[axis]=(plus-minus)/2e-6
            derivative.append(dict(geom1=a,geom2=b,signed_distance=distance,
                                   analytic_vs_centered_max_abs_error=float(np.max(np.abs(analytic-finite))),
                                   gradient_norm=float(np.linalg.norm(finite))))
        bounded,report=projector.solve(pose,goals,previous_pose=previous,maximum_step=maximum_step)
        static,static_report=projector.solve(pose,goals)
        record=dict(frame=frame,source_seconds=(frame-361)*.02,before_contacts=before,
                    before_original_relative_hand_errors_m=before_error.tolist(),signed_distance_derivative_checks=derivative,
                    speed_bounded=report,static_geometric_only=static_report,
                    root_waist_legs_bit_exact=bool(np.array_equal(bounded[:20],pose[:20])),
                    head_proxy_change_m=0.,original_relative_head_error_m=original_relative_head_error,
                    physical_rollout_performed=False)
        records.append(record);poses.append((pose,bounded,static))
        print(json.dumps(dict(frame=frame,bounded=report,static=static_report)),flush=True)
    result=dict(clip=args.clip,mujoco=mujoco.__version__,reference_sha256=sha256(path),all_frames_contact_audited=len(motion["joint_pos"]),
                physical_masks_and_exclusions_unchanged=True,contacts_from_actual_mj_fwdPosition=True,
                worst_active_pairs=ordered,probes=records,hand_error_gate_m=.15,step_limit_native_speed_fraction=.8,
                dt=.02,full_retarget_performed=False,hardware_authorized=False,elapsed_s=time.perf_counter()-started)
    np.savez_compressed(args.output/"probe_poses.npz",frames=frames,poses=np.asarray(poses))
    (args.output/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    for name,source in (("probe_snapshot.py",Path(__file__)),("projector_snapshot.py",Path(__file__).resolve().parents[1]/"utils/g1_true23_arm_collision_projection.py")):
        (args.output/name).write_bytes(source.read_bytes())


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--bundle",type=Path,required=True);p.add_argument("--references",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--clip",default="pico");p.add_argument("--frames",type=int,default=4)
    run(p.parse_args())
