"""Bounded short collision episode; emit no promoted full reference."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from gear_sonic.utils.g1_true23_mpc_student import load_inputs,sha256
from gear_sonic.utils.g1_true23_arm_collision_preview import TwoFrameArmProjection


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    model,contract,_,original,_=load_inputs(args.bundle,"pico")
    path=args.references/"pico/reference.npz"
    with np.load(path,allow_pickle=False) as a:motion={k:a[k].copy() for k in a.files}
    def pose(frame):return np.r_[motion["body_pos_w"][frame,0],motion["body_quat_w"][frame,0],motion["joint_pos"][frame]]
    projection=TwoFrameArmProjection(model,.8*np.asarray(contract["native_velocity"])[13:]*.02,
                                    max_iterations=args.iterations)
    previous=pose(args.start-1);seed=None;records=[];outputs=[];expected=[];started=time.perf_counter()
    if args.resume_poses:
        prior=json.loads((args.resume_poses.parent/"request.json").read_text())
        assert prior["frame"]+1==args.start and prior["reference_sha256"]==sha256(path)
        with np.load(args.resume_poses,allow_pickle=False) as a:resume=a["proposed"].copy()
        assert resume.shape==(2,30) and np.isfinite(resume).all()
        previous=resume[0]
        next_delta=motion["joint_pos"][args.start+1,13:]-motion["joint_pos"][args.start,13:]
        seed=np.array([resume[1,20:],resume[1,20:]+next_delta])
    request=dict(reference_sha256=sha256(path),start_frame=args.start,stop_frame=args.stop,source_clock_hz=50,
                 pose_preview_frames=1,pose_preview_seconds=.02,whole_source_export=False,root_waist_legs_fixed=True,
                 native_speed_fraction=.8,relative_hand_gate_m=.15,solver_hand_margin_m=.149,no_physical_rollout=True,
                 collision_scope="optimize arm-involving contacts only; fixed-body contacts remain reported and unresolved",
                 optimizer_max_iterations=args.iterations,
                 resume_poses_sha256=sha256(args.resume_poses) if args.resume_poses else None,
                 evaluator_sha256=sha256(Path(__file__)))
    (args.output/"request.json").write_text(json.dumps(request,indent=2)+"\n")
    (args.output/"runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (args.output/"projector_snapshot.py").write_bytes((Path(__file__).resolve().parents[1]/"utils/g1_true23_arm_collision_preview.py").read_bytes())
    for frame in range(args.start,args.stop):
        poses=np.array([pose(frame),pose(frame+1)])
        goals=np.array([poses[i,:3]+original["source_task_position_w"][f,:2]-original["source_qpos29"][f,:3] for i,f in enumerate((frame,frame+1))])
        proposed,report=projection.solve(poses,goals,previous,seed=seed)
        if proposed is not None:
            head_errors=[]
            for i,part in enumerate(projection.parts):
                part.forward(poses[i],proposed[i,20:]);torso=model.body("torso_link").id
                head=part.data.xpos[torso]+part.data.xmat[torso].reshape(3,3)@np.array([0.,0.,.35])
                desired=original["source_task_position_w"][frame+i,2]-original["source_qpos29"][frame+i,:3]
                head_errors.append(float(np.linalg.norm(head-proposed[i,:3]-desired)))
            report["original_relative_head_errors_m"]=head_errors
            report["passed"]&=max(head_errors)<=.15
        report.update(frame=frame,source_seconds=(frame-361)*.02,elapsed_s=time.perf_counter()-started)
        records.append(report)
        if proposed is not None:outputs.append(proposed[0]);expected.append(proposed[1])
        if not report["passed"]:
            print(json.dumps(report),flush=True);break
        previous=proposed[0].copy()
        next_delta=motion["joint_pos"][frame+2,13:]-motion["joint_pos"][frame+1,13:]
        seed=np.array([proposed[1,20:],proposed[1,20:]+next_delta])
        if (frame-args.start)%10==0:
            print(json.dumps(dict(frame=frame,handmax=max(max(m["hand_errors_m"]) for m in report["frames"]),
                                  joint_delta=report["joint_delta_max_rad"],elapsed_s=report["elapsed_s"])),flush=True)
    complete=len(records)==args.stop-args.start and all(r["passed"] for r in records)
    np.savez_compressed(args.output/"episode.npz",frames=np.arange(args.start,args.start+len(outputs)),
                        proposed_current=outputs,proposed_next=expected)
    result=dict(completed_episode=complete,requested_frames=args.stop-args.start,attempted_frames=len(records),
                pass_scope="arm-only geometry plus originalhand/head and adjacent speed; fixed-body collisions unresolved",
                all_self_collision_free=all(r.get("all_self_collision_free",False) for r in records),
                accepted_frames=sum(r["passed"] for r in records),records=records,elapsed_s=time.perf_counter()-started,
                full_reference_promoted=False,hardware_authorized=False)
    (args.output/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="records"}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--bundle",type=Path,required=True);p.add_argument("--references",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--start",type=int,default=3754);p.add_argument("--stop",type=int,default=3866)
    p.add_argument("--iterations",type=int,default=60);p.add_argument("--resume-poses",type=Path)
    run(p.parse_args())
