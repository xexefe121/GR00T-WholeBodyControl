"""Bounded, fully physical native23 inverse-dynamics QP replay."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils.g1_true23_wbc_qp import Native23WBCQP, QPConfig, digest, load_bundle


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    model,contract,motion,original,timeline,manifest=load_bundle(args.bundle,args.clip)
    controller=Native23WBCQP(model,contract,motion,original,QPConfig())
    phase=next(p for p in timeline["phases"] if p["name"]=="source_motion")
    requested=(round(args.seconds*50) if args.probe=="standing" else phase["control_start"]+round(args.seconds*50))
    requested=min(requested,len(motion["joint_pos"])-11)
    source=Path(__file__).resolve().parents[1]/"utils/g1_true23_wbc_qp.py"
    request=dict(kind="native23_whole_body_inverse_dynamics_qp_prototype",clip=args.clip,probe=args.probe,
                 requested_controls=requested,source_phase=phase,mujoco=mujoco.__version__,
                 source_timing_hz=50,physics_and_controller_hz=500,preview_seconds=0,
                 reference="immutable native_original.npz plus original29 hand/head tasks at unchanged source frames",
                 contact_contract="Only measured penetrating foot-ground contacts provide force variables; unilateral friction pyramid; source stance evidence controls soft contact acceleration weight.",
                 dynamics="M*qacc + bias - passive = joint_torque + sum(J_contact.T*force)",
                 actuation="direct joint torque under native effort caps, no BFM PD",
                 config=asdict(controller.config),manifest=manifest,
                 source_hashes={str(p):digest(p) for p in (Path(__file__),source,args.bundle/args.clip/"native_original.npz",args.bundle/args.clip/"original29.npz")},
                 root_assistance_forces=0,physical_state_rewrites_after_initialization=0,hardware_authorized=False)
    (args.output/"request.json").write_text(json.dumps(request,indent=2)+"\n")
    (args.output/"controller_snapshot.py").write_bytes(source.read_bytes())
    (args.output/"evaluator_snapshot.py").write_bytes(Path(__file__).read_bytes())
    data=mujoco.MjData(model)
    data.qpos[:]=np.r_[motion["body_pos_w"][10,0],motion["body_quat_w"][10,0],motion["joint_pos"][10]]
    root_rotation=Rotation.from_quat(data.qpos[[4,5,6,3]])
    data.qvel[:]=np.r_[motion["body_lin_vel_w"][10,0],root_rotation.inv().apply(motion["body_ang_vel_w"][10,0]),motion["joint_vel"][10]]
    mujoco.mj_forward(model,data)
    trace={k:[] for k in ("physics_qpos","physics_qvel","physics_torque","physics_qp_qacc","physics_actual_qacc",
                            "qpos","qvel","source_frame","joint_error","root_error","feet_error","original_task_error",
                            "range_excess","velocity_ratio","effort_ratio","contact_count","qp_total_ms")}
    for key,value in (("physics_qpos",data.qpos),("physics_qvel",data.qvel),("qpos",data.qpos),("qvel",data.qvel)):
        trace[key].append(value.copy())
    failure=None; records=[]; completed=0; started=time.perf_counter()
    for control in range(requested):
        frame=control+11
        for substep in range(10):
            mujoco.mj_forward(model,data)
            torque,record,predicted=controller.solve(data,frame)
            record.update(control=control,substep=substep,time=float(data.time),source_frame=frame)
            records.append(record)
            if torque is None:
                failure=dict(kind="qp_unsolved",control=control,substep=substep,time=float(data.time),status=record["status"],contacts=record["contacts"],acceleration_bound_conflict=record["acceleration_bound_conflict"])
                break
            if record["dynamics_equality_max"]>1e-4 or record["inequality_excess_max"]>1e-4:
                failure=dict(kind="qp_constraint_residual",control=control,substep=substep,record=record)
                break
            # Only joint motors receive the optimized torque. Predicted contact
            # forces and floating-base accelerations are never applied directly.
            data.ctrl[:]=np.clip(torque,-controller.effort,controller.effort)
            mujoco.mj_step(model,data)
            values=dict(physics_qpos=data.qpos,physics_qvel=data.qvel,physics_torque=data.ctrl,
                        physics_qp_qacc=predicted["qacc"],physics_actual_qacc=data.qacc)
            for key,value in values.items(): trace[key].append(value.copy())
            range_excess=float(max(0,np.max(controller.lo-data.qpos[7:]),np.max(data.qpos[7:]-controller.hi)))
            speed=float(np.max(np.abs(data.qvel[6:])/controller.velocity))
            effort=float(np.max(np.abs(data.qfrc_actuator[6:])/controller.effort))
            for key,value in (("range_excess",range_excess),("velocity_ratio",speed),("effort_ratio",effort),
                              ("contact_count",record["contacts"]),("qp_total_ms",record["total_ms"])): trace[key].append(value)
            tilt=float(np.arccos(np.clip(1-2*np.sum(data.qpos[4:6]**2),-1,1)))
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.qpos[2]<.25 or tilt>1.2:
                failure=dict(kind="fall_or_nonfinite",control=control,substep=substep,time=float(data.time),height=float(data.qpos[2]),tilt=tilt)
            elif range_excess>.01 or speed>1 or effort>1+1e-8:
                failure=dict(kind="physical_limit",control=control,substep=substep,time=float(data.time),range_excess=range_excess,velocity_ratio=speed,effort_ratio=effort)
            if failure: break
        mujoco.mj_forward(model,data)
        for key,value in (("qpos",data.qpos),("qvel",data.qvel)): trace[key].append(value.copy())
        trace["source_frame"].append(frame)
        trace["joint_error"].append(data.qpos[7:]-motion["joint_pos"][frame])
        trace["root_error"].append(data.qpos[:3]-motion["body_pos_w"][frame,0])
        trace["feet_error"].append(np.linalg.norm(data.xpos[controller.foot_ids]-motion["body_pos_w"][frame,np.asarray(controller.foot_ids)-1],axis=1))
        taskpos=np.array([controller.point(data,body,local)[0] for body,local in
                          [(controller.hand_ids[0],(.264,-.025,0)),(controller.hand_ids[1],(.264,.025,0)),(controller.torso_id,(0,0,.35))]])
        trace["original_task_error"].append(np.linalg.norm(taskpos-original["source_task_position_w"][frame],axis=1))
        completed=control+1
        if completed%50==0 or failure:
            print(json.dumps(dict(controls=completed,requested=requested,simulated_s=float(data.time),height=float(data.qpos[2]),failure=failure)),flush=True)
        if failure: break
        if time.perf_counter()-started>args.wall_seconds:
            failure=dict(kind="bounded_wall_timeout",control=control,time=float(data.time))
            break
    arrays={k:np.asarray(v) for k,v in trace.items()}
    np.savez_compressed(args.output/"trace.npz",**arrays)
    (args.output/"qp_records.json").write_text(json.dumps(records,indent=2)+"\n")
    metric_start=phase["control_start"] if completed>phase["control_start"] else 0
    idx=slice(metric_start,completed)
    metrics=dict(phase="source_motion" if metric_start else "standing_prefix",controls=completed-metric_start,
                 joint_rmse=float(np.sqrt(np.mean(arrays["joint_error"][idx]**2))),
                 root_p95_m=float(np.percentile(np.linalg.norm(arrays["root_error"][idx],axis=1),95)),
                 feet_p95_m=np.percentile(arrays["feet_error"][idx],95,axis=0).tolist(),
                 original_hand_head_p95_m=np.percentile(arrays["original_task_error"][idx],95,axis=0).tolist())
    solved=[r for r in records if r["solved"]]
    result=dict(probe_completed=completed==requested and failure is None,completed_controls=completed,
                requested_controls=requested,source_controls_completed=max(0,completed-phase["control_start"]),
                failure=failure,metrics=metrics,simulated_seconds=float(data.time),
                physical_substeps=len(trace["physics_torque"]),elapsed_wall_seconds=time.perf_counter()-started,
                qp_total_ms_p50_p95_max=np.percentile([r["total_ms"] for r in records],[50,95,100]).tolist(),
                qp_deadline_misses_2ms=sum(r["total_ms"]>2 for r in records),
                qp_dynamics_residual_max=max((r["dynamics_equality_max"] for r in solved),default=None),
                qp_inequality_residual_max=max((r["inequality_excess_max"] for r in solved),default=None),
                range_excess_max=max(trace["range_excess"],default=None),velocity_ratio_max=max(trace["velocity_ratio"],default=None),
                effort_ratio_max=max(trace["effort_ratio"],default=None),physical_state_rewrites_after_initialization=0,
                root_assistance_forces=0,source_frames_removed=0,source_timing_warped=False,
                mujoco=mujoco.__version__,full_body_tracking_qualified=False,hardware_authorized=False,
                request_sha256=digest(args.output/"request.json"),trace_sha256=digest(args.output/"trace.npz"))
    (args.output/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--clip",default="walk002")
    p.add_argument("--probe",choices=("standing","source-prefix"),default="standing")
    p.add_argument("--seconds",type=float,default=1.)
    p.add_argument("--wall-seconds",type=float,default=600.)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
