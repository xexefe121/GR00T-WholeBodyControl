"""Collect only actual physically executed bounded-feedback teacher samples."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.scripts.evaluate_g1_true23_mjbatch_plan_replay import load_plan, state_difference
from gear_sonic.utils.g1_true23_mpc_student import GoalFeatures,load_inputs,physical_failure,sha256,tracking_metrics,OFFSETS


def run(args):
    assert mujoco.__version__=="3.2.3"
    args.output.mkdir(parents=True,exist_ok=False)
    model,contract,motion,original,timeline=load_inputs(args.bundle)
    source_request,source_report,plan,_=load_plan(args.plan,model,args.bundle/"walk002/native_original.npz",args.bundle/"contract.json")
    baseline=None
    if args.baseline_replay:
        report=json.loads((args.baseline_replay/"report.json").read_text())
        assert report["plan_replayed"] and report["failure"] is None and report["range_excess_max"]==0
        assert report["trace_sha256"]==sha256(args.baseline_replay/"trace.npz")
        with np.load(args.baseline_replay/"trace.npz",allow_pickle=False) as data: baseline={k:data[k].copy() for k in ("qpos","qvel","target")}
    count=min(len(plan["target"]),args.max_controls)
    assert np.all(plan["physics_substeps"][:count]==10)
    features=GoalFeatures(motion,original,contract)
    kp,kd,effort=[np.asarray(contract[k]) for k in ("kp","kd","native_effort")]
    limits=np.asarray(contract["joint_limits"])
    request=dict(kind="bounded_K_teacher_actual_physical_perturbations",clip="walk002",cases=args.cases,
                 controls=count,feedback_correction_clip_rad=.1,mujoco=mujoco.__version__,seed=args.seed,
                 source_plan=str(args.plan.resolve()),source_plan_trace_sha256=sha256(args.plan/"trace.npz"),
                 source_plan_report_sha256=sha256(args.plan/"report.json"),source_plan_mujoco=source_request["mujoco"],
                 received_goal_frames=38,goal_offsets=OFFSETS.tolist(),declared_buffer_seconds=.74,
                 original_source_timing_hz=50,physics_hz=500,features=1023,
                 samples="pre-action observed state plus actual previous applied target -> actual applied target",
                 initial_perturbations="cases1..8: joints±.005rad,rootxy±.003m,z±.001m,roll/pitch/yaw±.002rad,jointspeed±.015rad/s,rootvel±.01",
                 observation_perturbations="cases9..16: every50Hzobservation jointsN(0,.002rad),rootorientationN(0,.001rad),qvelN(0,.01); plant and lower-levelPD use truephysicalstate",
                 retained_failed_rollouts=True,source_frames_removed=0,physical_state_rewrites_after_initialization=0,
                 root_assistance_forces=0,hardware_authorized=False,
                 code_sha256=sha256(Path(__file__)),feature_code_sha256=sha256(Path(__file__).resolve().parents[1]/"utils/g1_true23_mpc_student.py"))
    (args.output/"request.json").write_text(json.dumps(request,indent=2)+"\n")
    (args.output/"collector_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (args.output/"student_snapshot.py").write_bytes((Path(__file__).resolve().parents[1]/"utils/g1_true23_mpc_student.py").read_bytes())
    summaries=[]; dataset={key:[] for key in ("features","target","reference_joint_pos","case_id","source_frame")}
    started=time.perf_counter()
    for case in range(args.cases):
        rng=np.random.default_rng(args.seed+case)
        data=mujoco.MjData(model);data.qpos[:]=plan["qpos"][0];data.qvel[:]=plan["qvel"][0]
        delta_q=np.zeros(29);delta_v=np.zeros(29)
        if 1<=case<=8:
            delta_q[:3]=rng.uniform(-1,1,3)*[.003,.003,.001]
            delta_q[3:6]=rng.uniform(-.002,.002,3);delta_q[6:]=rng.uniform(-.005,.005,23)
            delta_v[:6]=rng.uniform(-.01,.01,6);delta_v[6:]=rng.uniform(-.015,.015,23)
            mujoco.mj_integratePos(model,data.qpos,delta_q,1.)
            data.qvel[:]+=delta_v
        mujoco.mj_forward(model,data)
        previous=motion["joint_pos"][10].copy()
        trace={k:[] for k in ("qpos","qvel","observed_qpos","observed_qvel","target","source_frame","features",
                              "joint_error","root_error","feet_error","original_task_error","original_relative_task_error",
                              "physics_qpos","physics_qvel","physics_torque","range_excess","velocity_ratio","effort_ratio")}
        for key,value in (("qpos",data.qpos),("qvel",data.qvel),("physics_qpos",data.qpos),("physics_qvel",data.qvel)):trace[key].append(value.copy())
        failure=None
        for control in range(count):
            frame=int(plan["source_frame"][control])
            observed_q,observed_v=data.qpos.copy(),data.qvel.copy()
            if case>=9:
                noise=np.r_[np.zeros(3),rng.normal(0,.001,3),rng.normal(0,.002,23)]
                mujoco.mj_integratePos(model,observed_q,noise,1.)
                observed_v+=rng.normal(0,.01,29)
            x=features(observed_q,observed_v,previous,frame)
            error=state_difference(model,plan["planned_state"][control],observed_q,observed_v)
            correction=np.clip(plan["feedback_gain"][control]@error,-.1,.1)
            target=np.clip(plan["planned_target"][control]+correction,limits[:,0],limits[:,1])
            for substep in range(10):
                data.ctrl[:]=np.clip(kp*(target-data.qpos[7:])-kd*data.qvel[6:],-effort,effort)
                mujoco.mj_step(model,data)
                for key,value in (("physics_qpos",data.qpos),("physics_qvel",data.qvel),("physics_torque",data.ctrl)):trace[key].append(value.copy())
                failure,physical=physical_failure(model,data,contract)
                for key,value in physical.items():trace[key].append(value)
                if failure:
                    failure.update(control=control,substep=substep+1,time=float(data.time));break
            values=dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),observed_qpos=observed_q,observed_qvel=observed_v,
                        target=target,source_frame=frame,features=x,
                        **tracking_metrics(model,data,motion,original,contract,frame))
            for key,value in values.items():trace[key].append(value)
            previous=target
            if failure:break
        arrays={k:np.asarray(v) for k,v in trace.items()}
        complete=len(trace["physics_torque"])//10
        valid=complete==count and failure is None and max(trace["range_excess"])<=1e-8
        phase=next(p for p in timeline["phases"] if p["name"]=="source_motion")
        first=phase["control_start"] if complete>phase["control_start"] else 0
        stop=min(complete,phase["control_stop"]) if first else complete
        idx=slice(first,stop)
        summary=dict(case_id=case,kind="nominal" if case==0 else "initial_perturbation" if case<=8 else "observation_noise",
                     expert_eligible=valid,completed_full_controls=complete,requested_controls=count,
                     metric_phase="source_motion" if first else "prefix",metric_controls=stop-first,
                     failure=failure,range_excess_max=max(trace["range_excess"]),velocity_ratio_max=max(trace["velocity_ratio"]),
                     effort_ratio_max=max(trace["effort_ratio"]),initial_delta_q=delta_q.tolist(),initial_delta_v=delta_v.tolist(),
                     root_p95_m=float(np.percentile(np.linalg.norm(arrays["root_error"][idx],axis=1),95)),
                     feet_p95_m=np.percentile(arrays["feet_error"][idx],95,axis=0).tolist(),
                     original_hand_head_p95_m=np.percentile(arrays["original_task_error"][idx],95,axis=0).tolist())
        if case==0 and baseline is not None:
            summary["baseline_replay_max_abs_error"]={k:float(np.max(np.abs(arrays[k]-baseline[k][:len(arrays[k])]))) for k in ("qpos","qvel","target")}
        np.savez_compressed(args.output/f"case_{case:02d}.npz",**arrays)
        summary["trace_sha256"]=sha256(args.output/f"case_{case:02d}.npz")
        summaries.append(summary)
        if valid:
            dataset["features"].append(arrays["features"])
            dataset["target"].append(arrays["target"].astype(np.float32))
            dataset["reference_joint_pos"].append(motion["joint_pos"][arrays["source_frame"]].astype(np.float32))
            dataset["case_id"].append(np.full(count,case,dtype=np.int16));dataset["source_frame"].append(arrays["source_frame"])
        print(json.dumps(summary),flush=True)
        (args.output/"cases.json").write_text(json.dumps(summaries,indent=2)+"\n")
    if dataset["features"]:
        np.savez_compressed(args.output/"expert_samples.npz",**{k:np.concatenate(v) for k,v in dataset.items()})
    result=dict(eligible_cases=sum(s["expert_eligible"] for s in summaries),total_cases=len(summaries),
                eligible_actual_samples=sum(len(v) for v in dataset["features"]),all_failures_retained=True,
                full_source_or_generalization_qualified=False,elapsed_s=time.perf_counter()-started,
                received_goal_buffer_seconds=.74,case_summaries=summaries)
    (args.output/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="case_summaries"}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--baseline-replay",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cases",type=int,default=17);p.add_argument("--max-controls",type=int,default=500);p.add_argument("--seed",type=int,default=623)
    run(p.parse_args())
