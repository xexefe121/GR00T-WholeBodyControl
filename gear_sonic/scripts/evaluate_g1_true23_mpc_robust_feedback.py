"""Physical perturbation test of capped feedback and measured ankle repulsion."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mpc_student import load_inputs,physical_failure,tracking_metrics,sha256
from gear_sonic.utils.g1_true23_mpc_robust_feedback import NoiseCappedFeedback
from gear_sonic.utils.g1_true23_bfmzero_ankle_barrier import AnkleRollRepulsion


def run(args):
    assert mujoco.__version__=="3.2.3"
    args.output.mkdir(parents=True,exist_ok=False)
    model,contract,motion,original,timeline=load_inputs(args.bundle)
    receipt=json.loads((args.plan/"report.json").read_text())
    assert receipt["trace_sha256"]==sha256(args.plan/"trace.npz")
    with np.load(args.plan/"trace.npz",allow_pickle=False) as a:plan={k:a[k].copy() for k in a.files}
    count=min(args.controls,len(plan["target"]))
    assert np.all(plan["physics_substeps"][:count]==10)
    kp,kd,effort=[np.asarray(contract[k]) for k in ("kp","kd","native_effort")]
    limits=np.asarray(contract["joint_limits"])
    feedback=NoiseCappedFeedback(plan["feedback_gain"],noise_gain_cap=args.noise_gain_cap)
    barrier=None if args.no_barrier else AnkleRollRepulsion([model.joint(i).name for i in range(1,24)],limits)
    phase=next(p for p in timeline["phases"] if p["name"]=="source_motion")
    request=dict(kind="native23_saved_plan_noise_capped_feedback_physical_probe",mujoco=mujoco.__version__,
                 plan_trace_sha256=receipt["trace_sha256"],plan=str(args.plan.resolve()),
                 controls=count,cases=args.cases,feedback=feedback.report(),ankle_repulsion=None if barrier is None else barrier.report(),
                 native_timestep_s=.002,goal_control_hz=50,native_effort_preserved=True,
                 contact_model_unchanged=True,physical_state_rewrites_after_initialization=0,root_assistance_forces=0,
                 fixed_saved_plan_not_online_replanning=True,hardware_authorized=False,
                 code_sha256=sha256(Path(__file__)))
    (args.output/"request.json").write_text(json.dumps(request,indent=2)+"\n")
    for name,path in (("evaluator_snapshot.py",Path(__file__)),("feedback_snapshot.py",Path(__file__).resolve().parents[1]/"utils/g1_true23_mpc_robust_feedback.py")):
        (args.output/name).write_bytes(path.read_bytes())
    results=[];started=time.perf_counter()
    for case in args.cases:
        rng=np.random.default_rng(623+case)
        data=mujoco.MjData(model);data.qpos[:]=plan["qpos"][0];data.qvel[:]=plan["qvel"][0]
        delta_q=np.zeros(29);delta_v=np.zeros(29)
        if 1<=case<=8:
            delta_q[:3]=rng.uniform(-1,1,3)*[.003,.003,.001]
            delta_q[3:6]=rng.uniform(-.002,.002,3);delta_q[6:]=rng.uniform(-.005,.005,23)
            delta_v[:6]=rng.uniform(-.01,.01,6);delta_v[6:]=rng.uniform(-.015,.015,23)
            mujoco.mj_integratePos(model,data.qpos,delta_q,1.);data.qvel[:]+=delta_v
        mujoco.mj_forward(model,data)
        trace={key:[] for key in ("qpos","qvel","target","source_frame","correction","raw_correction","feedback_ms",
            "joint_error","root_error","feet_error","original_task_error","original_relative_task_error",
            "physics_qpos","physics_qvel","physics_torque","barrier_torque","range_excess","velocity_ratio","effort_ratio")}
        for key,value in (("qpos",data.qpos),("qvel",data.qvel),("physics_qpos",data.qpos),("physics_qvel",data.qvel)):
            trace[key].append(value.copy())
        failure=None
        for control in range(count):
            frame=int(plan["source_frame"][control]);q,v=data.qpos.copy(),data.qvel.copy()
            if case>=9:
                noise=np.r_[np.zeros(3),rng.normal(0,.001,3),rng.normal(0,.002,23)]
                mujoco.mj_integratePos(model,q,noise,1.);v+=rng.normal(0,.01,29)
            tick=time.perf_counter();error=np.empty(58)
            mujoco.mj_differentiatePos(model,error[:29],1.,plan["planned_state"][control,:30],q)
            error[29:]=v-plan["planned_state"][control,30:]
            correction=feedback.correction(control,error)
            target=np.clip(plan["planned_target"][control]+correction,limits[:,0],limits[:,1])
            feedback_ms=1000*(time.perf_counter()-tick)
            for substep in range(10):
                barrier_torque=np.zeros(23) if barrier is None else barrier.torque(data.qpos[7:],data.qvel[6:])
                data.ctrl[:]=np.clip(kp*(target-data.qpos[7:])-kd*data.qvel[6:]+barrier_torque,-effort,effort)
                mujoco.mj_step(model,data)
                for key,value in (("physics_qpos",data.qpos),("physics_qvel",data.qvel),("physics_torque",data.ctrl),("barrier_torque",barrier_torque)):
                    trace[key].append(value.copy())
                failure,physical=physical_failure(model,data,contract)
                for key,value in physical.items():trace[key].append(value)
                if failure:
                    failure.update(control=control,substep=substep+1,time=float(data.time));break
            values=dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),target=target,source_frame=frame,
                        correction=correction,raw_correction=plan["feedback_gain"][control]@error,feedback_ms=feedback_ms,
                        **tracking_metrics(model,data,motion,original,contract,frame))
            for key,value in values.items():trace[key].append(value)
            if failure:break
        arrays={k:np.asarray(v) for k,v in trace.items()};np.savez_compressed(args.output/f"case_{case:02d}.npz",**arrays)
        complete=len(trace["physics_torque"])//10
        first=phase["control_start"] if complete>phase["control_start"] else 0
        stop=min(complete,phase["control_stop"]) if first else complete;idx=slice(first,stop)
        result=dict(case_id=case,completed_full_controls=complete,partial_substeps=len(trace["physics_torque"])%10,
                    failure=failure,range_excess_max=max(trace["range_excess"]),velocity_ratio_max=max(trace["velocity_ratio"]),
                    effort_ratio_max=max(trace["effort_ratio"]),strict_success=complete==count and failure is None and max(trace["range_excess"])<=1e-8,
                    metric_phase="source_motion" if first else "prefix",metric_controls=stop-first,
                    root_p95_m=float(np.percentile(np.linalg.norm(arrays["root_error"][idx],axis=1),95)),
                    leg_rmse_rad=float(np.sqrt(np.mean(arrays["joint_error"][idx,:12]**2))),
                    arm_rmse_rad=float(np.sqrt(np.mean(arrays["joint_error"][idx,13:]**2))),
                    feet_p95_m=np.percentile(arrays["feet_error"][idx],95,axis=0).tolist(),
                    original_hand_head_p95_m=np.percentile(arrays["original_task_error"][idx],95,axis=0).tolist(),
                    feedback_p95_ms=float(np.percentile(arrays["feedback_ms"],95)),
                    trace_sha256=sha256(args.output/f"case_{case:02d}.npz"))
        results.append(result);print(json.dumps(result),flush=True)
        (args.output/"cases.json").write_text(json.dumps(results,indent=2)+"\n")
    report=dict(strict_successes=sum(r["strict_success"] for r in results),cases=results,elapsed_s=time.perf_counter()-started,
                hardware_authorized=False,request_sha256=sha256(args.output/"request.json"))
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--bundle",type=Path,required=True);p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--controls",type=int,default=1417)
    p.add_argument("--cases",type=lambda x:[int(v) for v in x.split(',')],default=[0,1,9])
    p.add_argument("--noise-gain-cap",type=float,default=.04);p.add_argument("--no-barrier",action="store_true")
    run(p.parse_args())
