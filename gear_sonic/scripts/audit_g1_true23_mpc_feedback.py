"""Bounded numerical audit of saved MPC gains and true native contact dynamics."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mpc_student import load_inputs, sha256


def tangent(model,nominal,qpos,qvel):
    dx=np.empty(58)
    mujoco.mj_differentiatePos(model,dx[:29],1.,nominal[:30],qpos)
    dx[29:]=qvel-nominal[30:]
    return dx


def run(args):
    assert mujoco.__version__=="3.2.3"
    model,contract,*_=load_inputs(args.bundle)
    with np.load(args.plan/"trace.npz",allow_pickle=False) as a:
        plan={k:a[k].copy() for k in ("qpos","qvel","target","feedback_gain")}
    K=plan["feedback_gain"]
    sigma=np.r_[np.zeros(3),np.full(3,.001),np.full(23,.002),np.full(29,.01)]
    noise_rms=np.sqrt(np.sum((K*sigma)**2,axis=2))
    result=dict(mujoco=mujoco.__version__,plan_trace_sha256=sha256(args.plan/"trace.npz"),
                control_samples=len(K),abs_gain_p50_p95_p99_max=np.percentile(np.abs(K),[50,95,99,100]).tolist(),
                predicted_raw_target_noise_rms_p50_p95_p99_max=np.percentile(noise_rms,[50,95,99,100]).tolist(),
                gain_step_change_frobenius_p50_p95_p99_max=np.percentile(np.linalg.norm(np.diff(K,axis=0),axis=(1,2)),[50,95,99,100]).tolist(),
                finite_probe_state_scales=np.r_[np.full(29,.002),np.full(29,.01)].tolist(),
                finite_probe_direction_normalized_l2=True,microscopic_amplitude=.001,
                note="centered state directions have order1e-7 pose/order1e-6 velocity microscopic components; finite responses include actual contact transitions, not an explicit rigid-impact saltation matrix",
                probes=[])
    kp,kd,effort=[np.asarray(contract[k]) for k in ("kp","kd","native_effort")]
    limits=np.asarray(contract["joint_limits"])
    data=mujoco.MjData(model)
    def forward(qpos,qvel,target):
        mujoco.mj_resetData(model,data)
        data.qpos[:]=qpos;data.qvel[:]=qvel
        mujoco.mj_forward(model,data)
        active=[]
        for _ in range(10):
            data.ctrl[:]=np.clip(kp*(np.clip(target,limits[:,0],limits[:,1])-data.qpos[7:])-kd*data.qvel[6:],-effort,effort)
            mujoco.mj_step(model,data)
            active.append(tuple(sorted((min(c.geom1,c.geom2),max(c.geom1,c.geom2)) for c in data.contact if c.dist<0)))
        return np.r_[data.qpos,data.qvel],active
    rng=np.random.default_rng(20260910)
    # RMS-normalized errors use 2mm/2mrad poses and .01 rad/s or m/s velocities.
    physical_scale=np.r_[np.full(29,.002),np.full(29,.01)]
    for control in [50,249,300,350,500,750,873,950,1100]:
        q,v,u=plan["qpos"][control],plan["qvel"][control],plan["target"][control]
        nominal,contacts=forward(q,v,u)
        records=[]
        for _ in range(12):
            direction=rng.normal(size=58);direction/=np.linalg.norm(direction)
            outputs=[];modes=[]
            for amplitude in (.001,1.):
                delta=direction*physical_scale*amplitude
                perturbed=q.copy();mujoco.mj_integratePos(model,perturbed,delta[:29],1.)
                out,mode=forward(perturbed,v+delta[29:],u)
                outputs.append(tangent(model,nominal,out[:30],out[30:])/amplitude)
                modes.append(mode!=contacts)
            relative=float(np.linalg.norm((outputs[1]-outputs[0])/physical_scale)/max(1e-12,np.linalg.norm(outputs[1]/physical_scale)))
            delta=direction*physical_scale
            perturbed=q.copy();mujoco.mj_integratePos(model,perturbed,delta[:29],1.)
            correction=np.clip(K[control]@delta,-.1,.1)
            closed,_=forward(perturbed,v+delta[29:],u+correction)
            corrected=tangent(model,nominal,closed[:30],closed[30:])
            records.append(dict(finite_scale_derivative_relative_change=relative,contact_sequence_changed=modes,
                                raw_gain_correction_max=float(np.max(np.abs(K[control]@delta))),
                                open_scaled_error=float(np.linalg.norm(outputs[1]/physical_scale)),
                                gain_scaled_error=float(np.linalg.norm(corrected/physical_scale))))
        result["probes"].append(dict(control=control,noise_rms_max=float(np.max(noise_rms[control])),
                                     records=records))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    compact={k:v for k,v in result.items() if k!="probes"}
    compact["probes"]=[dict(control=p["control"],noise_rms_max=p["noise_rms_max"],
        median_derivative_change=float(np.median([r["finite_scale_derivative_relative_change"] for r in p["records"]])),
        physical_contact_changes=sum(r["contact_sequence_changed"][1] for r in p["records"]),
        median_open_error=float(np.median([r["open_scaled_error"] for r in p["records"]])),
        median_K_error=float(np.median([r["gain_scaled_error"] for r in p["records"]]))) for p in result["probes"]]
    print(json.dumps(compact),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--bundle",type=Path,required=True);p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);run(p.parse_args())
