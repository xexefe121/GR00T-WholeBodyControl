"""Bounded native23 factory locomotion reproduction; no robot commands."""
from pathlib import Path
import sys
import argparse
import json
import time
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from run_factory_mimic_sim import FIRMWARE,load_model
from gear_sonic.utils.g1_true23_factory_locomotion import FactoryHumanLoco12
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import assess,quiet
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--vx',type=float,default=0);ap.add_argument('--seconds',type=float,default=40)
ap.add_argument('--standing-phase',type=float,choices=[0,.25],default=0)
args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
model,c=load_model()
for k in ('joint_limits','native_effort','native_velocity'):c[k]=np.asarray(c[k])
policy=FactoryHumanLoco12(FIRMWARE,c['joint_limits'])
policy.standing_phase=args.standing_phase
data=mujoco.MjData(model)
with np.load(FIRMWARE.parent/'causal_dynamics_v1/bank/walk003.npz') as z:
    data.qpos[:]=z['states'][10,:30];data.qvel[:]=z['states'][10,30:]
mujoco.mj_forward(model,data)
trace={k:[] for k in ('qpos','qvel','target','physics_qpos','physics_qvel','inference_ms','command')}
failure=None;peak_speed=peak_effort=0.;stop_goal=None
for control in range(round(args.seconds*50)):
    t=control*.02
    velocity=np.array([args.vx if 3<=t<10 else 0,0,0])
    if t>=10 and stop_goal is None:stop_goal=data.qpos.copy()
    rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,data.qpos[3:7])
    begin=time.perf_counter_ns()
    target=policy.step(data.qpos,data.qvel,-rotation.reshape(3,3)[2],velocity)
    trace['inference_ms'].append((time.perf_counter_ns()-begin)/1e6)
    trace['target'].append(target.copy());trace['command'].append(velocity.copy())
    for substep in range(10):
        torque=policy.kp*(target-data.qpos[7:])-policy.kd*data.qvel[6:]
        band=np.minimum(.1,np.diff(c['joint_limits'],axis=1).ravel()*.2)
        penetration=data.qpos[7:]-np.clip(data.qpos[7:],c['joint_limits'][:,0]+band,c['joint_limits'][:,1]-band)
        outward=np.where(penetration*data.qvel[6:]>0,data.qvel[6:],0)
        data.ctrl[:]=np.clip(torque-100*penetration-2*outward,-c['native_effort'],c['native_effort'])
        mujoco.mj_step(model,data)
        trace['physics_qpos'].append(data.qpos.copy());trace['physics_qvel'].append(data.qvel.copy())
        reasons,values=assess(data,c,(control*10+substep+1)*.002)
        peak_speed=max(peak_speed,values['speed_ratio']);peak_effort=max(peak_effort,values['effort_ratio'])
        if reasons:failure={'time':float(data.time),'reasons':reasons,**values};break
    trace['qpos'].append(data.qpos.copy());trace['qvel'].append(data.qvel.copy())
    if failure:break
trace={k:np.asarray(v) for k,v in trace.items()};np.savez_compressed(args.output/'trace.npz',**trace)
report={'controller':'factory_human_loco_native12_plus_native_upper','requested_seconds':args.seconds,
    'standing_phase':args.standing_phase,
    'physical_complete':failure is None,'seconds':float(data.time),'failure':failure,
    'peak_speed_ratio':peak_speed,'peak_effort_ratio':peak_effort,
    'terminal_qpos':data.qpos.tolist(),'inference_ms_p50_p95_max':np.quantile(trace['inference_ms'],[.5,.95,1]).tolist(),
    'quiet':quiet(trace['physics_qpos'],trace['physics_qvel'],stop_goal) if stop_goal is not None else None,
    'recorded_full_body_tracking_tested':False,'simulation_ready':False,'hardware_commands':False}
(args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
