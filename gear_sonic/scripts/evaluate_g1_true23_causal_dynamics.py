"""Complete native MuJoCo3.2.3 attempts through the received-only actor path."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import mujoco
from gear_sonic.utils.g1_true23_causal_controller import CausalController
from gear_sonic.utils.g1_true23_bfmzero_stream import Packet

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('E:/codex-artifacts' if sys.platform=='win32' else '/mnt/e/codex-artifacts')
NEW=BASE/'sonic23_teleop_resume_20260911';OLD=BASE/'sonic23_teleop_six_hour_20260910'
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'


def archive(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def load_case(clip):
    if mujoco.__version__!='3.2.3':raise ValueError('native referee must be MuJoCo3.2.3')
    model=mujoco.MjModel.from_xml_path(str(BUNDLE/'native_prepared.xml'))
    with np.load(BUNDLE/'prepared_model_arrays.npz') as z:
        for k in z.files:getattr(model,k)[:]=z[k]
    mujoco.mj_setConst(model,mujoco.MjData(model))
    c=json.loads((BUNDLE/'contract.json').read_text())
    for k in ('default_q','kp','kd','native_effort','training_effort','native_velocity','joint_limits'):c[k]=np.asarray(c[k])
    motion=archive(OLD/'mjbatch_intent_floor_inputs_v1'/clip/'reference.npz')
    original=archive(BUNDLE/clip/'original29.npz')
    timeline=json.loads((BUNDLE/clip/'timeline.json').read_text())
    return model,c,motion,original,timeline


def assess(data,c,expected):
    q,v=data.qpos,data.qvel;lim=c['joint_limits']
    finite=np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(data.qfrc_actuator).all()
    if not finite:return ['nonfinite'],dict(speed_ratio=float('inf'))
    speed=float(np.max(np.abs(v[6:])/c['native_velocity']))
    excess=float(np.maximum(0,np.maximum(lim[:,0]-q[7:],q[7:]-lim[:,1])).max())
    effort=float(np.max(np.abs(data.qfrc_actuator[6:])/c['native_effort']))
    tilt=float(np.arccos(np.clip(1-2*np.sum(q[4:6]**2),-1,1)))
    reasons=[]
    if speed>1:reasons.append('native_joint_speed')
    if excess>1e-6:reasons.append('native_joint_bound')
    if effort>1+1e-9:reasons.append('native_effort')
    if q[2]<.25 or tilt>1.2:reasons.append('fall')
    if np.any(data.warning.number):reasons.append('engine_warning')
    if abs(np.linalg.norm(q[3:7])-1)>1e-10:reasons.append('quaternion')
    if abs(data.time-expected)>1e-10:reasons.append('clock')
    return reasons,dict(speed_ratio=speed,range_excess=excess,effort_ratio=effort,tilt=tilt)


def yaw(q):
    w,x,y,z=q.T;return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))


def quiet(q,v,goal,seconds=3):
    if not np.isfinite(seconds) or seconds<=0:raise ValueError('quiet window must be positive and finite')
    samples=int(round(seconds*500))
    if samples<1:raise ValueError('quiet window must contain at least one500Hz sample')
    if len(q)<samples:return dict(passed=False,reason=f'less_than_{seconds:g}s')
    q,v=q[-samples:],v[-samples:];dy=yaw(q[:,3:7])-yaw(goal[3:7][None])
    values=dict(xy_p95=float(np.percentile(np.linalg.norm(q[:,:2]-goal[:2],axis=-1),95)),
        yaw_p95_deg=float(np.percentile(np.abs(np.rad2deg(np.arctan2(np.sin(dy),np.cos(dy)))),95)),
        root_speed_p95=float(np.percentile(np.linalg.norm(v[:,:3],axis=-1),95)),
        joint_speed_p95=float(np.percentile(np.abs(v[:,6:]).max(-1),95)),joint_speed_max=float(np.abs(v[:,6:]).max()),
        tilt_max=float(np.arccos(np.clip(1-2*np.sum(q[:,4:6]**2,axis=-1),-1,1)).max()))
    limits=dict(xy_p95=.05,yaw_p95_deg=5,root_speed_p95=.05,joint_speed_p95=.5,joint_speed_max=2,tilt_max=.15)
    return dict(passed=all(values[k]<=limit for k,limit in limits.items()),values=values,limits=limits)


def metrics(model,poses,frames,controls,motion,original,timeline,tasks):
    phase=next(p for p in timeline['phases'] if p['name']=='source_motion')
    selected=(controls>=phase['control_start'])&(controls<phase['control_stop'])
    if not selected.any():return dict(passed=False,source_controls=0)
    q=poses[selected];frames=frames[selected];data=mujoco.MjData(model)
    ids=[model.body(t['target_body']).id for t in tasks];feet_ids=[model.body(s+'_ankle_roll_link').id for s in ('left','right')]
    points=[];feet=[]
    for pose in q:
        data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
        points.append([data.xpos[i]+data.xmat[i].reshape(3,3)@t['target_point'] for i,t in zip(ids,tasks)])
        feet.append(data.xpos[feet_ids].copy())
    root=q[:,:3];want_root=original['source_qpos29'][frames,:3]
    dy=yaw(q[:,3:7])-yaw(original['source_qpos29'][frames,3:7])
    errors=dict(root_p95_m=float(np.percentile(np.linalg.norm(root-want_root,axis=-1),95)),
        yaw_p95_deg=float(np.percentile(np.abs(np.rad2deg(np.arctan2(np.sin(dy),np.cos(dy)))),95)),
        hand_head_relative_p95_m=np.percentile(np.linalg.norm((np.asarray(points)-root[:,None])-(original['source_task_position_w'][frames]-want_root[:,None]),axis=-1),95,axis=0).tolist(),
        foot_relative_p95_m=np.percentile(np.linalg.norm((np.asarray(feet)-root[:,None])-(motion['body_pos_w'][frames][:,[6,12]]-motion['body_pos_w'][frames,0,None]),axis=-1),95,axis=0).tolist(),
        leg_rmse_rad=float(np.sqrt(np.mean((q[:,7:19]-motion['joint_pos'][frames,:12])**2))))
    passed=(len(q)==phase['requested_controls'] and errors['root_p95_m']<=.2 and errors['yaw_p95_deg']<=15
        and np.all(np.array(errors['hand_head_relative_p95_m'])<=[.15,.15,.1]) and max(errors['foot_relative_p95_m'])<=.12 and errors['leg_rmse_rad']<=.15)
    return dict(passed=bool(passed),source_controls=len(q),requested_source_controls=phase['requested_controls'],**errors)


def evaluate(args,clip):
    output=args.output/clip;output.mkdir(parents=True,exist_ok=False)
    model,c,motion,original,timeline=load_case(clip)
    bank=archive(args.bank/(clip+'.npz'));meta=json.loads((args.bank/'bank.json').read_text())
    data=mujoco.MjData(model);data.qpos[:]=bank['states'][10,:30];data.qvel[:]=bank['states'][10,30:]
    data.qvel[0]+=args.initial_velocity;mujoco.mj_forward(model,data)
    options=dict(now=-.22,model=model,standing_qpos=timeline['configured_standing_qpos'],tasks=meta['tasks'])
    if args.balanced:
        from gear_sonic.utils.g1_true23_causal_balance import BalancedCausalController
        neutral={k:v[:1] for k,v in archive(BUNDLE/'walk003/native_original.npz').items() if k!='fps'}
        weights=ROOT/'artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inference_v1/inference.safetensors'
        controller=BalancedCausalController(args.actor,c,weights=weights,neutral_motion=neutral,**options)
    else:controller=CausalController(args.actor,c,**options)
    receiver=controller.receiver
    total=timeline['total_requested_controls'];requested=total+args.hold*50
    for f in range(11):
        fields={k:motion[k][f] for k in motion if k!='fps'}
        assert receiver.receive(Packet(0,f,f*.02,fields),original['source_task_position_w'][f],original['source_task_quaternion_wxyz'][f],(f-11)*.02)
    trace=dict(qpos=[data.qpos.copy()],qvel=[data.qvel.copy()],physics_qpos=[data.qpos.copy()],physics_qvel=[data.qvel.copy()],
        physics_torque=[],physics_time=[0.],target=[],source_frame=[],global_control=[],physics_substeps=[],inference_ms=[],motion_fraction=[])
    failure=None;main_quiet=None;clock=0.;peak=0.;start=time.monotonic()
    for control in range(requested):
        frame=min(control+11,len(bank['joint'])-1)
        tick=time.perf_counter()
        if control+11<len(bank['joint']) and not (args.fault_control is not None and control>=args.fault_control):
            fields={k:motion[k][frame] for k in motion if k!='fps'}
            packet=Packet(0,frame,frame*.02,fields,frame==len(bank['joint'])-1)
            if not receiver.receive(packet,original['source_task_position_w'][frame],original['source_task_quaternion_wxyz'][frame],control*.02):
                # Gate faults remain latched; generated standing uses the same
                # causal policy and actual robot history, without a state reset.
                pass
        try:
            command=controller.command(data.qpos,data.qvel,control*.02)
            target=command.targets
        except Exception as e:failure=dict(kind='controller',control=control,error=repr(e));break
        controller.commit_applied(data.qpos,data.qvel,target)
        trace['inference_ms'].append((time.perf_counter()-tick)*1000)
        trace['motion_fraction'].append(command.status.get('motion_fraction',1.))
        trace['target'].append(target.copy());trace['source_frame'].append(frame);trace['global_control'].append(control)
        steps=0
        for sub in range(10):
            data.ctrl[:]=np.clip(c['kp']*(target-data.qpos[7:])-c['kd']*data.qvel[6:],-c['native_effort'],c['native_effort'])
            mujoco.mj_step(model,data);clock+=.002;steps+=1
            trace['physics_qpos'].append(data.qpos.copy());trace['physics_qvel'].append(data.qvel.copy())
            trace['physics_torque'].append(data.ctrl.copy());trace['physics_time'].append(float(data.time))
            reasons,values=assess(data,c,clock);peak=max(peak,values['speed_ratio'])
            if reasons:failure=dict(kind='physics',control=control,substep=sub+1,time=float(data.time),reasons=reasons,**values);break
        trace['physics_substeps'].append(steps);trace['qpos'].append(data.qpos.copy());trace['qvel'].append(data.qvel.copy())
        if control==total-1:main_quiet=quiet(np.asarray(trace['physics_qpos']),np.asarray(trace['physics_qvel']),original['source_qpos29'][-1])
        if failure:break
        if control%500==0:print(json.dumps(dict(clip=clip,control=control,simulation_time=float(data.time))),flush=True)
    arrays={k:np.asarray(v) for k,v in trace.items()};np.savez_compressed(output/'trace.npz',**arrays)
    complete=failure is None and len(arrays['target'])==requested
    completed=arrays['physics_substeps']==10
    source=metrics(model,arrays['qpos'][1:][completed],arrays['source_frame'][completed],arrays['global_control'][completed],motion,original,timeline,meta['tasks'])
    hold_quiet=quiet(arrays['physics_qpos'][total*10+1:],arrays['physics_qvel'][total*10+1:],original['source_qpos29'][-1]) if complete else None
    fault_quiet=quiet(arrays['physics_qpos'],arrays['physics_qvel'],receiver.stop.last) if complete and receiver.stop is not None else None
    report=dict(clip=clip,actor=str(args.actor),requested_controls=requested,completed_controls=int(completed.sum()),
        physical_complete=complete,source=source,main_quiet=main_quiet,hold_quiet=hold_quiet,failure=failure,
        passed=bool(complete and source['passed'] and main_quiet['passed'] and hold_quiet['passed']),
        elapsed_s=time.monotonic()-start,physics_steps=len(arrays['physics_torque']),maximum_speed_ratio=peak,
        policy_ms_p50_p95_max=np.percentile(arrays['inference_ms'],[50,95,100]).tolist() if len(arrays['inference_ms']) else [],
        packet_report=receiver.gate.epoch_report(),future_reference_frames=0,prepared_feedback_gains=False,
        initial_root_x_velocity_offset=args.initial_velocity,independent_realtime=False,hardware_authorized=False)
    report['controller_variant']='causal_full_range_with_neutral_balance' if args.balanced else 'causal_full_range_actor'
    report.update(fault_control=args.fault_control,fault_quiet=fault_quiet,
        fault_scenario_passed=bool(complete and receiver.gate.fault and fault_quiet and fault_quiet['passed']))
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--actor',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bank',type=Path,default=NEW/'causal_dynamics_v1/bank');p.add_argument('--clips',nargs='+',default=['walk003','walk002','pico','walk008'])
    p.add_argument('--hold',type=int,default=30);p.add_argument('--initial-velocity',type=float,default=0.)
    p.add_argument('--fault-control',type=int)
    p.add_argument('--balanced',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    result=[evaluate(a,c) for c in a.clips]
    scenario='nominal' if a.fault_control is None else 'input_loss'
    gate='passed' if scenario=='nominal' else 'fault_scenario_passed'
    (a.output/'report.json').write_text(json.dumps(dict(cases=result,scenario=scenario,
        passed=all(r[gate] for r in result),general_live_teleop_qualified=False),indent=2)+'\n')


if __name__=='__main__':main()
