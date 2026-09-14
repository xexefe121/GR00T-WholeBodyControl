"""Synchronized boundary evidence and one identical-state command probe.

Consumes the matched comparison; never changes a rollout or a reference.
"""
import argparse
import copy
import csv
import ctypes as ct
import json
from pathlib import Path
import sys
import mujoco
import numpy as np

from gear_sonic.scripts.compare_g1_true23_matched_walk003 import archive,dump,sha
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    MJ_TO_NATIVE,NATIVE_TO_MJ,SAFE_TARGET_DEFAULT_Q_HARDWARE,HARDWARE_23_ACTION_SCALE,
    safe_target_transform_numpy,_quaternion_matrix,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair-directory',type=Path,required=True)
    p.add_argument('--walk002-model',type=Path,required=True)
    p.add_argument('--walk002-report',type=Path,required=True)
    a=p.parse_args();out=a.pair_directory
    contract=json.loads((out/'comparison_contract.json').read_text())
    paths=list(contract['inputs_sha256'])
    def source(ending):return Path(next(x for x in paths if x.endswith(ending)))
    motion=archive(source('/walk003/reference.npz'));original=archive(source('/walk003/original29.npz'))
    audit=json.loads(source('/recorded_source_audit_v2.json').read_text())
    model=mujoco.MjModel.from_binary_path(str(out/'prepared_model.mjb'))
    oldmodel=mujoco.MjModel.from_binary_path(str(a.walk002_model))
    oldreport=json.loads(a.walk002_report.read_text())
    arms={k:archive(out/k/'trace.npz') for k in ('positive','sonic')}
    packets=json.loads((out/'references.json').read_text())
    tasks=[next(t for t in audit['original_intent']['convention']['tasks'] if t['name']==name)
           for name in ('left_hand','right_hand','head_proxy')]
    ids=[model.body(t['target_body']).id for t in tasks]
    feet=[model.body(side+'_ankle_roll_link').id for side in ('left','right')]
    names=['root','heading','left_hand','right_hand','head','left_foot','right_foot','leg_rmse']
    thresholds=np.array([.20,15,.15,.15,.10,.12,.12,.15]);errors={};d=mujoco.MjData(model)
    for arm,arr in arms.items():
        metrics=[]
        for control,pose in enumerate(arr['states'][10::10,:30]):
            frame=min(control+11,len(motion['joint_pos'])-1)
            d.qpos[:]=pose;mujoco.mj_kinematics(model,d)
            root=pose[:3];wanted=original['source_task_position_w'][frame];wr=original['source_qpos29'][frame,:3]
            points=np.array([d.xpos[i]+d.xmat[i].reshape(3,3)@np.array(t['target_point']) for i,t in zip(ids,tasks)])
            hand=np.linalg.norm((points-root)-(wanted-wr),axis=1)
            foot=np.linalg.norm((d.xpos[feet]-root)-(motion['body_pos_w'][frame,[6,12]]-motion['body_pos_w'][frame,0]),axis=1)
            actual_yaw=np.arctan2(d.xmat[1,3],d.xmat[1,0]);r=_quaternion_matrix(original['source_qpos29'][frame,3:7])
            dy=actual_yaw-np.arctan2(r[1,0],r[0,0])
            metrics.append(np.r_[np.linalg.norm(root-wr),abs(np.rad2deg(np.arctan2(np.sin(dy),np.cos(dy)))),hand,foot,
                np.sqrt(np.mean((pose[7:19]-motion['joint_pos'][frame,:12])**2))])
        errors[arm]=np.array(metrics)
    np.savez_compressed(out/'per_control_task_errors.npz',**errors,thresholds=thresholds,names=np.array(names))
    sonic=arms['sonic'];positive=arms['positive'];n=len(errors['sonic'])
    fail=(errors['sonic']>thresholds)&(errors['positive'][:n]<=thresholds)
    first={}
    for j,name in enumerate(names):
        indices=[i for i in range(n-4) if np.all(fail[i:i+5,j])]
        first[name]=indices[0] if indices else None
    earliest=min(i for i in first.values() if i is not None)
    # Verify the frozen output decoder and actual-command feedback conventions.
    for i,raw in enumerate(sonic['raw_native']):
        safe,target=safe_target_transform_numpy(raw)
        np.testing.assert_array_equal(target.astype(np.float64),sonic['proposals'][i])
        vr=np.r_[packets[i]['vr_3point_local_target'],packets[i]['vr_3point_local_orn_target']].astype(np.float32)
        np.testing.assert_array_equal(vr,sonic['encoder267'][i,240:261])
        np.testing.assert_array_equal(np.array(packets[i]['causal_history_lower_body'],np.float32),sonic['encoder267'][i,:240])
        if i:
            expected=((sonic['targets'][i*10-1]-np.array(SAFE_TARGET_DEFAULT_Q_HARDWARE))/np.array(HARDWARE_23_ACTION_SCALE))[MJ_TO_NATIVE].astype(np.float32)
            np.testing.assert_array_equal(expected,sonic['previous_safe_native'][i])
    schedule=np.load(out/'application_command_id.npy')
    # One CSV row per control update, both arms synchronized by packet/control
    # index. Positive normalized outputs are unavailable, never fabricated.
    knee_indices=[3,9];rows=[]
    for i in range(len(positive['proposals'])):
        packet=packets[i]
        row=dict(control=i,simulation_boundary_s=i*.02,reference_timestamp_ns=packet['control_monotonic_ns'],
            anchor_frame=packet['pico_anchor_source_frame_index'],scored_frame=min(i+11,len(motion['joint_pos'])-1))
        for side,j in zip(('left','right'),knee_indices):
            row[side+'_reference_anchor_rad']=np.array(packet['causal_history_lower_body'][:120]).reshape(10,12)[-1,j]
            row[side+'_reference_scored_rad']=motion['joint_pos'][row['scored_frame'],j]
            row[side+'_sonic_neutral_rad']=SAFE_TARGET_DEFAULT_Q_HARDWARE[j]
            row[side+'_sonic_scale_rad']=HARDWARE_23_ACTION_SCALE[j]
            for arm,arr in arms.items():
                prefix=side+'_'+arm+'_'
                if i>=len(arr['proposals']):continue
                row[prefix+'measured_rad']=arr['pre_control_state'][i,7+j]
                row[prefix+'velocity_rad_s']=arr['pre_control_state'][i,36+j]
                row[prefix+'proposed_target_rad']=arr['proposals'][i,j]
                stop=min((i+1)*10,len(arr['targets']))
                row[prefix+'applied_target_first_rad']=arr['targets'][i*10,j]
                row[prefix+'applied_target_last_rad']=arr['targets'][stop-1,j]
                row[prefix+'requested_torque_nm']=json.dumps(arr['requested_torque'][i*10:stop,j].tolist())
                row[prefix+'applied_torque_nm']=json.dumps(arr['applied_torque'][i*10:stop,j].tolist())
                row[prefix+'contact_count']=json.dumps(arr['contact_count'][i*10:stop].tolist())
                if arm=='sonic':
                    row[prefix+'raw_decoder_output']=arr['raw_native'][i,NATIVE_TO_MJ[j]]
                    row[prefix+'policy_presented_reference_rad']=arr['encoder267'][i,108+j]
                else:row[prefix+'raw_decoder_output']='not_applicable_applied_sequence'
        rows.append(row)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'synchronized_knee_boundaries.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    # Match every numeric array and option against the old walk002 plant.
    difference={}
    for k in dir(model):
        x,y=getattr(model,k),getattr(oldmodel,k)
        if isinstance(x,np.ndarray) and not np.array_equal(x,y):
            difference[k]=dict(prepared_shape=x.shape,walk002_shape=y.shape,
                changed_coordinates=int(np.count_nonzero(x!=y)) if x.shape==y.shape else None)
            if x.shape==y.shape and np.count_nonzero(x!=y)<100:
                where=np.argwhere(x!=y);difference[k].update(indices=where.tolist(),prepared_values=x[tuple(where.T)].tolist(),walk002_values=y[tuple(where.T)].tolist())
    options={}
    for k in dir(model.opt):
        if k.startswith('_'):continue
        x,y=getattr(model.opt,k),getattr(oldmodel.opt,k)
        if isinstance(x,(np.ndarray,int,float)) and not np.array_equal(x,y):options[k]=dict(prepared=x,walk002=y)
    gain_diff={}
    for new,old in [('kp','kp_hardware'),('kd','kd_hardware'),('native_effort','effort_hardware')]:
        gain_diff[new]=dict(prepared=contract['actuation'][new],walk002=oldreport[old])
    dump(out/'walk002_numeric_model_difference.json',dict(arrays=difference,solver_options=options,physical_pd_and_effort=gain_diff,
        xml_line_endings_used_as_numerical_evidence=False))
    # Reuse exact native step for a bounded identical-state command probe.
    lib=ct.CDLL(str(out/'libmatched_step.so'));ptr=ct.POINTER(ct.c_double)
    lib.matched_step.argtypes=[ct.c_void_p,ct.c_void_p]+[ptr]*5+[ct.c_double,ptr];lib.matched_step.restype=ct.c_int
    g=[np.ascontiguousarray(contract['actuation'][k],np.float64) for k in ('kp','kd','native_effort','native_velocity')]
    def step(data,target,index):
        target=np.ascontiguousarray(target,np.float64);requested=np.empty(23)
        return lib.matched_step(model._address,data._address,target.ctypes.data_as(ptr),*[v.ctypes.data_as(ptr) for v in g],(index+1)*.002,requested.ctypes.data_as(ptr))
    d=mujoco.MjData(model);d.qpos[:]=sonic['states'][0,:30];d.qvel[:]=sonic['states'][0,30:59];mujoco.mj_forward(model,d)
    for i in range(earliest*10):assert not step(d,sonic['targets'][i],i)
    np.testing.assert_array_equal(np.r_[d.qpos,d.qvel,d.time],sonic['pre_control_state'][earliest])
    probes={}
    for arm in ('sonic','positive'):
        probe=copy.copy(d)
        for i in range(earliest*10,(earliest+1)*10):
            target=arms[arm]['proposals'][earliest] if schedule[i]==earliest else sonic['targets'][earliest*10-1]
            assert not step(probe,target,i)
        expected=motion['joint_pos'][earliest+11,:12]
        probes[arm]=dict(post_leg_rmse_rad=float(np.sqrt(np.mean((probe.qpos[7:19]-expected)**2))),
            post_state=np.r_[probe.qpos,probe.qvel,probe.time],target=arms[arm]['proposals'][earliest],
            difference_from_nominal_sonic_post_max=float(np.max(abs(np.r_[probe.qpos,probe.qvel,probe.time]-sonic['states'][(earliest+1)*10]))))
    assert probes['sonic']['difference_from_nominal_sonic_post_max']==0
    # Existing 30-second acceptance: all 15,000 physics samples, same limits.
    hold=positive['states'][15691:30691];q=hold[:,:30];v=hold[:,30:59];wanted=original['source_qpos29'][-1]
    def yaw(q):return np.arctan2(2*(q[...,0]*q[...,3]+q[...,1]*q[...,2]),1-2*(q[...,2]**2+q[...,3]**2))
    dy=yaw(q[:,3:7])-yaw(wanted[3:7]);heading=np.abs(np.rad2deg(np.arctan2(np.sin(dy),np.cos(dy))))
    quiet_values=dict(samples=len(hold),root_xy_p95_m=float(np.percentile(np.linalg.norm(q[:,:2]-wanted[:2],axis=1),95)),
        heading_p95_deg=float(np.percentile(heading,95)),root_speed_p95_mps=float(np.percentile(np.linalg.norm(v[:,:3],axis=1),95)),
        joint_speed_p95_radps=float(np.percentile(np.max(abs(v[:,6:]),axis=1),95)),joint_speed_max_radps=float(np.max(abs(v[:,6:]))),
        tilt_max_rad=float(np.max(np.arccos(np.clip(1-2*np.sum(q[:,4:6]**2,axis=1),-1,1)))))
    quiet_pass=all(quiet_values[k]<=limit for k,limit in [('root_xy_p95_m',.05),('heading_p95_deg',5),('root_speed_p95_mps',.05),
        ('joint_speed_p95_radps',.5),('joint_speed_max_radps',2),('tilt_max_rad',.15)]) and len(hold)==15000
    result=dict(first_divergence_definition='five consecutive post-control errors above original threshold while positive control stays below it; localization only',
        first_by_metric_control_index=first,first_control=earliest,first_post_time_s=(earliest+1)*.02,
        phase='acquisition_ramp' if 250<=earliest<350 else 'source_motion' if 350<=earliest<1169 else 'other',
        first_synchronized_record=rows[earliest],error_names=names,sonic_errors=errors['sonic'][earliest],positive_errors=errors['positive'][earliest],
        original_goals_enter_encoder_exactly=True,frozen_action_decoding_exact=True,own_applied_command_history_verified=True,
        probe_common_state='SONIC pre-command state and history; earlier SONIC command retained until the common recorded application time',
        probe_positive_scope='archived positive command at this timestamp, not feedback reevaluated on SONIC state',probes=probes,
        positive_full_30s_quiet=dict(passed=quiet_pass,values=quiet_values),
        conclusion='Prepared task achievable on matched plant. Frozen SONIC conditioning/action path fails this paired test. Action sign alone does not identify a mapping bug; no policy adaptation run.',
        script_sha256=sha(__file__))
    dump(out/'first_divergence.json',result)
    print(json.dumps(dict(first_control=earliest,phase=result['phase'],probes={k:v['post_leg_rmse_rad'] for k,v in probes.items()},positive_30s=quiet_values)),flush=True)


if __name__=='__main__':main()
