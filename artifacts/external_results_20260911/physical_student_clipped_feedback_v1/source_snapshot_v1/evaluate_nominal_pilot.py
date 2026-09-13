"""One native323 student lifecycle, then a separately stored continuous5s hold."""
import json
from pathlib import Path
import time
import argparse
import mujoco
import numpy as np
from student_linear_runtime import *
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override,motion_states
from quiet_metrics import standing_windows,quiet_diagnostic,yaw


def get_state(model,data):
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    value=np.empty(mujoco.mj_stateSize(model,spec))
    mujoco.mj_getState(model,data,value,spec)
    return value


def assess(data,c,expected):
    limits=np.asarray(c['joint_limits']);q,dq=data.qpos,data.qvel
    finite=np.isfinite(q).all() and np.isfinite(dq).all()
    excess=float(np.maximum(0,np.maximum(limits[:,0]-q[7:],q[7:]-limits[:,1])).max()) if finite else float('inf')
    velocity=float(np.max(np.abs(dq[6:])/np.asarray(c['native_velocity']))) if finite else float('inf')
    force=data.qfrc_actuator[6:]
    effort=float(np.max(np.abs(force)/np.asarray(c['native_effort']))) if np.isfinite(force).all() else float('inf')
    tilt=float(np.arccos(np.clip(1-2*np.sum(q[4:6]**2),-1,1))) if finite else float('inf')
    reasons=[]
    if not finite or not np.isfinite(force).all():reasons.append('nonfinite_state_or_force')
    if finite and abs(np.linalg.norm(q[3:7])-1)>1e-10:reasons.append('invalid_root_quaternion')
    if excess>1e-6:reasons.append('native_joint_bound')
    if velocity>1:reasons.append('native_joint_speed')
    if effort>1+1e-9:reasons.append('native_actuator_force')
    if q[2]<.25 or tilt>1.2:reasons.append('fall')
    if np.any(data.warning.number):reasons.append('engine_warning')
    clock=float(abs(data.time-expected))
    if not np.isfinite(data.time) or clock>1e-10:reasons.append('physics_clock')
    return reasons,dict(range_excess=excess,velocity_ratio=velocity,effort_ratio=effort,tilt=tilt,clock_error=clock)


def new_trace(data):
    fields=('qpos','qvel','target','source_frame','global_control','controller_mode','joint_error','root_error',
        'state','history','previous_action','action','base_target','delta','features','inference_ms',
        'physics_qpos','physics_qvel','physics_torque','physics_actuator_torque','physics_time','physics_expected_time',
        'physics_warning_counts','physics_warning_lastinfo','physics_substeps','range_excess','velocity_ratio','effort_ratio','clock_error',
        'control_integration_before','control_history_before','control_previous_action_before')
    trace={k:[] for k in fields}
    for key in ('qpos','physics_qpos'):trace[key].append(data.qpos.copy())
    for key in ('qvel','physics_qvel'):trace[key].append(data.qvel.copy())
    trace['physics_time'].append(float(data.time));trace['physics_expected_time'].append(float(data.time))
    trace['physics_warning_counts'].append(data.warning.number.copy());trace['physics_warning_lastinfo'].append(data.warning.lastinfo.copy())
    return trace


def source_metrics(native,arrays,motion,original29,audit):
    complete=np.asarray(arrays['physics_substeps'])==10
    selected=complete&(arrays['global_control']>=350)&(arrays['global_control']<1169)
    if not np.any(selected):return dict(source_controls=0,no_source_samples=True)
    poses=arrays['qpos'][1:][selected];frames=arrays['source_frame'][selected]
    if not np.isfinite(poses).all():
        return dict(source_controls=int(selected.sum()),unavailable_reason='nonfinite measured source poses; failure retained')
    data=mujoco.MjData(native);points=[];feet=[]
    tasks=[next(t for t in audit['original_intent']['convention']['tasks'] if t['name']==name) for name in ('left_hand','right_hand','head_proxy')]
    ids=[native.body(t['target_body']).id for t in tasks]
    foot_ids=[native.body(side+'_ankle_roll_link').id for side in ('left','right')]
    for pose in poses:
        data.qpos[:]=pose;mujoco.mj_kinematics(native,data)
        points.append([data.xpos[i]+data.xmat[i].reshape(3,3)@np.asarray(t['target_point']) for i,t in zip(ids,tasks)])
        feet.append(data.xpos[foot_ids].copy())
    points=np.asarray(points);feet=np.asarray(feet);root=poses[:,:3]
    wanted=original29['source_task_position_w'][frames];wanted_root=original29['source_qpos29'][frames,:3]
    wanted_yaw=yaw(original29['source_qpos29'][frames,3:7]);actual_yaw=yaw(poses[:,3:7]);dy=actual_yaw-wanted_yaw
    ref_root=motion['body_pos_w'][frames,0];ref_feet=motion['body_pos_w'][frames][:,[6,12]]
    error=poses[:,7:]-motion['joint_pos'][frames]
    return dict(source_controls=len(poses),requested_source_controls=819,
        original_root_world_p95_m=float(np.percentile(np.linalg.norm(root-wanted_root,axis=1),95)),
        original_root_yaw_abs_p95_deg=float(np.percentile(np.abs(np.rad2deg(np.arctan2(np.sin(dy),np.cos(dy)))),95)),
        original_hand_head_relative_p95_m=np.percentile(np.linalg.norm((points-root[:,None])-(wanted-wanted_root[:,None]),axis=-1),95,axis=0).tolist(),
        original_hand_head_world_p95_m=np.percentile(np.linalg.norm(points-wanted,axis=-1),95,axis=0).tolist(),
        world_axis_relative_foot_p95_m=np.percentile(np.linalg.norm((feet-root[:,None])-(ref_feet-ref_root[:,None]),axis=-1),95,axis=0).tolist(),
        leg_rmse_rad=float(np.sqrt(np.mean(error[:,:12]**2))),arm_rmse_rad=float(np.sqrt(np.mean(error[:,13:]**2))),
        joint_rmse_by_hardware_joint=np.sqrt(np.mean(error**2,axis=0)).tolist())


def zero_parity(native,c,original,motion,original29):
    dest=BASE/'zero_parity';dest.mkdir(exist_ok=False)
    left=LinearStudentRuntime(native,c,original,motion,original29,BASE/'fit/zero_head.onnx')
    right=LinearStudentRuntime(native,c,original,motion,original29,BASE/'fit/zero_head.onnx')
    data=mujoco.MjData(native);initial=motion_states(motion)[10];data.qpos[:]=initial[:30];data.qvel[:]=initial[30:];mujoco.mj_forward(native,data)
    kp,kd,effort=[np.asarray(c[k]) for k in ('kp','kd','native_effort')]
    expected=float(data.time);states=[np.r_[data.qpos,data.qvel]];torques=[]
    for control in range(100):
        a=left.propose(control,data.qpos,data.qvel);b=right.propose(control,data.qpos,data.qvel,disable_head=True)
        for key in ('target','base_target','delta','previous_action','action','state','history','features'):np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(a['delta'],np.zeros(23,np.float32))
        for sub in range(10):
            data.ctrl[:]=np.clip(kp*(a['target']-data.qpos[7:])-kd*data.qvel[6:],-effort,effort)
            mujoco.mj_step(native,data);expected+=.002
            reasons,metrics=assess(data,c,expected)
            if reasons:raise AssertionError(('zero parity native witness failure',control,sub,reasons,metrics))
            states.append(np.r_[data.qpos,data.qvel]);torques.append(data.ctrl.copy())
    np.savez_compressed(dest/'trace.npz',physics_states=states,physics_torque=torques)
    report=dict(kind='zerohead_vs_disabledhead_same_frozen_ONNX_prior',controls=100,physics_steps=1000,
        all_control_state_history_rawaction_target_bitexact=True,all_zero_deltas_exact=True,native2msstrict_pass=True,
        historical_PyTorch_crossbackend_bitexact_claim=False,trace_sha256=sha(dest/'trace.npz'))
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


def main(zero_only=False):
    frozen=assert_frozen();assert mujoco.__version__=='3.2.3'
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,'walk003')
    motion,override=load_motion_override(REFERENCE,BUNDLE,'walk003',native,c,original,timeline,manifest)
    original29=archive(BUNDLE/'walk003/original29.npz');audit=json.loads((TEACHER/'recorded_source_audit_v2.json').read_text())
    if zero_only:
        zero_parity(native,c,original,motion,original29)
        return
    parity=json.loads((BASE/'zero_parity/report.json').read_text())
    assert parity['all_control_state_history_rawaction_target_bitexact'] and parity['native2msstrict_pass']
    fit=json.loads((BASE/'fit/report.json').read_text())
    assert sha(BASE/'fit/student_head.onnx')==fit['checkpoints']['student_head.onnx']
    runtime=LinearStudentRuntime(native,c,original,motion,original29,BASE/'fit/student_head.onnx')
    initial=motion_states(motion)[10];data=mujoco.MjData(native);data.qpos[:]=initial[:30];data.qvel[:]=initial[30:];mujoco.mj_forward(native,data)
    kp,kd,effort=[np.asarray(c[k]) for k in ('kp','kd','native_effort')]
    expected=float(data.time);initial_reasons,_=assess(data,c,expected)
    assert not initial_reasons,initial_reasons
    switch=next(p['control_start'] for p in timeline['phases'] if p['name']=='returned_standing')
    assert switch==1269 and len(motion['joint_pos'])-11==1569
    output=BASE/'nominal';output.mkdir(exist_ok=False)
    common_request=dict(kind=KIND,clip='walk003',motion_override=override,model_manifest=manifest,
        controller='frozenoriginalBFM+rangecomplete linearspan residual first1269; frozenBFMyaw4 terminal',
        head_sha256=sha(BASE/'fit/student_head.onnx'),zero_parity_sha256=sha(BASE/'zero_parity/report.json'),
        frozen_sources_sha256=sha(FROZEN_RECEIPT),teacher_audit_sha256=sha(TEACHER/'recorded_source_audit_v2.json'),
        student_goal='v4 native+original29 tasks',BFM_goal='originalnativeh8pos1yaw2, terminalyaw4',
        prior_action='preclipcombinedBFM+linearresidual while learned; BFMrawactor*5 terminal; unclippedhistory',
        received_goal_preview_seconds=.74,conservative_raw_pose_support_seconds=.76,
        physical_hz=500,control_hz=50,source_frames_removed=0,physical_statewrites_after_initialization=0,
        simulation_privileged_root_pose_velocity=True,hardware_authorized=False,DAgger_queries_launched=False)

    def segment(dest,start,count,expected_time):
        request=dict(common_request,initial_control=start,requested_controls=count,segment='original_lifecycle' if start==0 else 'separate5s_continuous_terminal_hold')
        (dest/'request.json').write_text(json.dumps(request,indent=2,allow_nan=False)+'\n')
        trace=new_trace(data);trace['physics_expected_time'][0]=expected_time
        segment_start_time=float(data.time)
        initial_vector=get_state(native,data);failure=None;started=time.perf_counter()
        for control in range(start,start+count):
            trace['control_integration_before'].append(get_state(native,data))
            trace['control_history_before'].append(np.concatenate([runtime.seed.history.data[k].reshape(-1) for k in sorted(runtime.seed.history.data)]).copy())
            trace['control_previous_action_before'].append(runtime.seed.previous_action.copy())
            try:
                proposed=runtime.propose(control,data.qpos,data.qvel,terminal=control>=switch)
            except Exception as exc:
                failure=dict(reasons=['policy_inference_fault'],global_control=control,local_control=control-start,
                    substep=0,time=float(data.time),exception_type=type(exc).__name__,message=str(exc))
                break
            target=proposed['target'];actual_substeps=0
            for sub in range(10):
                data.ctrl[:]=np.clip(kp*(target-data.qpos[7:])-kd*data.qvel[6:],-effort,effort)
                mujoco.mj_step(native,data);expected_time+=.002;actual_substeps+=1
                for key,value in (('physics_qpos',data.qpos),('physics_qvel',data.qvel),('physics_torque',data.ctrl),('physics_actuator_torque',data.qfrc_actuator[6:]),('physics_warning_counts',data.warning.number),('physics_warning_lastinfo',data.warning.lastinfo)):
                    trace[key].append(value.copy())
                trace['physics_time'].append(float(data.time));trace['physics_expected_time'].append(expected_time)
                reasons,metrics=assess(data,c,expected_time)
                for key in ('range_excess','velocity_ratio','effort_ratio','clock_error'):trace[key].append(metrics[key])
                if reasons:
                    failure=dict(reasons=reasons,global_control=control,local_control=control-start,substep=sub+1,time=float(data.time),**metrics)
                    break
            mujoco.mj_kinematics(native,data)
            frame=min(control+11,len(motion['joint_pos'])-1)
            values=dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),source_frame=frame,global_control=control,
                controller_mode=1 if control>=switch else 0,physics_substeps=actual_substeps,
                joint_error=data.qpos[7:]-motion['joint_pos'][frame],root_error=data.qpos[:3]-motion['body_pos_w'][frame,0],**proposed)
            for key,value in values.items():trace[key].append(value)
            if failure:break
        arrays={k:np.asarray(v) for k,v in trace.items()}
        np.savez_compressed(dest/'trace.npz',**arrays,initial_integration=initial_vector,final_integration=get_state(native,data),
            integration_state_spec=np.asarray(int(mujoco.mjtState.mjSTATE_INTEGRATION),dtype=np.int64),
            final_previous_action=runtime.seed.previous_action,final_recorded_controls=np.asarray(runtime.seed.recorded_controls),
            **{'final_history_'+k:v for k,v in runtime.seed.history.data.items()})
        complete=int(np.sum(arrays['physics_substeps']==10));strict=failure is None
        quiet=None
        if len(arrays['target']) and (start>0 or complete==count) and np.isfinite(arrays['physics_qpos']).all() and np.isfinite(arrays['physics_qvel']).all():
            windows=standing_windows(arrays,motion,original,original29);quiet=quiet_diagnostic(windows,strict)
        maximum=lambda key:float(np.max(arrays[key])) if len(arrays[key]) else 0.
        timings=np.percentile(arrays['inference_ms'],[50,95,100]).tolist() if len(arrays['inference_ms']) else []
        result=dict(kind=KIND,clip='walk003',mujoco=mujoco.__version__,segment=request['segment'],requested_controls=count,completed_full_controls=complete,completed_controls=complete,
            attempted_controls=len(arrays['target']),partial_substeps=len(arrays['physics_torque'])%10,physics_steps=len(arrays['physics_torque']),
            full_segment_completed=complete==count and strict,probe_completed=complete==count and strict,failure=failure,
            range_excess_max=maximum('range_excess'),velocity_ratio_max=maximum('velocity_ratio'),effort_ratio_max=maximum('effort_ratio'),
            engine_warning_counts=data.warning.number.tolist(),engine_warning_lastinfo=data.warning.lastinfo.tolist(),
            independent_accumulated_clock_max_error=maximum('clock_error'),
            policy_ms_p50_p95_max=timings,policy_20ms_deadline_misses=int(np.sum(arrays['inference_ms']>20)),
            source_metrics=source_metrics(native,arrays,motion,original29,audit) if len(arrays['target']) else dict(source_controls=0,no_source_samples=True),quiet_standing_diagnostic=quiet,
            simulation_start_time=segment_start_time,simulation_end_time=float(data.time),simulated_seconds=float(data.time)-segment_start_time,
            full_body_teleoperation_qualified=False,hardware_authorized=False,elapsed_seconds=time.perf_counter()-started,
            trace_sha256=sha(dest/'trace.npz'),request_sha256=sha(dest/'request.json'),motion_override=override,
            expert_query_snapshot='every precontrol fullintegration+history+previousaction retained; partialfailurestate also retained, noquerylaunched')
        result=finite_json(result)
        (dest/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result),flush=True)
        return result,expected_time

    result,expected=segment(output,0,1569,expected)
    extension=BASE/'post_lifecycle_hold_5s';extension.mkdir(exist_ok=False)
    if result['full_segment_completed']:
        extended,expected=segment(extension,1569,250,expected)
    else:
        extended=dict(requested_controls=250,attempted_controls=0,not_run_reason='original lifecycle stopped at firststrictphysicalfailure',
            no_reset_or_skip_to_terminal=True,full_segment_completed=False)
        (extension/'report.json').write_text(json.dumps(extended,indent=2)+'\n')
    (BASE/'pilot_outcome.json').write_text(json.dumps(dict(nominal=result,extension=extended,DAgger_not_launched=True,one_fixed_nominal_fit=True),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--zero-only',action='store_true')
    main(parser.parse_args().zero_only)
