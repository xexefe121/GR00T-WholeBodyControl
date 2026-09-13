"""Future selected canonical direct-target evaluation; source preparation only."""
import json
from pathlib import Path
import time
import argparse
import mujoco
import numpy as np
from direct_runtime import *
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override,motion_states
from quiet_metrics import standing_windows,quiet_diagnostic,yaw
from evaluation_gate import require_ready
from proposal_evidence import ProposalEvidence,exact,trace_arrays


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


def source_metrics(native,arrays,motion,original29,audit,timeline):
    complete=np.asarray(arrays['physics_substeps'])==10
    source_phase=next(p for p in timeline['phases'] if p['name']=='source_motion')
    selected=complete&(arrays['global_control']>=source_phase['control_start'])&(arrays['global_control']<source_phase['control_stop'])
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
    return dict(source_controls=len(poses),requested_source_controls=source_phase['requested_controls'],
        original_root_world_p95_m=float(np.percentile(np.linalg.norm(root-wanted_root,axis=1),95)),
        original_root_yaw_abs_p95_deg=float(np.percentile(np.abs(np.rad2deg(np.arctan2(np.sin(dy),np.cos(dy)))),95)),
        original_hand_head_relative_p95_m=np.percentile(np.linalg.norm((points-root[:,None])-(wanted-wanted_root[:,None]),axis=-1),95,axis=0).tolist(),
        original_hand_head_world_p95_m=np.percentile(np.linalg.norm(points-wanted,axis=-1),95,axis=0).tolist(),
        world_axis_relative_foot_p95_m=np.percentile(np.linalg.norm((feet-root[:,None])-(ref_feet-ref_root[:,None]),axis=-1),95,axis=0).tolist(),
        leg_rmse_rad=float(np.sqrt(np.mean(error[:,:12]**2))),arm_rmse_rad=float(np.sqrt(np.mean(error[:,13:]**2))),
        joint_rmse_by_hardware_joint=np.sqrt(np.mean(error**2,axis=0)).tolist())





def main():
    evaluation_binding=require_ready(BASE)
    HEAD=evaluation_binding['head']
    assert mujoco.__version__=='3.2.3'
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,'walk003')
    motion,override=load_motion_override(REFERENCE,BUNDLE,'walk003',native,c,original,timeline,manifest)
    original29=archive(BUNDLE/'walk003/original29.npz');audit=json.loads((TEACHER/'recorded_source_audit_v2.json').read_text())
    baseline=archive(BASE.parent/'original_bfm_entry250_v1/entry250/trace.npz')
    baseline_report=json.loads((BASE.parent/'original_bfm_entry250_v1/entry250/report.json').read_text())
    assert baseline_report['full_segment_completed'] and baseline_report['quiet_standing_diagnostic']['quiet_standing_diagnostic_pass']
    assert sha(BASE.parent/'original_bfm_entry250_v1/entry250/trace.npz')==baseline_report['trace_sha256']
    labels=archive(BASE.parent/'bfm_entry250_labels_v1/labels/labels.npz')
    label_report=json.loads((BASE.parent/'bfm_entry250_labels_v1/labels/report.json').read_text())
    assert label_report['samples']==1019 and label_report['expert_controls']==[250,1268]
    assert sha(BASE.parent/'bfm_entry250_labels_v1/labels/labels.npz')==label_report['labels_sha256']
    np.testing.assert_array_equal(labels['control'],np.arange(250,1269))
    np.testing.assert_array_equal(labels['source_frame'],np.arange(261,1280))
    fit=evaluation_binding['fit_report']
    assert fit['features']==1000 and fit['head_output']=='normalized_target'
    runtime=DirectStudentRuntime.from_native(native,c,original,motion,original29,HEAD,evaluation_binding['span'],timeline)
    assert runtime.ort_binary_sha256==str(evaluation_binding['first_export']['runtime_binary_sha256'].item())
    evidence=ProposalEvidence(runtime,c)
    initial=motion_states(motion)[10];data=mujoco.MjData(native);data.qpos[:]=initial[:30];data.qvel[:]=initial[30:];mujoco.mj_forward(native,data)
    kp,kd,effort=[np.asarray(c[k]) for k in ('kp','kd','native_effort')]
    expected=float(data.time);initial_reasons,_=assess(data,c,expected)
    assert not initial_reasons,initial_reasons
    switch=runtime.phases.terminal_start
    assert len(motion['joint_pos'])-runtime.phases.frame_offset==runtime.phases.total
    output=BASE/'nominal';output.mkdir(exist_ok=False)
    common_request=dict(kind=KIND,clip='walk003',motion_override=override,model_manifest=manifest,
        controller='originalBFM yaw2 startup; direct normalized absolute targets acquisition/source/return; originalBFM yaw4 terminal',
        head_sha256=sha(HEAD),baseline250_trace_sha256=sha(BASE.parent/'original_bfm_entry250_v1/entry250/trace.npz'),
        ordinary_final_step=fit.get('ordinary_final_step'),evaluation_binding_sha256=evaluation_binding['binding_sha256'],
        proposal_diagnostics='learned base_target is default_q, delta is span64*normalized_head64; BFM phases retain original base and zero delta; actual clipped target inverse enters learned history',
        frozen_sources_sha256=evaluation_binding['binding_sha256'],teacher_audit_sha256=sha(TEACHER/'recorded_source_audit_v2.json'),
        student_goal='v4 native+original29 tasks',BFM_goal='originalnativeh8pos1yaw2, terminalyaw4',
        prior_action='original rawBFM startup and terminal; inverse actual clipped native target during learned phase; no +/-5 history clamp',
        received_goal_preview_seconds=.74,conservative_raw_pose_support_seconds=.76,
        physical_hz=500,control_hz=50,source_frames_removed=0,physical_statewrites_after_initialization=0,
        simulation_privileged_root_pose_velocity=True,hardware_authorized=False,DAgger_queries_launched=False,
        initial_BFM_controls=runtime.phases.learned_start,learned_controls=[runtime.phases.learned_start,runtime.phases.terminal_start],first_activation_query250_labels_sha256=label_report['labels_sha256'],
        prefix_and_query_labels_comparison_only_no_injection=True,controller_mode_map={'0':'initial_BFM','1':'direct_absolute_target','2':'terminal_BFM'},learned_BFM_calls=0,clock_foundation_connected=False)

    class TransitionParityError(RuntimeError):pass

    def record_parity(name,pairs):
        checks={key:dict(bitexact=exact(actual,expected),actual_shape=list(np.shape(actual)),expected_shape=list(np.shape(expected)))
            for key,actual,expected in pairs}
        passed=all(v['bitexact'] for v in checks.values())
        result=dict(passed=passed,checks=checks,comparison_only_no_injection=True,hardware_authorized=False)
        (BASE/name).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        if not passed:raise TransitionParityError(name+' failed: '+str([k for k,v in checks.items() if not v['bitexact']]))

    def verify_generated_prefix250(trace):
        fields=('qpos','qvel','target','source_frame','global_control','controller_mode','joint_error','root_error',
            'state','history','previous_action','action','base_target','delta','features',
            'physics_qpos','physics_qvel','physics_torque','physics_actuator_torque','physics_time','physics_expected_time',
            'physics_warning_counts','physics_warning_lastinfo','physics_substeps','range_excess','velocity_ratio','effort_ratio','clock_error',
            'control_integration_before','control_history_before','control_previous_action_before')
        pairs=[(key,np.asarray(trace[key])[:len(baseline[key])],baseline[key][:,np.r_[0:52,75:1023]] if key=='features' else baseline[key]) for key in fields]
        pairs.extend([('final_integration',get_state(native,data),baseline['final_integration']),
            ('recorded_controls',np.asarray(runtime.seed.recorded_controls),baseline['final_recorded_controls']),
            ('final_previous_action',runtime.seed.previous_action,baseline['final_previous_action'])])
        pairs.extend(('final_history_'+key,value,baseline['final_history_'+key]) for key,value in runtime.seed.history.data.items())
        record_parity('canonical_prefix250_parity.json',pairs)

    def verify_actual_query250(proposed):
        pairs=[(key,proposed[key],labels[key][0]) for key in ('previous_action','history','state')]
        pairs.append(('features1000',proposed['features'],labels['features'][0,np.r_[0:52,75:1023]]))
        pairs.extend([('integration',get_state(native,data),labels['control_integration_before'][0]),
            ('qpos',data.qpos,labels['teacher_qpos'][0]),('qvel',data.qvel,labels['teacher_qvel'][0])])
        record_parity('actual_query250_input_parity.json',pairs)
        witness=evaluation_binding['first_export']
        record_parity('actual_query250_ownexport_output_parity.json',[(
            'features',proposed['features'],witness['features']),('normalized_target',proposed['normalized_head'],witness['normalized_target']),('preclamp_raw',proposed['base_target']+proposed['delta'],witness['raw_proposal']),('target',proposed['target'],witness['target'])])

    def segment(dest,start,count,expected_time):
        request=dict(common_request,initial_control=start,requested_controls=count,segment='original_lifecycle' if start==0 else 'separate5s_continuous_terminal_hold')
        (dest/'request.json').write_text(json.dumps(request,indent=2,allow_nan=False)+'\n')
        trace=new_trace(data);trace['physics_expected_time'][0]=expected_time
        trace['raw_proposal']=[];trace['actual_normalized_action']=[];trace['normalized_head']=[]
        segment_start_time=float(data.time)
        initial_vector=get_state(native,data);failure=None;started=time.perf_counter()
        for control in range(start,start+count):
            trace['control_integration_before'].append(get_state(native,data))
            trace['control_history_before'].append(np.concatenate([runtime.seed.history.data[k].reshape(-1) for k in sorted(runtime.seed.history.data)]).copy())
            trace['control_previous_action_before'].append(runtime.seed.previous_action.copy())
            evidence.begin(control,data.qpos,data.qvel,trace['control_integration_before'][-1],expected_time,data.warning.number,data.warning.lastinfo)
            try:
                if control==runtime.phases.learned_start:verify_generated_prefix250(trace)
                proposed=runtime.propose(control,data.qpos,data.qvel)
                evidence.accepted_proposal(proposed)
                if control==runtime.phases.learned_start:verify_actual_query250(proposed)
                runtime.commit(proposed)
            except Exception as exc:
                failure=dict(reasons=['transition_parity_fault' if isinstance(exc,TransitionParityError) else 'policy_inference_fault'],global_control=control,local_control=control-start,
                    substep=0,time=float(data.time),exception_type=type(exc).__name__,message=str(exc))
                evidence.preserve(dest/'rejected_precontrol.npz',repr(exc),data.qpos,data.qvel,get_state(native,data),expected_time,data.warning.number,data.warning.lastinfo)
                break
            target=proposed['target'];actual_substeps=0
            for sub in range(10):
                data.ctrl[:]=np.clip(kp*(target-data.qpos[7:])-kd*data.qvel[6:],-effort,effort)
                try:
                    mujoco.mj_step(native,data)
                except BaseException as exc:
                    evidence.current['attempted_native_ctrl']=data.ctrl.copy()
                    evidence.current['native_step_attempt_substep']=np.asarray(sub,np.int64)
                    evidence.preserve(dest/'native_exception_state.npz',repr(exc),data.qpos,data.qvel,get_state(native,data),expected_time,data.warning.number,data.warning.lastinfo)
                    np.savez_compressed(dest/'native_exception_completed_samples.npz',**{key:np.asarray(value) for key,value in trace.items()})
                    (dest/'fatal_failure.json').write_text(json.dumps(dict(reason='native_step_exception',exception_type=type(exc).__name__,message=str(exc),global_control=control,attempted_substep=sub,completed_prior_substeps=actual_substeps,requested_controls=count,full_segment_completed=False,unclassified_attempted_step=True),indent=2)+'\n')
                    raise
                expected_time+=.002;actual_substeps+=1
                for key,value in (('physics_qpos',data.qpos),('physics_qvel',data.qvel),('physics_torque',data.ctrl),('physics_actuator_torque',data.qfrc_actuator[6:]),('physics_warning_counts',data.warning.number),('physics_warning_lastinfo',data.warning.lastinfo)):
                    trace[key].append(value.copy())
                trace['physics_time'].append(float(data.time));trace['physics_expected_time'].append(expected_time)
                reasons,metrics=assess(data,c,expected_time)
                for key in ('range_excess','velocity_ratio','effort_ratio','clock_error'):trace[key].append(metrics[key])
                if reasons:
                    failure=dict(reasons=reasons,global_control=control,local_control=control-start,substep=sub+1,time=float(data.time),**metrics)
                    break
            mujoco.mj_kinematics(native,data)
            frame=min(control+runtime.phases.frame_offset,len(motion['joint_pos'])-1)
            values=dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),source_frame=frame,global_control=control,
                controller_mode=runtime.phases.mode(control),physics_substeps=actual_substeps,
                joint_error=data.qpos[7:]-motion['joint_pos'][frame],root_error=data.qpos[:3]-motion['body_pos_w'][frame,0],
                raw_proposal=evidence.last_raw_proposal.copy(),actual_normalized_action=evidence.last_actual_action.copy(),**proposed)
            for key,value in values.items():trace[key].append(value)
            if failure:
                evidence.preserve(dest/'strict_failure_state.npz',failure,data.qpos,data.qvel,get_state(native,data),expected_time,data.warning.number,data.warning.lastinfo)
                break
        arrays=trace_arrays(trace)
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
            attempted_precontrols=len(arrays['control_integration_before']),controller_recorded_controls=runtime.seed.recorded_controls,
            raw_proposal_and_actual_normalized_action_recorded=True,actual_normalized_action_used_for_history=True,actual_action_feedback_scope='learned phase only',inference_counts=dict(runtime.counts),forbidden_inference_calls=runtime.forbidden_calls,
            full_segment_completed=complete==count and strict,probe_completed=complete==count and strict,failure=failure,
            range_excess_max=maximum('range_excess'),velocity_ratio_max=maximum('velocity_ratio'),effort_ratio_max=maximum('effort_ratio'),
            engine_warning_counts=data.warning.number.tolist(),engine_warning_lastinfo=data.warning.lastinfo.tolist(),
            independent_accumulated_clock_max_error=maximum('clock_error'),
            policy_ms_p50_p95_max=timings,policy_20ms_deadline_misses=int(np.sum(arrays['inference_ms']>20)),
            source_metrics=source_metrics(native,arrays,motion,original29,audit,timeline) if len(arrays['target']) else dict(source_controls=0,no_source_samples=True),quiet_standing_diagnostic=quiet,
            simulation_start_time=segment_start_time,simulation_end_time=float(data.time),simulated_seconds=float(data.time)-segment_start_time,
            full_body_teleoperation_qualified=False,hardware_authorized=False,elapsed_seconds=time.perf_counter()-started,
            trace_sha256=sha(dest/'trace.npz'),request_sha256=sha(dest/'request.json'),motion_override=override,
            expert_query_snapshot='every precontrol fullintegration+history+previousaction retained; partialfailurestate also retained, noquerylaunched')
        result=finite_json(result)
        (dest/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result),flush=True)
        return result,expected_time

    initial_integration=get_state(native,data)
    assert initial_integration.shape==(291,)
    np.savez_compressed(BASE/'canonical_initial_snapshot.npz',integration=initial_integration,qpos=data.qpos,qvel=data.qvel,warning_counts=data.warning.number,warning_lastinfo=data.warning.lastinfo)
    record_parity('canonical_initial_full291_parity.json',[('integration',initial_integration,baseline['initial_integration']),('qpos',data.qpos,baseline['qpos'][0]),('qvel',data.qvel,baseline['qvel'][0]),('warning_counts',data.warning.number,baseline['physics_warning_counts'][0]),('warning_lastinfo',data.warning.lastinfo,baseline['physics_warning_lastinfo'][0])])
    result,expected=segment(output,0,runtime.phases.total,expected)
    extension=BASE/'post_lifecycle_hold_5s';extension.mkdir(exist_ok=False)
    if result['full_segment_completed']:
        extended,expected=segment(extension,runtime.phases.total,250,expected)
    else:
        extended=dict(requested_controls=250,attempted_controls=0,not_run_reason='original lifecycle stopped at first failure; saved failure identifies policy, parity or native limit',
            no_reset_or_skip_to_terminal=True,full_segment_completed=False)
        (extension/'report.json').write_text(json.dumps(extended,indent=2)+'\n')
    final_binding=require_ready(BASE)
    assert final_binding['binding_sha256']==evaluation_binding['binding_sha256']
    assert runtime.forbidden_calls==0
    (BASE/'pilot_outcome.json').write_text(json.dumps(dict(nominal=result,extension=extended,new_expert_query_launched=False,direct_absolute_target_trial=True,ordinary_final_global_step=fit.get('ordinary_final_step')),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
