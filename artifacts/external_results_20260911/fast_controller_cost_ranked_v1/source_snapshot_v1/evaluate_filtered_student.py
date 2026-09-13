"""One native lifecycle ranking four feasible forecasts by original tracking cost."""
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from student_linear_runtime import *
from evaluate_phase_student import get_state,assess,new_trace,source_metrics
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override,motion_states
from quiet_metrics import standing_windows,quiet_diagnostic
from transactional_student import TransactionalStudent
from ranked_admission import admit,CANDIDATE_NAMES
from native_forecast import NativeForecast
from prefix_tracking_cost import PrefixTrackingCost,COMPONENTS

PRIOR=BASE.parent/'fast_controller_phase_fit_v1'
EXTRA_FIELDS=('raw_bfm_action','proposed_target','proposed_action','applied_normalized_action','applied_delta',
    'primary_applied_unchanged','selected_candidate','candidate_count','forecast_count','admission_ms','control_loop_ms',
    'selected_prefix_cost','cost_evaluation_ms')

def write(path,value):path.write_text(json.dumps(finite_json(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')

def trace_arrays(trace):
    """Keep the native trace schema even when no command was admitted."""
    width=dict(qpos=30,qvel=29,target=23,joint_error=23,root_error=3,state=52,history=300,
        previous_action=23,action=23,base_target=23,delta=23,features=1069,physics_qpos=30,physics_qvel=29,
        physics_torque=23,physics_actuator_torque=23,physics_warning_counts=8,physics_warning_lastinfo=8,
        control_integration_before=291,control_history_before=300,control_previous_action_before=23,
        raw_bfm_action=23,proposed_target=23,proposed_action=23,applied_normalized_action=23,applied_delta=23)
    f32={'state','history','previous_action','action','delta','features','control_history_before',
        'control_previous_action_before','raw_bfm_action','proposed_action','applied_normalized_action'}
    i64={'source_frame','global_control','controller_mode','physics_substeps','selected_candidate','candidate_count','forecast_count'}
    i32={'physics_warning_counts','physics_warning_lastinfo'}
    result={}
    for key,values in trace.items():
        dtype=np.float32 if key in f32 else np.int64 if key in i64 else np.int32 if key in i32 else np.bool_ if key=='primary_applied_unchanged' else np.float64
        array=np.asarray(values,dtype=dtype)
        result[key]=array.reshape(-1,width[key]) if key in width else array.reshape(-1)
    controls=len(result['target']);steps=len(result['physics_torque'])
    for key,value in result.items():
        expected=steps+1 if key in {'physics_qpos','physics_qvel','physics_time','physics_expected_time','physics_warning_counts','physics_warning_lastinfo'} else steps if (key.startswith('physics_') and key!='physics_substeps') or key in {'range_excess','velocity_ratio','effort_ratio','clock_error'} else controls+1 if key in {'qpos','qvel'} else controls
        assert len(value)==expected,(key,len(value),expected)
    return result


def main():
    frozen=assert_frozen();assert mujoco.__version__=='3.2.3'
    clearance= json.loads((BASE/'runtime_clearance.json').read_text())
    assert clearance['canonical_rollout_authorized'] is True and clearance['canonical_trials']==1
    assert clearance['head_sha256']==sha(PRIOR/'fit/student_head.onnx')
    assert clearance['frozen_sources_sha256']==sha(FROZEN_RECEIPT)
    assert clearance['private_horizon_controls']==5 and clearance['actual_commit_controls']==1
    assert clearance['selection']=='lowest_exact_feasible_prefix_cost' and clearance['evaluate_all_four'] is True
    assert not (BASE/'nominal').exists()
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,'walk003')
    motion,override=load_motion_override(REFERENCE,BUNDLE,'walk003',native,c,original,timeline,manifest)
    original29=archive(BUNDLE/'walk003/original29.npz');audit=json.loads((TEACHER/'recorded_source_audit_v2.json').read_text())
    baseline=archive(BASE.parent/'original_bfm_entry250_v1/entry250/trace.npz')
    baseline_report=json.loads((BASE.parent/'original_bfm_entry250_v1/entry250/report.json').read_text())
    assert baseline_report['full_segment_completed'] and baseline_report['quiet_standing_diagnostic']['quiet_standing_diagnostic_pass']
    assert baseline_report['trace_sha256']==sha(BASE.parent/'original_bfm_entry250_v1/entry250/trace.npz')
    labels=archive(BASE.parent/'bfm_entry250_labels_v1/labels/labels.npz')
    label_report=json.loads((BASE.parent/'bfm_entry250_labels_v1/labels/report.json').read_text())
    assert label_report['samples']==1019 and label_report['labels_sha256']==sha(BASE.parent/'bfm_entry250_labels_v1/labels/labels.npz')
    np.testing.assert_array_equal(labels['control'],np.arange(250,1269))
    previous_run=archive(PRIOR/'nominal/trace.npz')
    assert sha(PRIOR/'nominal/trace.npz')=='bec4809cdb19c9b153d726485b7a77d2640c7bd399362681801bbd6abad086b9'
    fit=json.loads((PRIOR/'fit/report.json').read_text())
    assert fit['steps']==65000 and fit['full_objective_improved'] and fit['export_parity_passed']
    assert sha(PRIOR/'fit/student_head.onnx')==fit['checkpoints']['student_head.onnx']
    runtime=LinearStudentRuntime(native,c,original,motion,original29,PRIOR/'fit/student_head.onnx')
    transaction=TransactionalStudent(runtime);forecast=NativeForecast(native,c,maximum_controls=5)
    tracking_cost=PrefixTrackingCost(native,c,motion)
    np.testing.assert_array_equal(forecast.limits,runtime.limits)
    initial=motion_states(motion)[10];data=mujoco.MjData(native);data.qpos[:]=initial[:30];data.qvel[:]=initial[30:];mujoco.mj_forward(native,data)
    kp,kd,effort=[np.asarray(c[key]) for key in ('kp','kd','native_effort')]
    expected=float(data.time);assert not assess(data,c,expected)[0]
    switch=next(p['control_start'] for p in timeline['phases'] if p['name']=='returned_standing')
    assert switch==1269 and len(motion['joint_pos'])-11==1569
    output=BASE/'nominal';output.mkdir(exist_ok=False)
    last_applied=None;reference_observations=[]
    common=dict(kind='native23_fixed65000_cost_ranked_admission_student',clip='walk003',motion_override=override,
        model_manifest=manifest,head_sha256=sha(PRIOR/'fit/student_head.onnx'),frozen_sources_sha256=sha(FROZEN_RECEIPT),
        runtime_clearance_sha256=sha(BASE/'runtime_clearance.json'),candidate_order=list(CANDIDATE_NAMES),
        candidate_selection='lowest exact original five-control prefix state+input cost among complete feasible candidates',
        evaluate_all_four_including_duplicates=True,exact_cost_ties='first in original order',
        original_H30_objective_truncated_to_state0_to5_and_input0_to4=True,full_H30_cost_not_evaluated=True,
        original_cost_weights_and_v4_reference_unchanged=True,cost_runtime_optimization_calls=0,
        standing_bias_and_control_latency_require_complete_source_and_timing_audits=True,
        private_horizon_controls=5,private_horizon_seconds=.1,actual_commit_controls=1,actual_commit_seconds=.02,
        physical_hz=500,control_hz=50,initial_unfiltered_BFM_controls=250,terminal_yaw4_start=1269,
        moving_history='normalized actually selected native target; no +/-5 action clamp',
        terminal_history='raw BFM action when its target is applied unchanged; selected-target normalization otherwise',
        prefix_history='original raw BFM action',proposed_fields='delta,proposed_target,proposed_action,raw_bfm_action',
        applied_fields='target,action,applied_normalized_action,applied_delta',
        synchronous_frozen_plant_during_inference_and_preview=True,private_preview_calls_concurrent=False,
        live_asynchronous_timing_or_stream_qualified=False,simulation_privileged_root_pose_velocity=True,
        source_frames_removed=0,physical_statewrites_after_initialization=0,recorded_expert_targets_applied=False,
        learned_weights_changed=False,new_expert_queries=False,hardware_authorized=False)

    class TransitionParityError(RuntimeError):pass

    def parity(path,pairs):
        checks={key:bool(np.array_equal(actual,wanted)) for key,actual,wanted in pairs}
        passed=all(checks.values());write(path,dict(passed=passed,checks=checks,comparison_only_no_injection=True))
        if not passed:raise TransitionParityError(str(path.name)+' failed:'+str([key for key,val in checks.items() if not val]))

    def prefix250(trace):
        fields=('qpos','qvel','target','source_frame','global_control','controller_mode','joint_error','root_error',
            'state','history','previous_action','action','base_target','delta','features','physics_qpos','physics_qvel',
            'physics_torque','physics_actuator_torque','physics_time','physics_expected_time','physics_warning_counts',
            'physics_warning_lastinfo','physics_substeps','range_excess','velocity_ratio','effort_ratio','clock_error',
            'control_integration_before','control_history_before','control_previous_action_before')
        pairs=[(key,np.asarray(trace[key])[:len(baseline[key])],baseline[key]) for key in fields]
        pairs.extend([('final_integration',get_state(native,data),baseline['final_integration']),
            ('recorded_controls',np.asarray(runtime.seed.recorded_controls),baseline['final_recorded_controls']),
            ('final_previous_action',runtime.seed.previous_action,baseline['final_previous_action'])])
        pairs.extend(('final_history_'+key,value,baseline['final_history_'+key]) for key,value in runtime.seed.history.data.items())
        parity(BASE/'canonical_prefix250_parity.json',pairs)

    def query250(proposed):
        pairs=[(key,proposed[key],labels[key][0]) for key in ('features','base_target','previous_action','history','state')]
        pairs.extend([('integration',get_state(native,data),labels['control_integration_before'][0]),
            ('qpos',data.qpos,labels['teacher_qpos'][0]),('qvel',data.qvel,labels['teacher_qvel'][0])])
        parity(BASE/'actual_query250_input_parity.json',pairs)
        parity(BASE/'query250_frozen_head_proposal_parity.json',[(key,proposed[key],previous_run[key][250])
            for key in ('target','delta','base_target','features','history','state','previous_action','action')])

    def save_forecasts(dest,copies,metadata):
        arrays=dict(control=np.asarray([item['control'] for item in metadata],np.int64),
            candidate=np.asarray([item['candidate'] for item in metadata],np.int64),
            target=np.asarray([item['target'] for item in metadata],np.float64).reshape(-1,23),
            state_offsets=np.r_[0,np.cumsum([len(item['physics_qpos']) for item in copies],dtype=np.int64)],
            step_offsets=np.r_[0,np.cumsum([len(item['physics_torque']) for item in copies],dtype=np.int64)])
        shapes=dict(physics_qpos=(0,30),physics_qvel=(0,29),physics_time=(0,),physics_torque=(0,23),
            physics_actuator_force=(0,23),warning_counts=(0,8),warning_lastinfo=(0,8))
        for key in ('physics_qpos','physics_qvel','physics_time','physics_torque','physics_actuator_force','warning_counts','warning_lastinfo'):
            arrays[key]=np.concatenate([item[key] for item in copies]) if copies else np.empty(shapes[key],dtype=np.int32 if key.startswith('warning_') else np.float64)
        np.savez_compressed(dest/'private_forecasts.npz',**arrays)

    def save_costs(dest,copies,metadata):
        shapes=dict(state=(6,59),features=(6,101),residual=(6,157),component_cost=(6,7),state_cost=(6,),
            input_cost=(5,),target_reference=(5,23),state_goal_frame=(6,),input_goal_frame=(5,))
        arrays={key:np.full((len(copies),*shape),-1 if 'frame' in key else np.nan,
            dtype=np.int64 if 'frame' in key else np.float64) for key,shape in shapes.items()}
        arrays['cost_available']=np.zeros(len(copies),bool);arrays['state_knot_available']=np.zeros((len(copies),6),bool)
        arrays['control']=np.asarray([item['control'] for item in metadata],np.int64)
        arrays['candidate']=np.asarray([item['candidate'] for item in metadata],np.int64)
        for index,values in enumerate(copies):
            if values is None:continue
            count=len(values['state']);arrays['cost_available'][index]=True;arrays['state_knot_available'][index,:count]=True
            for key,value in values.items():arrays[key][index,:len(value)]=value
        np.savez_compressed(dest/'candidate_costs.npz',**arrays)

    def segment(dest,start,count,expected_time):
        nonlocal last_applied
        request=dict(common,initial_control=start,requested_controls=count,segment='original_lifecycle' if start==0 else 'separate5s_continuous_terminal_hold')
        write(dest/'request.json',request)
        trace=new_trace(data);trace['physics_expected_time'][0]=expected_time
        for key in EXTRA_FIELDS:trace[key]=[]
        initial_vector=get_state(native,data);segment_time=float(data.time);started=time.perf_counter()
        ledger=[];private_copies=[];private_meta=[];cost_copies=[];failure=None;selected_forecast_checks=0
        for control in range(start,start+count):
            loop_start=time.perf_counter();physical_elapsed=0.
            before=get_state(native,data);warning_before=data.warning.number.copy();info_before=data.warning.lastinfo.copy()
            named_history_before={key:value.copy() for key,value in runtime.seed.history.data.items()}
            history_before=np.concatenate([named_history_before[key].reshape(-1) for key in sorted(named_history_before)]).copy()
            previous_before=runtime.seed.previous_action.copy();recorded_before=runtime.seed.recorded_controls
            qpos_before=data.qpos.copy();qvel_before=data.qvel.copy()
            chosen_trace=None;records=[];admission_start=None;admission_ms=0.;proposal=None;candidate=-1;forecast_count=0
            selected_cost=0.;cost_elapsed=0.
            def save_rejected():
                arrays={} if proposal is None else {key:value for key,value in proposal.items() if isinstance(value,np.ndarray)}
                np.savez_compressed(dest/'rejected_proposal.npz',**arrays,control=np.asarray(control,np.int64),
                    integration_before=before,warning_before=warning_before,warning_lastinfo_before=info_before,
                    actual_qpos_before=qpos_before,actual_qvel_before=qvel_before,
                    actual_history_before=history_before,actual_previous_action_before=previous_before,
                    actual_recorded_controls_before=np.asarray(recorded_before,np.int64),
                    **{'actual_history_before_'+key:value for key,value in named_history_before.items()})
            try:
                reasons,_=assess(data,c,expected_time)
                if reasons:raise RuntimeError('Invalid precontrol native state:'+str(reasons))
                if control==250:prefix250(trace)
                proposal=transaction.begin(control,data.qpos,data.qvel,terminal=control>=switch,disable_head=control<250)
                if control==250:query250(proposal)
                if control<250:
                    target=proposal['target'].copy()
                else:
                    assert last_applied is not None
                    candidates=[proposal['target'].copy(),np.clip(proposal['base_target'],runtime.limits[:,0],runtime.limits[:,1]),
                        last_applied.copy(),np.clip(data.qpos[7:],runtime.limits[:,0],runtime.limits[:,1])]
                    admission_start=time.perf_counter()
                    def predict(candidate_index,target):
                        nonlocal forecast_count,cost_elapsed
                        report,views=forecast.predict(data,target,horizon_controls=5)
                        # Retain each candidate before the reusable views expire.
                        saved={key:value.copy() for key,value in views.items()}
                        index=len(private_copies);private_copies.append(saved)
                        private_meta.append(dict(control=control,candidate=candidate_index,target=target.copy()))
                        cost_copies.append(None)
                        forecast_count+=1
                        score,cost_arrays=tracking_cost.score(control,target,saved,report['feasible'])
                        cost_copies[index]=cost_arrays;cost_elapsed+=score['cost_evaluation_ms']
                        return dict(report,forecast_index=index,tracking_cost=score)
                    candidate,target,records=admit(candidates,predict)
                    assert len(records)==forecast_count==4
                    admission_ms=(time.perf_counter()-admission_start)*1000
                    np.testing.assert_array_equal(get_state(native,data),before)
                    np.testing.assert_array_equal(data.warning.number,warning_before);np.testing.assert_array_equal(data.warning.lastinfo,info_before)
                    if candidate is None:
                        save_rejected()
                        transaction.cancel()
                        ledger.append(dict(control=control,selected=None,records=records,admission_ms=admission_ms,
                            cost_evaluation_ms=cost_elapsed,
                            control_loop_ms=(time.perf_counter()-loop_start)*1000,actual_command_applied=False,
                            actual_history_and_count_unchanged=True))
                        failure=dict(reasons=['no_feasible_admission_candidate'],global_control=control,local_control=control-start,
                            substep=0,time=float(data.time),candidate_count=len(records),executed_rejected_controls=0)
                        break
                    selected_report=records[candidate]['witness']
                    assert selected_report is not None and selected_report['feasible'] and selected_report['physics_steps']==50
                    chosen_trace=private_copies[selected_report['forecast_index']]
                    selected_cost=selected_report['tracking_cost']['five_control_prefix_state_and_input_sum']
                proposed=transaction.commit(target)
            except Exception as exc:
                save_rejected();cancel_error=None
                if transaction.pending is not None:
                    try:transaction.cancel()
                    except Exception as cancellation:cancel_error=str(cancellation)
                if admission_start is not None:admission_ms=(time.perf_counter()-admission_start)*1000
                failure=dict(reasons=['transition_parity_fault' if isinstance(exc,TransitionParityError) else 'controller_admission_fault'],
                    global_control=control,local_control=control-start,substep=0,time=float(data.time),exception_type=type(exc).__name__,message=str(exc),cancellation_error=cancel_error)
                ledger.append(dict(control=control,selected=None,records=records,admission_ms=admission_ms,
                    cost_evaluation_ms=cost_elapsed,
                    control_loop_ms=(time.perf_counter()-loop_start)*1000,actual_command_applied=False,exception=failure))
                break
            # Only admitted commands enter actual-control arrays; rejections have a separate full snapshot.
            trace['control_integration_before'].append(before)
            trace['control_history_before'].append(history_before)
            trace['control_previous_action_before'].append(previous_before)
            actual_substeps=0;physical_before=time.perf_counter()
            for sub in range(10):
                data.ctrl[:]=np.clip(kp*(target-data.qpos[7:])-kd*data.qvel[6:],-effort,effort)
                mujoco.mj_step(native,data);expected_time+=.002;actual_substeps+=1
                for key,value in (('physics_qpos',data.qpos),('physics_qvel',data.qvel),('physics_torque',data.ctrl),
                    ('physics_actuator_torque',data.qfrc_actuator[6:]),('physics_warning_counts',data.warning.number),('physics_warning_lastinfo',data.warning.lastinfo)):
                    trace[key].append(value.copy())
                trace['physics_time'].append(float(data.time));trace['physics_expected_time'].append(expected_time)
                reasons,metrics=assess(data,c,expected_time)
                for key in ('range_excess','velocity_ratio','effort_ratio','clock_error'):trace[key].append(metrics[key])
                mismatch=[]
                if chosen_trace is not None:
                    compare=(('qpos',data.qpos,chosen_trace['physics_qpos'][sub+1]),('qvel',data.qvel,chosen_trace['physics_qvel'][sub+1]),
                        ('command',data.ctrl,chosen_trace['physics_torque'][sub]),('force',data.qfrc_actuator[6:],chosen_trace['physics_actuator_force'][sub]),
                        ('time',np.asarray(data.time),chosen_trace['physics_time'][sub+1]),
                        ('warning_counts',data.warning.number,chosen_trace['warning_counts'][sub+1]),
                        ('warning_lastinfo',data.warning.lastinfo,chosen_trace['warning_lastinfo'][sub+1]))
                    mismatch=[key for key,actual,wanted in compare if not np.array_equal(actual,wanted)]
                    selected_forecast_checks+=1
                if mismatch:reasons.append('selected_forecast_actual_mismatch')
                if reasons:
                    failure=dict(reasons=reasons,global_control=control,local_control=control-start,substep=sub+1,
                        time=float(data.time),forecast_mismatch_fields=mismatch,**metrics)
                    break
            physical_elapsed=(time.perf_counter()-physical_before)*1000
            mujoco.mj_kinematics(native,data);last_applied=target.copy()
            frame=min(control+11,len(motion['joint_pos'])-1)
            values=dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),source_frame=frame,global_control=control,
                controller_mode=2 if control>=switch else (0 if control<250 else 1),physics_substeps=actual_substeps,
                joint_error=data.qpos[7:]-motion['joint_pos'][frame],root_error=data.qpos[:3]-motion['body_pos_w'][frame,0],
                selected_candidate=candidate,candidate_count=len(records),forecast_count=forecast_count,admission_ms=admission_ms,
                selected_prefix_cost=selected_cost,cost_evaluation_ms=cost_elapsed,**proposed)
            for key,value in values.items():trace[key].append(value)
            # Diagnostic comparison only after the deliberate action-history change.
            if 250<=control<=252:
                reference_observations.append(dict(control=control,selected_candidate=candidate,
                    primary_selected=candidate==0,precontrol_qpos_equal=bool(np.array_equal(qpos_before,previous_run['qpos'][control])),
                    precontrol_qvel_equal=bool(np.array_equal(qvel_before,previous_run['qvel'][control])),
                    proposal_target_equal=bool(np.array_equal(proposal['target'],previous_run['target'][control])),
                    applied_target_equal=bool(np.array_equal(target,previous_run['target'][control])),
                    proposed_action_equal=bool(np.array_equal(proposal['action'],previous_run['action'][control])),
                    selected_history_action_equal=bool(np.array_equal(proposed['action'],previous_run['action'][control]))))
            elapsed=(time.perf_counter()-loop_start)*1000;trace['control_loop_ms'].append(elapsed)
            ledger.append(dict(control=control,selected=candidate,selected_name='unfiltered_initial_BFM' if candidate<0 else CANDIDATE_NAMES[candidate],
                records=records,admission_ms=admission_ms,proposal_ms=proposal['inference_ms'],actual_physics_and_checks_ms=physical_elapsed,
                selected_prefix_cost=selected_cost,cost_evaluation_ms=cost_elapsed,
                control_loop_ms=elapsed,actual_command_applied=True,actual_substeps=actual_substeps,
                selected_forecast_first10_bitexact=chosen_trace is not None and not failure))
            if failure:break
            if (control+1)%100==0:print(json.dumps(dict(control=control+1,selected=candidate,elapsed_seconds=time.perf_counter()-started)),flush=True)
        arrays=trace_arrays(trace)
        np.savez_compressed(dest/'trace.npz',**arrays,initial_integration=initial_vector,final_integration=get_state(native,data),
            integration_state_spec=np.asarray(int(mujoco.mjtState.mjSTATE_INTEGRATION),dtype=np.int64),
            final_previous_action=runtime.seed.previous_action,final_recorded_controls=np.asarray(runtime.seed.recorded_controls),
            **{'final_history_'+key:value for key,value in runtime.seed.history.data.items()})
        save_forecasts(dest,private_copies,private_meta);write(dest/'admission_ledger.json',ledger)
        assert len(cost_copies)==len(private_copies);save_costs(dest,cost_copies,private_meta)
        if start==0:write(BASE/'early_prior_phase_comparison.json',dict(mandatory=False,diagnostic_only=True,rows=reference_observations,
            moving_action_arithmetic_intentionally_changed=True,no_action_arithmetic_exceptions=True))
        complete=int(np.sum(arrays['physics_substeps']==10));strict=failure is None;quiet=None
        if len(arrays['target']) and (start>0 or complete==count):quiet=quiet_diagnostic(standing_windows(arrays,motion,original,original29),strict)
        maximum=lambda key:float(np.max(arrays[key])) if len(arrays[key]) else 0.
        percentile=lambda key:np.percentile(arrays[key],[50,95,100]).tolist() if len(arrays[key]) else []
        attempted_times=np.asarray([item['control_loop_ms'] for item in ledger],np.float64)
        attempted_admission_times=np.asarray([item['admission_ms'] for item in ledger if item['control']>=250],np.float64)
        ranked_loop_times=np.asarray([item['control_loop_ms'] for item in ledger if item['control']>=250],np.float64)
        result=dict(kind=common['kind'],clip='walk003',mujoco=mujoco.__version__,segment=request['segment'],requested_controls=count,
            completed_full_controls=complete,completed_controls=complete,attempted_controls=len(arrays['target']),
            partial_substeps=len(arrays['physics_torque'])%10,physics_steps=len(arrays['physics_torque']),
            full_segment_completed=complete==count and strict,probe_completed=complete==count and strict,failure=failure,
            range_excess_max=maximum('range_excess'),velocity_ratio_max=maximum('velocity_ratio'),effort_ratio_max=maximum('effort_ratio'),
            engine_warning_counts=data.warning.number.tolist(),engine_warning_lastinfo=data.warning.lastinfo.tolist(),
            independent_accumulated_clock_max_error=maximum('clock_error'),policy_ms_p50_p95_max=percentile('control_loop_ms'),
            proposal_ms_p50_p95_max=percentile('inference_ms'),admission_ms_p50_p95_max=percentile('admission_ms'),
            policy_20ms_deadline_misses=int(np.sum(arrays['control_loop_ms']>20)),
            attempted_policy_ms_p50_p95_max=np.percentile(attempted_times,[50,95,100]).tolist() if len(attempted_times) else [],
            attempted_admission_ms_p50_p95_max=np.percentile(attempted_admission_times,[50,95,100]).tolist() if len(attempted_admission_times) else [],
            attempted_policy_20ms_deadline_misses=int(np.sum(attempted_times>20)),
            attempted_admissions=len(attempted_admission_times),attempted_control_transactions=len(ledger),
            ranked_attempted_policy_ms_p50_p95_max=np.percentile(ranked_loop_times,[50,95,100]).tolist() if len(ranked_loop_times) else [],
            ranked_attempted_policy_20ms_deadline_misses=int(np.sum(ranked_loop_times>20)),
            cost_evaluation_ms_p50_p95_max=percentile('cost_evaluation_ms'),
            selection='lowest_exact_feasible_prefix_cost',all_four_candidates_each_admission=True,
            tracking_cost_component_order=list(COMPONENTS),runtime_optimizer_calls=0,
            timing_scope='Per-control proposals, forecasts and evidence copies, selection, actual physics/checks, and trace/ledger preparation; final compressed persistence included in elapsed_seconds only.',
            selected_forecast_actual_substeps_checked=selected_forecast_checks,
            private_forecasts=len(private_copies),private_physics_steps=sum(len(item['physics_torque']) for item in private_copies),
            selected_candidate_counts={name:int(np.sum(arrays['selected_candidate']==index)) for index,name in enumerate(CANDIDATE_NAMES)},
            source_metrics=source_metrics(native,arrays,motion,original29,audit) if len(arrays['target']) else dict(source_controls=0,no_source_samples=True),
            quiet_standing_diagnostic=quiet,simulation_start_time=segment_time,simulation_end_time=float(data.time),
            simulated_seconds=float(data.time)-segment_time,full_body_teleoperation_qualified=False,hardware_authorized=False,
            elapsed_seconds=time.perf_counter()-started,trace_sha256=sha(dest/'trace.npz'),request_sha256=sha(dest/'request.json'),
            private_forecasts_sha256=sha(dest/'private_forecasts.npz'),admission_ledger_sha256=sha(dest/'admission_ledger.json'),
            candidate_costs_sha256=sha(dest/'candidate_costs.npz'),
            motion_override=override,no_real_state_rollback=True,closed_loop_online_delay_qualification=False)
        write(dest/'report.json',result);print(json.dumps(finite_json(result)),flush=True)
        return result,expected_time
    nominal,expected=segment(output,0,1569,expected)
    extension=BASE/'post_lifecycle_hold_5s';extension.mkdir(exist_ok=False)
    if nominal['full_segment_completed']:hold,expected=segment(extension,1569,250,expected)
    else:
        hold=dict(requested_controls=250,attempted_controls=0,full_segment_completed=False,
            not_run_reason='Original lifecycle incomplete or failed; no reset or skip to terminal.')
        write(extension/'report.json',hold)
    write(BASE/'pilot_outcome.json',dict(nominal=nominal,extension=hold,one_fixed_controller_experiment=True,
        ordinary_final_global_step=65000,new_fitting=False,new_expert_query=False,hardware_authorized=False))

if __name__=='__main__':main()
