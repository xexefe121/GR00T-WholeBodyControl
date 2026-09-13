"""Saved-array audit only: no imported runtime, inference, model, or dynamics."""
import argparse, ast, hashlib, json, sys
from pathlib import Path
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation
import scipy

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
RUN=NEW/'velocity_chord_student_evaluation_v1'
FIT=NEW/'velocity_chord_student_v1'
SRC=RUN/'source_snapshot_v1'
def local(p):
    s=str(p).replace('\\','/')
    if sys.platform!='win32' and len(s)>2 and s[1]==':':s='/mnt/'+s[0].lower()+s[2:]
    return Path(s)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
CHECKS=[]
def exact(a,b,name):
    passed=equal(a,b);CHECKS.append(dict(name=name,passed=passed))
    if not passed:raise AssertionError(name)
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,np.float64)**2)))
def first(controls,mask):
    indices=np.flatnonzero(mask)
    return int(controls[indices[0]]) if len(indices) else None

def paths():
    bundle=local('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1')
    p=dict(nominal_trace=RUN/'nominal/trace.npz',nominal_report=RUN/'nominal/report.json',
        nominal_request=RUN/'nominal/request.json',outcome=RUN/'pilot_outcome.json',
        evaluation_binding=RUN/'evaluation_binding.json',evaluation_frozen=RUN/'frozen_inputs_v2.json',
        contract=bundle/'contract.json',original29=bundle/'walk003/original29.npz',
        motion=NEW.parent/'sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz',
        baseline=NEW/'original_bfm_entry250_v1/entry250/trace.npz',
        labels=NEW/'bfm_entry250_labels_v1/labels/labels.npz',
        centers=FIT/'generation/centers.npz',probes=FIT/'generation/features.npy',
        normalization=NEW/'fast_controller_phase_fit_v1/fit/teacher_fit.npz',
        final_predictions=FIT/'fit/final_predictions.npz',fit_report=FIT/'fit/report.json',
        final_export_review=NEW/'velocity_chord_final_export_review_v1/review.json',
        root_fit_review=NEW/'velocity_fit_evidence_independent_v1/report.json',
        witness=RUN/'head_witness/witness.npz',witness_report=RUN/'head_witness/report.json',
        goal_source=SRC/'gear_sonic/utils/g1_true23_mpc_student.py',
        observation_source=SRC/'gear_sonic/utils/g1_true23_bfm_seed_observations.py',
        runtime_source=SRC/'student_linear_runtime.py',driver=SRC/'evaluate_velocity_chord_student.py',
        evidence_source=SRC/'proposal_evidence.py')
    for segment in ('nominal','post_lifecycle_hold_5s'):
        for name in ('trace.npz','report.json','request.json','rejected_precontrol.npz','strict_failure_state.npz'):
            q=RUN/segment/name
            if q.exists():p[segment+'_'+name.replace('.','_')]=q
    for name in ('canonical_prefix250_parity.json','actual_query250_input_parity.json','actual_query250_ownexport_output_parity.json'):
        q=RUN/name
        if q.exists():p[name]=q
    return p

def freeze():
    assert not (BASE/'request.json').exists()
    p=paths();report=read(p['nominal_report']);read(p['outcome'])
    assert sha(p['nominal_trace'])==report['trace_sha256']
    # Frozen evaluator bindings include all original sources and model inputs.
    frozen=read(p['evaluation_frozen'])
    for name,digest in frozen['source_sha256'].items():assert sha(SRC/name)==digest,name
    for name,digest in frozen['input_sha256'].items():assert sha(local(name))==digest,name
    inputs={str(path):sha(path) for path in p.values()}
    write(BASE/'request.json',dict(kind='one_saved_final70000_outcome_audit',source_sha256=sha(__file__),
        input_sha256=inputs,paths={k:str(v) for k,v in p.items()},
        inference_calls=0,BFM_calls=0,physics_steps=0,optimizer_updates=0,new_queries=0,
        scope='Actual attempted state/features/raw action history; saved same-clock expert comparisons; no physical qualification.'))

def pure_definitions(p):
    """Extract only reviewed pure expressions; never execute a runtime module."""
    space=dict(np=np,Rotation=Rotation,Sequence=Sequence,OFFSETS=np.asarray([0,1,2,4,8,16,24,37]),FEATURES=1023)
    wanted={p['goal_source']:{'GoalFeatures'},p['observation_source']:{'state_and_terms','_quaternion_matrix'}}
    for source,names in wanted.items():
        tree=ast.parse(source.read_text())
        body=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
        assert {node.name for node in body}==names
        exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),space)
    return space['GoalFeatures'],space['state_and_terms']

def audit():
    request=read(BASE/'request.json');assert sha(__file__)==request['source_sha256']
    for name,digest in request['input_sha256'].items():assert sha(local(name))==digest,name
    assert not (BASE/'report.json').exists() and sys.platform!='win32','Use pinned WSL NumPy/SciPy arithmetic.'
    assert np.__version__=='1.26.4'
    p={k:local(v) for k,v in request['paths'].items()}
    GoalFeatures,state_and_terms=pure_definitions(p)
    c=read(p['contract']);motion=load(p['motion']);goals=GoalFeatures(motion,load(p['original29']),c)
    default,kp,effort=[np.asarray(c[k]) for k in ('default_q','kp','training_effort')]
    limits=np.asarray(c['joint_limits']);names=c['joint_names']
    history={k:np.zeros((4,n),np.float32) for k,n in dict(actions=23,base_ang_vel=3,dof_pos=23,dof_vel=23,projected_gravity=3).items()}
    previous=np.zeros(23,np.float32);recorded=0;all_rows=[];named_rows={k:[] for k in history};segment_reports=[]
    def flat():return np.concatenate([history[k].reshape(-1) for k in sorted(history)]).copy()
    def proposal(q,dq,control,base,delta):
        sensed,terms=state_and_terms(q[7:],dq[6:],q[3:7],dq[3:6],previous,default)
        previous_target=np.clip(default+previous*.25*effort/kp,limits[:,0],limits[:,1])
        features=np.zeros(1069,np.float32) if control>=1269 else np.r_[goals(q,dq,previous_target,control+11),base-default,previous].astype(np.float32)
        raw=(np.asarray(base)+delta)
        target=np.clip(raw,limits[:,0],limits[:,1])
        recovered_base_action=((base-default)*kp/(.25*effort)).astype(np.float32)
        exact(default+recovered_base_action*.25*effort/kp,base,'base inverse round trip '+str(control))
        combined=(recovered_base_action+delta*kp/(.25*effort)).astype(np.float32)
        actual=((target-default)*kp/(.25*effort)).astype(np.float32)
        return dict(state=sensed,features=features,target=target,action=combined,raw_proposal=raw,actual_normalized_action=actual),terms,recovered_base_action
    def push(terms):
        for key in history:history[key][1:]=history[key][:-1].copy();history[key][0]=terms[key]
    segments=[('nominal',p['nominal_trace'])]
    hold=RUN/'post_lifecycle_hold_5s/trace.npz'
    if hold.exists():segments.append(('hold',hold))
    traces=[]
    for label,path in segments:
        a=load(path);traces.append(a);start=recorded;n=len(a['target']);pre=len(a['control_integration_before'])
        exact(a['global_control'],np.arange(start,start+n,dtype=a['global_control'].dtype),label+' continuous control clock')
        assert len(a['qpos'])==n+1 and len(a['qvel'])==n+1 and pre in (n,n+1)
        if label=='hold':
            exact(traces[0]['final_integration'],a['initial_integration'],'continuous hold integration')
            exact(traces[0]['qpos'][-1],a['qpos'][0],'continuous hold qpos')
            exact(traces[0]['qvel'][-1],a['qvel'][0],'continuous hold qvel')
        for i,control in enumerate(a['global_control']):
            control=int(control);assert control==recorded
            assert int(a['source_frame'][i])==min(control+11,len(motion['joint_pos'])-1)
            assert int(a['controller_mode'][i])==(0 if control<250 else (1 if control<1269 else 2))
            q,dq=a['qpos'][i],a['qvel'][i]
            exact(a['control_integration_before'][i,1:31],q,'integration qpos '+str(control))
            exact(a['control_integration_before'][i,31:60],dq,'integration qvel '+str(control))
            exact(previous,a['previous_action'][i],'prior '+str(control));exact(previous,a['control_previous_action_before'][i],'precontrol prior '+str(control))
            before=flat();exact(before,a['history'][i],'actor lag history '+str(control));exact(before,a['control_history_before'][i],'complete precontrol history '+str(control))
            for key in history:named_rows[key].append(history[key].copy())
            got,terms,recovered=proposal(q,dq,control,a['base_target'][i],a['delta'][i])
            for key,value in got.items():exact(value,a[key][i],key+' '+str(control))
            if control<250 or control>=1269:exact(a['delta'][i],np.zeros(23,np.float32),'disabled residual '+str(control))
            all_rows.append({**{key:a[key][i].copy() for key in ('features','previous_action','action','base_target','delta','target','raw_proposal','actual_normalized_action','state','history')},
                'control':control,'qpos':q.copy(),'qvel':dq.copy(),'physics_substeps':int(a['physics_substeps'][i]),'recovered_base_action':recovered})
            push(terms);previous=got['action'].copy();recorded+=1
        rejected=None
        if pre==n+1:
            rejected_path=path.parent/'rejected_precontrol.npz';assert rejected_path.exists()
            r=load(rejected_path);control=int(r['global_control']);assert control==recorded
            exact(previous,r['previous_action_before'],'rejected prior');exact(flat(),a['control_history_before'][-1],'rejected precontrol flat history')
            for key in history:exact(history[key],r['history_before_'+key],'rejected named before '+key)
            exact(r['qpos'],a['qpos'][-1],'rejected qpos');exact(r['qvel'],a['qvel'][-1],'rejected qvel')
            state,terms=state_and_terms(r['qpos'][7:],r['qvel'][6:],r['qpos'][3:7],r['qvel'][3:6],previous,default)
            if 'actor_input_state' in r:exact(state[None],r['actor_input_state'],'rejected actual actor state')
            if 'actor_input_history' in r:exact(flat()[None],r['actor_input_history'],'rejected actual actor history')
            if 'actor_input_last_action' in r:exact(previous[None],r['actor_input_last_action'],'rejected actual actor prior')
            if 'head_input_features' in r:
                assert 'actor_output_0' in r
                rejected_raw=r['actor_output_0'][0]*5
                rejected_base=default+rejected_raw*.25*effort/kp
                previous_target=np.clip(default+previous*.25*effort/kp,limits[:,0],limits[:,1])
                features=np.r_[goals(r['qpos'],r['qvel'],previous_target,control+11),rejected_base-default,previous].astype(np.float32)
                exact(features[None],r['head_input_features'],'rejected exact head input from returned actor output')
            if 'proposal_base_target' in r:
                got,terms,recovered=proposal(r['qpos'],r['qvel'],control,r['proposal_base_target'],r['proposal_delta'])
                for key,value in got.items():exact(value,r['proposal_'+key] if 'proposal_'+key in r else r[key],'rejected '+key)
            before={key:value.copy() for key,value in history.items()};push(terms)
            after_push=all(equal(history[k],r['history_after_'+k]) for k in history)
            before_same=all(equal(before[k],r['history_after_'+k]) for k in history)
            assert after_push or before_same,'Rejected history is neither original nor single exact measured update'
            if before_same:history=before
            recorded_after=int(r['recorded_controls_after']);assert recorded_after in (recorded,recorded+1)
            if recorded_after==recorded+1:
                assert 'proposal_action' in r;previous=r['proposal_action'].copy();recorded=recorded_after
            exact(previous,r['previous_action_after'],'rejected final prior')
            rejected=dict(control=control,proposal_outputs_available='proposal_base_target' in r,history_updated=after_push and not before_same,recorded_controls_after=recorded)
        exact(previous,a['final_previous_action'],label+' final prior')
        assert int(a['final_recorded_controls'])==recorded
        for key in history:exact(history[key],a['final_history_'+key],label+' final named history '+key)
        segment_reports.append(dict(segment=label,commands_with_native_steps=n,attempted_precontrols=pre,rejected=rejected))

    nominal=traces[0];baseline=load(p['baseline']);prefix_n=min(250,len(nominal['target']))
    for key in ('qpos','qvel','target','source_frame','global_control','controller_mode','joint_error','root_error','state','history','previous_action','action','base_target','delta','features','physics_qpos','physics_qvel','physics_torque','physics_actuator_torque','physics_time','physics_expected_time','physics_warning_counts','physics_warning_lastinfo','physics_substeps','range_excess','velocity_ratio','effort_ratio','clock_error','control_integration_before','control_history_before','control_previous_action_before'):
        if key in ('range_excess','velocity_ratio','effort_ratio','clock_error'):count=prefix_n*10
        else:count=prefix_n*10+(0 if key in ('physics_torque','physics_actuator_torque') else 1) if key.startswith('physics_') and key!='physics_substeps' else prefix_n+(1 if key in ('qpos','qvel') else 0)
        exact(nominal[key][:count],baseline[key][:count],'original BFM prefix '+key)
    labels=load(p['labels']);witness=load(p['witness']);centers=load(p['centers']);predictions=load(p['final_predictions'])
    query=next((row for row in all_rows if row['control']==250),None)
    if query:
        for key in ('features','base_target','previous_action','history','state'):exact(query[key],labels[key][0],'query250 '+key)
        exact(query['qpos'],labels['teacher_qpos'][0],'query250 qpos');exact(query['qvel'],labels['teacher_qvel'][0],'query250 qvel')
        exact(nominal['control_integration_before'][250],labels['control_integration_before'][0],'query250 integration')
        exact(nominal['control_integration_before'][250],baseline['final_integration'],'full baseline integration at250')
        exact(query['previous_action'],baseline['final_previous_action'],'full baseline prior at250')
        for key in named_rows:exact(named_rows[key][250],labels['history_'+key][0],'query250 named '+key)
        for key in named_rows:exact(named_rows[key][250],baseline['final_history_'+key],'full baseline final named '+key)
        exact(query['features'],witness['features'],'first activation witness input');exact(query['delta'],witness['onnx_delta'],'first activation witness output')
    # Last physical failure capsule includes direct raw actor returns: verify it
    # separately from the base-target round-trip used for earlier saved controls.
    for label,path in segments:
        capsule=path.parent/'strict_failure_state.npz'
        if capsule.exists():
            cap=load(capsule);row=next(v for v in all_rows if v['control']==int(cap['global_control']))
            exact(cap['actor_output_0'][0]*5,row['recovered_base_action'],'failure direct actor raw output')
            exact(cap['actor_input_state'][0],row['state'],'failure direct actor state')
            exact(cap['actor_input_history'][0],row['history'],'failure direct actor history')
            if 'head_output_0' in cap:exact(cap['head_output_0'][0],row['delta'],'failure direct head output')

    selected=[row for row in all_rows if 250<=row['control']<1269]
    controls=np.asarray([v['control'] for v in selected],np.int64);n=len(selected)
    diagnostic=[];arrays={}
    if n:
        x=np.asarray([v['features'] for v in selected]);norm=load(p['normalization']);std=norm['feature_std'].astype(np.float64)
        nominal_lo=centers['features'].min(0);nominal_hi=centers['features'].max(0)
        probes=np.load(p['probes'],mmap_mode='r').reshape(-1,1069)
        augmented_lo=np.minimum(nominal_lo,probes.min(0));augmented_hi=np.maximum(nominal_hi,probes.max(0))
        envelope=lambda lo,hi:np.maximum(np.maximum(lo-x,x-hi),0).astype(np.float64)/std
        nominal_excursion=envelope(nominal_lo,nominal_hi);augmented_excursion=envelope(augmented_lo,augmented_hi)
        clip=np.asarray([v['target']-v['raw_proposal'] for v in selected])
        action_difference=np.asarray([v['action']-v['actual_normalized_action'] for v in selected])
        prior_difference=np.asarray([v['previous_action']-all_rows[v['control']-1]['actual_normalized_action'] for v in selected])
        applied_action_history=np.asarray([[all_rows[int(control)-2-lag]['actual_normalized_action'] if int(control)-2-lag>=0 else np.zeros(23,np.float32) for lag in range(4)] for control in controls])
        raw_applied_history_difference=np.asarray(named_rows['actions'])[controls]-applied_action_history
        qindices=controls-250;matching_delta=predictions['predicted_delta'][2038+qindices]
        matched_target=np.clip(labels['base_target'][qindices]+matching_delta,limits[:,0],limits[:,1])
        actual_target=np.asarray([v['target'] for v in selected]);target_error=actual_target-labels['expert_target'][qindices]
        feature_delta=(x.astype(np.float64)-labels['features'][qindices])/std
        named_diff={key:np.asarray(named_rows[key])[controls]-labels['history_'+key][qindices] for key in named_rows}
        for i,row in enumerate(selected):
            control=row['control'];entry=dict(control=control,physics_substeps=row['physics_substeps'],
                nominal_envelope_outside_coordinates=int(np.count_nonzero(nominal_excursion[i])),augmented_envelope_outside_coordinates=int(np.count_nonzero(augmented_excursion[i])),
                augmented_max_standardized_excursion=float(augmented_excursion[i].max()),
                actual_vs_query250_same_clock_feature_rms=float(np.sqrt(np.mean(feature_delta[i]**2))),
                actual_vs_query250_same_clock_target_rmse_rad=rms(target_error[i]),
                saved_fit_at_query250_expert_input_rmse_rad=rms(matched_target[i]-labels['expert_target'][qindices[i]]),
                clipped_joints=[names[j] for j in np.flatnonzero(clip[i]!=0)],max_target_clipping_rad=float(np.abs(clip[i]).max()),
                raw_minus_applied_action_max=float(np.abs(action_difference[i]).max()),
                actual_vs_same_clock_joint_velocity_rms=float(rms(row['qvel'][6:]-labels['teacher_qvel'][qindices[i],6:])),
                same_clock_teacher_targets=[])
            for dataset in range(3):
                idx=dataset*1019+control-250
                entry['same_clock_teacher_targets'].append(dict(dataset=dataset,
                    actual_target_error_rms_rad=rms(row['target']-centers['expert_target'][idx]),
                    actual_input_standardized_rms=rms((row['features']-centers['features'][idx])/std)))
            diagnostic.append(entry)
        arrays=dict(control=controls,nominal_envelope_excursion=nominal_excursion,augmented_envelope_excursion=augmented_excursion,
            clipping_adjustment=clip,raw_minus_applied_action=action_difference,prior_minus_previous_applied_action=prior_difference,
            raw_action_history_minus_applied_action_history=raw_applied_history_difference,
            actual_minus_query250_same_clock_target=target_error,query250_same_clock_standardized_feature_delta=feature_delta,
            **{'same_clock_named_history_delta_'+key:value for key,value in named_diff.items()})
        events=dict(first_target_differs_from_query250_teacher=first(controls,np.any(target_error!=0,axis=1)),
            first_input_differs_from_query250_teacher=first(controls,np.any(feature_delta!=0,axis=1)),
            first_native_target_clipping=first(controls,np.any(clip!=0,axis=1)),
            first_raw_applied_action_byte_difference=first(controls,np.any(action_difference!=0,axis=1)),
            first_raw_applied_action_difference_above_1e_minus5=first(controls,np.any(np.abs(action_difference)>1e-5,axis=1)),
            first_prior_vs_previous_applied_difference_above_1e_minus5=first(controls,np.any(np.abs(prior_difference)>1e-5,axis=1)),
            first_raw_vs_applied_action_history_difference_above_1e_minus5=first(controls,np.any(np.abs(raw_applied_history_difference).reshape(n,-1)>1e-5,axis=1)),
            first_nominal_envelope_excursion=first(controls,np.any(nominal_excursion>0,axis=1)),
            first_augmented_training_envelope_excursion=first(controls,np.any(augmented_excursion>0,axis=1)),
            first_augmented_direct_velocity_excursion=first(controls,np.any(augmented_excursion[:,23:46]>0,axis=1)),
            first_augmented_raw_prior_excursion=first(controls,np.any(augmented_excursion[:,-23:]>0,axis=1)),
            first_named_history_vs_query250_teacher={key:first(controls,np.any(value.reshape(n,-1)!=0,axis=1)) for key,value in named_diff.items()})
    else:events={}
    for name,digest in request['input_sha256'].items():assert sha(local(name))==digest,name
    failure_states=[]
    for (label,path),a in zip(segments,traces):
        failure=read(path.parent/'report.json')['failure']
        if failure and len(a['physics_qvel']):
            ratios=np.abs(a['physics_qvel'][-1,6:])/np.asarray(c['native_velocity']);j=int(ratios.argmax())
            failure_states.append(dict(segment=label,producer_failure=failure,max_speed_joint=names[j],joint_index=j,
                joint_velocity_radps=float(a['physics_qvel'][-1,6+j]),speed_cap_radps=float(c['native_velocity'][j]),
                speed_ratio=float(ratios[j]),qpos=float(a['physics_qpos'][-1,7+j])))
    np.savez_compressed(BASE/'arrays.npz',**arrays)
    write(BASE/'report.json',dict(passed=True,request_sha256=sha(BASE/'request.json'),source_sha256=sha(__file__),
        checks=len(CHECKS),segments=segment_reports,applied_controls=len(all_rows),moving_controls=n,
        original_BFM_prefix_controls_verified=prefix_n,query250_exact=query is not None,
        history_convention='Original raw combined action; normalized applied target is diagnostic only.',
        full_named_final_history_exact=True,all_available_attempted_state_features_history_actions_exact=True,
        base_action_provenance='Earlier raw actor actions recovered from recorded float64 base target with exact round trip; final failure capsule additionally checks direct recorded actor output.',
        events=events,failure_states=failure_states,rows=diagnostic,
        training_envelope='Both3057 nominal centers and full143679 nominal+probe feature universe; coordinate ranges are descriptive, not a feasibility or stability gate.',
        comparison_limitation='Same-clock targets and saved head predictions belong to distinct expert trajectories after control250; no expert counterfactual evaluated at actual student states.',
        physics_qualification='Root independent replay and intent audits own physical qualification; this audit performs no dynamics.',
        runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
        arrays_sha256=sha(BASE/'arrays.npz'),all_frozen_inputs_unchanged=True,
        inference_calls=0,BFM_calls=0,physics_steps=0,optimizer_updates=0,new_queries=0))
    write(BASE/'checks.json',CHECKS)
    print(json.dumps(dict(passed=True,controls=len(all_rows),moving_controls=n,events=events)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:freeze()
    else:
        try:audit()
        except BaseException as error:
            write(BASE/'audit_failure.json',dict(error=repr(error),completed_checks=CHECKS,inference_calls=0,physics_steps=0));raise
