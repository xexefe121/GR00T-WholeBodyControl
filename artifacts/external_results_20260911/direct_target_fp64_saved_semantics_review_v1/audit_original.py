"""One pure saved direct-outcome audit. No model/runtime imports or inference."""
import ast,hashlib,json,sys
from pathlib import Path
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
def local(p):
    s=str(p).replace('\\','/')
    return Path('/mnt/'+s[0].lower()+s[2:] if sys.platform!='win32' and len(s)>2 and s[1]==':' else s)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def paths():
    run=NEW/'direct_target_student_evaluation_v1';src=run/'source_draft_v1'
    bundle=local('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1')
    p=dict(trace=run/'nominal/trace.npz',report=run/'nominal/report.json',failure=run/'nominal/strict_failure_state.npz',
        frozen=run/'frozen_inputs_v2.json',owner=run/'evaluation_completion_verification.json',hold=run/'post_lifecycle_hold_5s/report.json',
        physics=NEW/'direct_target_independent_physics_v1/report.json',contract=bundle/'contract.json',original29=bundle/'walk003/original29.npz',
        motion=NEW.parent/'sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz',
        baseline=NEW/'original_bfm_entry250_v1/entry250/trace.npz',labels=NEW/'bfm_entry250_labels_v1/labels/labels.npz',
        centers=NEW/'velocity_chord_student_v1/generation/centers.npz',witness=run/'head_witness/witness.npz',witness_report=run/'head_witness/report.json',
        norm=NEW/'direct_target_student_v1/fit/normalization.npz',feature_source=src/'direct_features.py',
        observation_source=src/'gear_sonic/utils/g1_true23_bfm_seed_observations.py',runtime_source=src/'direct_runtime.py')
    for name in ('canonical_initial_full291_parity','canonical_prefix250_parity','actual_query250_input_parity','actual_query250_ownexport_output_parity'):
        p[name]=run/(name+'.json')
    return p
def freeze():
    p=paths();assert sha(p['trace'])=='449a31d17aa7df6b23c20d6b0394237fc36b97d779f46544e6375975e0cdf427'
    assert sha(p['physics'])=='8538c0187f881a0662eaa2ee0f970bc69881921a24709575c571fec2b333db8c'
    frozen=read(p['frozen'])
    for key,relative in [('feature_source','direct_features.py'),('observation_source','gear_sonic/utils/g1_true23_bfm_seed_observations.py'),('runtime_source','direct_runtime.py')]:
        assert sha(p[key])==frozen['source_sha256'][relative]
    write(BASE/'request.json',dict(kind='one_pure_direct_target_saved_outcome_audit',source_sha256=sha(__file__),
        paths={k:str(v) for k,v in p.items()},input_sha256={str(v):sha(v) for v in p.values()},
        model_calls=0,BFM_calls=0,physics_steps=0,optimizer_updates=0,scope='316 actual controls,66 learned, no recovered trajectory'))
def pure(p):
    scope=dict(np=np,Rotation=Rotation,Sequence=Sequence,OFFSETS=np.array([0,1,2,4,8,16,24,37],np.int64),FEATURES=1000)
    for key,names in [('feature_source',{'DirectFeatures'}),('observation_source',{'state_and_terms','_quaternion_matrix'})]:
        tree=ast.parse(p[key].read_text());body=[v for v in tree.body if isinstance(v,(ast.ClassDef,ast.FunctionDef)) and v.name in names]
        assert {v.name for v in body}==names
        exec(compile(ast.Module(body=body,type_ignores=[]),str(p[key]),'exec'),scope)
    return scope['DirectFeatures'],scope['state_and_terms']
def audit():
    assert sys.platform!='win32' and np.__version__=='1.26.4','Pinned WSL NumPy arithmetic required'
    req=read(BASE/'request.json');reqsha=sha(BASE/'request.json');assert sha(__file__)==req['source_sha256']
    for path,digest in req['input_sha256'].items():assert sha(local(path))==digest,path
    p={k:local(v) for k,v in req['paths'].items()};a=load(p['trace']);r=read(p['report']);cap=load(p['failure'])
    checks=[]
    def exact(x,y,name):
        x,y=np.asarray(x),np.asarray(y);ok=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
        checks.append(dict(name=name,passed=ok))
        if not ok:raise AssertionError(name)
    Features,terms_fn=pure(p);c=read(p['contract']);goals=Features(load(p['motion']),load(p['original29']),c)
    default,kp,effort=[np.asarray(c[k]) for k in ('default_q','kp','training_effort')]
    norm=load(p['norm']);span=norm['joint_span'].astype(np.float64);limits=np.asarray(c['joint_limits'])
    history={k:np.zeros((4,n),np.float32) for k,n in dict(actions=23,base_ang_vel=3,dof_pos=23,dof_vel=23,projected_gravity=3).items()}
    previous=np.zeros(23,np.float32);named={k:[] for k in history}
    flat=lambda:np.concatenate([history[k].reshape(-1) for k in sorted(history)]).copy()
    assert len(a['target'])==316 and len(a['qpos'])==317 and r['attempted_controls']==316 and r['physics_steps']==3158
    exact(a['global_control'],np.arange(316,dtype=np.int64),'control clock');exact(a['source_frame'],np.arange(11,327,dtype=np.int64),'source clock')
    exact(a['controller_mode'],np.r_[np.zeros(250,np.int64),np.ones(66,np.int64)],'phase schedule')
    exact(a['physics_substeps'],np.r_[np.full(315,10,np.int64),np.array([8],np.int64)],'actual substeps')
    expected=0.;times=[expected]
    for _ in range(3158):expected+=.002;times.append(expected)
    exact(a['physics_expected_time'],np.array(times),'independent expected clock');exact(a['physics_time'],np.array(times),'actual native clock')
    for i in range(316):
        q,dq=a['qpos'][i],a['qvel'][i];integration=a['control_integration_before'][i]
        exact(q,integration[1:31],'integration q '+str(i));exact(dq,integration[31:60],'integration dq '+str(i))
        exact(integration[0],a['physics_time'][i*10],'precontrol time '+str(i))
        exact(q,a['physics_qpos'][i*10],'physics boundary q '+str(i));exact(dq,a['physics_qvel'][i*10],'physics boundary dq '+str(i))
        sensed,terms=terms_fn(q[7:],dq[6:],q[3:7],dq[3:6],previous,default)
        exact(sensed,a['state'][i],'state '+str(i));exact(goals(q,dq,i+11),a['features'][i],'features '+str(i))
        exact(previous,a['previous_action'][i],'prior '+str(i));exact(previous,a['control_previous_action_before'][i],'precontrol prior '+str(i))
        exact(flat(),a['history'][i],'lag history '+str(i));exact(flat(),a['control_history_before'][i],'precontrol history '+str(i))
        for key in history:named[key].append(history[key].copy())
        if i<250:
            raw=a['base_target'][i]+np.zeros(23,np.float32)
            exact(a['normalized_head'][i],np.zeros(23,np.float32),'disabled head '+str(i))
            exact(a['delta'][i],np.zeros(23,np.float64),'disabled delta promoted trace '+str(i))
            outgoing=((a['base_target'][i]-default)*kp/(.25*effort)).astype(np.float32)
            exact(default+outgoing*.25*effort/kp,a['base_target'][i],'original base round trip '+str(i))
        else:
            exact(a['base_target'][i],default,'direct default base '+str(i))
            delta=span*a['normalized_head'][i].astype(np.float64);raw=default+delta
            exact(delta,a['delta'][i],'promoted absolute delta '+str(i))
            outgoing=((np.clip(raw,limits[:,0],limits[:,1])-default)*kp/(.25*effort)).astype(np.float32)
        target=np.clip(raw,limits[:,0],limits[:,1]);actual=((target-default)*kp/(.25*effort)).astype(np.float32)
        for key,value in [('raw_proposal',raw),('target',target),('actual_normalized_action',actual),('action',outgoing)]:exact(value,a[key][i],key+' '+str(i))
        for key in history:history[key][1:]=history[key][:-1].copy();history[key][0]=terms[key]
        previous=outgoing
    exact(previous,a['final_previous_action'],'final prior');assert int(a['final_recorded_controls'])==316
    for key in history:exact(history[key],a['final_history_'+key],'final named '+key)
    b=load(p['baseline'])
    for key in ('qpos','qvel','target','source_frame','global_control','controller_mode','state','history','previous_action','action','base_target',
                'control_integration_before','control_history_before','control_previous_action_before','physics_qpos','physics_qvel','physics_torque','physics_actuator_torque',
                'physics_time','physics_expected_time','physics_warning_counts','physics_warning_lastinfo','physics_substeps'):
        count=2501 if key.startswith('physics_') and key!='physics_substeps' else 251 if key in ('qpos','qvel') else 250
        if key in ('physics_torque','physics_actuator_torque'):count=2500
        exact(a[key][:count],b[key][:count],'original BFM250 '+key)
    exact(a['control_integration_before'][250],b['final_integration'],'baseline end291')
    labels=load(p['labels']);centers=load(p['centers']);kept=np.r_[np.arange(52),np.arange(75,1023)]
    for key in ('state','history','previous_action'):exact(a[key][250],labels[key][0],'query250 '+key)
    exact(a['features'][250],labels['features'][0,kept],'query250 features');exact(a['features'][250],centers['features'][2038,kept],'center2038 features')
    exact(a['qpos'][250],labels['teacher_qpos'][0],'query250 q');exact(a['qvel'][250],labels['teacher_qvel'][0],'query250 dq')
    for key in named:exact(named[key][250],labels['history_'+key][0],'query250 named '+key)
    w=load(p['witness'])
    for actual,witness in [('features','features'),('normalized_head','normalized_target'),('raw_proposal','raw_proposal'),('target','target'),('delta','delta')]:exact(a[actual][250],w[witness],'activation witness '+actual)
    for name in ('canonical_initial_full291_parity','canonical_prefix250_parity','actual_query250_input_parity','actual_query250_ownexport_output_parity'):assert read(p[name])['passed'] is True
    assert r['forbidden_inference_calls']==0
    for mode in range(3):
        for graph in ('backward','actor','head'):
            count=250 if mode==0 and graph in ('backward','actor') else 66 if mode==1 and graph=='head' else 0
            for stage in ('attempted','returned'):assert r['inference_counts'][f'{mode}_{graph}_{stage}']==count
    exact(cap['qpos_at_failure'],a['qpos'][-1],'failure q');exact(cap['qvel_at_failure'],a['qvel'][-1],'failure dq')
    exact(cap['integration_at_failure'],a['final_integration'],'failure full291');exact(cap['head_input_features'][0],a['features'][315],'failure head input')
    exact(cap['head_output_0'][0],a['normalized_head'][315],'failure returned head output');exact(cap['previous_action_after'],previous,'failure prior after')
    for key in history:
        exact(cap['history_before_'+key],named[key][315],'failure prehistory '+key);exact(cap['history_after_'+key],history[key],'failure final history '+key)
    selected=np.arange(250,316);features=a['features'][250:];teacher_features=labels['features'][:66,kept];teacher_target=labels['expert_target'][:66]
    clipped=np.any(a['raw_proposal'][250:]!=a['target'][250:],axis=1)
    changed=np.any(features!=teacher_features,axis=1)
    first=lambda mask:int(selected[np.flatnonzero(mask)[0]]) if np.any(mask) else None
    rows=[]
    for j,control in enumerate(selected):
        rows.append(dict(control=int(control),target_clipped=bool(clipped[j]),feature_changed=bool(changed[j]),
            feature_RMS_standardized=float(np.sqrt(np.mean(((features[j].astype(np.float64)-teacher_features[j])/norm['feature_std'])**2))),
            same_clock_expert_target_RMSE_rad=float(np.sqrt(np.mean((a['target'][control]-teacher_target[j])**2))),
            same_clock_joint_velocity_RMSE=float(np.sqrt(np.mean((a['qvel'][control,6:]-labels['teacher_qvel'][j,6:])**2)))))
    np.savez_compressed(BASE/'actual_rows.npz',control=selected,features=features,teacher_features=teacher_features,teacher_target=teacher_target,
        target=a['target'][250:],raw_proposal=a['raw_proposal'][250:],normalized_head=a['normalized_head'][250:],previous_action=a['previous_action'][250:],
        action=a['action'][250:],state=a['state'][250:],history=a['history'][250:],qpos=a['qpos'][250:316],qvel=a['qvel'][250:316])
    for path,digest in req['input_sha256'].items():assert sha(local(path))==digest,path
    assert sha(BASE/'request.json')==reqsha and sha(__file__)==req['source_sha256']
    report=dict(passed=True,moving_controls=66,actual_controls=316,exact_checks=len(checks),checks=checks,
        inference_calls=0,BFM_calls=0,physics_steps=0,optimizer_updates=0,all_inputs_unchanged=True,
        first_feature_departure=first(changed),first_learned_target_clip=first(clipped),learned_clipped_commands=int(clipped.sum()),rows=rows,
        failure=r['failure'],terminal_or_hold_reached=False,limitations='Same-clock expert comparisons use different actual states after departure; no replanned targets or recovered trajectory.',
        request_sha256=reqsha,source_sha256=sha(__file__),trace_sha256=sha(p['trace']),actual_rows_sha256=sha(BASE/'actual_rows.npz'))
    write(BASE/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('checks','rows')},indent=2))
if __name__=='__main__':
    if sys.argv[1:] == ['--freeze']:freeze()
    elif sys.argv[1:] == ['--run']:audit()
    else:raise SystemExit('Use --freeze or --run')
