"""One selected fixed set of 3054 nominal-verified one-control policy branches."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
from collection_arrays import Store,ROWS,HISTORY_WIDTHS,sha,atomic
from branch_inputs import local,read,archive,exact,Inputs,advance_history,frozen_native,fixed_map_function
from stateless_adapter import prepare

BASE=Path(__file__).resolve().parent.parent;DEST=BASE/'collection'
BUDGET=dict(backward=3054,actor=3054,head=6108)
COUNTS={key:dict(attempted=0,returned=0) for key in BUDGET}
PROGRESS=dict(stage='PREFLIGHT',nominal_verified=0,policy_attempted=0,policy_completed=0,strict_failed=0,labels_completed=0,
    native_attempted=0,native_returned=0,calls=COUNTS,active_row=None)
ACTIVE={};STORE=None;ENGINE=None;NATIVE_ACTIVE=False

def active_save():
    temp=DEST/'active.npz.tmp'
    with temp.open('wb') as f:np.savez_compressed(f,**{k:np.asarray(v) for k,v in ACTIVE.items()});f.flush();os.fsync(f.fileno())
    temp.replace(DEST/'active.npz')
def progress():
    if ENGINE is not None:
        PROGRESS['native_attempted']=ENGINE.total_attempted;PROGRESS['native_returned']=ENGINE.total_returned
    atomic(DEST/'progress.json',PROGRESS)
def json_finite(value):
    if isinstance(value,dict):return {key:json_finite(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [json_finite(item) for item in value]
    if isinstance(value,float) and not np.isfinite(value):return 'NaN' if np.isnan(value) else ('Infinity' if value>0 else '-Infinity')
    return value
def ledger(name,value):
    with (DEST/(name+'.jsonl')).open('a',encoding='utf-8') as f:f.write(json.dumps(json_finite(value),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def evidence_files():
    return {p.name:dict(path='../'+p.name,sha256=sha(p)) for p in DEST.iterdir() if p.is_file() and p.name not in ('report.json','failure.json','progress.json') and not p.name.endswith('.tmp')}
def frozen(request):
    for name,digest in request['source_sha256'].items():assert sha(Path(__file__).parent/name)==digest,name
    for name,digest in request['input_sha256'].items():assert sha(local(name))==digest,name
class Counted:
    def __init__(self,session,name):self.session=session;self.name=name
    def run(self,outputs,feed):
        count=COUNTS[self.name]
        if count['attempted']>=BUDGET[self.name]:raise RuntimeError('Graph budget exhausted '+self.name)
        if PROGRESS['nominal_verified']!=ROWS:raise RuntimeError('Graph call before all nominal transitions verified')
        for key in list(ACTIVE):
            if key.startswith(self.name+'_output_'):del ACTIVE[key]
        for key,value in feed.items():ACTIVE[self.name+'_input_'+key]=np.asarray(value).copy()
        ACTIVE['active_graph']=np.asarray(self.name);ACTIVE['active_graph_attempt']=np.int64(count['attempted']+1)
        ACTIVE['active_graph_returned']=np.asarray(False)
        count['attempted']+=1;active_save();progress()
        result=self.session.run(outputs,feed)
        count['returned']+=1
        ACTIVE['active_graph_returned']=np.asarray(True)
        for i,value in enumerate(result):ACTIVE[self.name+'_output_'+str(i)]=np.asarray(value).copy()
        active_save();progress()
        ledger('graph_calls',dict(row=PROGRESS['active_row'],stage=PROGRESS['stage'],graph=self.name,attempt=count['attempted'],returned=count['returned']))
        return result
def output23(result,label):
    assert len(result)==1 and result[0].shape==(1,23) and result[0].dtype==np.float32,label
    assert np.isfinite(result[0]).all(),label
    return result[0][0].copy()
def persist_native(phase,row,evidence,status):
    STORE.row(row,**{phase+'_'+key:value for key,value in evidence.items()},**{phase+'_status':status});STORE.flush()
def forecast(phase,row,start,warn,info,target,expected):
    global NATIVE_ACTIVE
    ACTIVE.update(native_phase=np.asarray(phase),start_integration=start.copy(),start_warning=warn.copy(),start_lastinfo=info.copy(),applied_target=target.copy())
    active_save();progress();NATIVE_ACTIVE=True
    try:
        report,evidence=ENGINE.run(start,warn,info,target,expected)
        persist_native(phase,row,evidence,1 if report['feasible'] else 2)
        ledger('native_rows',dict(phase=phase,row=row,report=report,attempted=evidence['attempted_steps'],returned=evidence['returned_steps'],captured=evidence['valid_steps']))
        return report,evidence
    except BaseException as error:
        evidence=ENGINE.evidence();persist_native(phase,row,evidence,3)
        for key,value in evidence.items():ACTIVE['native_exception_'+key]=np.asarray(value).copy()
        ACTIVE['native_exception_private_warning']=ENGINE.engine.private.warning.number.copy()
        ACTIVE['native_exception_private_lastinfo']=ENGINE.engine.private.warning.lastinfo.copy()
        ACTIVE['native_exception_private_command']=ENGINE.engine.private.ctrl.copy()
        ACTIVE['native_exception_end_state_unclassified']=np.asarray(True)
        ledger('native_exceptions',dict(phase=phase,row=row,error=repr(error),attempted=evidence['attempted_steps'],returned=evidence['returned_steps'],captured=evidence['valid_steps'],end_state_unclassified=True))
        active_save();raise
    finally:NATIVE_ACTIVE=False;progress()
def calibration_start(inputs,row,index,delta,raw,target,outgoing,advanced):
    assert row==2036 and index==2038
    a=inputs.actual;c=inputs.centers;w=inputs.witness
    assert int(w['control'])==250 and int(w['source_frame'])==261
    for key in ('qpos','qvel','state','features','history','previous_action','base_target'):
        exact(c[key][index],a[key][250],'calibration input '+key)
    exact(c['features'][index],w['features'],'calibration witness input')
    exact(delta,w['onnx_delta'],'calibration witness output')
    for key,value in dict(delta=delta,raw_proposal=raw,target=target,action=outgoing).items():exact(value,a[key][250],'calibration command '+key)
    exact(outgoing,a['previous_action'][251],'calibration endpoint prior')
    exact(advanced,a['history'][251],'calibration advanced actor history')
    exact(inputs.integration(2,250),a['control_integration_before'][250],'calibration full start291')
def calibration_end(inputs,evidence,state,base,features,delta):
    a=inputs.actual
    expected=inputs.expected(a,250)
    for key,value in expected.items():exact(evidence[key],value,'calibration native '+key)
    exact(evidence['end_integration'],a['control_integration_before'][251],'calibration full successor291')
    for key,value in dict(state=state,base_target=base,features=features,delta=delta).items():exact(value,a[key][251],'calibration endpoint '+key)

def main():
    global STORE,ENGINE
    parser=argparse.ArgumentParser();parser.add_argument('--clearance-sha256',required=True);args=parser.parse_args()
    assert sha(BASE/'clearance.json')==args.clearance_sha256
    clear=read(BASE/'clearance.json');assert clear['approved'] is True
    request_sha=sha(BASE/'request.json');assert request_sha==clear['request_sha256']
    assert sha(local(clear['review_path']))==clear['review_sha256']
    request=read(BASE/'request.json');frozen(request)
    assert request['graph_call_ceiling']==BUDGET and request['native_step_ceiling']==61080 and request['rows']==ROWS
    paths={k:local(v) for k,v in request['paths'].items()}
    DEST.mkdir(exist_ok=False);progress();started=time.perf_counter()
    try:
        assert sys.platform!='win32' and np.__version__=='1.26.4'
        sys.path.insert(0,str(paths['onnx_dependencies']))
        import mujoco,onnxruntime as ort
        assert mujoco.__version__=='3.2.3' and ort.__version__=='1.23.2'
        binaries=list((Path(ort.__file__).parent/'capi').glob('*pybind11_state*.so'));assert len(binaries)==1
        assert sha(binaries[0]) in request['input_sha256'].values()
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        sessions={name:Counted(ort.InferenceSession(str(paths[name]),sess_options=options,providers=['CPUExecutionProvider']),name) for name in BUDGET}
        model,contract,original,timeline,manifest=frozen_native(paths['bundle'])
        c,builder,latent,sensed=prepare(original,archive(paths['motion']),archive(paths['original29']),contract,sessions)
        fixed_map=fixed_map_function();inputs=Inputs(paths);centers=inputs.centers;limits=np.asarray(contract['joint_limits'],np.float64)
        old_report=read(paths['old_report']);assert old_report['passed'] and old_report['completed_controls']==1268 and old_report['snapshots_sha256']==sha(paths['old_snapshots'])
        assert str(inputs.old['original_trace_sha256'])==sha(paths['trace0'])
        assert str(inputs.witness['head_sha256'])==sha(paths['head']) and str(inputs.witness['centers_sha256'])==sha(paths['centers'])
        PROGRESS['stage']='PURE_CENTER_INPUT_CHECKS';progress();inputs.validate_centers(builder,sensed,contract)
        STORE=Store(DEST/'data')
        for row,(dataset,control,index) in enumerate(inputs.rows):
            successor=index+1
            values=dict(dataset=dataset,start_control=control,successor_control=control+1,source_frame=control+11,successor_frame=control+12,
                center_index=index,successor_center_index=successor,successor_plan_control=centers['plan_control'][successor],
                successor_plan_local=centers['plan_local'][successor],successor_plan_accepted_update=centers['plan_accepted_update'][successor],
                successor_replan_boundary=centers['plan_control'][successor]!=centers['plan_control'][index],successor_zero_gain=not np.any(centers['gain'][successor]),
                incoming_raw_prior=centers['previous_action'][index],nominal_base_action=centers['base_action'][index],nominal_base_target=centers['base_target'][index],
                nominal_target=centers['expert_target'][index],nominal_features=centers['features'][index],nominal_state=centers['state'][index],incoming_history=centers['history'][index])
            for key in HISTORY_WIDTHS:values['incoming_history_'+key]=centers['history_'+key][index]
            named,advanced=advance_history(centers,index);values['advanced_history']=advanced
            for key in HISTORY_WIDTHS:values['advanced_history_'+key]=named[key]
            STORE.row(row,**values)
        STORE.flush()
        from native_capture import CapturedForecast
        ENGINE=CapturedForecast(model,contract)
        PROGRESS['stage']='ALL_NOMINAL_TRANSITIONS_FIRST';progress()
        for row,(dataset,control,index) in enumerate(inputs.rows):
            PROGRESS['active_row']=row;ACTIVE.clear();ACTIVE.update(row=np.int64(row),dataset=np.int64(dataset),control=np.int64(control),center=np.int64(index))
            expected=inputs.expected(inputs.traces[dataset],control)
            report,evidence=forecast('nominal',row,inputs.integration(dataset,control),expected['warning_counts'][0],expected['warning_lastinfo'][0],centers['expert_target'][index].copy(),inputs.clock[control*10])
            assert report['feasible'] and evidence['attempted_steps']==evidence['returned_steps']==evidence['valid_steps']==10
            for key,value in expected.items():exact(evidence[key],value,f'nominal {row} {key}')
            exact(evidence['end_integration'],inputs.integration(dataset,control+1),f'nominal {row} full successor291')
            STORE.row(row,nominal_verified=True);PROGRESS['nominal_verified']+=1;progress()
            if row%100==0:print(json.dumps(dict(stage=PROGRESS['stage'],verified=PROGRESS['nominal_verified'])),flush=True)
        STORE.flush();assert PROGRESS['nominal_verified']==ROWS and sum(v['attempted'] for v in COUNTS.values())==0
        atomic(DEST/'all_nominal_parity.json',dict(passed=True,rows=ROWS,physics_steps=30540,full_successor291=True,all_samples_byteexact=True,graph_calls=0,warning_source_dtypes=inputs.warning_source_dtypes,native_warning_dtype='int32',warning_comparison='Value-preserving explicit int64-to-int32 cast for legacy source only'))
        order=[2036]+[row for row in range(ROWS) if row!=2036]
        np.save(DEST/'policy_execution_order.npy',np.asarray(order,np.int64));PROGRESS['stage']='POLICY_BRANCHES'
        for sequence,row in enumerate(order):
            dataset,control,index=inputs.rows[row];successor=index+1;PROGRESS['active_row']=row
            ACTIVE.clear();ACTIVE.update(row=np.int64(row),sequence=np.int64(sequence),dataset=np.int64(dataset),control=np.int64(control),center=np.int64(index),
                qpos=centers['qpos'][index].copy(),qvel=centers['qvel'][index].copy(),incoming_prior=centers['previous_action'][index].copy(),
                incoming_history=centers['history'][index].copy(),advanced_history=STORE.arrays['advanced_history'][row].copy())
            for key in HISTORY_WIDTHS:ACTIVE['incoming_history_'+key]=centers['history_'+key][index].copy()
            PROGRESS['stage']='NOMINAL_HEAD';progress()
            returned=sessions['head'].run(None,dict(features=centers['features'][index][None]))
            delta=output23(returned,'nominal head');STORE.row(row,nominal_head_output=returned[0],policy_head_delta=delta)
            raw=centers['base_target'][index]+delta;target=np.clip(raw,limits[:,0],limits[:,1])
            outgoing=(centers['base_action'][index]+delta*np.asarray(contract['kp'])/(.25*np.asarray(contract['training_effort']))).astype(np.float32)
            normalized=((target-np.asarray(contract['default_q']))*np.asarray(contract['kp'])/(.25*np.asarray(contract['training_effort']))).astype(np.float32)
            STORE.row(row,policy_raw_target=raw,policy_applied_target=target,outgoing_raw_prior=outgoing,policy_actual_normalized_action=normalized,policy_native_target_clipped=raw!=target)
            ACTIVE.update(raw_proposal=raw.copy(),outgoing_prior=outgoing.copy());active_save()
            if row==2036:calibration_start(inputs,row,index,delta,raw,target,outgoing,STORE.arrays['advanced_history'][row])
            PROGRESS['stage']='POLICY_NATIVE';PROGRESS['policy_attempted']+=1;progress()
            expected=inputs.expected(inputs.traces[dataset],control)
            report,evidence=forecast('policy',row,inputs.integration(dataset,control),expected['warning_counts'][0],expected['warning_lastinfo'][0],target,inputs.clock[control*10])
            if not report['feasible']:
                if row==2036:raise AssertionError('Fixed actual251 branch witness became infeasible')
                PROGRESS['strict_failed']+=1;progress();continue
            assert evidence['attempted_steps']==evidence['returned_steps']==evidence['valid_steps']==10
            PROGRESS['policy_completed']+=1;PROGRESS['stage']='ENDPOINT_GRAPHS';progress()
            q,v=evidence['qpos'][10].copy(),evidence['qvel'][10].copy();ACTIVE.update(endpoint_qpos=q.copy(),endpoint_qvel=v.copy())
            z=latent(control+12,q);ACTIVE['endpoint_latent']=z.copy();active_save()
            assert z.shape==(1,256) and z.dtype==np.float32 and np.isfinite(z).all()
            state=sensed(q,v,outgoing);history=STORE.arrays['advanced_history'][row].copy()
            STORE.row(row,endpoint_state=state,endpoint_latent=z)
            returned=sessions['actor'].run(None,dict(state=state[None],last_action=outgoing[None],history=history[None],z=z))
            actor=output23(returned,'endpoint actor');base_action=actor*5
            base=c['default_q']+base_action*.25*c['training_effort']/c['kp'];features=builder(q,v,control+12,base,outgoing)
            assert np.isfinite(base).all() and np.isfinite(features).all()
            STORE.row(row,endpoint_actor_output=returned[0],endpoint_base_action=base_action,endpoint_base_target=base,endpoint_features=features)
            teacher=fixed_map(centers,successor,q,v,limits)
            for key,value in teacher.items():assert np.isfinite(value).all(),key
            STORE.row(row,**teacher,label_residual_rad=teacher['label_fixed_map_target']-base)
            returned=sessions['head'].run(None,dict(features=features[None]));endpoint_delta=output23(returned,'endpoint head')
            STORE.row(row,endpoint_head_output=returned[0],endpoint_head_delta=endpoint_delta,endpoint_raw_proposal=base+endpoint_delta,endpoint_applied_target=np.clip(base+endpoint_delta,limits[:,0],limits[:,1]))
            if row==2036:
                calibration_end(inputs,evidence,state,base,features,endpoint_delta)
                atomic(DEST/'query250_actual251_calibration.json',dict(passed=True,row=2036,center=2038,start_control=250,successor_control=251,ten_native_samples_and_full291_byteexact=True,
                    nominal_head_witness_byteexact=True,endpoint_state_base_features_head_history_prior_byteexact=True,counts_included_in_selected_budget=True))
            STORE.row(row,label_valid=True);PROGRESS['labels_completed']+=1;STORE.flush();progress()
            if sequence%100==0:print(json.dumps(dict(sequence=sequence,labels=PROGRESS['labels_completed'],strict_failed=PROGRESS['strict_failed'],native_returned=ENGINE.total_returned)),flush=True)
        assert PROGRESS['policy_attempted']==ROWS and PROGRESS['policy_completed']+PROGRESS['strict_failed']==ROWS
        assert PROGRESS['labels_completed']==PROGRESS['policy_completed']
        expected_calls=dict(backward=PROGRESS['labels_completed'],actor=PROGRESS['labels_completed'],head=ROWS+PROGRESS['labels_completed'])
        assert COUNTS=={key:dict(attempted=value,returned=value) for key,value in expected_calls.items()}
        assert ENGINE.total_attempted==ENGINE.total_returned<=61080
        PROGRESS['stage']='FINAL_REHASH';progress()
        assert sha(BASE/'request.json')==request_sha and sha(BASE/'clearance.json')==args.clearance_sha256
        frozen(request)
        manifest=STORE.manifest(dict(complete=True,request_sha256=request_sha,rows=ROWS,labels_valid=PROGRESS['labels_completed'],strict_failed=PROGRESS['strict_failed'],
            graph_calls=COUNTS,native_attempted=ENGINE.total_attempted,native_returned=ENGINE.total_returned,
            evidence_files=evidence_files(),
            schema_sha256=sha(DEST/'data/schema.json'),native_rows_sha256=sha(DEST/'native_rows.jsonl'),
            graph_calls_sha256=sha(DEST/'graph_calls.jsonl'),nonfinite_report_encoding='Nonfinite native report scalars use explicit NaN/Infinity/-Infinity strings; raw arrays retain IEEE values.',
            policy_order_sha256=sha(DEST/'policy_execution_order.npy'),row_status_definition='Physical feasibility is separate from nominal_verified and label_valid. Invalid labels retain NaN endpoint fields; no failed row removed.'))
        valid=STORE.arrays['label_valid'];error=STORE.arrays['endpoint_raw_proposal'][valid]-STORE.arrays['label_fixed_map_target'][valid]
        report=dict(completed=True,passed=True,rows=ROWS,nominal_verified=ROWS,policy_branches=ROWS,labels_valid=int(valid.sum()),strict_failed=PROGRESS['strict_failed'],
            graph_calls=COUNTS,total_graph_calls=sum(v['returned'] for v in COUNTS.values()),native_attempted=ENGINE.total_attempted,native_returned=ENGINE.total_returned,
            query250_actual251_calibration_passed=True,all_frozen_inputs_unchanged=True,request_sha256=request_sha,clearance_sha256=args.clearance_sha256,
            manifest_sha256=sha(DEST/'data/manifest.json'),seconds=time.perf_counter()-started,optimizer_updates=0,new_mpc_queries=0,
            evidence_files=evidence_files(),native_rows_sha256=sha(DEST/'native_rows.jsonl'),schema_sha256=sha(DEST/'data/schema.json'),
            graph_calls_sha256=sha(DEST/'graph_calls.jsonl'),
            policy_native_clip_rows=int(np.any(STORE.arrays['policy_native_target_clipped'],axis=1).sum()),successor_replan_rows=int(STORE.arrays['successor_replan_boundary'].sum()),
            teacher_feedback_clip_rows=int(np.any(STORE.arrays['teacher_feedback_clipped'][valid],axis=1).sum()),teacher_native_clip_rows=int(np.any(STORE.arrays['teacher_native_clipped'][valid],axis=1).sum()),
            endpoint_raw_vs_fixed_map_rmse_rad=float(np.sqrt(np.mean(error**2))) if len(error) else None,
            runtime=dict(numpy=np.__version__,mujoco=mujoco.__version__,onnxruntime=ort.__version__,binary_sha256=sha(binaries[0]),threads=1,head_batch=1),
            limitation='Independent one-control branches from qualified expert states. Raw combined-action history retained. Targets use same-clock saved committed feedback with both original clips, not replanning or a qualified connected trajectory.')
        atomic(DEST/'report.json',report);PROGRESS['stage']='COMPLETE';progress();print(json.dumps(report),flush=True)
    except BaseException as error:
        if STORE is not None:STORE.flush()
        active_save();PROGRESS['failed_stage']=PROGRESS['stage'];PROGRESS['stage']='FAILED';progress()
        if STORE is not None:STORE.manifest(dict(complete=False,request_sha256=request_sha,progress=PROGRESS,evidence_files=evidence_files(),forensic_state_warning='An exception row end_integration may be partially mutated/unclassified. Only valid_steps samples were captured; attempted and returned counts remain distinct.'))
        atomic(DEST/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),progress=PROGRESS,active_sha256=sha(DEST/'active.npz'),
            committed_rows='nominal_verified and label_valid masks only; other returned data are failure evidence, not labels. No retry or resume selected.'))
        raise
    finally:
        if ENGINE is not None:ENGINE.close()
if __name__=='__main__':
    try:main()
    except BaseException as error:
        if not DEST.exists():atomic(BASE/'preflight_failure.json',dict(error=repr(error),progress=PROGRESS))
        raise
