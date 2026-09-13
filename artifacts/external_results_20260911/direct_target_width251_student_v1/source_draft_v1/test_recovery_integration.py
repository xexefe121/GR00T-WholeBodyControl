"""Dedicated fake-data/CPU-tensor tests; no actual task model or checkpoint."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import types
import numpy as np
import pytest
import torch
import recovery_protocol as contract
import recovery_data as loader
import recovery_metrics as metrics
from recovery_objective import phase_cells,recovery_loss
from recovery_diagnostics import parity

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'direct_target_causal_width512_student_v1/source_prepared_v1'

def test_selected_protocol_values_have_no_numeric_defaults():
    request=dict(updates=2,learning_rate_values=[2e-6,1e-6],recovery_coefficient=.25)
    p=contract.protocol(request)
    assert p.final_step==81002 and p.optimizer_final==16002
    assert p.budgets['training_forward_calls']==8 and p.budgets['training_forward_rows']==31408
    assert p.budgets['diagnostic_Torch_calls']==5764 and p.budgets['diagnostic_Torch_rows']==1474352
    assert p.budgets['diagnostic_ORT_calls']==1441 and p.budgets['diagnostic_ORT_rows']==368588
    assert p.rate(0)==2e-6 and p.rate(1)==1e-6
    with pytest.raises(ValueError):p.rate(2)
    with pytest.raises(KeyError):contract.protocol({})

@pytest.mark.parametrize('field,value',[
    ('updates',0),('updates',10001),('updates',True),('learning_rate_values',[1e-6]),
    ('learning_rate_values',[1e-6,0.]),('learning_rate_values',[1e-6,float('nan')]),
    ('recovery_coefficient',0.),('recovery_coefficient',float('inf')),('recovery_coefficient',True)])
def test_protocol_rejects_unbounded_missing_or_adaptive_values(field,value):
    request=dict(updates=2,learning_rate_values=[2e-6,1e-6],recovery_coefficient=.25);request[field]=value
    with pytest.raises(ValueError):contract.protocol(request)

def test_three_phase_loss_and_gradient_match_explicit_balanced_formula():
    phase=np.repeat(np.arange(3,dtype=np.int8),[99,819,100]);control=np.arange(251,1269,dtype=np.int64)
    cells=[torch.from_numpy(ids) for ids in phase_cells(phase,control)]
    prediction=torch.from_numpy(np.repeat(np.array([1.,2.,3.],np.float32),[99,819,100])[:,None].repeat(23,axis=1)).requires_grad_()
    loss,byphase=recovery_loss(prediction,torch.zeros_like(prediction),cells)
    torch.testing.assert_close(byphase,torch.tensor([1.,4.,9.]))
    torch.testing.assert_close(loss,torch.tensor(14/3))
    assert not torch.isclose(loss,prediction.square().mean())
    loss.backward()
    for value,count,ids in zip((1.,2.,3.),(99,819,100),cells):
        torch.testing.assert_close(prediction.grad[ids],torch.full((count,23),2*value/(3*count*23)))

@pytest.mark.parametrize('kind',['phase','clock','dtype'])
def test_three_phase_partition_rejects_wrong_provenance(kind):
    phase=np.repeat(np.arange(3,dtype=np.int8),[99,819,100]);control=np.arange(251,1269,dtype=np.int64)
    if kind=='phase':phase[0]=1
    if kind=='clock':control[0]=250
    if kind=='dtype':phase=phase.astype(np.int64)
    with pytest.raises(ValueError):phase_cells(phase,control)

def new_rows():
    x=np.zeros((1018,1323),np.float32);x[:,0]=.25;x[:,1000:1023]=.5;x[:,1023:]=.125;c=np.arange(251,1269,dtype=np.int64)
    return dict(causal_features=x,features=x[:,:1000],context=x[:,1000:],incoming_prior=x[:,1000:1023],incoming_history=x[:,1023:],
        expert_target=np.zeros((1018,23)),control=c,source_frame=c+11,phase=np.repeat(np.arange(3,dtype=np.int8),[99,819,100]),
        first_student_state_query=np.r_[True,np.zeros(1017,bool)],plan_control=251+5*((c-251)//5),plan_local=(c-251)%5,
        feedback_clipped=np.zeros((1018,23),bool),native_clipped=np.zeros((1018,23),bool))

@pytest.mark.parametrize('change',[None,'prior','history','plan','prefix','target','phase'])
def test_qualified_recovery_loader_preserves_full1018_or_rejects_corruption(tmp_path,monkeypatch,change):
    rows=new_rows()
    if change=='prior':rows['incoming_prior']=rows['incoming_prior'].copy();rows['incoming_prior'][0,0]=1
    if change=='history':rows['incoming_history']=rows['incoming_history'].copy();rows['incoming_history'][0,0]=1
    if change=='plan':rows['plan_local']=rows['plan_local'].copy();rows['plan_local'][0]=1
    if change=='prefix':rows['control'][0]=250
    if change=='target':rows['expert_target'][0,0]=2
    if change=='phase':rows['phase'][0]=1
    path=tmp_path/'fake_rows.npz';np.savez(path,**rows)
    monkeypatch.setattr(loader,'check_recovery_subjects',lambda *a:dict(fake=True))
    data=dict(limits=np.tile([-1.,1.],(23,1)));request=dict(subjects=dict(recovery_rows=dict(path=str(path))))
    if change is not None:
        with pytest.raises(ValueError):loader.attach_recovery(data,request,{})
    else:
        result=loader.attach_recovery(data,request,{})
        assert result is data and result['recovery_features'].shape==(1018,1323)
        assert [len(v) for v in result['recovery_cells']]==[99,819,100]
        np.testing.assert_array_equal(result['recovery_features'][:,1000:1023],rows['incoming_prior'])
        assert len(result['recovery_target'])==1018 and np.count_nonzero(result['recovery_metadata']['first_student_state_query'])==1

def test_append_metrics_leaves_old_values_unchanged(monkeypatch):
    old=dict(nominal_objective=.1,physical_objective=.2,full_state_objective=.3,balanced_weighted_objective=.9,
        nominal_cells=[dict(kept=True)],full_state_cells=[dict(kept=True)],physical_cells=[dict(kept=True)])
    monkeypatch.setattr(metrics,'summarize_balanced',lambda *a:copy.deepcopy(old))
    rows=new_rows();data=dict(recovery_target=rows['expert_target'],recovery_cells=phase_cells(rows['phase'],rows['control']),
        default=np.zeros(23),span=np.ones(23,np.float32),limits=np.tile([-10.,10.],(23,1)))
    output=dict(recovery=np.ones((1018,23),np.float32))
    result=metrics.summarize_recovery(output,data,.25)
    for key,value in old.items():assert result[key]==value
    assert result['recovery_objective']==1. and result['combined_recovery_objective']==1.15
    assert [c['rows'] for c in result['recovery_cells']]==[99,819,100]
    assert all(c['first24']['applied_RMSE_rad']==1. for c in result['recovery_cells'])

def test_all_twelve_fp64_comparisons_gate_recovery_without_loosening_old():
    corpora=('nominal','full_state','physical','recovery');backends=('CPU64','GPU64','ORT64')
    output={b:{c:np.zeros((2,23),np.float32) for c in corpora} for b in backends}
    data=dict(default=np.zeros(23),span=np.ones(23,np.float32),limits=np.tile([-1.,1.],(23,1)))
    good=parity(output,data);assert good['passed'] and len(good['comparisons'])==12 and good['tolerance_rad']==1e-5
    output['ORT64']['recovery'][0,0]=2e-5
    bad=parity(output,data);assert not bad['passed'] and bad['maximum_preclamp_rad']>1e-5
    assert all(v==0 for k,v in bad['comparisons'].items() if not k.startswith('recovery_'))

def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

def test_four_corpus_diagnostic_function_bodies_unchanged():
    assert functions(HERE/'recovery_diagnostics.py')==functions(OLD/'context_diagnostics.py')
    tree=ast.parse((HERE/'recovery_diagnostics.py').read_text());values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'CORPORA','SIZES','CALLS'}}
    assert values==dict(CORPORA=('nominal','full_state','physical','recovery'),SIZES=(9904,354612,3054,1018),CALLS=1441)

def test_promoted_model_and_export_math_unchanged_except_graph_label():
    before=functions(OLD/'width512_promoted.py');after=functions(HERE/'recovery_promoted.py')
    assert before['PromotedDirect']==after['PromotedDirect']
    assert before['export_onnx'].replace('same_81000_width512_context_weights_fp64_execution','selected_recovery_width512_context_weights_fp64_execution')==after['export_onnx']

def test_main_four_counted_forwards_single_update_no_expansion():
    tree=ast.parse((HERE/'train_recovery.py').read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_continuation')
    loop=next(n for n in ast.walk(run) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='index')
    calls=[n for n in ast.walk(loop) if isinstance(n,ast.Call)]
    forward=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='counted_forward']
    assert [ast.literal_eval(n.args[2]) for n in forward]==['nominal','full_state','physical','recovery']
    assert len([n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='step'])==1
    assert len([n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='backward'])==1
    allnames=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert not {'expand_source','verify_width_restoration','calibrate'} & set(allnames)
    assert allnames.count('restore_warm512')==1
    initial=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='initial_drift')
    assert next(n for n in initial.body if isinstance(n,ast.For)).iter.id=='OLD_CORPORA'

def test_fourth_forward_failure_keeps_first_three_returns_and_exact_attempts():
    source=ast.parse((HERE/'training_support.py').read_text())
    fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='counted_forward')
    fake_torch=types.SimpleNamespace(float32=torch.float32,isfinite=torch.isfinite,cuda=types.SimpleNamespace(synchronize=lambda:None))
    namespace={'torch':fake_torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'qualified_counter','exec'),namespace)
    counter=dict(calls_attempted=0,calls_returned=0,calls_synchronized=0,calls_verified=0,rows_attempted=0,rows_returned=0,rows_verified=0);active={}
    for name,n in [('nominal',9904),('full_state',1728),('physical',3054)]:
        namespace['counted_forward'](lambda values:torch.zeros((len(values),23),dtype=torch.float32),np.empty((n,1)),name,counter,active,4,15704)
    def fail(_):raise RuntimeError('injected fourth call')
    with pytest.raises(RuntimeError):namespace['counted_forward'](fail,np.empty((1018,1)),'recovery',counter,active,4,15704)
    assert counter['calls_attempted']==4 and counter['calls_returned']==3
    assert counter['rows_attempted']==15704 and counter['rows_returned']==14686
    assert {'nominal_prediction','full_state_prediction','physical_prediction'}<=set(active)
    assert active['current_forward']['name']=='recovery' and not active['current_forward']['returned']

def test_saved_schedule_prefix_is_not_generated_or_wrapped(monkeypatch):
    centers=np.zeros((10000,864),np.int32);axes=np.zeros((10000,864),np.int8);centers[:,0]=np.arange(10000)
    old=dict(schedule_lineage=dict(full_updates=10000))
    monkeypatch.setattr(loader,'load_response_data',lambda *a:(old,{},centers,axes));monkeypatch.setattr(loader,'attach_recovery',lambda data,*a:data)
    data,norm,sc,sa=loader.load_data(dict(updates=2,learning_rate_values=[1e-6,1e-6],recovery_coefficient=.25),{})
    assert np.shares_memory(sc,centers) and np.shares_memory(sa,axes)
    assert sc[:,0].tolist()==[0,1] and data['schedule_lineage']['wrap_or_regeneration'] is False

def test_protocol_configuration_not_read_at_module_import():
    for name in ('train_recovery.py','recovery_protocol.py','recovery_data.py','recovery_support.py'):
        tree=ast.parse((HERE/name).read_text())
        for n in tree.body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):continue
            for call in ast.walk(n):
                if isinstance(call,ast.Call):
                    method=call.func.attr if isinstance(call.func,ast.Attribute) else call.func.id if isinstance(call.func,ast.Name) else ''
                    assert method not in {'read','load','load_data','gate','protocol','check_protocol','from_checkpoint','restore_warm512'},name

def fake_admission(tmp_path):
    subjects={};pins={}
    def save(role,value,name=None,raw=False):
        path=tmp_path/(name or role+'.json')
        path.write_bytes(value if raw else json.dumps(value).encode())
        item=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest());subjects[role]=item;pins[str(path)]=item['sha256']
        return item
    subjects['normalization']=dict(path=str(tmp_path/'normalization.npz'),sha256='a'*64)
    save('recovery_rows',b'synthetic rows placeholder','expert_rows.npz',True)
    save('collection_qualification',dict(root_authorized_extraction=True,model_fitting_authorized=False,control_start=251,control_stop_exclusive=1269,rows=1018))
    source=dict(source_review_pass=True,source_sha256={'collector.py':'b'*64});save('collection_source_review',source)
    save('collection_request',dict(root_selected_collection=True,model_fitting_authorized=False,
        subjects={name:subjects[role] for role,name in [('collection_qualification','qualification'),('collection_source_review','source_review')]},source_sha256=source['source_sha256']))
    save('collection_report',dict(passed=True,collection_completed=True,model_fitting_authorized=False,rows=1018,control_start=251,control_stop_exclusive=1269,fresh_student_state_queries=1,
        request_sha256=subjects['collection_request']['sha256'],outputs={'expert_rows.npz':subjects['recovery_rows']['sha256'],'normalization.npz':'a'*64},
        input_sha256={subjects[r]['path']:subjects[r]['sha256'] for r in ('collection_qualification','collection_source_review')},source_sha256=source['source_sha256']),name='report.json')
    save('consistency_report',dict(passed=True,evidence_diagnosis_completed=True,old_rows=12958,new_rows=1018,rows=13976,
        input_sha256={subjects[r]['path']:subjects[r]['sha256'] for r in ('recovery_rows','collection_report','collection_request','collection_qualification','collection_source_review','normalization')}))
    save('warm_restore_review',dict(source_review_pass=True,source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        for name in ('warm512_restore.py','width512.py','restoration_support.py','balance_contract.py')}))
    return dict(subjects=subjects),pins

def test_minimal_new_data_gate_accepts_exact_chain_and_extra_role_annotations(tmp_path):
    request,pins=fake_admission(tmp_path)
    request['subjects']['collection_qualification']['pass_field']='root_authorized_extraction'
    evidence=loader.check_recovery_subjects(request,pins)
    assert evidence['fresh_student_state_queries']==1 and evidence['conflicts_not_automatically_repaired']

@pytest.mark.parametrize('role,field,value',[
    ('collection_report','collection_completed',False),('collection_report','fresh_student_state_queries',1018),
    ('collection_qualification','rows',1019),('consistency_report','evidence_diagnosis_completed',False),
    ('warm_restore_review','source_review_pass',False),('warm_restore_review','source_sha256',{})])
def test_new_data_gate_rejects_changed_qualification(tmp_path,role,field,value):
    request,pins=fake_admission(tmp_path);item=request['subjects'][role];path=Path(item['path'])
    record=json.loads(path.read_text());record[field]=value;path.write_text(json.dumps(record));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    item['sha256']=digest;pins[str(path)]=digest
    with pytest.raises((ValueError,KeyError)):loader.check_recovery_subjects(request,pins)
