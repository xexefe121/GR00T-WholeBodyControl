"""Synthetic saved values and static producer contracts only; no task evidence."""
import ast,copy
from pathlib import Path
import numpy as np
import pytest
import torch
import audit_recovery_math as math
from audit_recovery_bindings import exact_member
from audit_width_math import rate
from audit_balanced_math import PRODUCER_WEIGHT_RULE

HERE=Path(__file__).resolve().parent
PRODUCER=HERE.parents[1]/'direct_target_width251_student_v1/source_prepared_v1'

def source():
    group=dict(params=list(range(6)),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=1e-5,
        amsgrad=False,maximize=False,foreach=False,capturable=False,differentiable=False,fused=False,decoupled_weight_decay=True)
    return dict(kind='direct_absolute_native23_target',condition='causal',ordinary_final_step=81000,
        optimizer_step=16000,fresh_optimizer=False,hidden_width=512,architecture=[1323,512,512,23],context_blinded=False,
        full_state_coefficient=math.WEIGHT,context_order='previous_action23_then_incoming_history300',response_weight_rule=PRODUCER_WEIGHT_RULE,
        actor_state={name:torch.full(shape,.125) for name,shape in zip(math.NAMES,math.SHAPES)},
        optimizer_state=dict(param_groups=[group],state={i:dict(step=torch.tensor(16000.),exp_avg=torch.full(shape,.2),exp_avg_sq=torch.full(shape,.3)) for i,shape in enumerate(math.SHAPES)}),
        feature_mean=torch.zeros(1323),feature_std=torch.ones(1323),
        rng=dict(torch_cpu=torch.arange(16,dtype=torch.uint8),torch_cuda=[torch.arange(8,dtype=torch.uint8)],numpy={'fake':1},python=(3,(1,2),None)),
        request=dict(ordinary_final_step=81000,optimizer_final_step=16000,architecture=[1323,512,512,23],first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323'))

def initial(saved):
    out={k:copy.deepcopy(saved[k]) for k in ('actor_state','optimizer_state','feature_mean','feature_std')}
    out.update(rng_after_restoration=copy.deepcopy(saved['rng']),expansion_performed=False,ordinary_start_step=81000,optimizer_start_step=16000,hidden_width=512,condition='causal')
    return out

@pytest.mark.parametrize('change',[None,'context','added_neuron','old_moment','new_moment','step','group','norm','rng','expansion'])
def test_all_warm512_state_must_be_exact(change):
    s=source();i=initial(s)
    if change=='context':i['actor_state']['0.weight'][0,1000]+=1
    if change=='added_neuron':i['actor_state']['0.weight'][511,0]+=1
    if change=='old_moment':i['optimizer_state']['state'][0]['exp_avg'][0,0]+=1
    if change=='new_moment':i['optimizer_state']['state'][0]['exp_avg_sq'][511,1000]=0
    if change=='step':i['optimizer_state']['state'][5]['step']+=1
    if change=='group':i['optimizer_state']['param_groups'][0]['lr']=1e-5
    if change=='norm':i['feature_std'][0]+=1
    if change=='rng':i['rng_after_restoration']['torch_cuda'][0][0]+=1
    if change=='expansion':i['expansion_performed']=True
    assert bool(math.warm_initial_errors(i,s))==(change is not None)

@pytest.mark.parametrize('change',['source_age','source_shape','source_group','source_request'])
def test_source_contract_rejects_wrong_endpoint(change):
    s=source()
    if change=='source_age':s['optimizer_state']['state'][0]['step']=torch.tensor(6000.)
    if change=='source_shape':s['actor_state']['0.weight']=torch.ones(256,1323)
    if change=='source_group':s['optimizer_state']['param_groups'][0]['fused']=True
    if change=='source_request':s['request']['first_layer_execution']='dense'
    assert math.source_errors(s)

def rows():
    control=np.arange(251,1269,dtype=np.int64);features=np.zeros((1018,1323),np.float32)
    return dict(causal_features=features,features=features[:,:1000],context=features[:,1000:],incoming_prior=features[:,1000:1023],incoming_history=features[:,1023:],
        expert_target=np.zeros((1018,23),np.float64),control=control,source_frame=control+11,phase=np.repeat(np.arange(3,dtype=np.int8),(99,819,100)),
        first_student_state_query=np.arange(1018)==0,plan_control=251+5*((control-251)//5),plan_local=(control-251)%5,
        feedback_clipped=np.zeros((1018,23),bool),native_clipped=np.zeros((1018,23),bool))

@pytest.mark.parametrize('field',[None,'control','source_frame','phase','first_student_state_query','plan_control','plan_local','incoming_prior','incoming_history','expert_target'])
def test_recovery_exact_connected_provenance(field):
    value=rows()
    if field is not None:
        value[field]=value[field].copy();value[field].flat[0]=True if field=='first_student_state_query' else 2
        if field=='first_student_state_query':value[field][1]=True
    if field is None:assert [len(x) for x in math.recovery_schema(value,np.tile([-1.,1.],(23,1)))]==[99,819,100]
    else:
        with pytest.raises(AssertionError):math.recovery_schema(value,np.tile([-1.,1.],(23,1)))

def test_twelve_parities_include_new_corpus():
    output={b:{c:np.zeros((2,23),np.float32) for c in math.CORPORA} for b in ('CPU64','GPU64','ORT64')}
    data=dict(default=np.zeros(23),span=np.ones(23,np.float32),limits=np.tile([-1.,1.],(23,1)))
    output['ORT64']['recovery'][0,0]=2e-5
    result=math.numerical_comparisons(output,data)
    assert len(result)==12 and result['recovery_CPU64_ORT64']>1e-5
    assert all(v==0 for k,v in result.items() if not k.startswith('recovery_'))

def test_independent_D3_equal_cell_metrics(monkeypatch):
    original=dict(nominal_objective=.1,balanced_weighted_objective=.9,full_state_cells=[{'untouched':True}])
    monkeypatch.setattr(math,'metrics',lambda *a:copy.deepcopy(original));monkeypatch.setattr(math,'balanced_metrics',lambda m,*a:m)
    r=rows();ids=math.recovery_schema(r,np.tile([-10.,10.],(23,1)))
    output=dict(recovery=np.repeat(np.array([1.,2.,3.],np.float32),(99,819,100))[:,None].repeat(23,1))
    data=dict(default=np.zeros(23),span=np.ones(23,np.float32),limits=np.tile([-10.,10.],(23,1)),recovery_teacher=r['expert_target'],recovery_cells=ids)
    result=math.recovery_metrics(output,data,None,None,.2)
    assert result['recovery_objective']==14/3
    assert result['combined_recovery_objective']==.9+.2*(14/3)
    for k,v in original.items():assert result[k]==v
    assert [c['normalized_MSE'] for c in result['recovery_cells']]==[1.,4.,9.]

def test_ledger_float32_product_and_float64_combination():
    d=np.array([.123456789,.33333334],np.float32).astype(np.float64)
    weighted=(d.astype(np.float32)*np.float32(.2)).astype(np.float64);old=np.array([.01,.02],np.float64)
    cells=np.repeat(d[:,None],3,1);objectives=np.column_stack((d,weighted,old+weighted))
    result=math.recovery_ledger_expectations(cells,objectives,old,.2)
    np.testing.assert_array_equal(result['weighted'],objectives[:,1]);np.testing.assert_array_equal(result['combined'],objectives[:,2])
    assert np.any(weighted!=d*.2)

@pytest.mark.parametrize('good',[True,False])
def test_path_and_sha_both_bind(good):
    subject=dict(path='/mnt/e/folder/report.json',sha256='a'*64)
    mapping={'E:/folder/report.json':'a'*64 if good else 'b'*64}
    assert exact_member(mapping,subject)==good
    assert not exact_member({'E:/wrong/report.json':'a'*64},subject)

def test_all_selected_ramp_and_cosine_rates():
    rates=[rate(i) for i in range(10000)]
    assert rates[0]==rates[-1]==1e-6 and rates[249]==rates[250]==1e-5
    assert all(a<=b for a,b in zip(rates[:249],rates[1:250]))
    assert all(a>=b for a,b in zip(rates[250:-1],rates[251:]))

def test_literal_producer_restoration_fields():
    tree=ast.parse((PRODUCER/'warm512_restore.py').read_text())
    verify=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_restoration')
    call=next(n.value for n in verify.body if isinstance(n,ast.Return))
    values={k.arg:({'OPTIMIZER_STEP':16000,'SOURCE_STEP':81000,'FORWARD':'split_old256_new256_original1000_plus323'}[k.value.id] if isinstance(k.value,ast.Name) else ast.literal_eval(k.value)) for k in call.keywords}
    assert values==math.restoration_fields()

def test_main_static_corpus_budget_and_no_model_construction():
    tree=ast.parse((HERE/'audit_saved_warm.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    names=[n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else '' for n in calls]
    assert not set(names)&{'forward','backward','step','InferenceSession','WiderContextTarget','AdamW','from_checkpoint','expand_source'}
    text=(HERE/'audit_saved_warm.py').read_text()
    assert 'counts(40000,157040000)' in text and 'counts(1441,368588)' in text and 'len(ledger)==7205' in text
    assert 'for corpus in OLD_CORPORA}' in text and 'len(drift_report)==12' in text
