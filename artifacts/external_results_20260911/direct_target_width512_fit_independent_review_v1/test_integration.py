"""Independent synthetic integration checks. No task checkpoint/data/CUDA calls."""
import ast
import copy
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch

SOURCE=Path(os.environ['WIDTH_SOURCE'])
sys.path.insert(0,str(SOURCE))
from width512 import NAMES,OLD_SHAPES,NEW_SHAPES,SLICES,expand_source,WiderContextTarget,proposed_rate
from width_restore import verify_width_restoration
from width512_promoted import from_checkpoint
from response_contract import BUDGETS,START_STEP,FINAL_STEP,UPDATES,OPTIMIZER_START,OPTIMIZER_FINAL
torch.set_num_threads(1)

def source_fixture():
    generator=torch.Generator().manual_seed(375)
    actor={name:torch.randn(shape,generator=generator)*.01 for name,shape in zip(NAMES,OLD_SHAPES)}
    state={i:dict(step=torch.tensor(6000.,dtype=torch.float32),exp_avg=torch.randn(shape,generator=generator)*1e-4,
        exp_avg_sq=torch.rand(shape,generator=generator)*1e-3) for i,shape in enumerate(OLD_SHAPES)}
    group=dict(params=list(range(6)),lr=1e-6,weight_decay=1e-5,betas=(.9,.999),eps=1e-8,
        foreach=False,fused=False,amsgrad=False,maximize=False,capturable=False,differentiable=False,decoupled_weight_decay=True)
    return dict(kind='direct_absolute_native23_target',condition='causal',ordinary_final_step=71000,optimizer_step=6000,
        context_blinded=False,full_state_coefficient=1.8188207859141674,context_order='previous_action23_then_incoming_history300',
        actor_state=actor,optimizer_state=dict(state=state,param_groups=[group]),feature_mean=torch.zeros(1323),
        feature_std=torch.ones(1323),rng=dict(synthetic=torch.tensor([4,2,1],dtype=torch.uint8)))

def restoration_fixture():
    source=source_fixture();capsule=expand_source(source)
    model=WiderContextTarget(source['feature_mean'].numpy(),source['feature_std'].numpy())
    model.actor.load_state_dict(capsule['actor_state'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,foreach=False,fused=False)
    optimizer.load_state_dict(copy.deepcopy(capsule['optimizer_state']))
    return source,capsule,model,optimizer

def function(name):
    tree=ast.parse((SOURCE/'train_response_balanced.py').read_text())
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)

def initial_drift():
    scope=dict(np=np,CORPORA=('nominal','full_state','physical'))
    exec(compile(ast.Module(body=[function('initial_drift')],type_ignores=[]),'actual_initial_drift','exec'),scope)
    return scope['initial_drift']

def test_actual_warm_restoration_and_capsule_ownership():
    source,capsule,model,optimizer=restoration_fixture()
    result=verify_width_restoration(model,optimizer,source,capsule,copy.deepcopy(source['rng']))
    assert result['optimizer_start_step']==result['shared_optimizer_step_for_new_entries']==6000
    assert result['old_optimizer_blocks_exact'] and result['new_optimizer_moments_zero']
    optimizer.state[next(iter(model.parameters()))]['exp_avg'][0,0]+=1
    assert not torch.equal(optimizer.state_dict()['state'][0]['exp_avg'],capsule['optimizer_state']['state'][0]['exp_avg'])
    with pytest.raises(ValueError):verify_width_restoration(model,optimizer,source,capsule,source['rng'])

@pytest.mark.parametrize('mutation',['old_actor','old_moment','new_moment','new_outgoing','step','rng','group','normalization'])
def test_coherent_capsule_corruption_still_rejected(mutation):
    source,capsule,model,optimizer=restoration_fixture()
    if mutation=='old_actor':capsule['actor_state']['0.weight'][0,0]+=1;model.actor.load_state_dict(capsule['actor_state'])
    elif mutation in ('old_moment','new_moment'):
        column=0 if mutation=='old_moment' else 256
        capsule['optimizer_state']['state'][2]['exp_avg'][0,column]+=1
        optimizer.load_state_dict(copy.deepcopy(capsule['optimizer_state']))
    elif mutation=='new_outgoing':
        capsule['actor_state']['4.weight'][0,256]=1;model.actor.load_state_dict(capsule['actor_state'])
    elif mutation=='step':
        capsule['optimizer_state']['state'][0]['step'].fill_(6001)
        optimizer.load_state_dict(copy.deepcopy(capsule['optimizer_state']))
    elif mutation=='group':optimizer.param_groups[0]['lr']=1e-5
    elif mutation=='normalization':model.feature_mean[0]=1
    rng=copy.deepcopy(source['rng'])
    if mutation=='rng':rng['synthetic'][0]=9
    with pytest.raises(ValueError):verify_width_restoration(model,optimizer,source,capsule,rng)

def test_initial_gate_accepts_disclosed_subthreshold_nonbyte(tmp_path):
    prior=np.zeros((2,23),np.float32);actual=prior.copy();actual[0,0]=np.float32(5e-6)
    path=tmp_path/'synthetic.npy';np.save(path,prior)
    result=initial_drift()({k:actual for k in ('nominal','full_state','physical')},
        {'restoration_predictions':{k:str(path) for k in ('nominal','full_state','physical')}},{'span':np.ones(23,np.float32)})
    assert result['passed'] and not result['byte_gate_required']
    assert all(not row['byte_equal'] for row in result['corpora'].values())

@pytest.mark.parametrize('value',[np.float32(1.01e-5),np.float32(np.inf),np.float32(np.nan)])
def test_initial_gate_rejects_above_threshold_or_nonfinite(tmp_path,value):
    prior=np.zeros((1,23),np.float32);actual=prior.copy();actual[0,0]=value
    path=tmp_path/'synthetic.npy';np.save(path,prior)
    args=({k:actual for k in ('nominal','full_state','physical')},
        {'restoration_predictions':{k:str(path) for k in ('nominal','full_state','physical')}},{'span':np.ones(23,np.float32)})
    if np.isfinite(value):assert initial_drift()(*args)['passed'] is False
    else:
        with pytest.raises(ValueError):initial_drift()(*args)

def test_full_schedule_and_inclusive_rate_budget():
    assert (START_STEP,FINAL_STEP,UPDATES,OPTIMIZER_START,OPTIMIZER_FINAL)==(71000,81000,10000,6000,16000)
    rates=[proposed_rate(i) for i in range(10000)]
    assert rates[0]==rates[-1]==1e-6 and rates[249]==rates[250]==1e-5
    assert all(a<=b for a,b in zip(rates[:249],rates[1:250]))
    assert all(a>=b for a,b in zip(rates[250:-1],rates[251:]))
    assert BUDGETS['training_forward_calls']==3*10000
    assert BUDGETS['training_forward_rows']==10000*(9904+2*864+3054)
    assert BUDGETS['calibration_forward_calls']==BUDGETS['calibration_gradient_calls']==0
    for i in (-1,10000,True):
        with pytest.raises(ValueError):proposed_rate(i)

def test_fp64_export_uses_opened_new_paths_and_owned_weights():
    source=source_fixture();capsule=expand_source(source)
    saved=dict(kind='direct_absolute_native23_target',ordinary_final_step=81000,actor_state=capsule['actor_state'],
        feature_mean=capsule['feature_mean'],feature_std=capsule['feature_std'])
    baseline=from_checkpoint(saved)
    saved['actor_state']['2.weight'][:256,256:].fill_(.002)
    saved['actor_state']['4.weight'][:,256:].fill_(.003)
    promoted=from_checkpoint(saved)
    x=torch.randn((7,1323),generator=torch.Generator().manual_seed(993))
    a={name:value.double() for name,value in saved['actor_state'].items()}
    F=torch.nn.functional
    v=(x.double()-saved['feature_mean'].double())/saved['feature_std'].double()
    old=F.elu(F.linear(v[:,:1000],a['0.weight'][:256,:1000],a['0.bias'][:256])+F.linear(v[:,1000:],a['0.weight'][:256,1000:]))
    new=F.elu(F.linear(v,a['0.weight'][256:],a['0.bias'][256:]))
    old2=F.elu(F.linear(old,a['2.weight'][:256,:256],a['2.bias'][:256])+F.linear(new,a['2.weight'][:256,256:]))
    new2=F.elu(F.linear(torch.cat((old,new),1),a['2.weight'][256:],a['2.bias'][256:]))
    split=F.linear(old2,a['4.weight'][:,:256],a['4.bias'])+F.linear(new2,a['4.weight'][:,256:])
    dense=(x.double()-promoted.feature_mean)/promoted.feature_std
    for i in range(3):
        dense=F.linear(dense,getattr(promoted,'w'+str(i)),getattr(promoted,'b'+str(i)))
        if i<2:dense=F.elu(dense)
    torch.testing.assert_close(dense,split,rtol=1e-11,atol=1e-12)
    with torch.no_grad():
        actual=promoted(x);old_output=baseline(x)
    assert not torch.equal(actual,old_output)
    assert torch.equal(actual,dense.float())
    before=actual.clone()
    saved['actor_state']['4.weight'].zero_()
    with torch.no_grad():assert torch.equal(promoted(x),before)

@pytest.mark.parametrize('failure',['inside_step','synchronize',None])
def test_actual_optimizer_transition_preserves_partial_flags(failure):
    loop=next(n for n in ast.walk(function('run_continuation')) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='index')
    begin=next(i for i,n in enumerate(loop.body) if ast.unparse(n)=="state['optimizer_call_started'] = True")
    end=next(i for i,n in enumerate(loop.body) if ast.unparse(n)=="state['completed_updates'] = index + 1")
    state=dict(optimizer_call_started=False,optimizer_call_returned=False,optimizer_synchronized=False,completed_updates=0)
    warm_steps=[6000]*6
    def step():
        warm_steps[0]+=1
        if failure=='inside_step':raise RuntimeError('synthetic partial AdamW')
        for i in range(1,6):warm_steps[i]+=1
    def sync():
        if failure=='synchronize':raise RuntimeError('synthetic synchronization failure')
    scope=dict(state=state,index=0,optimizer=SimpleNamespace(step=step),torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=sync)))
    code=compile(ast.Module(body=loop.body[begin:end+1],type_ignores=[]),'actual_optimizer_transition','exec')
    if failure:
        with pytest.raises(RuntimeError):exec(code,scope)
        assert state['completed_updates']==0 and state['optimizer_synchronized'] is False
        assert state['optimizer_call_started'] is True
        assert state['optimizer_call_returned']==(failure=='synchronize')
    else:exec(code,scope);assert state['completed_updates']==1 and state['optimizer_synchronized']
    assert warm_steps==([6001,6000,6000,6000,6000,6000] if failure=='inside_step' else [6001]*6)
    assert begin<end<next(i for i,n in enumerate(loop.body) if ast.unparse(n)=="used_centers[index] = sc[index]")

def test_gate_before_training_and_fp64_graph_release_scope():
    tree=function('run_continuation');text=ast.unparse(tree)
    assert text.index("if not drift['passed']")<text.index('for index in range(UPDATES)')
    assert text.index('verify_width_restoration(')<text.index("run_backend('initial_GPU32'")
    assert text.index('copy.deepcopy(expansion[\'optimizer_state\'])')<text.index('verify_width_restoration(')
    assert "state['optimizer_step_counters']" in text and 'resumable=False' in text and 'automatic_retry=False' in text
    parity_tree=ast.parse((SOURCE/'context_diagnostics.py').read_text())
    parity=next(n for n in parity_tree.body if isinstance(n,ast.FunctionDef) and n.name=='parity')
    assert ast.dump(parity,include_attributes=False)==ast.dump(next(n for n in ast.parse((SOURCE.parent.parent/'direct_target_causal_response_balanced_student_v2/source_snapshot_v1/context_diagnostics.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='parity'),include_attributes=False)
