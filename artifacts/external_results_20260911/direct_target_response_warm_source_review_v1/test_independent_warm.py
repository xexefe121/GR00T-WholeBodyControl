"""Synthetic CPU state/array checks only; no task checkpoint or data reads."""
import ast
import copy
import math
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parent.parent / 'direct_target_causal_response_balanced_student_v1/source_draft_v1'
sys.path.insert(0, str(SOURCE))
from context_model import ContextTarget
from response_restore import check_warm_source, verify_warm_restoration
from response_contract import cosine_rate, BUDGETS, COEFFICIENT
from response_diagnostics import add_balanced_metrics
from balance_contract import GROUP_WEIGHTS
from restoration_support import exact_saved, restore_rng
from training_support import rng_save

torch.set_num_threads(1)

def fixture():
    mean=np.zeros(1323,np.float32); std=np.ones(1323,np.float32)
    model=ContextTarget(mean,std)
    with torch.no_grad():
        for i,p in enumerate(model.parameters()):p.fill_(.01*(i+1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6,weight_decay=1e-5,foreach=False,fused=False)
    for i,p in enumerate(model.parameters()):
        optimizer.state[p]=dict(step=torch.tensor(3000.,dtype=torch.float32),
            exp_avg=torch.full_like(p,.002*(i+1)),exp_avg_sq=torch.full_like(p,.003*(i+1)))
    rng={'synthetic_marker':17}
    saved=dict(kind='direct_absolute_native23_target',condition='causal',ordinary_final_step=68000,
        optimizer_step=3000,context_blinded=False,full_state_coefficient=COEFFICIENT,
        context_order='previous_action23_then_incoming_history300',
        actor_state=copy.deepcopy(model.actor.state_dict()),optimizer_state=copy.deepcopy(optimizer.state_dict()),
        feature_mean=model.feature_mean.clone(),feature_std=model.feature_std.clone(),rng=copy.deepcopy(rng))
    return model,optimizer,saved,rng

def test_exact_warm_state_and_nonzero_context():
    model,optimizer,saved,rng=fixture()
    other=ContextTarget(np.zeros(1323,np.float32),np.ones(1323,np.float32))
    other.actor.load_state_dict(saved['actor_state'])
    warm=torch.optim.AdamW(other.parameters(),lr=1e-5,weight_decay=1e-5,foreach=False,fused=False)
    warm.load_state_dict(copy.deepcopy(saved['optimizer_state']))
    result=verify_warm_restoration(other,warm,saved,rng)
    assert result['warm_optimizer_exact'] and not result['fresh_optimizer']
    assert warm.param_groups[0]['lr']==1e-6
    assert torch.count_nonzero(other.actor[0].weight[:,1000:])==256*323
    exact_saved(warm.state_dict(),saved['optimizer_state'])

@pytest.mark.parametrize('damage',['context','first_moment','second_moment','step','norm','rng'])
def test_corruption_is_rejected(damage):
    model,optimizer,saved,rng=fixture()
    p=next(model.parameters())
    if damage=='context':
        with torch.no_grad():p[:,1000:].zero_()
    elif damage=='first_moment':optimizer.state[p]['exp_avg'].zero_()
    elif damage=='second_moment':optimizer.state[p]['exp_avg_sq'].zero_()
    elif damage=='step':optimizer.state[p]['step'].zero_()
    elif damage=='norm':model.feature_mean[1000]=1
    else:rng['synthetic_marker']=18
    with pytest.raises(ValueError):verify_warm_restoration(model,optimizer,saved,rng)

def test_warm_update_keeps_moments_and_advances_all_six():
    model,optimizer,saved,rng=fixture()
    verify_warm_restoration(model,optimizer,saved,rng)
    before=[p.detach().clone() for p in model.parameters()]
    for group in optimizer.param_groups:group['lr']=cosine_rate(0)
    for p in model.parameters():p.grad=torch.zeros_like(p)
    optimizer.step()  # Synthetic tensors only; no model forward.
    for i,(p,old) in enumerate(zip(model.parameters(),before)):
        state=optimizer.state[p]
        assert state['step'].item()==3001
        torch.testing.assert_close(state['exp_avg'],torch.full_like(p,.002*(i+1)*.9),rtol=2e-7,atol=1e-10)
        torch.testing.assert_close(state['exp_avg_sq'],torch.full_like(p,.003*(i+1)*.999),rtol=2e-7,atol=1e-10)
        # Zero current gradient still moves via retained momentum; a fresh optimizer would not.
        pure_decay=old*(1-cosine_rate(0)*1e-5)
        assert torch.any(p!=pure_decay)
    assert optimizer.param_groups[0]['weight_decay']==1e-5

def test_complete_rng_restore_without_cuda_calls(monkeypatch):
    monkeypatch.setattr(torch.cuda,'device_count',lambda:0)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda values:None)
    before=rng_save()
    torch.rand(5);np.random.rand(5)
    import random
    random.random()
    restore_rng(before)
    exact_saved(rng_save(),before)

def test_inclusive_schedule_and_exact_call_budget():
    rates=np.array([cosine_rate(i) for i in range(3000)])
    expected=np.array([1e-6+.5*(1e-5-1e-6)*(1+math.cos(math.pi*i/2999)) for i in range(3000)])
    np.testing.assert_array_equal(rates,expected)
    assert rates[0]==1e-5 and rates[-1]==1e-6 and np.all(np.diff(rates)<0)
    assert BUDGETS['training_forward_calls']==3*3000
    assert BUDGETS['training_forward_rows']==(9904+1728+3054)*3000
    assert BUDGETS['diagnostic_Torch_calls']==1437*4
    assert BUDGETS['diagnostic_Torch_rows']==367570*4
    assert BUDGETS['diagnostic_ORT_calls']==1437 and BUDGETS['diagnostic_ORT_rows']==367570
    for key in ('calibration_forward_calls','calibration_gradient_calls','native_calls','BFM_calls','manual_export_trace_calls'):assert BUDGETS[key]==0

def metrics_fixture():
    cells=[]
    for d in range(3):
        for p in range(3):
            for g in range(6):
                x=(d*18+p*6+g+1)*.00001
                cells.append(dict(dataset=d,phase=p,tangent_group=g,response_MSE=x,
                    first24_response_MSE=x*2,odd_response_MSE=x*.8,even_response_MSE=x*.2,zero_response_MSE=x*.5))
    return dict(nominal_objective=.125,physical_objective=.25,
        full_state_objective=float(np.mean([c['response_MSE'] for c in cells])),full_state_cells=cells)

def test_balanced_metrics_keep_originals_and_group_fastest():
    source=metrics_fixture();before=copy.deepcopy(source);actual=add_balanced_metrics(source)
    for key in before:assert actual[key]==before[key]
    expected=np.array([c['response_MSE'] for c in before['full_state_cells']])*np.tile(GROUP_WEIGHTS,9)
    np.testing.assert_array_equal([c['weighted_response_MSE'] for c in actual['balanced_full_state_cells']],expected)
    assert actual['balanced_full_state_objective']==expected.mean()
    assert actual['weighted_objective']==.125+COEFFICIENT*before['full_state_objective']+.25
    assert actual['balanced_weighted_objective']==.125+COEFFICIENT*expected.mean()+.25
    assert not np.allclose(expected,np.array([c['response_MSE'] for c in before['full_state_cells']])*np.repeat(GROUP_WEIGHTS,9))

def test_wrong_metric_order_rejected():
    source=metrics_fixture();source['full_state_cells'][0],source['full_state_cells'][1]=source['full_state_cells'][1],source['full_state_cells'][0]
    with pytest.raises(ValueError):add_balanced_metrics(source)

def isolated_initial_drift():
    tree=ast.parse((SOURCE/'train_response_balanced.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='initial_drift')
    namespace={'np':np,'CORPORA':('nominal','full_state','physical')}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<source-only initial_drift>','exec'),namespace)
    return namespace['initial_drift']

@pytest.mark.parametrize('error,passed',[(1e-6,True),(2e-5,False)])
def test_initial_gate_remains_tolerance_not_byte_gate(tmp_path,error,passed):
    request={'restoration_predictions':{}};outputs={}
    for corpus in ('nominal','full_state','physical'):
        path=tmp_path/(corpus+'.npy');np.save(path,np.zeros((2,23),np.float32))
        request['restoration_predictions'][corpus]=str(path);outputs[corpus]=np.full((2,23),error,np.float32)
    result=isolated_initial_drift()(outputs,request,{'span':np.ones(23,np.float32)})
    assert result['passed']==passed and result['byte_gate_required'] is False
    assert all(c['byte_equal'] is False for c in result['corpora'].values())

def test_nonfinite_initial_comparison_rejected(tmp_path):
    request={'restoration_predictions':{}};outputs={}
    for corpus in ('nominal','full_state','physical'):
        path=tmp_path/(corpus+'.npy');np.save(path,np.zeros((2,23),np.float32))
        request['restoration_predictions'][corpus]=str(path);outputs[corpus]=np.full((2,23),np.nan,np.float32)
    with pytest.raises(ValueError):isolated_initial_drift()(outputs,request,{'span':np.ones(23,np.float32)})
