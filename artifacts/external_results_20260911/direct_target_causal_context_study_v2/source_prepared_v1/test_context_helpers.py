"""Synthetic data/model tests only; never load task checkpoints or execute physics."""
import copy
import numpy as np
import pytest
import torch
torch.set_num_threads(1)
from context_contract import (CONDITIONS,UPDATES,COEFFICIENT,cosine_rate,join_context,weighted_context_moments,
    history_flat,advance_history,HISTORY_WIDTHS,inverse_applied_action,fixed_schedule,ExpandedFeatures)
from context_model import ContextTarget,expand_actor,assert_blinded_columns

def test_exact_fixed_protocol():
    assert CONDITIONS==('blinded','causal') and UPDATES==3000 and COEFFICIENT==1.8188207859141674
    assert cosine_rate(0)==1e-5 and cosine_rate(2999)==1e-6
    rates=np.array([cosine_rate(i) for i in range(3000)])
    assert np.all(np.diff(rates)<0)
    with pytest.raises(ValueError):cosine_rate(3000)

def test_balanced_moments_include_between_cell_variance_and_unequal_sizes():
    counts=np.arange(1,16);cell=np.repeat(np.arange(15),counts)
    values=np.repeat(cell[:,None],323,axis=1).astype(np.float32)
    ids=[np.flatnonzero(cell==i) for i in range(15)]
    mean,std,mean64,var64=weighted_context_moments(values,ids)
    assert np.array_equal(mean64,np.full(323,7.))
    assert np.allclose(var64,np.full(323,np.var(np.arange(15,dtype=np.float64))))
    assert mean.dtype==std.dtype==np.float32
    assert not np.isclose(mean64[0],values[:,0].mean())

def test_std_floor_and_invalid_cell_partition():
    x=np.ones((15,323),np.float32);cells=[np.array([i]) for i in range(15)]
    m,s,_,v=weighted_context_moments(x,cells)
    assert np.all(s==np.float32(.05)) and np.all(v==0)
    cells[-1]=cells[0]
    with pytest.raises(ValueError):weighted_context_moments(x,cells)

def test_prior_before_history_and_no_clip_or_roundtrip():
    p=np.arange(46,dtype=np.float32).reshape(2,23)-12
    h=np.arange(600,dtype=np.float32).reshape(2,300)
    out=join_context(p,h)
    assert out[:,:23].tobytes()==p.tobytes() and out[:,23:].tobytes()==h.tobytes()
    assert out.max()>5
    with pytest.raises(ValueError):join_context(p.astype(np.float64),h)

def test_history_advances_once_with_original_bfm_state_order():
    named={key:np.arange(2*4*w,dtype=np.float32).reshape(2,4,w) for key,w in HISTORY_WIDTHS.items()}
    original={key:v.copy() for key,v in named.items()};state=np.arange(104,dtype=np.float32).reshape(2,52)+1000;prior=np.full((2,23),99,np.float32)
    advanced=advance_history(named,state,prior)
    assert np.array_equal(advanced['actions'][:,0],prior)
    assert np.array_equal(advanced['base_ang_vel'][:,0],state[:,49:52])
    assert np.array_equal(advanced['projected_gravity'][:,0],state[:,46:49])
    for key in named:
        assert np.array_equal(advanced[key][:,1:],named[key][:,:3])
        assert np.array_equal(named[key],original[key])
    assert history_flat(named).shape==(2,300)
    assert np.array_equal(history_flat(named)[:,:92],named['actions'].reshape(2,92))

def test_actual_target_inverse_differs_from_raw_proposal_when_clipped():
    contract=dict(default_q=[0.]*23,kp=[2.]*23,training_effort=[4.]*23)
    raw=np.full((1,23),2.,np.float64);applied=np.clip(raw,-.5,.5)
    assert np.array_equal(inverse_applied_action(applied,contract),np.ones((1,23),np.float32))
    assert not np.array_equal(inverse_applied_action(raw,contract),inverse_applied_action(applied,contract))

def test_probe_context_fixed_and_blind_normalizes_to_zero_at_both_precisions():
    current=np.arange(8*1000,dtype=np.float32).reshape(8,1000);context=np.arange(3*323,dtype=np.float32).reshape(3,323)
    indices=np.array([0,0,0,1,1,1,2,2],np.int64);mean=np.linspace(-3,7,323,dtype=np.float32);std=np.linspace(.05,4,323,dtype=np.float32)
    true=ExpandedFeatures(current,context,indices,'causal',mean);blind=ExpandedFeatures(current,context,indices,'blinded',mean)
    assert np.array_equal(true[:][:,:1000],current)
    assert np.array_equal(true[0][1000:],true[2][1000:])
    assert np.array_equal(true[3][1000:],context[1])
    for dtype in (np.float32,np.float64):assert np.all((blind[:][:,1000:].astype(dtype)-mean.astype(dtype))/std.astype(dtype)==0)
    assert current[0,0]==0 and context[0,0]==0

def test_saved_schedule_prefix_only():
    rows=np.broadcast_to(np.arange(864,dtype=np.int32),(10000,864)).copy();axes=np.zeros((10000,864),np.int8)
    a,b=fixed_schedule(rows,axes)
    assert a.shape==(3000,864) and a.tobytes()==rows[:3000].tobytes() and b.tobytes()==axes[:3000].tobytes()
    a[0,0]=-1;assert rows[0,0]==0

def synthetic_actor():
    torch.manual_seed(17)
    net=torch.nn.Sequential(torch.nn.Linear(1000,256),torch.nn.ELU(),torch.nn.Linear(256,256),torch.nn.ELU(),torch.nn.Linear(256,23))
    return net

def test_zero_column_expansion_preserves_all_old_tensors():
    old=synthetic_actor().state_dict();expanded=expand_actor(old)
    assert expanded['0.weight'].shape==(256,1323)
    for key,value in old.items():assert torch.equal(expanded[key][:,:1000] if key=='0.weight' else expanded[key],value)
    assert_blinded_columns(expanded)

def test_identical_initial_context_conditions_and_blinded_gradient_zero():
    old=synthetic_actor();expanded=expand_actor(old.state_dict())
    model=ContextTarget(np.zeros(1323,np.float32),np.ones(1323,np.float32));model.actor.load_state_dict(expanded)
    x=torch.randn(4,1000);h=torch.randn(4,323)
    blind=torch.cat((x,torch.zeros_like(h)),1);causal=torch.cat((x,h),1)
    assert torch.equal(model(blind),model(causal))
    assert torch.max(torch.abs(old(x)-model(blind)))<1e-5
    model(blind).square().mean().backward()
    assert torch.count_nonzero(model.actor[0].weight.grad[:,1000:])==0
    model.zero_grad(set_to_none=True);model(causal).square().mean().backward()
    assert torch.count_nonzero(model.actor[0].weight.grad[:,1000:])>0

def test_context_shape_rejects_old_or_wrong_dtype_public_input():
    model=ContextTarget(np.zeros(1323,np.float32),np.ones(1323,np.float32))
    with pytest.raises(ValueError):model(torch.zeros(1,1000))
    with pytest.raises(ValueError):model(torch.zeros(1,1323,dtype=torch.float64))

def test_chronology_detects_shifted_named_and_flat_history():
    from context_data import verify_chronology
    c=dict(default_q=[0.]*23,kp=[2.]*23,training_effort=[4.]*23)
    state=np.arange(3*52,dtype=np.float32).reshape(3,52);target=np.full((3,23),.1,np.float64)
    prior=np.full((3,23),.2,np.float32)
    named={key:np.zeros((3,4,w),np.float32) for key,w in HISTORY_WIDTHS.items()}
    for i in range(2):
        advanced=advance_history({key:value[i:i+1] for key,value in named.items()},state[i:i+1],prior[i:i+1])
        for key in named:named[key][i+1]=advanced[key][0]
    rows=dict(control=np.arange(250,253),state=state,previous_action=prior,expert_target=target,named=named)
    assert verify_chronology(rows,c)==2
    bad=copy.deepcopy(rows)
    for key in named:bad['named'][key][1:]=bad['named'][key][:-1].copy()
    with pytest.raises(ValueError):verify_chronology(bad,c)

def test_actual_ort_option_fields_used_by_fitter_exist():
    import ast
    from pathlib import Path
    import onnxruntime as ort
    tree=ast.parse((Path(__file__).parent/'train_context_pair.py').read_text())
    attributes={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='options'}
    assert attributes=={'intra_op_num_threads','inter_op_num_threads','execution_mode'}
    options=ort.SessionOptions()
    assert all(hasattr(options,name) for name in attributes)
