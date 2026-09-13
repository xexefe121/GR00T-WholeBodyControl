"""Synthetic CPU tensors only; no task checkpoints or task GPU probes."""
import numpy as np
import pytest
import torch
from context_model import ContextTarget,expand_actor
torch.set_num_threads(1)

def make():
    torch.manual_seed(7331)
    old=torch.nn.Sequential(torch.nn.Linear(1000,256),torch.nn.ELU(),torch.nn.Linear(256,256),torch.nn.ELU(),torch.nn.Linear(256,23))
    mean=np.linspace(-.7,.9,1323,dtype=np.float32);std=np.linspace(.1,1.2,1323,dtype=np.float32)
    new=ContextTarget(mean,std);new.actor.load_state_dict(expand_actor(old.state_dict()))
    return old,new

@pytest.mark.parametrize('batch',[1,52,176,238,256,1728,3054,9904])
def test_zero_columns_exact_original_for_diagnostic_and_training_shapes(batch):
    old,new=make();features=torch.randn(batch,1323)
    with torch.no_grad():
        baseline=old(((features[:,:1000]-new.feature_mean[:1000])/new.feature_std[:1000]).contiguous())
        causal=new(features)
        blind=features.clone();blind[:,1000:]=new.feature_mean[1000:]
        blinded=new(blind)
    assert baseline.numpy().tobytes()==causal.numpy().tobytes()==blinded.numpy().tobytes()
    assert len(list(new.parameters()))==6

def test_split_layout_bias_and_differentiable_gradient_contract(monkeypatch):
    old,new=make();features=torch.randn(7,1323);calls=[]
    original=torch.nn.functional.linear
    def record(x,w,b=None):
        calls.append((tuple(x.shape),tuple(w.shape),x.is_contiguous(),w.is_contiguous(),b is None))
        return original(x,w,b)
    monkeypatch.setattr(torch.nn.functional,'linear',record)
    predicted=new(features)
    assert calls[:2]==[((7,1000),(256,1000),True,True,False),((7,323),(256,323),True,True,True)]
    assert len(calls)==4
    predicted.square().mean().backward()
    old(((features[:,:1000]-new.feature_mean[:1000])/new.feature_std[:1000]).contiguous()).square().mean().backward()
    for (name,value),(oldname,oldvalue) in zip(new.actor.named_parameters(),old.named_parameters()):
        assert name==oldname
        actual=value.grad[:,:1000] if name=='0.weight' else value.grad
        assert torch.equal(actual,oldvalue.grad)
    assert torch.count_nonzero(new.actor[0].weight.grad[:,1000:])>0
    new.zero_grad(set_to_none=True);features[:,1000:]=new.feature_mean[1000:]
    new(features).square().mean().backward()
    assert torch.count_nonzero(new.actor[0].weight.grad[:,1000:])==0

def test_nonzero_context_split_matches_mathematical_double_map():
    _,new=make();features=torch.randn(17,1323)
    with torch.no_grad():new.actor[0].weight[:,1000:].normal_(std=.03)
    normalized=(features.double()-new.feature_mean.double())/new.feature_std.double()
    weights={k:v.detach().double() for k,v in new.actor.state_dict().items()}
    linear=torch.nn.functional.linear
    split=linear(normalized[:,:1000].contiguous(),weights['0.weight'][:,:1000].contiguous(),weights['0.bias'])+linear(normalized[:,1000:].contiguous(),weights['0.weight'][:,1000:].contiguous())
    mono=linear(normalized,weights['0.weight'],weights['0.bias'])
    assert torch.max(torch.abs(split-mono))<1e-12
    assert not torch.equal(new(features),new(torch.cat((features[:,:1000],new.feature_mean[1000:].expand(17,-1)),1)))
