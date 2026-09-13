"""CPU synthetic tensors only. No actual checkpoint, dataset, head, or native use."""
import copy
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from width512 import (NAMES,OLD_SHAPES,NEW_SHAPES,SLICES,SEED,expand_source,
                      validate_source,WiderContextTarget,zero_preserving_add,proposed_rate)

torch.set_num_threads(1)


def same(a,b):
    if isinstance(a,torch.Tensor):
        return a.dtype==b.dtype and a.shape==b.shape and a.numpy().tobytes()==b.numpy().tobytes()
    if isinstance(a,np.ndarray):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def source():
    generator=torch.Generator().manual_seed(81317)
    actor={name:torch.randn(shape,generator=generator)*.025 for name,shape in zip(NAMES,OLD_SHAPES)}
    state={index:dict(step=torch.tensor(6000.,dtype=torch.float32),
                      exp_avg=torch.randn(shape,generator=generator)*.0001,
                      exp_avg_sq=torch.rand(shape,generator=generator)*.0001)
           for index,shape in enumerate(OLD_SHAPES)}
    return dict(kind='direct_absolute_native23_target',condition='causal',ordinary_final_step=71000,
                optimizer_step=6000,context_blinded=False,full_state_coefficient=1.8188207859141674,
                context_order='previous_action23_then_incoming_history300',actor_state=actor,
                feature_mean=torch.randn(1323,generator=generator)*.01,
                feature_std=torch.rand(1323,generator=generator)+.5,
                optimizer_state=dict(state=state,param_groups=[dict(params=list(range(6)),lr=1e-6,
                    weight_decay=1e-5,betas=(.9,.999),eps=1e-8,foreach=False,fused=False,
                    amsgrad=False,maximize=False,capturable=False,differentiable=False)]),
                rng=dict(torch_cpu=generator.get_state().clone(),torch_cuda=[torch.tensor([7,4],dtype=torch.uint8)],
                         python=(3,(1,2,3),None),numpy=('synthetic',np.arange(8,dtype=np.uint32),0,0,0.)))


def network(capsule):
    model=WiderContextTarget(capsule['feature_mean'].numpy(),capsule['feature_std'].numpy())
    model.actor.load_state_dict(capsule['actor_state'])
    return model


def old_forward(saved,x):
    a=saved['actor_state'];v=(x-saved['feature_mean'])/saved['feature_std']
    v=F.linear(v[:,:1000].contiguous(),a['0.weight'][:,:1000].contiguous(),a['0.bias'])+F.linear(v[:,1000:].contiguous(),a['0.weight'][:,1000:].contiguous())
    return F.linear(F.elu(F.linear(F.elu(v),a['2.weight'],a['2.bias'])),a['4.weight'],a['4.bias'])


@pytest.mark.parametrize('batch',[1,52,176,238,256,1728,3054,9904])
def test_full_initial_output_byteexact_old_contractions(batch):
    saved=source();capsule=expand_source(saved);model=network(capsule)
    x=torch.randn((batch,1323),generator=torch.Generator().manual_seed(44))
    with torch.no_grad():
        old,new=old_forward(saved,x),model(x)
    assert same(old,new)


def test_seed_global_rng_source_immutability_and_six_blocks():
    saved=source();snapshot=copy.deepcopy(saved);rng=torch.get_rng_state().clone()
    first=expand_source(saved);second=expand_source(saved)
    assert same(saved,snapshot) and same(first,second) and same(torch.get_rng_state(),rng)
    assert same(first['rng_to_restore'],saved['rng']) and first['expansion_seed']==SEED
    assert same(first['optimizer_state']['param_groups'],saved['optimizer_state']['param_groups'])
    for index,(name,shape,block) in enumerate(zip(NAMES,NEW_SHAPES,SLICES)):
        assert tuple(first['actor_state'][name].shape)==shape
        assert same(first['actor_state'][name][block],saved['actor_state'][name])
        for key in ('exp_avg','exp_avg_sq'):
            value=first['optimizer_state']['state'][index][key]
            assert tuple(value.shape)==shape and same(value[block],saved['optimizer_state']['state'][index][key])
            mask=torch.ones(shape,dtype=torch.bool);mask[block]=False
            assert torch.count_nonzero(value[mask])==0
        assert same(first['optimizer_state']['state'][index]['step'],saved['optimizer_state']['state'][index]['step'])
    assert torch.count_nonzero(first['actor_state']['2.weight'][:256,256:])==0
    assert torch.count_nonzero(first['actor_state']['4.weight'][:,256:])==0
    assert torch.count_nonzero(first['actor_state']['0.weight'][256:])>0
    assert torch.count_nonzero(first['actor_state']['2.weight'][256:])>0
    assert not torch.equal(first['actor_state']['0.weight'][256],first['actor_state']['0.weight'][257])
    first['actor_state']['0.weight'][0,0]+=1
    first['optimizer_state']['state'][0]['exp_avg'][0,0]+=1
    first['rng_to_restore']['torch_cpu'][0]=0
    assert same(saved,snapshot)


def test_model_constructor_preserves_global_cpu_rng():
    saved=source();before=torch.get_rng_state().clone();network(expand_source(saved))
    assert same(before,torch.get_rng_state())


def test_gradients_open_zero_outgoing_paths_then_reach_new_inputs():
    saved=source();model=network(expand_source(saved))
    generator=torch.Generator().manual_seed(90)
    x=torch.randn((17,1323),generator=generator)
    cotangent=torch.randn((17,23),generator=generator)
    (model(x)*cotangent).sum().backward()
    grad={name:p.grad.clone() for name,p in model.actor.named_parameters()}
    assert torch.count_nonzero(grad['2.weight'][:256,256:])>0
    assert torch.count_nonzero(grad['4.weight'][:,256:])>0
    assert torch.count_nonzero(grad['0.weight'][256:])==0
    assert torch.count_nonzero(grad['2.weight'][256:])==0
    # Mathematical old-block gradients are compared with their original path.
    old=copy.deepcopy(saved)
    for p in old['actor_state'].values():p.requires_grad_(True)
    (old_forward(old,x)*cotangent).sum().backward()
    for name,block in zip(NAMES,SLICES):
        torch.testing.assert_close(grad[name][block],old['actor_state'][name].grad,rtol=2e-5,atol=1e-6)
    # One explicitly synthetic update, never a task optimizer/model.
    with torch.no_grad():
        for p in model.parameters():p.add_(p.grad,alpha=-.001)
    incoming_after_first=model.actor[0].weight[256:].detach().clone()
    model.zero_grad(set_to_none=True)
    (model(x)*cotangent).sum().backward()
    for name,block in [('0.weight',(slice(256,None),slice(None))),
                       ('0.bias',(slice(256,None),)),
                       ('2.weight',(slice(256,None),slice(None))),
                       ('2.bias',(slice(256,None),))]:
        assert torch.count_nonzero(dict(model.actor.named_parameters())[name].grad[block])>0
    with torch.no_grad():
        for p in model.parameters():p.add_(p.grad,alpha=-.001)
    assert not same(incoming_after_first,model.actor[0].weight[256:].detach())


def test_warm_optimizer_load_and_synthetic_zero_gradient_step():
    saved=source();snapshot=copy.deepcopy(saved);capsule=expand_source(saved);model=network(capsule)
    old_parameters=[torch.nn.Parameter(saved['actor_state'][name].clone()) for name in NAMES]
    old_optimizer=torch.optim.AdamW(old_parameters,lr=1e-6,foreach=False,fused=False)
    new_optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6,foreach=False,fused=False)
    old_optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state']))
    new_optimizer.load_state_dict(copy.deepcopy(capsule['optimizer_state']))
    assert same(new_optimizer.state_dict(),capsule['optimizer_state'])
    for parameters,opt in [(old_parameters,old_optimizer),(list(model.parameters()),new_optimizer)]:
        for p in parameters:p.grad=torch.zeros_like(p)
        opt.step()
    for old,new,block in zip(old_parameters,model.parameters(),SLICES):
        assert same(old.detach(),new.detach()[block])
    assert all(float(value['step'])==6001 for value in new_optimizer.state.values())
    assert same(saved,snapshot)


def test_signed_zero_and_added_path_gradient_preserved():
    original=torch.tensor([-0.,+0.,1.,-1.],requires_grad=True)
    added=torch.tensor([+0.,-0.,+0.,-0.],requires_grad=True)
    result=zero_preserving_add(original,added)
    assert same(result.detach(),original.detach())
    result.sum().backward()
    assert torch.equal(original.grad,torch.ones(4)) and torch.equal(added.grad,torch.ones(4))


def test_zero_add_first_and_second_derivatives():
    a=torch.tensor([.3,-.2],dtype=torch.float64,requires_grad=True)
    b=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(zero_preserving_add,(a,b))
    assert torch.autograd.gradgradcheck(zero_preserving_add,(a,b))


def test_nonzero_added_path_matches_monolithic_real_map():
    capsule=expand_source(source());a=capsule['actor_state']
    a['2.weight'][:256,256:].fill_(.0001);a['4.weight'][:,256:].fill_(.0002)
    model=network(capsule);x=torch.randn((31,1323),generator=torch.Generator().manual_seed(71))
    with torch.no_grad():
        value=(x.to(torch.float64)-capsule['feature_mean'].double())/capsule['feature_std'].double()
        for index in (0,2,4):
            value=F.linear(value,a[str(index)+'.weight'].double(),a[str(index)+'.bias'].double())
            if index!=4:value=F.elu(value)
        torch.testing.assert_close(model(x).double(),value,rtol=1e-5,atol=1e-6)


def test_nonzero_outputs_and_all_six_gradients_match_dense512():
    capsule=expand_source(source());a=capsule['actor_state']
    a['2.weight'][:256,256:].fill_(.0001);a['4.weight'][:,256:].fill_(.0002)
    model=network(capsule)
    dense={name:value.clone().requires_grad_(True) for name,value in a.items()}
    generator=torch.Generator().manual_seed(110)
    x=torch.randn((13,1323),generator=generator)
    cotangent=torch.randn((13,23),generator=generator)
    value=(x-capsule['feature_mean'])/capsule['feature_std']
    for index in (0,2,4):
        value=F.linear(value,dense[str(index)+'.weight'],dense[str(index)+'.bias'])
        if index!=4:value=F.elu(value)
    split=model(x)
    # FP32 contractions have different reduction layouts. This synthetic
    # tolerance checks algebra; it is not the later1e-5rad export gate.
    torch.testing.assert_close(split,value,rtol=2e-5,atol=1e-6)
    (split*cotangent).sum().backward();(value*cotangent).sum().backward()
    for name,param in model.actor.named_parameters():
        torch.testing.assert_close(param.grad,dense[name].grad,rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize('case',['step','moment_shape','negative_variance','duplicate_id','dtype','norm','condition','order','lr','missing_rng'])
def test_corrupted_warm_metadata_rejected(case):
    saved=source()
    if case=='step':saved['optimizer_state']['state'][0]['step'].fill_(5999)
    elif case=='moment_shape':saved['optimizer_state']['state'][0]['exp_avg']=torch.zeros(1)
    elif case=='negative_variance':saved['optimizer_state']['state'][0]['exp_avg_sq'][0,0]=-1
    elif case=='duplicate_id':saved['optimizer_state']['param_groups'][0]['params'][-1]=0
    elif case=='dtype':saved['actor_state']['0.weight']=saved['actor_state']['0.weight'].double()
    elif case=='norm':saved['feature_std'][0]=0
    elif case=='condition':saved['condition']='blinded'
    elif case=='order':saved['actor_state']=dict(reversed(list(saved['actor_state'].items())))
    elif case=='lr':saved['optimizer_state']['param_groups'][0]['lr']=1e-5
    elif case=='missing_rng':saved['rng']={}
    with pytest.raises(ValueError):expand_source(saved)


def test_fixed_seed_and_schedule_boundaries():
    with pytest.raises(ValueError):expand_source(source(),seed=1)
    rates=np.array([proposed_rate(i) for i in range(10000)])
    assert rates[0]==rates[-1]==1e-6 and rates[249]==rates[250]==1e-5
    assert np.all(np.diff(rates[:250])>0) and np.all(np.diff(rates[250:])<0)
    assert 10000*(9904+1728+3054)==146860000
    for index in [-1,10000,False,1.]:
        with pytest.raises(ValueError):proposed_rate(index)
