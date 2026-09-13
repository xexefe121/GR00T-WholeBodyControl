"""Synthetic predictions/targets only: no task models, checkpoints or arrays."""
import math
import numpy as np
import pytest
import torch
from balance_contract import (GROUP_NAMES, ZERO_RESPONSE_ENERGIES, MEAN_ZERO_RESPONSE_ENERGY,
                              GROUP_WEIGHTS, CELL_GROUPS)
from balanced_response_objective import balanced_full_state_loss
from full_state_objective import full_state_loss
from direct_objective import nominal_loss

torch.set_num_threads(1)

def fixture():
    g=torch.Generator().manual_seed(914)
    nominal=torch.randn((97,23),generator=g,dtype=torch.float32,requires_grad=True)
    endpoint=torch.randn((1728,23),generator=g,dtype=torch.float32,requires_grad=True)
    indices=torch.arange(864,dtype=torch.int64)%97
    teacher=torch.randn((864,2,23),generator=g,dtype=torch.float64)*.07
    return nominal,endpoint,indices,teacher

def test_fixed_teacher_energies_and_weights_are_exact_float64_rule():
    e=np.array(ZERO_RESPONSE_ENERGIES,np.float64)
    assert e.shape==(6,) and np.isfinite(e).all() and np.all(e>0)
    assert float(e.mean())==MEAN_ZERO_RESPONSE_ENERGY
    assert tuple((e.mean()/e).tolist())==GROUP_WEIGHTS
    assert GROUP_NAMES==('root_position','root_rotation','joint_position','root_linear_velocity','root_angular_velocity','joint_velocity')

def test_fixed_9_by_6_group_broadcast_order():
    assert CELL_GROUPS==tuple(range(6))*9
    nominal=torch.zeros((1,23),dtype=torch.float32)
    endpoint=torch.zeros((1728,23),dtype=torch.float32)
    indices=torch.zeros(864,dtype=torch.int64)
    teacher=torch.arange(1,55,dtype=torch.float64).repeat_interleave(16)[:,None,None].expand(864,2,23)
    got=balanced_full_state_loss(nominal,endpoint,indices,teacher)
    expected=torch.arange(1,55,dtype=torch.float64).square()
    weights=torch.tensor(GROUP_WEIGHTS,dtype=torch.float64).repeat(9)
    assert torch.equal(got.original_cells,expected)
    assert torch.equal(got.weighted_cells,expected*weights)
    assert not torch.equal(got.weighted_cells,expected*torch.tensor(GROUP_WEIGHTS,dtype=torch.float64).repeat_interleave(9))
    assert torch.equal(got.weighted_objective,(expected*weights).mean())

def test_zero_response_energy_preserved_and_equalized_across_groups():
    e=torch.tensor(ZERO_RESPONSE_ENERGIES,dtype=torch.float64)
    w=torch.tensor(GROUP_WEIGHTS,dtype=torch.float64)
    expected=torch.full((6,),MEAN_ZERO_RESPONSE_ENERGY,dtype=torch.float64)
    torch.testing.assert_close(e*w,expected,rtol=2e-16,atol=1e-19)
    nominal=torch.zeros((1,23),dtype=torch.float32);endpoint=torch.zeros((1728,23),dtype=torch.float32)
    teacher=e.sqrt().repeat(9).repeat_interleave(16)[:,None,None].expand(864,2,23)
    got=balanced_full_state_loss(nominal,endpoint,torch.zeros(864,dtype=torch.int64),teacher)
    torch.testing.assert_close(got.weighted_objective,got.original_objective,rtol=4e-16,atol=1e-19)
    torch.testing.assert_close(got.weighted_cells.reshape(9,6).mean(0),expected,rtol=4e-16,atol=1e-19)

def test_original_metrics_and_input_tensors_unchanged():
    args=fixture();before=[v.detach().clone() for v in args]
    original,cells=full_state_loss(*args);got=balanced_full_state_loss(*args)
    assert torch.equal(original,got.original_objective) and torch.equal(cells,got.original_cells)
    for a,b in zip(args,before):assert torch.equal(a,b)
    assert got.weighted_objective.dtype==got.original_objective.dtype==torch.float64

def test_analytic_endpoint_and_repeated_center_gradients():
    nominal,endpoint,indices,teacher=fixture()
    got=balanced_full_state_loss(nominal,endpoint,indices,teacher)
    got.weighted_objective.backward()
    error=endpoint.detach().double().reshape(864,2,23)-nominal.detach()[indices].double()[:,None,:]-teacher
    weights=torch.tensor(GROUP_WEIGHTS,dtype=torch.float64).repeat(9).repeat_interleave(16)[:,None,None]
    expected=2*error*weights/(54*16*2*23)
    torch.testing.assert_close(endpoint.grad,expected.reshape(1728,23).float(),rtol=1e-6,atol=1e-9)
    center_expected=torch.zeros_like(nominal)
    center_expected.index_add_(0,indices,(-expected.sum(1)).float())
    torch.testing.assert_close(nominal.grad,center_expected,rtol=1e-6,atol=2e-9)
    assert torch.count_nonzero(nominal.grad)>0 and torch.count_nonzero(endpoint.grad)>0

@pytest.mark.parametrize('group',range(6))
def test_individual_group_gradient_matches_weighted_original(group):
    args=fixture();nominal,endpoint,indices,teacher=args
    original,cells=full_state_loss(*args)
    chosen=torch.arange(group,54,6)
    expected_loss=cells[chosen].sum()*GROUP_WEIGHTS[group]/54
    expected=torch.autograd.grad(expected_loss,(nominal,endpoint),retain_graph=True)
    got=balanced_full_state_loss(*args)
    actual=torch.autograd.grad(got.weighted_cells[chosen].sum()/54,(nominal,endpoint))
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-6,atol=2e-9)

def test_nominal_term_value_and_gradient_untouched():
    nominal,endpoint,indices,teacher=fixture()
    labels=torch.zeros_like(nominal);cells=[torch.arange(i,97,15) for i in range(15)]
    before,per_before=nominal_loss(nominal,labels,cells)
    before_gradient=torch.autograd.grad(before,nominal)[0]
    balanced_full_state_loss(nominal,endpoint,indices,teacher)
    after,per_after=nominal_loss(nominal,labels,cells)
    after_gradient=torch.autograd.grad(after,nominal)[0]
    assert torch.equal(before,after) and torch.equal(per_before,per_after) and torch.equal(before_gradient,after_gradient)

def test_zero_error_has_zero_weighted_metric_and_gradients():
    nominal=torch.zeros((1,23),dtype=torch.float32,requires_grad=True)
    endpoint=torch.zeros((1728,23),dtype=torch.float32,requires_grad=True)
    result=balanced_full_state_loss(nominal,endpoint,torch.zeros(864,dtype=torch.int64),torch.zeros((864,2,23),dtype=torch.float64))
    assert result.weighted_objective.item()==result.original_objective.item()==0
    result.weighted_objective.backward()
    assert not torch.count_nonzero(nominal.grad) and not torch.count_nonzero(endpoint.grad)

@pytest.mark.parametrize('change', ['endpoint_shape','teacher_dtype','prediction_dtype'])
def test_original_schema_rejections_remain(change):
    args=list(fixture())
    if change=='endpoint_shape':args[1]=args[1][:-1]
    elif change=='teacher_dtype':args[3]=args[3].float()
    else:args[0]=args[0].double()
    with pytest.raises(ValueError):balanced_full_state_loss(*args)
