import copy
import numpy as np
import pytest
import torch
from audit_restoration import differences

def tree():
    return {'actor':torch.tensor([[1.,2.]],dtype=torch.float32),
            'optimizer':{'state':{0:{'step':torch.tensor(5000.),'exp_avg':torch.tensor([.125]),'exp_avg_sq':torch.tensor([.25])}},
                         'param_groups':[{'lr':3e-5,'betas':(.9,.999),'foreach':False}]},
            'rng':{'cpu':torch.tensor([3,8],dtype=torch.uint8),'numpy':np.array([4,5],dtype=np.uint32),'python':(3,(1,2),None)}}

def test_equal_independent_tree():
    original=tree()
    assert differences(copy.deepcopy(original),original)==[]

@pytest.mark.parametrize('mutation',[
    lambda x:x['actor'].add_(.01),
    lambda x:x['optimizer']['state'][0]['step'].add_(1),
    lambda x:x['optimizer']['state'][0]['exp_avg_sq'].zero_(),
    lambda x:x['rng']['cpu'].zero_(),
    lambda x:x['rng'].__setitem__('python',[3,(1,2),None]),
    lambda x:x['rng'].__setitem__('numpy',np.array([4,5],dtype=np.int32)),
    lambda x:x['optimizer']['param_groups'][0].__setitem__('foreach',0),
    lambda x:x['optimizer']['state'].clear(),
])
def test_detects_restoration_fault(mutation):
    expected=tree();actual=copy.deepcopy(expected);mutation(actual)
    assert differences(actual,expected)

def test_signed_zero_bytes_preserved():
    assert differences({'value':0.},{'value':-0.})
