"""Bounded synthetic restoration and fixed-schedule tests; no task model."""
import copy
import random
import unittest
from unittest.mock import patch
import numpy as np
import torch
from continuation_contract import source_row,continuation_rate,exact_saved,restore_rng,verify_start

def rng():
    n=np.random.get_state()
    return dict(torch_cpu=torch.get_rng_state().clone(),torch_cuda=[],
        numpy=dict(name=n[0],keys=n[1].tolist(),position=n[2],has_gauss=n[3],cached_gaussian=n[4]),python=random.getstate())

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.actor=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.ELU(),torch.nn.Linear(3,3),torch.nn.ELU(),torch.nn.Linear(3,1))
        self.register_buffer('feature_mean',torch.tensor([.25,-.25]));self.register_buffer('feature_std',torch.tensor([.5,2.]))

def fixture():
    model=Tiny();opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-5,foreach=False,fused=False)
    # Exactly one synthetic update creates six moment states, then a fixture counter is assigned.
    model.actor(torch.ones(2,2)).sum().backward();opt.step()
    for value in opt.state.values():value['step'].fill_(5000)
    saved=dict(kind='direct_absolute_native23_target',ordinary_final_step=5000,
        actor_state=copy.deepcopy(model.actor.state_dict()),optimizer_state=copy.deepcopy(opt.state_dict()),rng=rng(),
        feature_mean=model.feature_mean.clone(),feature_std=model.feature_std.clone())
    return model,opt,saved

class ContinuationTests(unittest.TestCase):
    def test_all_50000_schedule_ids(self):
        actual=np.array([source_row(i) for i in range(50000)])
        np.testing.assert_array_equal(actual,np.tile(np.arange(5000),10))
        for invalid in [-1,50000,True,1.0]:
            with self.assertRaises(ValueError):source_row(invalid)
    def test_inclusive_cosine(self):
        values=np.array([continuation_rate(i) for i in range(50000)])
        self.assertAlmostEqual(values[0],3e-5,places=19);self.assertEqual(values[-1],3e-6)
        self.assertTrue(np.all(np.diff(values)<0));self.assertAlmostEqual(values[24999]+values[25000],3.3e-5,places=19)
    def test_exact_bytes_and_schema(self):
        value={'tensor':torch.tensor([0.,-0.]),'array':np.array([1,2],np.int32),'nested':(True,[7])}
        exact_saved(copy.deepcopy(value),value)
        for key,new in [('tensor',torch.tensor([0.,0.])),('array',np.array([1,2],np.int64)),('nested',[True,[7]])]:
            changed=copy.deepcopy(value);changed[key]=new
            with self.assertRaises(ValueError):exact_saved(changed,value)
    def test_full_six_state_restore(self):
        _,_,saved=fixture();model=Tiny();model.actor.load_state_dict(saved['actor_state'])
        opt=torch.optim.AdamW(model.parameters(),lr=1.,foreach=False,fused=False);opt.load_state_dict(saved['optimizer_state'])
        with patch('torch.cuda.device_count',return_value=0),patch('torch.cuda.set_rng_state_all') as set_cuda:
            restore_rng(saved['rng']);set_cuda.assert_called_once_with([])
        result=verify_start(model,opt,saved,rng())
        self.assertTrue(result['optimizer_exact']);self.assertEqual(result['optimizer_state_count'],6)
    def test_reject_weight_step_norm_rng(self):
        for mode in ['weight','step','norm','rng','missing']:
            model,opt,saved=fixture();current=copy.deepcopy(saved['rng'])
            if mode=='weight':
                with torch.no_grad():next(model.parameters()).add_(1.)
            elif mode=='step':next(iter(opt.state.values()))['step'].fill_(4999)
            elif mode=='norm':model.feature_mean.add_(1.)
            elif mode=='rng':current['torch_cpu'][0]^=1
            elif mode=='missing':opt.state.pop(next(iter(opt.state)))
            with self.assertRaises(ValueError):verify_start(model,opt,saved,current)
    def test_reject_other_checkpoint(self):
        model,opt,saved=fixture();saved['ordinary_final_step']=4999
        with self.assertRaises(ValueError):verify_start(model,opt,saved,saved['rng'])
    def test_rng_device_mismatch(self):
        with patch('torch.cuda.device_count',return_value=1):
            with self.assertRaises(ValueError):restore_rng(rng())

if __name__=='__main__':unittest.main()
