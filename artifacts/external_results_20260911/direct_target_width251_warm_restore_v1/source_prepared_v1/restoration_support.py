"""Fixed continuation schedule and exact saved-state comparisons; no fitting."""
import math
import random
import numpy as np
import torch

START_STEP=5000
ADDITIONAL_UPDATES=50000
FINAL_STEP=55000
SCHEDULE_ROWS=5000

def source_row(index):
    if type(index) is not int or not 0<=index<ADDITIONAL_UPDATES:raise ValueError('fixed additional-update index')
    return index%SCHEDULE_ROWS

def continuation_rate(index):
    source_row(index)
    return 3e-6+.5*(3e-5-3e-6)*(1+math.cos(math.pi*index/(ADDITIONAL_UPDATES-1)))

def exact_saved(actual,expected,path='state'):
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or actual.dtype!=expected.dtype or actual.shape!=expected.shape:raise ValueError(path+' tensor schema differs')
        if actual.detach().cpu().numpy().tobytes()!=expected.detach().cpu().numpy().tobytes():raise ValueError(path+' tensor bytes differ')
    elif isinstance(expected,np.ndarray):
        if not isinstance(actual,np.ndarray) or actual.dtype!=expected.dtype or actual.shape!=expected.shape or actual.tobytes()!=expected.tobytes():raise ValueError(path+' array differs')
    elif isinstance(expected,dict):
        if not isinstance(actual,dict) or actual.keys()!=expected.keys():raise ValueError(path+' dict keys differ')
        for key in expected:exact_saved(actual[key],expected[key],path+'.'+str(key))
    elif isinstance(expected,(tuple,list)):
        if type(actual)!=type(expected) or len(actual)!=len(expected):raise ValueError(path+' sequence schema differs')
        for i,(a,b) in enumerate(zip(actual,expected)):exact_saved(a,b,path+'.'+str(i))
    elif type(actual)!=type(expected) or actual!=expected:raise ValueError(path+' scalar differs')

def restore_rng(saved):
    torch.set_rng_state(saved['torch_cpu'].cpu())
    if len(saved['torch_cuda'])!=torch.cuda.device_count():raise ValueError('CUDA RNG device count differs')
    torch.cuda.set_rng_state_all([v.cpu() for v in saved['torch_cuda']])
    n=saved['numpy'];np.random.set_state((n['name'],np.asarray(n['keys'],np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
    random.setstate(saved['python'])

def verify_start(model,optimizer,saved,current_rng):
    if saved['ordinary_final_step']!=START_STEP or saved['kind']!='direct_absolute_native23_target':raise ValueError('exact ordinary5000 direct checkpoint required')
    exact_saved(model.actor.state_dict(),saved['actor_state'],'actor')
    exact_saved(optimizer.state_dict(),saved['optimizer_state'],'AdamW')
    states=list(optimizer.state.values())
    if len(states)!=6 or any(int(v['step'])!=START_STEP for v in states):raise ValueError('all six AdamW states must be at5000')
    exact_saved(current_rng,saved['rng'],'RNG')
    for key in ('feature_mean','feature_std'):exact_saved(getattr(model,key).cpu(),saved[key],key)
    return dict(model_exact=True,optimizer_exact=True,optimizer_state_count=6,optimizer_step=START_STEP,RNG_exact=True,normalization_exact=True)
