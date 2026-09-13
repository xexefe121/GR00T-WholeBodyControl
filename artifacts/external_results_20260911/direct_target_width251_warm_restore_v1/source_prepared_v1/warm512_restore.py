"""Strict warm512 state restoration; no checkpoint I/O or forward/update calls."""
import copy
import math
import random
import numpy as np
import torch
from width512 import WiderContextTarget,NAMES,NEW_SHAPES
from restoration_support import exact_saved,restore_rng
from balance_contract import GROUP_WEIGHTS,WEIGHT_RULE

SOURCE_STEP=81000
OPTIMIZER_STEP=16000
COEFFICIENT=1.8188207859141674
FORWARD='split_old256_new256_original1000_plus323'

def tensor(value,shape,label,dtype=torch.float32):
    if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=dtype or tuple(value.shape)!=shape:
        raise ValueError(label+': expected CPU tensor schema')
    if not bool(torch.isfinite(value).all()):raise ValueError(label+': nonfinite source tensor')

def validate_rng(rng):
    if not isinstance(rng,dict) or set(rng)!= {'torch_cpu','torch_cuda','numpy','python'}:
        raise ValueError('Exact source global RNG fields required')
    for label,value in [('torch_cpu',rng['torch_cpu']),*[(f'torch_cuda_{i}',v) for i,v in enumerate(rng['torch_cuda'])]]:
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=torch.uint8 or value.ndim!=1 or value.numel()==0:
            raise ValueError('Saved RNG byte tensor schema: '+label)
    if not isinstance(rng['torch_cuda'],list):raise ValueError('CUDA RNG list required')
    try:torch.Generator(device='cpu').set_state(rng['torch_cpu'])
    except RuntimeError as exc:raise ValueError('Saved CPU RNG state encoding') from exc
    n=rng['numpy']
    if (set(n)!= {'name','keys','position','has_gauss','cached_gaussian'} or n['name']!='MT19937'
        or not isinstance(n['keys'],list) or len(n['keys'])!=624
        or any(type(v) is not int or not 0<=v<=2**32-1 for v in n['keys'])
        or type(n['position']) is not int or not 0<=n['position']<=624
        or type(n['has_gauss']) is not int or n['has_gauss'] not in (0,1)
        or type(n['cached_gaussian']) is not float or not math.isfinite(n['cached_gaussian'])):
        raise ValueError('Saved NumPy RNG schema')
    # Validate Python's opaque RNG tuple on a local object, not global state.
    if not isinstance(rng['python'],tuple):raise ValueError('Saved Python RNG tuple required')
    local=random.Random(0)
    try:local.setstate(rng['python'])
    except (TypeError,ValueError) as exc:raise ValueError('Saved Python RNG schema') from exc

def validate_source(saved):
    expected=dict(kind='direct_absolute_native23_target',condition='causal',ordinary_final_step=SOURCE_STEP,
        optimizer_step=OPTIMIZER_STEP,fresh_optimizer=False,hidden_width=512,architecture=[1323,512,512,23],
        context_blinded=False,context_order='previous_action23_then_incoming_history300',
        full_state_coefficient=COEFFICIENT,response_group_weights=GROUP_WEIGHTS,response_weight_rule=WEIGHT_RULE)
    for key,value in expected.items():exact_saved(saved.get(key),value,'source81000 '+key)
    request=saved['request']
    for key,value in dict(ordinary_final_step=SOURCE_STEP,optimizer_final_step=OPTIMIZER_STEP,architecture=[1323,512,512,23],
                          first_layer_execution=FORWARD,export_first_layer_execution='monolithic_float64_1323').items():
        exact_saved(request.get(key),value,'source81000 request '+key)
    actor=saved['actor_state']
    if tuple(actor)!=NAMES:raise ValueError('Six ordered full512 actor tensors required')
    for name,shape in zip(NAMES,NEW_SHAPES):tensor(actor[name],shape,name)
    for name in ('feature_mean','feature_std'):tensor(saved[name],(1323,),name)
    if not bool((saved['feature_std']>0).all()):raise ValueError('Positive original1323 normalization required')
    optimizer=saved['optimizer_state']
    if set(optimizer)!= {'state','param_groups'} or len(optimizer['param_groups'])!=1:raise ValueError('One whole AdamW group required')
    group=optimizer['param_groups'][0];ids=group['params']
    if ids!=list(range(6)) or set(optimizer['state'])!=set(ids):raise ValueError('Six ordered source optimizer identities required')
    options=dict(lr=1e-6,weight_decay=1e-5,betas=(.9,.999),eps=1e-8,foreach=False,fused=False,amsgrad=False,
                 maximize=False,capturable=False,differentiable=False,decoupled_weight_decay=True)
    for key,value in options.items():exact_saved(group.get(key),value,'source AdamW '+key)
    for identifier,shape in zip(ids,NEW_SHAPES):
        state=optimizer['state'][identifier]
        if set(state)!= {'step','exp_avg','exp_avg_sq'}:raise ValueError('Complete three-field AdamW state required')
        tensor(state['step'],(),'step')
        if float(state['step'])!=OPTIMIZER_STEP:raise ValueError('Every source optimizer age must be16000')
        for field in ('exp_avg','exp_avg_sq'):tensor(state[field],shape,field)
        if not bool((state['exp_avg_sq']>=0).all()):raise ValueError('Negative squared moment')
    validate_rng(saved['rng'])

def validate_destination(model,optimizer):
    if type(model) is not WiderContextTarget or 'forward' in model.__dict__:
        raise ValueError('Exact unchanged WiderContextTarget forward required')
    if type(model.actor) is not torch.nn.Sequential or len(model.actor)!=5:
        raise ValueError('Exact original five-module actor required')
    for i in (0,2,4):
        if type(model.actor[i]) is not torch.nn.Linear:raise ValueError('Unchanged original Linear modules required')
    for i in (1,3):
        if type(model.actor[i]) is not torch.nn.ELU or model.actor[i].alpha!=1. or model.actor[i].inplace:
            raise ValueError('Unchanged ELU modules required')
    for module in model.modules():
        if 'forward' in module.__dict__ or module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
            raise ValueError('No unreviewed model hooks')
    actual=model.actor.state_dict()
    if tuple(actual)!=NAMES:raise ValueError('Actual ordered actor names differ')
    for name,shape in zip(NAMES,NEW_SHAPES):
        if tuple(actual[name].shape)!=shape or actual[name].dtype!=torch.float32:
            raise ValueError('Actual full512 actor schema differs')
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups)!=1:
        raise ValueError('Actual AdamW single group required')
    actual_params=optimizer.param_groups[0]['params'];expected=list(model.actor.parameters())
    if len(actual_params)!=6 or any(a is not b for a,b in zip(actual_params,expected)):
        raise ValueError('Actual AdamW parameters must match six actor tensors in order')
    if not all(p.requires_grad for p in expected):raise ValueError('All six original actor tensors must remain trainable')

def verify_restoration(model,optimizer,saved,current_rng):
    validate_source(saved);validate_destination(model,optimizer)
    exact_saved(model.actor.state_dict(),saved['actor_state'],'all six full512 learned actor tensors')
    exact_saved(optimizer.state_dict(),saved['optimizer_state'],'all full512 AdamW tensors and whole parameter group')
    exact_saved(current_rng,saved['rng'],'source81000 global RNG')
    for name in ('feature_mean','feature_std'):exact_saved(getattr(model,name).cpu(),saved[name],'frozen '+name)
    return dict(restoration_completed=True,actor_exact=True,all_full512_parameters_preserved=True,
        learned_context_columns_preserved=True,learned_added_neurons_preserved=True,normalization_exact=True,
        warm_optimizer_exact=True,whole_optimizer_group_exact=True,optimizer_state_count=6,optimizer_start_step=OPTIMIZER_STEP,
        fresh_optimizer=False,RNG_exact=True,ordinary_start_step=SOURCE_STEP,architecture=[1323,512,512,23],
        expansion_performed=False,weights_or_moments_zeroed=False,forward_execution=FORWARD,
        learning_rate_override_performed=False,model_forward_calls=0,optimizer_updates=0,native_steps=0)

def restore(model,optimizer,saved,current_rng,progress):
    """Restore only after a caller's concrete fit gate. Caller preserves failures.

    `current_rng` is the reviewed RNG capture function; `progress` remains owned
    by the caller so any partial restore can be recorded in its failure capsule.
    No file I/O, model forward, optimizer update, LR override, or retry occurs.
    """
    progress.update(stage='warm512_validate',actor_load_attempted=False,actor_load_returned=False,
        optimizer_load_attempted=False,optimizer_load_returned=False,rng_restore_attempted=False,rng_restore_returned=False,
        restoration_verified=False)
    validate_source(saved);validate_destination(model,optimizer)
    # Both expected normalization buffers must already be the frozen source.
    for name in ('feature_mean','feature_std'):exact_saved(getattr(model,name).cpu(),saved[name],'destination frozen '+name)
    progress['stage']='warm512_actor_load';progress['actor_load_attempted']=True
    model.actor.load_state_dict(copy.deepcopy(saved['actor_state']),strict=True);progress['actor_load_returned']=True
    progress['stage']='warm512_optimizer_load';progress['optimizer_load_attempted']=True
    # Deep copy prevents CPU step/moment aliasing from mutating expected evidence.
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state']));progress['optimizer_load_returned']=True
    progress['stage']='warm512_rng_restore';progress['rng_restore_attempted']=True
    restore_rng(copy.deepcopy(saved['rng']));progress['rng_restore_returned']=True
    result=verify_restoration(model,optimizer,saved,current_rng())
    progress['stage']='warm512_verified';progress['restoration_verified']=True
    return result
