"""Warm restoration checks. No task checkpoint is opened by this module."""
import torch
from restoration_support import exact_saved
from response_contract import START_STEP,OPTIMIZER_START,COEFFICIENT

def check_warm_source(saved):
    for key,value in [('kind','direct_absolute_native23_target'),('condition','causal'),
                      ('ordinary_final_step',START_STEP),('optimizer_step',OPTIMIZER_START),
                      ('context_blinded',False),('full_state_coefficient',COEFFICIENT),
                      ('context_order','previous_action23_then_incoming_history300')]:
        if saved.get(key)!=value:raise ValueError('Source causal68000 metadata differs: '+key)
    expected={'0.weight':(256,1323),'0.bias':(256,),'2.weight':(256,256),
              '2.bias':(256,),'4.weight':(23,256),'4.bias':(23,)}
    actor=saved['actor_state']
    if list(actor)!=list(expected):raise ValueError('Six ordered causal actor tensors required.')
    for key,shape in expected.items():
        value=actor[key]
        if value.dtype!=torch.float32 or tuple(value.shape)!=shape or not bool(torch.isfinite(value).all()):
            raise ValueError('Source actor tensor schema: '+key)
    groups=saved['optimizer_state']['param_groups'];states=saved['optimizer_state']['state']
    if len(groups)!=1 or len(groups[0]['params'])!=6 or len(states)!=6:
        raise ValueError('Single six-state source AdamW group required.')
    group=groups[0]
    if len(set(group['params']))!=6 or set(group['params'])!=set(states):
        raise ValueError('Six unique ordered AdamW parameter identities required.')
    for key,value in [('lr',1e-6),('weight_decay',1e-5),('foreach',False),('fused',False),
                      ('betas',(0.9,0.999)),('eps',1e-8),('amsgrad',False),('maximize',False),
                      ('capturable',False),('differentiable',False)]:
        if group.get(key)!=value:raise ValueError('Source AdamW configuration differs: '+key)
    for identifier,(key,param) in zip(group['params'],actor.items()):
        state=states[identifier]
        if (set(state)!= {'step','exp_avg','exp_avg_sq'} or state['step'].dtype!=torch.float32
            or state['step'].shape!=torch.Size([]) or float(state['step'])!=OPTIMIZER_START):
            raise ValueError('Source warm AdamW state/step differs.')
        for name in ('exp_avg','exp_avg_sq'):
            value=state[name]
            if value.shape!=param.shape or value.dtype!=param.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('Source warm moment tensor differs: '+key+'/'+name)

def verify_warm_restoration(model,optimizer,saved,rng):
    check_warm_source(saved)
    exact_saved(model.actor.state_dict(),saved['actor_state'],'all six learned causal tensors')
    exact_saved(optimizer.state_dict(),saved['optimizer_state'],'all warm AdamW states and groups')
    exact_saved(rng,saved['rng'],'restored causal RNG')
    for key in ('feature_mean','feature_std'):
        exact_saved(getattr(model,key).cpu(),saved[key],key)
    if len(optimizer.state)!=6 or any(int(value['step'])!=OPTIMIZER_START for value in optimizer.state.values()):
        raise ValueError('Actual restored optimizer steps differ.')
    return dict(actor_exact=True,learned_context_columns_preserved=True,normalization_exact=True,
                warm_optimizer_exact=True,optimizer_state_count=6,optimizer_start_step=3000,
                fresh_optimizer=False,RNG_exact=True,learning_rate_restart_after_restoration=[1e-5,1e-6])
