"""Independent saved tensor reconstruction. No model or optimizer construction."""
import copy
import math
import torch
from audit_restoration import differences

NAMES=('0.weight','0.bias','2.weight','2.bias','4.weight','4.bias')
OLD=((256,1323),(256,),(256,256),(256,),(23,256),(23,))
NEW=((512,1323),(512,),(512,512),(512,),(23,512),(23,))
SEED=20260912
GROUP=dict(params=list(range(6)),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=1e-5,
    amsgrad=False,maximize=False,foreach=False,capturable=False,differentiable=False,
    fused=False,decoupled_weight_decay=True)

def rate(index):
    if type(index) is not int or not 0<=index<10000:raise ValueError('Fixed10000 index required')
    if index<=249:return 1e-6+(1e-5-1e-6)*index/249
    return 1e-6+.5*(1e-5-1e-6)*(1+math.cos(math.pi*(index-250)/9749))

def source_errors(source):
    errors=[]
    expected=dict(kind='direct_absolute_native23_target',condition='causal',ordinary_final_step=71000,
        optimizer_step=6000,context_blinded=False,full_state_coefficient=1.8188207859141674,
        context_order='previous_action23_then_incoming_history300')
    for key,value in expected.items():errors+=differences(source.get(key),value,'source.'+key)
    actor=source['actor_state'];opt=source['optimizer_state']
    if tuple(actor)!=NAMES:errors.append('source.actor order')
    errors+=differences(opt['param_groups'],[GROUP],'source.optimizer group')
    if set(opt['state'])!=set(range(6)):errors.append('source.optimizer state identities');return errors
    def finite(t,shape):return torch.is_tensor(t) and t.device.type=='cpu' and t.dtype==torch.float32 and tuple(t.shape)==shape and bool(torch.isfinite(t).all())
    for i,(name,shape) in enumerate(zip(NAMES,OLD)):
        if not finite(actor.get(name),shape):errors.append('source.actor '+name)
        state=opt['state'][i]
        if set(state)!=set(('step','exp_avg','exp_avg_sq')):errors.append('source.state keys '+name);continue
        if not finite(state['step'],()) or float(state['step'])!=6000:errors.append('source.step '+name)
        for key in ('exp_avg','exp_avg_sq'):
            if not finite(state[key],shape):errors.append('source.moment '+name+'/'+key)
        if finite(state['exp_avg_sq'],shape) and not bool((state['exp_avg_sq']>=0).all()):errors.append('source.negative variance '+name)
    for key in ('feature_mean','feature_std'):
        if not finite(source[key],(1323,)):errors.append('source.normalization '+key)
    if finite(source['feature_std'],(1323,)) and not bool((source['feature_std']>0).all()):errors.append('source.normalization positivity')
    if not isinstance(source.get('rng'),dict) or not source['rng']:errors.append('source.RNG missing')
    return errors

def expand_zero(value,index):
    """Construct explicit old/new blocks; never use producer expansion code."""
    if index in (0,1,3):return torch.cat((value,torch.zeros_like(value)),dim=0)
    if index==2:
        return torch.cat((torch.cat((value,torch.zeros_like(value)),dim=1),torch.zeros((256,512),dtype=torch.float32)),dim=0)
    if index==4:return torch.cat((value,torch.zeros_like(value)),dim=1)
    return value.clone()

def reconstruct_expansion(source):
    errors=source_errors(source)
    if errors:raise ValueError(errors)
    global_before=torch.get_rng_state().clone()
    gen=torch.Generator(device='cpu');gen.manual_seed(SEED)
    def draw(shape,fan):
        bound=1/math.sqrt(fan)
        return torch.empty(shape,dtype=torch.float32).uniform_(-bound,bound,generator=gen)
    # These four draws are ordered and independent of global CPU/CUDA RNG.
    first_w=draw((256,1323),1323);first_b=draw((256,),1323)
    second_w=draw((256,512),512);second_b=draw((256,),512)
    old=source['actor_state']
    actor={'0.weight':torch.cat((old['0.weight'],first_w),0),
        '0.bias':torch.cat((old['0.bias'],first_b),0),
        '2.weight':torch.cat((torch.cat((old['2.weight'],torch.zeros((256,256),dtype=torch.float32)),1),second_w),0),
        '2.bias':torch.cat((old['2.bias'],second_b),0),
        '4.weight':torch.cat((old['4.weight'],torch.zeros((23,256),dtype=torch.float32)),1),
        '4.bias':old['4.bias'].clone()}
    optimizer=copy.deepcopy(source['optimizer_state'])
    for i in range(6):
        for key in ('exp_avg','exp_avg_sq'):optimizer['state'][i][key]=expand_zero(source['optimizer_state']['state'][i][key],i)
    if differences(torch.get_rng_state(),global_before):raise AssertionError('Auditor global RNG changed')
    return dict(actor_state=actor,optimizer_state=optimizer,expansion_generator_state=gen.get_state().clone())

def width_initial_errors(initial,source):
    errors=source_errors(source)
    if errors:return errors
    expected=reconstruct_expansion(source)
    for key,value in expected.items():errors+=differences(initial[key],value,'initial.'+key)
    for key in ('feature_mean','feature_std'):errors+=differences(initial[key],source[key],'initial.'+key)
    errors+=differences(initial['rng_after_restoration'],source['rng'],'initial.RNG')
    for key,value in dict(expansion_seed=SEED,ordinary_start_step=71000,optimizer_start_step=6000,hidden_width=512,condition='causal').items():errors+=differences(initial[key],value,'initial.'+key)
    return errors

def restoration_fields():
    return dict(old_actor_blocks_exact=True,expanded_actor_exact=True,learned_context_columns_preserved=True,
        normalization_exact=True,old_optimizer_blocks_exact=True,new_optimizer_moments_zero=True,
        whole_optimizer_group_exact=True,optimizer_state_count=6,optimizer_start_step=6000,
        shared_optimizer_step_for_new_entries=6000,fresh_optimizer=False,RNG_exact=True,
        expansion_seed=SEED,old_hidden_width=256,new_hidden_width=512,new_outgoing_paths_zero=True,
        global_rng_unchanged_by_expansion=True,zero_preserving_addition=True,old_block_contractions_preserved=True,
        learning_rate_restart_after_restoration=dict(ramp_updates=250,ramp_inclusive=[1e-6,1e-5],cosine_updates=9750,cosine_inclusive=[1e-5,1e-6]))
