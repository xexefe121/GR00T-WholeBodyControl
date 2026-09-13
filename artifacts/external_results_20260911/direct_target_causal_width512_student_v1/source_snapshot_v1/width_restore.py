"""Actual expansion verification, called only inside an explicitly cleared fit."""
import torch
from width512 import validate_source,NAMES,SLICES,OLD_SHAPES,NEW_SHAPES,SEED
from restoration_support import exact_saved
from training_support import cpu_tree


def verify_width_restoration(model,optimizer,saved,expansion,rng):
    validate_source(saved)
    actual=cpu_tree(model.actor.state_dict())
    states=cpu_tree(optimizer.state_dict())
    exact_saved(actual,expansion['actor_state'],'actual complete width512 actor')
    exact_saved(states['param_groups'],saved['optimizer_state']['param_groups'],'original whole AdamW group')
    exact_saved(states,expansion['optimizer_state'],'actual expanded six AdamW states')
    exact_saved(rng,saved['rng'],'source71000 restored global RNG')
    exact_saved(expansion['rng_to_restore'],saved['rng'],'expansion retained source RNG')
    for key in ('feature_mean','feature_std'):
        exact_saved(getattr(model,key).cpu(),saved[key],'retained1323 '+key)
    identifiers=states['param_groups'][0]['params']
    for identifier,name,new_shape,block in zip(identifiers,NAMES,NEW_SHAPES,SLICES):
        if tuple(actual[name].shape)!=new_shape:raise ValueError('Expanded actor shape differs: '+name)
        exact_saved(actual[name][block],saved['actor_state'][name],'old learned block '+name)
        state=states['state'][identifier]
        if state['step'].dtype!=torch.float32 or state['step'].shape!=torch.Size([]) or float(state['step'])!=6000:
            raise ValueError('Expanded optimizer step differs.')
        for field in ('exp_avg','exp_avg_sq'):
            value=state[field]
            exact_saved(value[block],saved['optimizer_state']['state'][identifier][field],'old moment block '+name+'/'+field)
            mask=torch.ones(new_shape,dtype=torch.bool);mask[block]=False
            if torch.count_nonzero(value[mask])!=0:raise ValueError('New moment slot not zero.')
    if torch.count_nonzero(actual['2.weight'][:256,256:]) or torch.count_nonzero(actual['4.weight'][:,256:]):
        raise ValueError('Added outgoing path not initially zero.')
    if expansion['expansion_seed']!=SEED:raise ValueError('Expansion seed differs.')
    return dict(old_actor_blocks_exact=True,expanded_actor_exact=True,learned_context_columns_preserved=True,
        normalization_exact=True,old_optimizer_blocks_exact=True,new_optimizer_moments_zero=True,
        whole_optimizer_group_exact=True,optimizer_state_count=6,optimizer_start_step=6000,
        shared_optimizer_step_for_new_entries=6000,fresh_optimizer=False,RNG_exact=True,
        expansion_seed=SEED,old_hidden_width=256,new_hidden_width=512,
        new_outgoing_paths_zero=True,global_rng_unchanged_by_expansion=True,
        zero_preserving_addition=True,old_block_contractions_preserved=True,
        learning_rate_restart_after_restoration=dict(ramp_updates=250,ramp_inclusive=[1e-6,1e-5],
            cosine_updates=9750,cosine_inclusive=[1e-5,1e-6]))
