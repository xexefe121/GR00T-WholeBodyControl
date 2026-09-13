"""Independent saved-cell weighting and exact warm-state comparisons only."""
import copy
import numpy as np
from audit_context_math import WEIGHT
from audit_restoration import differences

ENERGY_PROVENANCE_RULE='Emean/Eg'
PRODUCER_WEIGHT_RULE='mean_six_teacher_group_energies_over_group_energy'

GROUPS=('root_position','root_rotation','joint_position','root_linear_velocity','root_angular_velocity','joint_velocity')

def energy_weights(energy):
    assert tuple(energy['group_names'])==GROUPS and energy['rule']==ENERGY_PROVENANCE_RULE
    values=np.asarray(energy['zero_response_energies'],np.float64)
    assert values.shape==(6,) and np.isfinite(values).all() and np.all(values>0)
    mean=float(values.mean());weights=mean/values
    assert energy['mean_zero_response_energy']==mean
    assert np.array_equal(np.asarray(energy['group_weights'],np.float64),weights)
    return weights

def balanced_metrics(original,weights,energy):
    result=copy.deepcopy(original);cells=result['full_state_cells']
    assert [(c['dataset'],c['phase'],c['tangent_group']) for c in cells]==[(d,p,g) for d in range(3) for p in range(3) for g in range(6)]
    measured=np.array([np.mean([c['zero_response_MSE'] for c in cells if c['tangent_group']==g]) for g in range(6)])
    assert np.allclose(measured,energy['zero_response_energies'],rtol=5e-12,atol=1e-14)
    weighted=[]
    for cell in cells:
        group=cell['tangent_group'];weight=float(weights[group])
        row={name:cell[name] for name in ('dataset','phase','tangent_group')};row['weight']=weight
        for key in ('response_MSE','first24_response_MSE','odd_response_MSE','even_response_MSE','zero_response_MSE'):row['weighted_'+key]=weight*cell[key]
        weighted.append(row)
    for source,target in [('response_MSE','objective'),('odd_response_MSE','odd_MSE'),('even_response_MSE','even_MSE'),('zero_response_MSE','zero_response_MSE')]:
        result['balanced_full_state_'+target]=float(np.mean([c['weighted_'+source] for c in weighted]))
    result['balanced_full_state_cells']=weighted
    result['balanced_full_state_group_comparison']=[dict(tangent_group=g,group_name=GROUPS[g],weight=float(weights[g]),
        teacher_zero_response_energy=energy['zero_response_energies'][g],
        weighted_response_MSE=float(np.mean([c['weighted_response_MSE'] for c in weighted if c['tangent_group']==g])),
        weighted_zero_response_MSE=float(np.mean([c['weighted_zero_response_MSE'] for c in weighted if c['tangent_group']==g]))) for g in range(6)]
    result['balanced_weighted_objective']=(result['nominal_objective']+WEIGHT*result['balanced_full_state_objective'])+result['physical_objective']
    result['optimization_objective']='balanced_weighted_objective';result['response_weight_rule']=PRODUCER_WEIGHT_RULE
    return result

def warm_initial_errors(initial,source):
    errors=[]
    for a,b in [('actor_state','actor_state'),('optimizer_state','optimizer_state'),('rng_after_restoration','rng'),('feature_mean','feature_mean'),('feature_std','feature_std')]:
        errors.extend(differences(initial[a],source[b],'warm.'+a))
    if source.get('condition')!='causal' or source.get('ordinary_final_step')!=68000 or source.get('optimizer_step')!=3000 or source.get('context_blinded') is not False:errors.append('source causal68000 metadata')
    names=['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']
    if list(source['actor_state'])!=names:errors.append('source six ordered actor tensors')
    state=source['optimizer_state']['state'];groups=source['optimizer_state']['param_groups']
    if set(state)!=set(range(6)) or len(groups)!=1 or groups[0]['params']!=list(range(6)):errors.append('source six ordered optimizer states')
    for index,value in state.items():
        step=np.asarray(value['step'])
        if step.shape!=() or step.dtype!=np.float32 or float(step)!=3000:errors.append('source exact step:'+str(index))
        if set(value)!= {'step','exp_avg','exp_avg_sq'}:errors.append('source moment keys:'+str(index))
        for key in ('exp_avg','exp_avg_sq'):
            array=np.asarray(value[key]);parameter=np.asarray(source['actor_state'][names[index]])
            if array.shape!=parameter.shape or array.dtype!=parameter.dtype or not np.isfinite(array).all():errors.append('source moment schema:'+str(index)+key)
        if np.any(np.asarray(value['exp_avg_sq'])<0):errors.append('source negative second moment')
    return errors

def ledger_expectations(nominal,full,physical,weights):
    assert nominal.ndim==full.ndim==physical.ndim==2 and len(nominal)==len(full)==len(physical)
    assert nominal.shape[1]==15 and full.shape[1]==54 and physical.shape[1]==9
    assert all(a.dtype==np.float64 and np.isfinite(a).all() and np.all(a>=0) for a in (nominal,full,physical))
    assert weights.shape==(6,) and weights.dtype==np.float64 and np.isfinite(weights).all() and np.all(weights>0)
    # Producer nominal cell entries are saved float32 means. Recreate its float32
    # outer mean before promotion, then preserve left-associated float64 totals.
    n=nominal.astype(np.float32).mean(axis=1).astype(np.float64)
    original=full.mean(axis=1);weighted=full*np.tile(weights,9);balanced=weighted.mean(axis=1);p=physical.mean(axis=1)
    return dict(nominal=n,original_full=original,balanced_full=balanced,physical=p,weighted_cells=weighted,
        original_total=(n+WEIGHT*original)+p,balanced_total=(n+WEIGHT*balanced)+p)
