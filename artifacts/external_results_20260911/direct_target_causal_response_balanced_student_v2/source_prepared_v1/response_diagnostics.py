"""Add fixed weighted summaries without changing any original diagnostic value."""
import numpy as np
from balance_contract import GROUP_WEIGHTS,GROUP_NAMES,WEIGHT_RULE,ZERO_RESPONSE_ENERGIES
from response_contract import COEFFICIENT
from context_diagnostics import summarize
from training_support import save_loss_prefix

def add_balanced_metrics(result):
    cells=result['full_state_cells']
    if len(cells)!=54 or [(c['dataset'],c['phase'],c['tangent_group']) for c in cells]!=[
            (d,p,g) for d in range(3) for p in range(3) for g in range(6)]:
        raise ValueError('Original54 cell order differs.')
    weighted=[]
    for cell in cells:
        weight=GROUP_WEIGHTS[cell['tangent_group']]
        weighted.append(dict(dataset=cell['dataset'],phase=cell['phase'],tangent_group=cell['tangent_group'],
            weight=weight,weighted_response_MSE=weight*cell['response_MSE'],
            weighted_first24_response_MSE=weight*cell['first24_response_MSE'],
            weighted_odd_response_MSE=weight*cell['odd_response_MSE'],
            weighted_even_response_MSE=weight*cell['even_response_MSE'],
            weighted_zero_response_MSE=weight*cell['zero_response_MSE']))
    for source,target in [('weighted_response_MSE','balanced_full_state_objective'),
                          ('weighted_odd_response_MSE','balanced_full_state_odd_MSE'),
                          ('weighted_even_response_MSE','balanced_full_state_even_MSE'),
                          ('weighted_zero_response_MSE','balanced_full_state_zero_response_MSE')]:
        result[target]=float(np.mean([c[source] for c in weighted]))
    result['balanced_full_state_cells']=weighted
    result['balanced_full_state_group_comparison']=[dict(tangent_group=g,group_name=GROUP_NAMES[g],
        weight=GROUP_WEIGHTS[g],teacher_zero_response_energy=ZERO_RESPONSE_ENERGIES[g],
        weighted_response_MSE=float(np.mean([c['weighted_response_MSE'] for c in weighted if c['tangent_group']==g])),
        weighted_zero_response_MSE=float(np.mean([c['weighted_zero_response_MSE'] for c in weighted if c['tangent_group']==g]))) for g in range(6)]
    result['weighted_objective']=result['nominal_objective']+COEFFICIENT*result['full_state_objective']+result['physical_objective']
    result['balanced_weighted_objective']=result['nominal_objective']+COEFFICIENT*result['balanced_full_state_objective']+result['physical_objective']
    result['optimization_objective']='balanced_weighted_objective'
    result['response_weight_rule']=WEIGHT_RULE
    return result

def summarize_balanced(outputs,data):
    return add_balanced_metrics(summarize(outputs,data))

def save_balanced_prefix(dest,losses,nc,fc,pc,original_losses,weighted_cells):
    save_loss_prefix(dest,losses,nc,fc,pc)
    np.save(dest/'original_objectives.npy',np.asarray(original_losses,np.float64).reshape(-1,2))
    np.save(dest/'balanced_full_state_cell_losses.npy',np.asarray(weighted_cells,np.float64).reshape(-1,54))
