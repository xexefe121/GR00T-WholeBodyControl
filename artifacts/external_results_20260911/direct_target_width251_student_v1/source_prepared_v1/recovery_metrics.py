"""Append D3 measurements; preserve all old15/9/54 summaries exactly."""
import numpy as np
from direct_contract import normalized_labels
from context_diagnostics import target_errors
from response_diagnostics import summarize_balanced,save_balanced_prefix

def summarize_recovery(outputs,data,coefficient):
    result=summarize_balanced(outputs,data)
    prediction=np.asarray(outputs['recovery']);target=data['recovery_target']
    square=(prediction-normalized_labels(target,data['default'],data['span']))**2
    cells=[]
    for phase,ids in enumerate(data['recovery_cells']):
        cells.append(dict(phase=phase,rows=len(ids),normalized_MSE=float(square[ids].mean()),
            first24=target_errors(prediction[ids[:24]],target[ids[:24]],data),**target_errors(prediction[ids],target[ids],data)))
    value=float(np.mean([c['normalized_MSE'] for c in cells]))
    result.update(recovery_cells=cells,recovery_objective=value,recovery_coefficient=coefficient,
        optimization_objective='balanced_old_objective_plus_recovery',
        combined_recovery_objective=result['balanced_weighted_objective']+coefficient*value)
    return result

def save_prefix(dest,losses,nc,fc,pc,original_losses,weighted_cells,recovery_losses,recovery_cells):
    save_balanced_prefix(dest,losses,nc,fc,pc,original_losses,weighted_cells)
    np.save(dest/'recovery_objectives.npy',np.asarray(recovery_losses,np.float64).reshape(-1,3))
    np.save(dest/'recovery_cell_losses.npy',np.asarray(recovery_cells,np.float64).reshape(-1,3))
