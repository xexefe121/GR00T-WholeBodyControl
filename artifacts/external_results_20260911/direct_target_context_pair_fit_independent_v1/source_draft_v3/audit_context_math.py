"""Independent context and paired-result algebra. No task runtime imports."""
import math
import numpy as np
from audit_math import phase_indices
from audit_full_state_math import AXES, metrics as original_metrics

WEIGHT=1.8188207859141674
WIDTHS=(('actions',23),('base_ang_vel',3),('dof_pos',23),('dof_vel',23),('projected_gravity',3))

def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def inverse(target,contract):
    a=np.asarray(target)
    assert a.ndim==2 and a.shape[1]==23 and a.dtype==np.float64 and np.isfinite(a).all()
    return ((a-np.asarray(contract['default_q'],np.float64))*np.asarray(contract['kp'],np.float64)
            /(.25*np.asarray(contract['training_effort'],np.float64))).astype(np.float32)

def source_context(archive,contract):
    """Check literal history ordering and every within-dataset contiguous transition."""
    n=len(archive['control']);pieces=[]
    for name,width in WIDTHS:
        value=archive['history_'+name]
        assert value.shape==(n,4,width) and value.dtype==np.float32 and np.isfinite(value).all()
        pieces.append(value.reshape(n,4*width))
    flat=np.concatenate(pieces,axis=1)
    prior=archive['previous_action'];state=archive['state']
    assert exact(flat,archive['history'])
    assert prior.shape==(n,23) and prior.dtype==np.float32 and np.isfinite(prior).all()
    assert state.shape==(n,52) and state.dtype==np.float32 and np.isfinite(state).all()
    dataset=archive.get('dataset',np.zeros(n,dtype=np.int64))
    selected=np.flatnonzero((dataset[1:]==dataset[:-1])&(np.diff(archive['control'])==1))
    terms=(prior,state[:,49:52],state[:,:23],state[:,23:46],state[:,46:49])
    for (name,width),term in zip(WIDTHS,terms):
        before=archive['history_'+name][selected];after=archive['history_'+name][selected+1]
        assert exact(after[:,0],term[selected]) and exact(after[:,1:],before[:,:3])
    assert exact(prior[selected+1],inverse(archive['expert_target'][selected],contract))
    return np.concatenate((prior,flat),axis=1),len(selected)

def endpoint_context(centers,arrays,contract):
    start=arrays['center_index'];n=len(start)
    assert exact(arrays['incoming_history'],centers['history'][start])
    assert exact(arrays['incoming_raw_prior'],centers['previous_action'][start])
    state=centers['state'][start];prior=centers['previous_action'][start]
    terms=(prior,state[:,49:52],state[:,:23],state[:,23:46],state[:,46:49]);pieces=[]
    for (name,width),term in zip(WIDTHS,terms):
        original=centers['history_'+name][start]
        expected=np.empty((n,4,width),np.float32)
        expected[:,0]=term;expected[:,1:]=original[:,:3]
        assert exact(expected,arrays['advanced_history_'+name])
        pieces.append(expected.reshape(n,-1))
    history=np.concatenate(pieces,axis=1)
    assert exact(history,arrays['advanced_history'])
    applied_prior=inverse(arrays['policy_applied_target'],contract)
    assert exact(applied_prior,arrays['policy_actual_normalized_action'])
    return np.concatenate((applied_prior,history),axis=1)

def moments(context):
    assert context.shape==(9904,323) and context.dtype==np.float32 and np.isfinite(context).all()
    x=context.astype(np.float64);cells=phase_indices()
    mean=sum(x[ids].sum(axis=0)/len(ids) for ids in cells)/15
    variance=sum(np.square(x[ids]-mean).sum(axis=0)/len(ids) for ids in cells)/15
    return mean,variance,mean.astype(np.float32),np.maximum(np.sqrt(variance),.05).astype(np.float32)

def rate(index):
    if not 0<=index<3000:raise ValueError('Fixed3000 index')
    return 1e-6+.5*(1e-5-1e-6)*(1+math.cos(math.pi*index/2999))

def drift(a,b,span):
    assert a.shape==b.shape and a.dtype==b.dtype==np.float32
    delta=(a.astype(np.float64)-b.astype(np.float64))*span.astype(np.float64)
    assert np.isfinite(delta).all()
    return dict(max_preclip_error_rad=float(np.abs(delta).max()),
        RMS_preclip_error_rad=float(np.sqrt(np.square(delta).mean())),byte_equal=exact(a,b))

def sign_components(error,wanted):
    assert error.shape==wanted.shape and error.shape[-2:]==(2,23)
    assert error.dtype==wanted.dtype==np.float64 and np.isfinite(error).all() and np.isfinite(wanted).all()
    odd=.5*(error[...,1,:]-error[...,0,:]);even=.5*(error[...,1,:]+error[...,0,:])
    return dict(odd_response_MSE=float(np.square(odd).mean()),even_response_MSE=float(np.square(even).mean()),
        zero_response_MSE=float(np.square(wanted).mean()))

def metrics(predictions,data):
    """Retain qualified independent full58 metrics, add requested sign decomposition."""
    result=original_metrics(predictions,data)
    n=predictions['nominal'].astype(np.float64)[:3057]
    f=predictions['full_state'].astype(np.float64).reshape(3057,58,2,23)
    teacher=data['nominal_teacher'][:3057]
    wanted=(data['full_state_teacher'].reshape(3057,58,2,23)-teacher[:,None,None,:])/data['span'].astype(np.float64)
    error=f-n[:,None,None,:]-wanted
    for cell in result['full_state_cells']:
        ids=phase_indices()[cell['dataset']*3+cell['phase']];axes=AXES[cell['tangent_group']]
        cell.update(sign_components(error[ids][:,axes],wanted[ids][:,axes]))
    for key,cell_key in [('full_state_odd_MSE','odd_response_MSE'),('full_state_even_MSE','even_response_MSE'),('full_state_zero_response_MSE','zero_response_MSE')]:
        result[key]=float(np.mean([v[cell_key] for v in result['full_state_cells']]))
    result['full_state_group_comparison']=[dict(tangent_group=g,
        response_MSE=float(np.mean([v['response_MSE'] for v in result['full_state_cells'] if v['tangent_group']==g])),
        zero_response_MSE=float(np.mean([v['zero_response_MSE'] for v in result['full_state_cells'] if v['tangent_group']==g]))) for g in range(6)]
    result['weighted_objective']=result['nominal_objective']+WEIGHT*result['full_state_objective']+result['physical_objective']
    return result
