"""Fixed two-condition context utility protocol; no adaptive decisions."""
import math
import numpy as np

CONDITIONS = ('blinded', 'causal')
UPDATES = 3000
PAIRS = 864
FEATURES = 1323
CONTEXT = 323
COEFFICIENT = 1.8188207859141674
BUDGETS = dict(training_forward_rows=88116000,training_forward_calls=18000,training_updates=6000,
    diagnostic_Torch_rows=2940560,diagnostic_Torch_calls=11496,diagnostic_ORT_rows=735140,
    diagnostic_ORT_calls=2874,calibration_forward_calls=0,calibration_gradient_calls=0,
    native_calls=0,BFM_calls=0,manual_export_trace_calls=0)
HISTORY_WIDTHS = {'actions':23,'base_ang_vel':3,'dof_pos':23,'dof_vel':23,'projected_gravity':3}

def cosine_rate(index):
    if not 0 <= index < UPDATES: raise ValueError('Selected3000 update index required.')
    return 1e-6 + .5 * (1e-5 - 1e-6) * (1 + math.cos(math.pi * index / (UPDATES - 1)))

def exact(a,b,name):
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
        raise ValueError('Context bytes differ: '+name)

def join_context(prior,history):
    if prior.ndim!=2 or prior.shape[1]!=23 or history.shape!=(len(prior),300):
        raise ValueError('Prior23/history300 schema.')
    if prior.dtype!=np.float32 or history.dtype!=np.float32 or not np.isfinite(prior).all() or not np.isfinite(history).all():
        raise ValueError('Finite float32 causal context required.')
    return np.concatenate((prior,history),axis=1)

def weighted_context_moments(context,cells):
    if context.ndim!=2 or context.shape[1]!=CONTEXT or context.dtype!=np.float32 or not np.isfinite(context).all():
        raise ValueError('Context normalization input schema.')
    if len(cells)!=15 or any(len(ids)==0 for ids in cells):raise ValueError('All15 nominal cells required.')
    ids=np.concatenate(cells)
    if not np.array_equal(np.sort(ids),np.arange(len(context))):raise ValueError('Cells must partition all nominal rows once.')
    values=context.astype(np.float64)
    mean64=np.stack([values[ids].mean(axis=0) for ids in cells]).mean(axis=0)
    variance64=np.stack([((values[ids]-mean64)**2).mean(axis=0) for ids in cells]).mean(axis=0)
    mean=mean64.astype(np.float32)
    std=np.maximum(np.sqrt(variance64),.05).astype(np.float32)
    return mean,std,mean64,variance64

def history_flat(named):
    if set(named)!=set(HISTORY_WIDTHS):raise ValueError('Complete named history required.')
    n=len(named['actions'])
    for key,width in HISTORY_WIDTHS.items():
        value=named[key]
        if value.shape!=(n,4,width) or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('History schema: '+key)
    return np.concatenate([named[key].reshape(n,-1) for key in sorted(named)],axis=1)

def advance_history(named,state,prior):
    if state.shape!=(len(prior),52) or state.dtype!=np.float32:raise ValueError('BFM state52 ordering required.')
    terms=dict(actions=prior,base_ang_vel=state[:,49:52],dof_pos=state[:,:23],dof_vel=state[:,23:46],projected_gravity=state[:,46:49])
    return {key:np.concatenate((terms[key][:,None,:],named[key][:,:3,:]),axis=1) for key in HISTORY_WIDTHS}

def inverse_applied_action(target,contract):
    if target.ndim!=2 or target.shape[1]!=23 or target.dtype!=np.float64 or not np.isfinite(target).all():raise ValueError('Actual float64 target required.')
    default=np.asarray(contract['default_q'],np.float64);kp=np.asarray(contract['kp'],np.float64);effort=np.asarray(contract['training_effort'],np.float64)
    return ((target-default)*kp/(.25*effort)).astype(np.float32)

def fixed_schedule(centers,axes):
    if centers.shape!=(10000,864) or centers.dtype!=np.int32 or axes.shape!=centers.shape or axes.dtype!=np.int8:
        raise ValueError('Original10000×864 saved schedule required.')
    return centers[:UPDATES].copy(),axes[:UPDATES].copy()

class ExpandedFeatures:
    """Same current1000 bytes; context is gathered by explicit index, never shifted."""
    def __init__(self,current,context,indices,condition,mean):
        if current.ndim!=2 or current.shape[1]!=1000 or current.dtype!=np.float32:raise ValueError('Current1000 schema.')
        if context.ndim!=2 or context.shape[1]!=323 or context.dtype!=np.float32 or indices.shape!=(len(current),) or indices.dtype!=np.int64:raise ValueError('Context lookup schema.')
        if condition not in CONDITIONS or mean.shape!=(323,) or mean.dtype!=np.float32:raise ValueError('Condition/mean schema.')
        if np.any(indices<0) or np.any(indices>=len(context)):raise ValueError('Context lookup out of range.')
        self.current=current;self.context=context;self.indices=indices;self.condition=condition;self.mean=mean
        self.shape=(len(current),1323);self.dtype=np.dtype(np.float32)
    def __len__(self):return self.shape[0]
    def __getitem__(self,index):
        current=self.current[index];context=self.context[self.indices[index]]
        if self.condition=='blinded':context=np.broadcast_to(self.mean,context.shape)
        return np.concatenate((current,context),axis=-1)
    def __array__(self,dtype=None):
        value=self[:]
        if dtype is not None and np.dtype(dtype)!=np.float32:raise ValueError('No implicit context feature dtype change.')
        return value
