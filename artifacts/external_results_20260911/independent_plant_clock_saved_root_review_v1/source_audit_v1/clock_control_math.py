"""Independent owned evidence decoding and measured control/history arithmetic."""
import base64
import hashlib
import json
import copy
import numpy as np
from clock_saved_math import array,exact,first_difference

SIZES=dict(actions=23,base_ang_vel=3,dof_pos=23,dof_vel=23,projected_gravity=3)


def decode(value):
    if isinstance(value,list):return [decode(v) for v in value]
    if not isinstance(value,dict):return value
    if set(value)=={'type','hex'} and value['type']=='float':return float.fromhex(value['hex'])
    if set(value)=={'type','base64','sha256','length'} and value['type']=='bytes':
        raw=base64.b64decode(value['base64'],validate=True)
        if len(raw)!=value['length'] or hashlib.sha256(raw).hexdigest()!=value['sha256']:
            raise AssertionError('Owned bytes length/hash mismatch')
        return raw
    if set(value)=={'type','class','fields'} and value['type']=='dataclass':
        return {'dataclass':value['class'],'fields':decode(value['fields'])}
    return {k:decode(v) for k,v in value.items()}


def terms(state,incoming,default):
    q,v=state[1:31],state[31:60]
    quat=q[3:7].copy();quat/=np.linalg.norm(quat)
    w,x,y,z=quat
    rotation=np.asarray(((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)),
                         (2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)),
                         (2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y))))
    values=dict(actions=incoming,base_ang_vel=v[3:6]*.25,dof_pos=q[7:]-default,
                dof_vel=v[6:],projected_gravity=rotation.T@np.array([0.,0.,-1.]))
    return {k:np.asarray(v,np.float32) for k,v in values.items()}


def named_bytes(items,sizes,lags):
    if [item[0] for item in items]!=sorted(sizes):raise AssertionError('Named history order')
    result={}
    for key,raw in items:
        if type(raw) is not bytes or len(raw)!=4*lags*sizes[key]:raise AssertionError('History byte schema')
        result[key]=np.frombuffer(raw,np.float32).reshape((lags,sizes[key]) if lags>1 else (sizes[key],))
        if not np.isfinite(result[key]).all():raise AssertionError('Finite history values')
    return result


def controls(a,metadata,initial,contract,full,hold,epoch_ns):
    c=len(a['control_control']);n=len(a['step_index'])
    if not 0<=c<=1819 or c>(n+9)//10+int(n%10==0):raise AssertionError('Actual boundary count')
    exact(a['control_control'],np.arange(c,dtype=np.int64),'Every actual control index')
    exact(a['control_physics'],np.arange(c,dtype=np.int64)*10,'Fixed50Hz control boundary')
    array(a['control_integration'],np.float64,(c,291),'Measured control integration')
    array(a['control_target'],np.float64,(c,23),'Actual control target')
    array(a['control_raw_action'],np.float32,(c,23),'Actual control raw action')
    array(a['control_incoming_raw'],np.float32,(c,23),'Incoming previous raw action')
    array(a['control_flat_history_before'],np.float32,(c,300),'Measured four-lag history')
    array(a['control_held'],np.bool_,(c,),'Held flag')
    frames=np.r_[np.arange(11,1580,dtype=np.int64),np.full(250,1579,np.int64)]
    exact(a['control_nominal_source_frame'],frames[:c],'Original nominal source frames')
    expected_target=np.concatenate([full['target'],hold['target']])
    expected_action=np.concatenate([full['action'],hold['action']])
    history={k:np.zeros((4,size),np.float32) for k,size in SIZES.items()}
    expected_prior=np.zeros(23,np.float32)
    for key in ('control_history_before','control_history_after','control_measured_terms',
                'control_command_ids','control_nominal_windows','control_active_windows'):
        if len(metadata[key])!=c:raise AssertionError('Control metadata length: '+key)
    for i in range(c):
        state=initial if i==0 else a['step_integration'][10*i-1]
        exact(a['control_integration'][i],state,'Actual boundary follows prior captured full291')
        exact(a['control_incoming_raw'][i],expected_prior,'Previous actually activated action')
        flat=np.concatenate([history[k].reshape(-1) for k in sorted(history)])
        exact(a['control_flat_history_before'][i],flat,'Old history before update')
        measured=terms(state,expected_prior,contract['default_q'])
        before=named_bytes(metadata['control_history_before'][i],SIZES,4)
        reported_terms=named_bytes(metadata['control_measured_terms'][i],SIZES,1)
        after=named_bytes(metadata['control_history_after'][i],SIZES,4)
        for key in sorted(SIZES):
            exact(before[key],history[key],'Named old history '+key)
            exact(reported_terms[key],measured[key],'Independently measured term '+key)
            history[key]=np.concatenate([measured[key][None],history[key][:3]],axis=0)
            exact(after[key],history[key],'Actual history update '+key)
        if a['control_held'][i]:
            if i==0:raise AssertionError('Initial command cannot be a held interval')
            exact(a['control_target'][i],a['control_target'][i-1],'Held applied target')
            exact(a['control_raw_action'][i],a['control_raw_action'][i-1],'Held raw feedback')
            if metadata['control_command_ids'][i]!=metadata['control_command_ids'][i-1] or metadata['control_active_windows'][i]!=metadata['control_active_windows'][i-1]:
                raise AssertionError('Held active identity changes')
        expected_prior=a['control_raw_action'][i].copy()
    if n:
        indices=np.arange(n)//10
        exact(a['step_target'],a['control_target'][indices],'Every substep uses actually activated target')
        exact(a['step_raw_action'],a['control_raw_action'][indices],'Every substep retains actual raw feedback')
    deadlines=epoch_ns+np.arange(c,dtype=np.int64)*20_000_000
    exact(a['control_nominal_activation_ns'],deadlines,'Fixed original activation schedule')
    array(a['control_actual_activation_ns'],np.int64,(c,),'Actual activation time')
    array(a['control_admitted_ns'],np.int64,(c,),'Actual admission time')
    if np.any(a['control_actual_activation_ns']<deadlines):raise AssertionError('Activation before fixed deadline')
    for i in range(1,c):
        if not a['control_held'][i] and a['control_admitted_ns'][i]>=deadlines[i]:raise AssertionError('New command admitted late')
    held=np.flatnonzero(a['control_held'])
    if len(held) and not np.all(a['control_held'][held[0]:]):raise AssertionError('Latched hold was silently cleared')
    target_difference=first_difference(a['control_target'],expected_target[:c])
    action_difference=first_difference(a['control_raw_action'],expected_action[:c])
    return dict(control_count=c,history_updates_verified=c,held_controls=held.tolist(),
        first_target_difference=target_difference,
        first_raw_action_difference=action_difference,
        main_hold_boundary_reached=c>1569,
        exact_recorded_commands=(c==1819 and not len(held) and target_difference is None and action_difference is None))


def reserved_control(a,metadata,initial,contract,full,hold,epoch_ns):
    """Audit a logger's reserved boundary without giving it committed credit."""
    item=metadata.get('control_ledger_overflow')
    if item is None:return None
    if item.get('dataclass')!='ControlRecord':raise AssertionError('Reserved control type')
    row=item['fields'];n=len(a['control_control'])
    if row['control']!=n or row['physics']!=10*n:raise AssertionError('Reserved boundary index')
    b={k:v.copy() for k,v in a.items()};m=copy.deepcopy(metadata)
    for key in ('control','physics','nominal_activation_ns','actual_activation_ns','admitted_ns','held'):
        b['control_'+key]=np.append(b['control_'+key],np.asarray(row[key],dtype=b['control_'+key].dtype))
    command=row['command']
    if command.get('dataclass')!='Command':raise AssertionError('Reserved command type')
    command=command['fields']
    for field,key,dtype,width in [('snapshot_state','integration',np.float64,291),('incoming_raw','incoming_raw',np.float32,23),
                                 ('flat_history_before','flat_history_before',np.float32,300)]:
        value=np.frombuffer(row[field],dtype)
        if value.shape!=(width,):raise AssertionError('Reserved control byte shape')
        b['control_'+key]=np.concatenate([b['control_'+key],value[None]])
    for key,field,dtype in [('target','target',np.float64),('raw_action','raw_action',np.float32)]:
        b['control_'+key]=np.concatenate([b['control_'+key],np.frombuffer(command[field],dtype)[None]])
    b['control_nominal_source_frame']=np.append(b['control_nominal_source_frame'],np.int64(min(n+11,1579)))
    for key,value in [('control_history_before',row['history_before']),('control_history_after',row['history_after']),
        ('control_measured_terms',row['terms']),('control_command_ids',command['command_id']),
        ('control_nominal_windows',row['nominal_window_id']),('control_active_windows',row['active_window_id'])]:m[key].append(value)
    verified=controls(b,m,initial,contract,full,hold,epoch_ns)
    return {'reserved_control':n,'independently_checked_history_entries':verified['history_updates_verified'],
            'committed_control_credit':0,'native_step_credit':0}
