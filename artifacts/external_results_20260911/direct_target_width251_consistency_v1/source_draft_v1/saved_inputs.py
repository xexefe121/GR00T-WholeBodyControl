"""Minimal original nominal/physical reconstruction; no full58 arrays loaded."""
from pathlib import Path
import numpy as np
from input_schema import load_numeric,archive_headers
from direct_contract import COUNTS,RETAINED,normalized_labels
from consistency_math import NOMINAL,PHYSICAL,NEW,OLD,TOTAL

def exact(a,b,name):
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():raise ValueError('Exact saved data identity: '+name)

def checked_npy(path,shape,dtype):
    with Path(path).open('rb') as stream:
        from input_schema import header,check_spec
        check_spec(header(stream),(shape,dtype),str(path))
    result=np.load(path,allow_pickle=False)
    if not np.isfinite(result).all():raise ValueError('Nonfinite numeric input')
    return result

def nominal_archive(path,n):
    spec={'features':((n,1069),'float32'),'expert_target':((n,23),'float64'),
        'control':((n,),'int64'),'source_frame':((n,),'int64'),'previous_action':((n,23),'float32'),
        'history':((n,300),'float32'),'joint_span':((23,),'float32'),'joint_limits':((23,2),'float64')}
    rows,_=load_numeric(path,spec)
    return rows

def assemble(paths,physical_arrays,read_json):
    """All paths must already be directly pinned and qualified by the runner."""
    norm_spec={'feature_mean':((1323,),'float32'),'feature_std':((1323,),'float32'),
        'context_mean':((323,),'float32'),'context_std':((323,),'float32'),
        'default_q':((23,),'float64'),'joint_span':((23,),'float32'),'joint_limits':((23,2),'float64')}
    norm,_=load_numeric(paths['normalization'],norm_spec)
    exact(norm['context_mean'],norm['feature_mean'][1000:],'frozen context mean')
    exact(norm['context_std'],norm['feature_std'][1000:],'frozen context std')
    if np.any(norm['feature_std']<=0):raise ValueError('Positive frozen normalization')
    contract=read_json(paths['contract'])
    for key in ('default_q','joint_limits'):exact(norm[key],np.asarray(contract[key],np.float64),'normalization '+key)
    exact(norm['joint_span'],(norm['joint_limits'][:,1]-norm['joint_limits'][:,0]).astype(np.float32),'native rounded span')
    x=np.empty((TOTAL,1323),np.float32);y=np.empty((TOTAL,23),np.float64)
    origin=np.r_[np.zeros(NOMINAL,np.int8),np.ones(PHYSICAL,np.int8),np.full(NEW,2,np.int8)]
    local_index=np.r_[np.arange(NOMINAL),np.arange(PHYSICAL),np.arange(NEW)].astype(np.int64)
    dataset=np.empty(TOTAL,np.int64);phase=np.empty(TOTAL,np.int8);control=np.empty(TOTAL,np.int64);frame=np.empty(TOTAL,np.int64)
    start=0
    for role,n,codes in [('old_centers',3057,[0,1,2]),('old_pico',5980,[3]),('old_walk002',867,[4])]:
        rows=nominal_archive(paths[role],n);stop=start+n
        exact(rows['joint_span'],norm['joint_span'],role+' span');exact(rows['joint_limits'],norm['joint_limits'],role+' limits')
        x[start:stop,:1000]=np.take(rows['features'],RETAINED,axis=1)
        x[start:stop,1000:1023]=rows['previous_action'];x[start:stop,1023:]=rows['history'];y[start:stop]=rows['expert_target']
        ds=np.concatenate([np.full(sum(COUNTS[d]),d,np.int64) for d in codes])
        cs=np.concatenate([np.arange(250,250+sum(COUNTS[d]),dtype=np.int64) for d in codes])
        ps=np.concatenate([np.repeat(np.arange(3,dtype=np.int64),COUNTS[d]) for d in codes])
        exact(rows['control'],cs,role+' controls');exact(rows['source_frame'],cs+11,role+' source frames')
        if role=='old_centers':
            values,_=load_numeric(paths[role],{'dataset':((3057,),'int64')});exact(values['dataset'],ds,'center dataset')
        else:
            values,_=load_numeric(paths[role],{'phase':((n,),'int64')});exact(values['phase'],ps,role+' saved phase')
        dataset[start:stop]=ds;phase[start:stop]=ps;control[start:stop]=cs;frame[start:stop]=rows['source_frame'];start=stop
        del rows
    assert start==NOMINAL
    context=checked_npy(paths['old_nominal_context'],(NOMINAL,323),'float32')
    exact(x[:NOMINAL,1000:],context,'all original nominal contexts');del context
    # Absolute endpoint teacher target, not endpoint-minus-successor training label.
    pf=checked_npy(physical_arrays['endpoint_features'],(PHYSICAL,1069),'float32')
    x[NOMINAL:OLD,:1000]=np.take(pf,RETAINED,axis=1);del pf
    y[NOMINAL:OLD]=checked_npy(physical_arrays['label_fixed_map_target'],(PHYSICAL,23),'float64')
    context=checked_npy(paths['old_physical_context'],(PHYSICAL,323),'float32')
    actual=checked_npy(physical_arrays['policy_actual_normalized_action'],(PHYSICAL,23),'float32')
    advanced=checked_npy(physical_arrays['advanced_history'],(PHYSICAL,300),'float32')
    exact(context[:,:23],actual,'physical actual applied prior');exact(context[:,23:],advanced,'physical once-advanced history')
    applied=checked_npy(physical_arrays['policy_applied_target'],(PHYSICAL,23),'float64')
    inverse=((applied-norm['default_q'])*np.asarray(contract['kp'])/(.25*np.asarray(contract['training_effort']))).astype(np.float32)
    exact(actual,inverse,'physical inverse applied target')
    x[NOMINAL:OLD,1000:]=context;del context,actual,advanced,applied,inverse
    d=np.repeat(np.arange(3,dtype=np.int64),1018);c=np.tile(np.arange(251,1269,dtype=np.int64),3)
    exact(checked_npy(physical_arrays['dataset'],(PHYSICAL,),'int64'),d,'physical dataset')
    exact(checked_npy(physical_arrays['successor_control'],(PHYSICAL,),'int64'),c,'physical endpoint clock')
    exact(checked_npy(physical_arrays['start_control'],(PHYSICAL,),'int64'),c-1,'physical branch start')
    dataset[NOMINAL:OLD]=d;control[NOMINAL:OLD]=c;frame[NOMINAL:OLD]=c+11;phase[NOMINAL:OLD]=np.where(c<350,0,np.where(c<1169,1,2))
    # Clipping is known for physical and new labels. Old nominal flags are unknown.
    feedback=np.zeros((TOTAL,23),bool);native=np.zeros((TOTAL,23),bool);clip_known=np.zeros(TOTAL,bool)
    feedback[NOMINAL:OLD]=checked_npy(physical_arrays['teacher_feedback_clipped'],(PHYSICAL,23),'bool')
    native[NOMINAL:OLD]=checked_npy(physical_arrays['teacher_native_clipped'],(PHYSICAL,23),'bool');clip_known[NOMINAL:OLD]=True
    new_spec={'causal_features':((NEW,1323),'float32'),'features':((NEW,1000),'float32'),'context':((NEW,323),'float32'),
        'incoming_prior':((NEW,23),'float32'),'incoming_history':((NEW,300),'float32'),'expert_target':((NEW,23),'float64'),
        'control':((NEW,),'int64'),'source_frame':((NEW,),'int64'),'phase':((NEW,),'int8'),
        'first_student_state_query':((NEW,),'bool'),'plan_control':((NEW,),'int64'),'plan_local':((NEW,),'int64'),
        'feedback_clipped':((NEW,23),'bool'),'native_clipped':((NEW,23),'bool')}
    new,_=load_numeric(paths['new_rows'],new_spec)
    cs=np.arange(251,1269,dtype=np.int64);exact(new['control'],cs,'new controls');exact(new['source_frame'],cs+11,'new frame')
    exact(new['phase'],np.repeat(np.arange(3,dtype=np.int8),[99,819,100]),'new phase')
    exact(new['first_student_state_query'],np.r_[True,np.zeros(NEW-1,bool)],'one fresh student state')
    exact(new['plan_control'],251+5*((cs-251)//5),'new committed plan');exact(new['plan_local'],(cs-251)%5,'new local plan')
    exact(new['causal_features'][:,:1000],new['features'],'new current prefix');exact(new['causal_features'][:,1000:],new['context'],'new context suffix')
    exact(new['context'][:,:23],new['incoming_prior'],'new incoming prior');exact(new['context'][:,23:],new['incoming_history'],'new incoming history')
    x[OLD:]=new['causal_features'];y[OLD:]=new['expert_target'];dataset[OLD:]=5;phase[OLD:]=new['phase'];control[OLD:]=cs;frame[OLD:]=cs+11
    feedback[OLD:]=new['feedback_clipped'];native[OLD:]=new['native_clipped'];clip_known[OLD:]=True
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite assembled records')
    if np.any(y<norm['joint_limits'][:,0]) or np.any(y>norm['joint_limits'][:,1]):raise ValueError('Target outside native interval')
    plan_control=np.full(TOTAL,-1,np.int64);plan_local=plan_control.copy();plan_control[OLD:]=new['plan_control'];plan_local[OLD:]=new['plan_local']
    labels=normalized_labels(y,norm['default_q'],norm['joint_span'])
    metadata=dict(origin=origin,origin_row=local_index,dataset=dataset,phase=phase,control=control,source_frame=frame,
        plan_control=plan_control,plan_local=plan_local,teacher_feedback_clipped=feedback,teacher_native_clipped=native,
        clipping_flags_known=clip_known,target=y,normalized_target=labels,incoming_prior=x[:,1000:1023],incoming_history=x[:,1023:])
    return x,y,labels,norm,metadata
