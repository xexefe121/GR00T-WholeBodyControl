"""Build direct supervision solely from a qualified completed saved trajectory."""
from types import SimpleNamespace
import numpy as np
from causal_features import incoming_context
from observations import BFMHistory,state_and_terms
from collection_math import START,STOP,MAIN,HOLD,WIDTHS,array,exact,phase,plan_index,named,committed_target

def trace_contract(trace,n,start):
    fields={'global_control':((n,),'int64'),'source_frame':((n,),'int64'),'physics_substeps':((n,),'int64'),
            'controller_mode':((n,),'int64'),'qpos':((n+1,30),'float64'),'qvel':((n+1,29),'float64'),
            'target':((n,23),'float64'),'control_integration_before':((n,291),'float64'),
            'control_previous_action_before':((n,23),'float32'),'control_history_before':((n,300),'float32'),
            'previous_action':((n,23),'float32'),'action':((n,23),'float32'),
            'initial_integration':((291,),'float64'),'final_integration':((291,),'float64'),
            'final_previous_action':((23,),'float32')}
    for key,(shape,dtype) in fields.items():array(trace[key],shape,dtype,key)
    exact(trace['global_control'],np.arange(start,start+n,dtype=np.int64),'absolute controls')
    exact(trace['source_frame'],np.minimum(trace['global_control']+11,1579),'original frame clock')
    exact(trace['physics_substeps'],np.full(n,10,np.int64),'complete controls')
    exact(trace['control_integration_before'][:,1:31],trace['qpos'][:-1],'integration qpos')
    exact(trace['control_integration_before'][:,31:60],trace['qvel'][:-1],'integration qvel')
    exact(trace['control_integration_before'][0],trace['initial_integration'],'initial integration')
    exact(trace['final_integration'][1:31],trace['qpos'][-1],'final qpos')
    exact(trace['final_integration'][31:60],trace['qvel'][-1],'final qvel')
    exact(trace['previous_action'],trace['control_previous_action_before'],'prior aliases')
    exact(trace['previous_action'][1:],trace['action'][:-1],'prior chronology')
    exact(trace['final_previous_action'],trace['action'][-1],'final prior')
    for key,width in WIDTHS.items():
        array(trace['control_history_'+key],(n,4,width),'float32','named history '+key)
        array(trace['final_history_'+key],(4,width),'float32','final history '+key)
    flat=np.concatenate([trace['control_history_'+key].reshape(n,-1) for key in sorted(WIDTHS)],axis=1)
    exact(flat,trace['control_history_before'],'named/flat history')
    if int(trace['integration_state_spec'])!=8191:raise ValueError('Full integration spec')
    if int(trace['final_recorded_controls'])!=start+n:raise ValueError('Final history clock')

def validate_history(trace,default):
    # Before-update context is checked from saved measured states, without inference.
    history=BFMHistory();history.data=named(trace['control_history_before'][0])
    for i in range(len(trace['global_control'])):
        q,v,prior=trace['qpos'][i],trace['qvel'][i],trace['previous_action'][i]
        _,terms=state_and_terms(q[7:],v[6:],q[3:7],v[3:6],prior,default)
        before=history.before_update(terms)
        exact(before,trace['control_history_before'][i],'one history update '+str(i))
    for key in WIDTHS:exact(history.data[key],trace['final_history_'+key],'final history '+key)

def validate_completed(main,hold,snapshot,contract):
    trace_contract(main,MAIN,0);trace_contract(hold,HOLD,MAIN)
    exact(main['controller_mode'][:250],np.zeros(250,np.int64),'BFM prefix')
    exact(main['controller_mode'][250:STOP],np.ones(STOP-250,np.int64),'learned250 then expert251')
    exact(main['controller_mode'][STOP:],np.full(MAIN-STOP,2,np.int64),'terminal mode')
    exact(hold['controller_mode'],np.full(HOLD,2,np.int64),'hold mode')
    exact(main['control_integration_before'][START],snapshot['integration'],'qualified actual pre251')
    exact(main['previous_action'][START],snapshot['previous_action'],'qualified incoming applied prior251')
    exact(main['control_history_before'][START],snapshot['history_flat'],'qualified incoming history251')
    exact(main['final_integration'],hold['initial_integration'],'continuous full291')
    exact(main['final_previous_action'],hold['previous_action'][0],'continuous prior')
    for key in WIDTHS:
        exact(main['final_history_'+key],hold['control_history_'+key][0],'continuous history '+key)
    default,kp,effort=[np.asarray(contract[key],np.float64) for key in ('default_q','kp','training_effort')]
    for value in (default,kp,effort):array(value,(23,),'float64','native vector')
    if np.any(kp<=0) or np.any(effort<=0):raise ValueError('Positive native gains/effort')
    inverse=((main['target'][250:STOP]-default)*kp/(.25*effort)).astype(np.float32)
    exact(inverse,main['action'][250:STOP],'actual applied-target inverse')
    validate_history(main,default);validate_history(hold,default)

def collect(main,hold,snapshot,contract,plans,plan_records,current_features,difference):
    validate_completed(main,hold,snapshot,contract)
    expected=list(range(START,STOP,5))
    if sorted(plans)!=expected or [r['control'] for r in plan_records]!=expected:raise ValueError('All204 original committed plans required')
    for record in plan_records:
        count=min(5,STOP-record['control'])
        if record['requested_commit']!=count or record['executed_controls']!=count or not np.isfinite(record['cost']):
            raise ValueError('Completed finite committed plan required')
    limits=np.asarray(contract['joint_limits'],np.float64);array(limits,(23,2),'float64','joint limits')
    rows={}
    for control in range(START,STOP):
        owner,local=plan_index(control);plan=plans[owner]
        for key,shape in [('nominal_states',(31,59)),('targets',(30,23)),('gains',(30,23,58))]:
            array(plan[key],shape,'float64','plan '+key)
        q,v=main['qpos'][control],main['qvel'][control]
        target,raw,correction,preclip=committed_target(difference,plan['nominal_states'][local],plan['targets'][local],plan['gains'][local],q,v,limits)
        exact(target,main['target'][control],'actual applied expert map '+str(control))
        features=current_features(q,v,int(main['source_frame'][control]));array(features,(1000,),'float32','current features')
        context=incoming_context(SimpleNamespace(previous_action=main['previous_action'][control],
            history=SimpleNamespace(data=named(main['control_history_before'][control]))))
        values=dict(control=np.int64(control),source_frame=main['source_frame'][control],phase=np.int8(phase(control)),
            first_student_state_query=np.bool_(control==START),features=features,context=context,
            causal_features=np.concatenate([features,context]),qpos=q,qvel=v,
            integration=main['control_integration_before'][control],incoming_prior=main['previous_action'][control],
            incoming_history=main['control_history_before'][control],expert_target=target,
            plan_control=np.int64(owner),plan_local=np.int64(local),planned_state=plan['nominal_states'][local],
            planned_target=plan['targets'][local],gain=plan['gains'][local],feedback_raw=raw,
            feedback_correction=correction,feedback_clipped=raw!=correction,native_clipped=preclip!=target)
        values.update({'incoming_history_'+k:main['control_history_'+k][control] for k in WIDTHS})
        for key,value in values.items():rows.setdefault(key,[]).append(np.asarray(value).copy())
    result={key:np.stack(value) for key,value in rows.items()}
    exact(result['control'],np.arange(START,STOP,dtype=np.int64),'exact1018 admitted indices')
    assert np.bincount(result['phase'],minlength=3).tolist()==[99,819,100]
    return result
