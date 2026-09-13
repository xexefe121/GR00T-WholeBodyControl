"""Small synthetic states/receipts only; no task dataset, model or native work."""
import ast
import copy
from pathlib import Path
import numpy as np
import pytest
from collection_math import START,STOP,MAIN,HOLD,WIDTHS,phase,plan_index,committed_target
from collection_adapter import collect,validate_completed
from qualification_gate import validate_reports,member,QUALIFIED_ROLES
from observations import BFMHistory,state_and_terms

def fake_segment(n,start,history=None):
    q=np.zeros((n+1,30),np.float64);q[:,2]=.75;q[:,3]=1
    v=np.zeros((n+1,29),np.float64)
    integ=np.zeros((n+1,291),np.float64);integ[:,0]=(np.arange(n+1)+start)*.02;integ[:,1:31]=q;integ[:,31:60]=v
    mode=np.where(np.arange(start,start+n)<250,0,np.where(np.arange(start,start+n)<STOP,1,2)).astype(np.int64)
    h=BFMHistory() if history is None else copy.deepcopy(history)
    rows=[];names={k:[] for k in WIDTHS}
    for i in range(n):
        for k in WIDTHS:names[k].append(h.data[k].copy())
        _,terms=state_and_terms(q[i,7:],v[i,6:],q[i,3:7],v[i,3:6],np.zeros(23,np.float32),np.zeros(23))
        rows.append(h.before_update(terms))
    a=dict(global_control=np.arange(start,start+n,dtype=np.int64),source_frame=np.minimum(np.arange(start,start+n,dtype=np.int64)+11,1579),
           physics_substeps=np.full(n,10,np.int64),controller_mode=mode,qpos=q,qvel=v,target=np.zeros((n,23),np.float64),
           control_integration_before=integ[:-1].copy(),control_previous_action_before=np.zeros((n,23),np.float32),
           control_history_before=np.asarray(rows),previous_action=np.zeros((n,23),np.float32),action=np.zeros((n,23),np.float32),
           initial_integration=integ[0].copy(),final_integration=integ[-1].copy(),final_previous_action=np.zeros(23,np.float32),
           integration_state_spec=np.asarray(8191),final_recorded_controls=np.asarray(start+n))
    a.update({'control_history_'+k:np.asarray(v) for k,v in names.items()})
    a.update({'final_history_'+k:v.copy() for k,v in h.data.items()})
    return a,h

@pytest.fixture(scope='module')
def fixture():
    main,h=fake_segment(MAIN,0);hold,_=fake_segment(HOLD,MAIN,h)
    snapshot=dict(integration=main['control_integration_before'][START].copy(),previous_action=main['previous_action'][START].copy(),
                  history_flat=main['control_history_before'][START].copy())
    contract=dict(default_q=np.zeros(23),kp=np.ones(23),training_effort=np.full(23,4.),joint_limits=np.tile([-1.,1.],(23,1)))
    state=np.r_[main['qpos'][START],main['qvel'][START]]
    plans={c:dict(nominal_states=np.tile(state,(31,1)),targets=np.zeros((30,23)),gains=np.zeros((30,23,58))) for c in range(START,STOP,5)}
    records=[dict(control=c,cost=0.,requested_commit=min(5,STOP-c),executed_controls=min(5,STOP-c)) for c in plans]
    return main,hold,snapshot,contract,plans,records

@pytest.mark.parametrize('control',[0,250,1269,1569])
def test_excludes_prefix_and_terminal(control):
    with pytest.raises(ValueError):phase(control)

def test_exact_phases_and_plan_boundaries():
    assert np.bincount([phase(c) for c in range(START,STOP)]).tolist()==[99,819,100]
    assert plan_index(251)==(251,0) and plan_index(255)==(251,4) and plan_index(256)==(256,0)
    assert plan_index(1268)==(1266,2)

def test_complete_synthetic_collection_has_only_selected_rows(fixture):
    calls=[]
    def features(q,v,frame):calls.append(frame);return np.full(1000,frame,np.float32)
    result=collect(*fixture,features,lambda x,y:np.zeros(58))
    assert calls==list(range(262,1280)) and result['causal_features'].shape==(1018,1323)
    assert result['control'].tolist()==list(range(251,1269))
    assert result['first_student_state_query'].tolist()==[True]+[False]*1017
    assert np.array_equal(result['causal_features'][:,:1000],result['features'])
    assert np.array_equal(result['causal_features'][:,1000:1023],result['incoming_prior'])
    assert np.array_equal(result['causal_features'][:,1023:],result['incoming_history'])

@pytest.mark.parametrize('fault',['boundary','hold_state','hold_prior','hold_history','history_shift','applied_inverse'])
def test_saved_context_corruption_rejected(fixture,fault):
    main,hold,snapshot,contract,*_=copy.deepcopy(fixture)
    if fault=='boundary':snapshot['integration'][100]+=1
    elif fault=='hold_state':hold['initial_integration'][100]+=1;hold['control_integration_before'][0,100]+=1
    elif fault=='hold_prior':hold['previous_action'][0,0]=1;hold['control_previous_action_before'][0,0]=1
    elif fault=='hold_history':hold['control_history_before'][0,0]+=1;hold['control_history_actions'][0,0,0]+=1
    elif fault=='history_shift':main['control_history_before'][252,0]+=1;main['control_history_actions'][252,0,0]+=1
    else:main['target'][251,0]=.25
    with pytest.raises(ValueError):validate_completed(main,hold,snapshot,contract)

def test_nested_feedback_and_native_clips():
    gain=np.zeros((23,58));gain[0,0]=2.;u=np.zeros(23);u[0]=.95
    target,raw,corr,pre=committed_target(lambda x,y:np.ones(58),np.zeros(59),u,gain,np.zeros(30),np.zeros(29),np.tile([-1.,1.],(23,1)))
    assert raw[0]==2. and corr[0]==.1 and pre[0]==1.05 and target[0]==1.

def test_map_functions_ast_exact_to_qualified_saved_math():
    root=Path(__file__).resolve().parents[2]
    original=root/'direct_target_width512_saved_semantics_review_v1/fixed_maps.py'
    def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    before=functions(original);after=functions(Path(__file__).with_name('collection_math.py'))
    assert before['difference_function']==after['difference_function']
    assert before['committed_target']==after['committed_target']

def fake_reports():
    subjects={k:dict(path='E:/fake/'+k+'.json',sha256=k+'-digest') for k in (*QUALIFIED_ROLES,'qualification')}
    owner=dict(completion_accounting_passed=True,requested_recovery_completed=True,raw_exit_known=True,all_postrun_pins_exact=True,
               processes_absent=True,raw_python_exit_code=0,exit_code=0,accounting_uncertainty=[],
               output_sha256={subjects[k]['path']:subjects[k]['sha256'] for k in ('main_trace','main_report','hold_trace','hold_report','recovery_request')})
    reports={'owner':owner,'source_review':dict(source_review_pass=True),'boundary_review':dict(boundary_review_pass=True),
             'selection_receipt':dict(passed=True,outputs={'precontrol251.npz':subjects['boundary_snapshot']['sha256']}),
             'recovery_request':dict(root_selected=True,kind='fixed_width81000_pre251_actual_expert_recovery',
                 protocol=dict(initial_global_control=251,prefix_controls=251,learned_prefix_controls=1,requested_branch_controls=1318,
                     MPC_controls=1018,conditional_hold_controls=250,original_source_controls=819,requested_combined_main_controls=1569,prefix_native_steps=2510),
                 subjects=dict(selected_snapshot=subjects['boundary_snapshot'],input_selection=subjects['selection_receipt']))}
    reports['boundary_review'].update(snapshot_subject=subjects['boundary_snapshot'],selection_subject=subjects['selection_receipt'])
    comparisons={k:True for k in ('all_qpos_bitexact','all_qvel_bitexact','all_command_torque_bitexact','physics_actuator_force_bitexact',
                                 'all_physics_time_bitexact','physics_warning_number_bitexact','physics_warning_lastinfo_bitexact')}
    for part,n,start in [('main',1569,0),('hold',250,1569)]:
        reports[part+'_report']=dict(full_segment_completed=True,requested_controls=n,trace_sha256=subjects[part+'_trace']['sha256'])
        reports[part+'_physics']=dict(independent_segment_pass=True,recorded_trace_reproduced_through_last_sample=True,
            requested_segment_completed=True,intended_segment_controls=n,physics_steps=n*10,recorded_partial_substeps=0,
            original_trace_comparison=comparisons.copy(),input_hashes={subjects[part+'_trace']['path']:subjects[part+'_trace']['sha256']})
        reports[part+'_intent']=dict(independent_physical_pass=True,intended_segment_completed=True,requested_segment_quiet_pass=True,
            requested_controls=n,global_start=start,full_lifecycle_source_intent_pass=True,source_metrics=dict(source_controls=819),
            hashes={subjects[k]['path']:subjects[k]['sha256'] for k in (part+'_trace',part+'_physics','main_trace')})
    reports['qualification']=dict(root_authorized_extraction=True,model_fitting_authorized=False,control_start=251,
        control_stop_exclusive=1269,rows=1018,subjects={k:subjects[k] for k in QUALIFIED_ROLES})
    return subjects,reports

def test_complete_literal_report_chain():validate_reports(*fake_reports())

@pytest.mark.parametrize('fault',['owner','raw_bool','main_quiet','hold_quiet','source','wrong_trace','wrong_physics','wrong_scope','partial','recovery_scope','boundary_hash'])
def test_qualification_failures_are_closed(fault):
    subjects,reports=fake_reports()
    if fault=='owner':reports['owner']['processes_absent']=False
    elif fault=='raw_bool':reports['owner']['raw_python_exit_code']=False
    elif fault=='main_quiet':reports['main_intent']['requested_segment_quiet_pass']=False
    elif fault=='hold_quiet':reports['hold_intent']['requested_segment_quiet_pass']=False
    elif fault=='source':reports['main_intent']['full_lifecycle_source_intent_pass']=False
    elif fault=='wrong_trace':reports['main_physics']['input_hashes']={'E:/other.json':subjects['main_trace']['sha256']}
    elif fault=='wrong_physics':reports['hold_intent']['hashes'].pop(subjects['hold_physics']['path'])
    elif fault=='wrong_scope':reports['qualification']['control_start']=250
    elif fault=='partial':reports['main_physics']['recorded_partial_substeps']=1
    elif fault=='recovery_scope':reports['recovery_request']['protocol']['initial_global_control']=250
    else:reports['boundary_review']['snapshot_subject']=dict(path='E:/fake/wrong',sha256='wrong')
    with pytest.raises(ValueError):validate_reports(subjects,reports)

def test_windows_wsl_literal_aliases():
    member({'/mnt/e/fake/a.json':'digest'},dict(path='E:\\fake\\a.json',sha256='digest'))
    with pytest.raises(ValueError):member({'E:/other.json':'digest'},dict(path='E:/fake/a.json',sha256='digest'))

def test_real_pure_features_use_measured_heading_and_preserve_inputs():
    from direct_features import DirectFeatures
    n=100;quat=np.zeros((n,24,4));quat[:,:,0]=1
    task=np.zeros((n,3,4));task[:,:,0]=1
    motion=dict(joint_pos=np.tile(np.arange(n)[:,None],(1,23)).astype(float),joint_vel=np.zeros((n,23)),
        body_pos_w=np.zeros((n,24,3)),body_quat_w=quat,body_lin_vel_w=np.zeros((n,24,3)),body_ang_vel_w=np.zeros((n,24,3)))
    original=dict(source_task_position_w=np.zeros((n,3,3)),source_task_quaternion_wxyz=task)
    contract=dict(default_q=np.zeros(23),body_names=['left_ankle_roll_link','right_ankle_roll_link']+['b'+str(i) for i in range(22)])
    builder=DirectFeatures(motion,original,contract)
    q=np.zeros(30);q[0]=1;q[2]=.75;q[3]=1;v=np.zeros(29);before=q.tobytes()
    result=builder(q,v,11);assert result.shape==(1000,) and result.dtype==np.float32 and q.tobytes()==before
    goals=result[56:].reshape(8,118)
    assert goals[:,0].tolist()==[11.,12.,13.,15.,19.,27.,35.,48.]
    assert goals[0,46]==-1.
    q[3]=q[6]=np.sqrt(.5);rotated=builder(q,v,11)[56:].reshape(8,118)
    assert abs(float(rotated[0,46]))<1e-6 and rotated[0,47]==1.
