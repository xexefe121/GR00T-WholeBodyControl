"""Pure math, recorded-input and stub tests. No model/native construction."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import json
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from direct_features import DirectFeatures
from direct_runtime import DirectStudentRuntime,Phases,native_output,actual_previous_action,exact
from evaluation_gate import require_model_ready
from proposal_evidence import ProposalEvidence,trace_arrays
from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory,state_and_terms

SOURCE=Path(__file__).resolve().parent


def timeline():
    names=['initial_standing','acquisition_ramp','source_motion','return_ramp','returned_standing','standing_proof_margin']
    ranges=[(0,250),(250,350),(350,1169),(1169,1269),(1269,1519),(1519,1569)]
    return dict(total_requested_controls=1569,prehistory_frames=11,phases=[dict(name=n,control_start=a,control_stop=b,requested_controls=b-a) for n,(a,b) in zip(names,ranges)])


def contract():
    return dict(default_q=np.linspace(-.2,.2,23),joint_limits=np.tile([-1.123456789,1.234567891],(23,1)),
                kp=np.linspace(10,50,23),training_effort=np.linspace(5,20,23))


class StubSession:
    def __init__(self,output,fail=False):self.output=output;self.fail=fail;self.feeds=[]
    def run(self,names,feed):
        self.feeds.append({k:v.copy() for k,v in feed.items()})
        if self.fail:raise RuntimeError('scripted graph fault')
        return [self.output.copy()]


class Seed:
    def __init__(self,c):
        self.contract=c;self.history=BFMHistory();self.previous_action=np.zeros(23,np.float32);self.recorded_controls=0
        raw=np.linspace(-1,1,23,dtype=np.float32);raw[0]=-0.0
        self.sessions={'backward':StubSession(np.ones((1,256),np.float32)), 'actor':StubSession(raw[None])}
    def _terms(self,q,dq,previous):return state_and_terms(q[7:],dq[6:],q[3:7],dq[3:6],previous,self.contract['default_q'])
    def _goal(self,frame,q):return self.sessions['backward'].run(None,{'state':np.zeros((1,52),np.float32)})[0]


def original_infer():
    text=(SOURCE/'student_linear_runtime.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='infer_base')
    scope={'np':np,'terminal_goal_yaw4':lambda seed,frame,q:seed._goal(frame,q)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<unchanged original infer_base>', 'exec'),scope)
    return scope['infer_base']


def runtime(head=None,features=None):
    c=contract();seed=Seed(c)
    span=np.diff(c['joint_limits'],axis=1)[:,0].astype(np.float32)
    head=StubSession(np.zeros((1,23),np.float32)) if head is None else head
    features=(lambda q,dq,frame:np.r_[q[7:],dq,np.zeros(948)].astype(np.float32)) if features is None else features
    result=DirectStudentRuntime(seed,head,features,c,span,timeline(),original_infer())
    q=np.r_[0.,0.,.76,1.,0.,0.,0.,c['default_q']];dq=np.zeros(29)
    return result,q,dq


def test_output_promotes_span_and_head_before_multiplication():
    c=contract();span=np.diff(c['joint_limits'],axis=1)[:,0].astype(np.float32)
    head=np.linspace(-.1,.1,23,dtype=np.float32)
    raw,target,delta=native_output(head,c['default_q'],span,c['joint_limits'])
    expected_delta=span.astype(np.float64)*head.astype(np.float64)
    assert exact(delta,expected_delta) and exact(raw,c['default_q']+expected_delta)
    assert np.any(delta!=(span*head).astype(np.float64))
    recomputed=(c['joint_limits'][:,1]-c['joint_limits'][:,0])*head.astype(np.float64)
    assert np.any(delta!=recomputed)


def test_final_clamp_repairs_float32_limit_rounding():
    default=np.zeros(23,np.float64);span=np.ones(23,np.float32)
    limits=np.tile([-0.1,0.1],(23,1));head=np.full(23,.1,np.float32)
    raw,target,_=native_output(head,default,span,limits)
    assert np.all(raw>limits[:,1]) and exact(target,limits[:,1])


@pytest.mark.parametrize('bad',[np.zeros(23,np.float64),np.zeros(24,np.float32),np.full(23,np.nan,np.float32)])
def test_invalid_head_rejected(bad):
    c=contract()
    with pytest.raises(ValueError):native_output(bad,c['default_q'],np.ones(23,np.float32),c['joint_limits'])


def test_learned_zero_bfm_calls_and_actual_target_prior():
    r,q,dq=runtime(head=StubSession(np.full((1,23),10,np.float32)))
    r.seed.recorded_controls=250;r.seed.previous_action[:]=77
    original_history={k:v.copy() for k,v in r.seed.history.data.items()}
    proposed=r.propose(250,q,dq)
    assert r.counts['1_head_attempted']==r.counts['1_head_returned']==1
    assert all(value==0 for key,value in r.counts.items() if 'head' not in key)
    assert all(exact(r.seed.history.data[k],v) for k,v in original_history.items())
    assert np.all(r.seed.previous_action==77) and r.seed.recorded_controls==250
    assert exact(proposed['target'],r.limits[:,1])
    r.commit(proposed)
    assert exact(r.seed.previous_action,actual_previous_action(proposed['target'],r.c))
    assert np.any(np.abs(r.seed.previous_action)>5)
    assert np.all(r.seed.history.data['actions'][0]==77)
    assert r.seed.recorded_controls==251


def test_different_prior_history_same_measured_feature_head_and_target():
    left,q,dq=runtime();right,_,_=runtime()
    for r in (left,right):r.seed.recorded_controls=250
    right.seed.previous_action[:]=999
    for v in right.seed.history.data.values():v[:]=-123
    a=left.propose(250,q,dq);b=right.propose(250,q,dq)
    for key in ('features','normalized_head','target'):assert exact(a[key],b[key])
    assert not exact(a['history'],b['history']) and not exact(a['previous_action'],b['previous_action'])


@pytest.mark.parametrize('control',[0,249,1269,1569,1818])
def test_original_bfm_startup_and_terminal_arithmetic(control):
    r,q,dq=runtime();r.seed.recorded_controls=control
    prior=r.seed.previous_action.copy();staged=copy.deepcopy(r.seed.history)
    _,terms=r.seed._terms(q,dq,prior);wanted_history=staged.before_update(terms)
    raw=np.linspace(-1,1,23,dtype=np.float32);raw[0]=-0.0;raw=raw*5
    base=r.c['default_q']+raw*.25*r.c['training_effort']/r.c['kp']
    zero=np.zeros(23,np.float32)
    wanted_action=(raw+zero*r.c['kp']/(.25*r.c['training_effort'])).astype(np.float32)
    proposed=r.propose(control,q,dq)
    assert exact(proposed['base_target'],base) and exact(proposed['target'],np.clip(base+zero,r.limits[:,0],r.limits[:,1]))
    assert exact(proposed['action'],wanted_action) and exact(proposed['history'],wanted_history)
    r.commit(proposed)
    assert all(exact(r.seed.history.data[k],v) for k,v in staged.data.items())
    assert exact(r.seed.previous_action,wanted_action)
    mode=r.phases.mode(control)
    assert r.counts[str(mode)+'_backward_returned']==r.counts[str(mode)+'_actor_returned']==1
    assert all(value==0 for key,value in r.counts.items() if 'head' in key)


def test_before_update_once_across_learned_to_terminal():
    r,q,dq=runtime();r.seed.recorded_controls=1267
    original=BFMHistory();previous=np.zeros(23,np.float32)
    for control in (1267,1268,1269):
        _,terms=r.seed._terms(q,dq,previous);expected=original.before_update(terms)
        proposed=r.propose(control,q,dq)
        assert exact(proposed['history'],expected)
        r.commit(proposed);previous=proposed['action'].copy()
        assert all(exact(r.seed.history.data[k],v) for k,v in original.data.items())
    assert r.counts['1_head_returned']==2 and r.counts['2_actor_returned']==1


def test_head_exception_preserves_input_and_does_not_commit_history(tmp_path):
    r,q,dq=runtime(head=StubSession(np.zeros((1,23),np.float32),fail=True));r.seed.recorded_controls=250
    evidence=ProposalEvidence(r,r.c)
    evidence.begin(250,q,dq,np.zeros(291),5.,np.zeros(8,np.int32),np.zeros(8,np.int32))
    with pytest.raises(RuntimeError):r.propose(250,q,dq)
    assert r.seed.recorded_controls==250 and not np.any(r.seed.history.data['actions'])
    evidence.preserve(tmp_path/'failure.npz','fault',q,dq,np.zeros(291),5.,np.zeros(8,np.int32),np.zeros(8,np.int32))
    with np.load(tmp_path/'failure.npz') as z:
        assert z['runtime_head_input_features'].shape==(1,1000)
        assert not z['runtime_head_current_call_returned']
        assert z['count_1_head_attempted']==1 and z['count_1_head_returned']==0


def test_parity_rejection_pending_proposal_does_not_commit(tmp_path):
    r,q,dq=runtime();r.seed.recorded_controls=250
    evidence=ProposalEvidence(r,r.c);evidence.begin(250,q,dq,np.zeros(291),5.,np.zeros(8,np.int32),np.zeros(8,np.int32))
    proposed=r.propose(250,q,dq);evidence.accepted_proposal(proposed)
    assert r.pending is not None and r.seed.recorded_controls==250
    assert not np.any(r.seed.history.data['projected_gravity'])
    evidence.preserve(tmp_path/'failure.npz','witness mismatch',q,dq,np.zeros(291),5.,np.zeros(8,np.int32),np.zeros(8,np.int32))
    with np.load(tmp_path/'failure.npz') as z:
        assert z['uncommitted_proposal'] and z['head_output_0'].shape==(1,23)


def test_mutated_proposal_and_duplicate_commit_rejected():
    r,q,dq=runtime();r.seed.recorded_controls=250
    p=r.propose(250,q,dq);saved=p['target'].copy();p['target'][0]+=.1
    with pytest.raises(ValueError):r.commit(p)
    assert r.seed.recorded_controls==250
    p['target'][:]=saved;r.commit(p)
    with pytest.raises(ValueError):r.commit(p)


def test_guard_forbids_bfm_during_learned():
    r,_,_=runtime();r.current_mode=1
    with pytest.raises(ValueError):r.seed.sessions['actor'].run(None,{})
    assert r.forbidden_calls==1 and r.counts['1_actor_attempted']==0


def test_unbound_witness_and_evaluator_cannot_create_model(tmp_path):
    for purpose in ('witness','evaluation'):
        with pytest.raises(ValueError,match='no model/native'):require_model_ready(tmp_path,purpose)


def test_timeline_is_authoritative_and_no_clock_id_is_a_feature():
    phases=Phases(timeline())
    assert [phases.mode(c) for c in (0,249,250,1268,1269,1818)]==[0,0,1,1,2,2]
    wrong=timeline();wrong['phases'][2]['control_start']=351
    with pytest.raises(ValueError):Phases(wrong)
    node=ast.parse((SOURCE/'direct_features.py').read_text())
    call=next(n for n in ast.walk(node) if isinstance(n,ast.FunctionDef) and n.name=='__call__')
    assert [a.arg for a in call.args.args]==['self','qpos','qvel','frame']


@pytest.mark.parametrize('frame',[0,2,20,39,100])
def test_direct_features_byte_match_original_slices_with_signed_zero(frame):
    original=(SOURCE/'gear_sonic/utils/g1_true23_mpc_student.py').read_text()
    node=next(n for n in ast.parse(original).body if isinstance(n,ast.ClassDef) and n.name=='GoalFeatures')
    scope={'np':np,'Rotation':Rotation,'OFFSETS':np.array([0,1,2,4,8,16,24,37]),'FEATURES':1023}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<pure original GoalFeatures>', 'exec'),scope)
    rng=np.random.default_rng(123);n=40
    motion=dict(joint_pos=rng.normal(size=(n,23)),joint_vel=rng.normal(size=(n,23)),
        body_pos_w=rng.normal(size=(n,24,3)),body_quat_w=np.tile([1.,0.,0.,0.],(n,24,1)),
        body_lin_vel_w=rng.normal(size=(n,24,3)),body_ang_vel_w=rng.normal(size=(n,24,3)))
    original29=dict(source_task_position_w=rng.normal(size=(n,3,3)),source_task_quaternion_wxyz=np.tile([1.,0.,0.,0.],(n,3,1)))
    c=contract();c['body_names']=['left_ankle_roll_link','right_ankle_roll_link']+[str(i) for i in range(22)]
    q=np.r_[0.,-.0,.76,1.,0.,0.,0.,c['default_q']];dq=rng.normal(size=29);dq[6]=-0.0
    old=scope['GoalFeatures'](motion,original29,c)
    new=DirectFeatures(motion,original29,c)
    for previous in (np.zeros(23),np.full(23,99.),np.full(23,-123.)):
        expected=old(q,dq,previous,frame)[np.r_[0:52,75:1023]]
        assert exact(new(q,dq,frame),expected)


def test_native_strict_oracle_and_PD_expression_are_preserved():
    original=ast.parse((SOURCE/'original_evaluator.py').read_text())
    current=ast.parse((SOURCE/'evaluate_direct_target_student.py').read_text())
    for name in ('get_state','assess'):
        nodes=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name) for tree in (original,current)]
        assert ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False)
    def controls(tree):
        return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Assign)
                and any(ast.unparse(target)=='data.ctrl[:]' for target in n.targets)]
    # Original contains an unused zero-parity routine; selected main expression is identical.
    assert len(controls(current))==1 and controls(current)[0] in controls(original)


def test_preparation_imports_do_not_load_native_or_onnx_modules():
    assert 'mujoco' not in sys.modules and 'onnxruntime' not in sys.modules and 'torch' not in sys.modules


def evaluator_function(name):
    text=(SOURCE/'evaluate_direct_target_student.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    scope={'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<unchanged evaluator helper>', 'exec'),scope)
    return scope[name]


def test_empty_failure_trace_keeps_direct_schema():
    data=SimpleNamespace(qpos=np.zeros(30),qvel=np.zeros(29),time=0.,warning=SimpleNamespace(number=np.zeros(8,np.int32),lastinfo=np.zeros(8,np.int32)))
    trace=evaluator_function('new_trace')(data)
    trace.update(raw_proposal=[],actual_normalized_action=[],normalized_head=[])
    arrays=trace_arrays(trace)
    assert arrays['features'].shape==(0,1000) and arrays['features'].dtype==np.float32
    assert arrays['normalized_head'].shape==(0,23) and arrays['normalized_head'].dtype==np.float32
    assert arrays['physics_qpos'].shape==(1,30) and arrays['control_integration_before'].shape==(0,291)


def test_native_strict_speed_boundary_and_warning_retention():
    c=contract();c.update(native_velocity=np.full(23,20.),native_effort=np.full(23,40.))
    q=np.r_[0.,0.,.76,1.,0.,0.,0.,c['default_q']];dq=np.zeros(29)
    data=SimpleNamespace(qpos=q,qvel=dq,qfrc_actuator=np.zeros(29),time=.002,
        warning=SimpleNamespace(number=np.zeros(8,np.int32),lastinfo=np.zeros(8,np.int32)))
    assess=evaluator_function('assess')
    data.qvel[6]=20.;assert assess(data,c,.002)[0]==[]
    data.qvel[6]=np.nextafter(20.,np.inf);assert 'native_joint_speed' in assess(data,c,.002)[0]
    data.qvel[6]=0.;data.warning.number[0]=1;assert 'engine_warning' in assess(data,c,.002)[0]
    assert data.warning.number[0]==1


def test_native_exception_preserves_pre_attempt_clock_and_partial_schema(tmp_path):
    text=(SOURCE/'evaluate_direct_target_student.py').read_text();tree=ast.parse(text)
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='sub' and ast.unparse(n.iter)=='range(10)']
    assert len(loops)==1
    data=SimpleNamespace(qpos=np.r_[0.,0.,.76,1.,0.,0.,0.,np.zeros(23)],qvel=np.zeros(29),ctrl=np.zeros(23),time=0.,
        qfrc_actuator=np.zeros(29),warning=SimpleNamespace(number=np.zeros(8,np.int32),lastinfo=np.zeros(8,np.int32)))
    trace=evaluator_function('new_trace')(data);trace.update(raw_proposal=[],actual_normalized_action=[],normalized_head=[])
    r,q,dq=runtime();evidence=ProposalEvidence(r,r.c)
    evidence.begin(0,data.qpos,data.qvel,np.zeros(291),0.,data.warning.number,data.warning.lastinfo)
    def fake_step(native,actual):raise RuntimeError('fake native call did not return')
    scope=dict(np=np,json=json,data=data,kp=np.ones(23),kd=np.ones(23),effort=np.ones(23),target=np.full(23,.2),
        mujoco=SimpleNamespace(mj_step=fake_step),native=None,evidence=evidence,dest=tmp_path,trace=trace,
        get_state=lambda native,data:np.zeros(291),expected_time=0.,actual_substeps=0,count=1569,control=0,start=0)
    with pytest.raises(RuntimeError,match='did not return'):
        exec(compile(ast.Module(body=loops,type_ignores=[]),'<same native loop with fake step>', 'exec'),scope)
    assert scope['expected_time']==0. and scope['actual_substeps']==0 and len(trace['physics_time'])==1
    assert (tmp_path/'native_exception_state.npz').is_file() and (tmp_path/'native_exception_completed_samples.npz').is_file()
    failure=json.loads((tmp_path/'fatal_failure.json').read_text())
    assert failure['unclassified_attempted_step'] and failure['completed_prior_substeps']==0
    with np.load(tmp_path/'native_exception_state.npz') as saved:
        assert saved['expected_time_at_failure']==0. and saved['attempted_native_ctrl'].shape==(23,)


def test_initial_complete291_comparison_precedes_segment_call():
    tree=ast.parse((SOURCE/'evaluate_direct_target_student.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    expressions=[ast.unparse(n) for n in main.body]
    gate=next(i for i,s in enumerate(expressions) if "record_parity('canonical_initial_full291_parity.json'" in s)
    segment=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
                 and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
                 and n.value.func.id=='segment')
    assert gate<segment
