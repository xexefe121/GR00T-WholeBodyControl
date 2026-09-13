"""Synthetic saved arrays and injected native/planner/session stubs only."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from recovery_contract import START,PLAN_CONTROLS,budgets,protocol,SOURCE_TRACE_SHA
from recovery_inputs import extract,named,exact,CONTROL,STEP_OUTPUTS
from counted_work import WorkLedger,Hooks,BatchProxy,SessionProxy,BudgetExceeded

SOURCE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('pure_history',SOURCE/'gear_sonic/utils/g1_true23_bfm_seed_observations.py')
history_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(history_module)
BFMHistory=history_module.BFMHistory


def terms(q,v,prior):
    return dict(actions=prior.copy(),base_ang_vel=v[3:6].astype(np.float32),
        dof_pos=q[7:].astype(np.float32),dof_vel=v[6:].astype(np.float32),
        projected_gravity=np.asarray([0.,0.,-1.],np.float32))


def synthetic_trace():
    n=START+1
    q=np.zeros((n+1,30),np.float64);q[:,2]=.8;q[:,3]=1
    v=np.zeros((n+1,29),np.float64)
    for row in range(n+1):
        q[row,7:]=row*.0001;v[row,6:]=row*.0002
    times=np.zeros(n*10+1,np.float64)
    for i in range(n*10):times[i+1]=times[i]+.002
    h=BFMHistory();previous=np.zeros(23,np.float32);hist=[];priors=[]
    actions=np.full((n,23),.25,np.float32);actions[250]=np.arange(23,dtype=np.float32)*.01
    for control in range(n):
        priors.append(previous.copy());hist.append(h.before_update(terms(q[control],v[control],previous)))
        previous=actions[control].copy()
    contract=dict(default_q=np.zeros(23),kp=np.ones(23)*2,training_effort=np.ones(23)*8,
                  kd=np.ones(23),native_effort=np.ones(23)*10)
    integration=np.zeros((n,291),np.float64)
    integration[:,0]=times[np.arange(n)*10];integration[:,1:31]=q[:n];integration[:,31:60]=v[:n]
    integration[:,60:]=np.arange(231)[None]*.125  # Preserve nonzero hidden state/forces.
    trace=dict(control_integration_before=integration,control_history_before=np.asarray(hist),
        history=np.asarray(hist).copy(),control_previous_action_before=np.asarray(priors),
        previous_action=np.asarray(priors).copy(),action=actions,target=actions.astype(np.float64),
        qpos=q,qvel=v,global_control=np.arange(n,dtype=np.int64),source_frame=np.arange(n,dtype=np.int64)+11,
        physics_substeps=np.full(n,10,np.int64),controller_mode=np.r_[np.zeros(250,np.int64),np.ones(2,np.int64)],
        delta=np.zeros((n,23),np.float64),joint_error=np.zeros((n,23)),root_error=np.zeros((n,3)),
        physics_qpos=np.repeat(q,10,axis=0)[:n*10+1],physics_qvel=np.repeat(v,10,axis=0)[:n*10+1],
        physics_time=times,physics_expected_time=times.copy(),
        physics_warning_counts=np.zeros((n*10+1,8),np.int32),physics_warning_lastinfo=np.zeros((n*10+1,8),np.int32),
        initial_integration=integration[0].copy())
    for key in STEP_OUTPUTS:trace[key]=np.zeros((n*10,23) if key in ('physics_torque','physics_actuator_torque') else (n*10,),np.float64)
    return trace,contract


def functions(path,names,env):
    tree=ast.parse(Path(path).read_text())
    nodes=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name in names]
    assert {v.name for v in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env


def test_full251_prefix_and_hidden_state():
    trace,c=synthetic_trace();before={k:v.copy() for k,v in trace.items()}
    snap,prefix=extract(trace,c)
    for key in CONTROL:assert len(prefix[key])==251
    for key in ('qpos','qvel'):assert len(prefix[key])==252
    assert len(prefix['physics_qpos'])==2511 and len(prefix['physics_torque'])==2510
    exact(snap['integration'],trace['control_integration_before'][251],'full291')
    assert np.any(snap['integration'][60:]) and prefix['controller_mode'][-1]==1
    exact(snap['previous_action'],trace['action'][250],'applied250 prior')
    for k in trace:exact(trace[k],before[k],k)


@pytest.mark.parametrize('field,index',[
    ('global_control',(251,)),('source_frame',(251,)),('physics_substeps',(250,)),
    ('controller_mode',(250,)),('control_previous_action_before',(251,0)),
    ('history',(251,0)),('target',(250,0)),('physics_time',(2510,)),
    ('physics_expected_time',(2510,)),('control_integration_before',(251,60)),
])
def test_corrupt_saved_boundary(field,index):
    trace,c=synthetic_trace()
    if field=='control_integration_before':
        # Hidden components have no fabricated oracle: they must be preserved.
        trace[field][index]+=1;snap,_=extract(trace,c)
        exact(snap['integration'],trace[field][251],'all full291 components');return
    trace[field][index]+=1
    with pytest.raises((ValueError,AssertionError)):extract(trace,c)


def test_named_offsets_and_no_aliases():
    flat=np.arange(300,dtype=np.float32);parts=named(flat)
    assert list(parts)==['actions','base_ang_vel','dof_pos','dof_vel','projected_gravity']
    assert [parts[k].reshape(-1)[0] for k in parts]==[0,92,104,196,288]
    parts['actions'][0,0]=-99;assert flat[0]==0
    with pytest.raises(ValueError):named(flat.astype(np.float64))


def test_protocol_counts_and_last_commit():
    b=budgets();p=protocol()
    assert len(PLAN_CONTROLS)==204 and PLAN_CONTROLS[0]==251 and PLAN_CONTROLS[-1]==1266
    assert sum(min(5,1269-c) for c in PLAN_CONTROLS)==1018
    assert p['final_MPC_commit']==3 and p['prefix_native_steps']+p['actual_native_step_max']==18190
    assert b['batch_fd_step']==(12240,150552000)
    assert b['batch_line_step']==(1114140,20933100)
    assert b['native_private_step']==(200180,200180)
    assert b['BFM_backward']==(6670,53360)


def test_attempt_return_failure_and_refusal():
    ledger=WorkLedger({'x':(2,5)});sentinel=object()
    assert ledger.invoke('x',2,lambda:sentinel) is sentinel
    def fail():raise RuntimeError('injected')
    with pytest.raises(RuntimeError):ledger.invoke('x',3,fail)
    with pytest.raises(BudgetExceeded):ledger.invoke('x',1,lambda:None)
    assert ledger.counts['x']==dict(attempted_calls=2,returned_calls=1,attempted_units=5,returned_units=2)
    assert ledger.first_failure['message']=='injected' and ledger.active==[]
    assert ledger.refused['kind']=='budget_refusal'
    snapshot=ledger.snapshot();snapshot['counts']['x']['attempted_calls']=0
    assert ledger.counts['x']['attempted_calls']==2


def test_nested_failure_inner_evidence():
    ledger=WorkLedger({'outer':(1,1),'inner':(1,1)})
    def fail():raise ValueError('inner')
    with pytest.raises(ValueError):ledger.invoke('outer',1,ledger.invoke,'inner',1,fail)
    assert [v['category'] for v in ledger.first_failure['stack']]==['outer','inner']
    assert ledger.counts['outer']['returned_calls']==0


def fake_modules():
    actual_calls=[]
    class Batch:
        def __init__(self,model,n,**kwargs):self.n=n;self.field=object();actual_calls.append(('construct',n,kwargs))
        def bind(self,key):return self.field
        def step(self,*args,**kwargs):actual_calls.append(('batch_step',args,kwargs));return self.field
        def forward(self,*args,**kwargs):actual_calls.append(('forward',args,kwargs));return self.field
    class Planner:
        def __init__(self,feasible):self.feasibility=feasible
        def rollout(self,*args,**kwargs):return ('unchanged_rollout',args,kwargs)
        def linearize(self,*args,**kwargs):return ('unchanged_derivative',args,kwargs)
    def ilqr(p,*args,**kwargs):return p.rollout(*args,**kwargs)
    core=SimpleNamespace(Batch=Batch,Planner=Planner,ilqr=ilqr)
    native=SimpleNamespace(mj_step=lambda m,d:actual_calls.append(('native',d)))
    inspect=lambda model,data,*a,**k:native.mj_step(model,data)
    restoration=SimpleNamespace(ilqr=ilqr,inspect_native_segment=inspect)
    driver=SimpleNamespace(ilqr=ilqr,inspect_native_segment=inspect,preview_native_control=inspect)
    return core,restoration,driver,native,actual_calls


def test_hooks_preserve_calls_live_private_restoration_and_close():
    core,restoration,driver,native,calls=fake_modules();oldstep=native.mj_step;oldbatch=core.Batch
    ledger=WorkLedger();hooks=Hooks(ledger,native,core,restoration,driver)
    live=object();private=object();hooks.register_live(live)
    native.mj_step(None,live);driver.inspect_native_segment(None,private)
    fd=core.Batch(None,2460,num_threads=2);line=core.Batch(None,9,forward=False)
    assert fd.bind('qpos') is fd._actual.field
    assert fd.step(np.arange(4),nstep=9) is fd._actual.field
    line.step(nstep=10);line.forward(np.arange(3));fd.forward(np.arange(59))
    assert driver.ilqr(core.Planner(object()),'state')[0]=='unchanged_rollout'
    assert restoration.ilqr(core.Planner(None),'state')[0]=='unchanged_rollout'
    core.Planner(None).linearize('x')
    assert ledger.counts['native_live_step']['returned_units']==1
    assert ledger.counts['native_private_step']['returned_units']==1
    assert ledger.counts['native_initial_certificate_step']['returned_units']==1
    assert ledger.counts['batch_fd_step']['returned_units']==36
    assert ledger.counts['batch_line_step']['returned_units']==90
    assert ledger.counts['batch_explicit_forward']['returned_units']==62
    assert ledger.counts['restoration_rollout']['returned_calls']==1
    assert ledger.counts['ordinary_rollout']['returned_calls']==1
    assert ledger.counts['linearize']['returned_calls']==1
    hooks.close();assert native.mj_step is oldstep and core.Batch is oldbatch


def test_private_subcontexts_and_refusal_before_native_attempt():
    core,restoration,driver,native,calls=fake_modules();ledger=WorkLedger()
    hooks=Hooks(ledger,native,core,restoration,driver);hooks.register_live(object());private=object()
    with pytest.raises(ValueError):native.mj_step(None,private)
    driver.inspect_native_segment(None,private)
    restoration.inspect_native_segment(None,private)
    driver.preview_native_control(None,private)
    class Session:
        def run(self,*args,**kwargs):return None
    seed=SimpleNamespace(sessions={'actor':Session(),'backward':Session()},propose=lambda:native.mj_step(None,private))
    hooks.instrument_seed(seed);seed.propose()
    assert ledger.counts['native_private_step']['returned_calls']==4
    for category in ('initial_certificate','restoration_certificate','preview','BFM'):
        assert ledger.counts['native_'+category+'_step']['returned_calls']==1
    ledger.limits['native_preview_step']=(1,1)
    with pytest.raises(BudgetExceeded):driver.preview_native_control(None,private)
    assert ledger.counts['native_private_step']['attempted_calls']==4
    assert ledger.counts['imminent_native_preview']['attempted_calls']==2
    assert ledger.counts['imminent_native_preview']['returned_calls']==1
    assert len([v for v in calls if v[0]=='native'])==4
    hooks.close()


def test_seed_sessions_count_rows_without_extra_calls():
    ledger=WorkLedger();calls=[];sentinel=object()
    class Session:
        def run(self,names,inputs):calls.append(inputs);return sentinel
    actor=SessionProxy(Session(),'BFM_actor',ledger);back=SessionProxy(Session(),'BFM_backward',ledger)
    assert actor.run(None,{'state':np.zeros((1,52))}) is sentinel
    assert back.run(None,{'state':np.zeros((8,52))}) is sentinel
    assert len(calls)==2 and ledger.counts['BFM_backward']['returned_units']==8
    with pytest.raises(ValueError):back.run(None,{'state':np.zeros((9,52))})
    assert len(calls)==2


def setup_fixture(tmp_path):
    trace,c=synthetic_trace();snap,prefix=extract(trace,c)
    inputs=tmp_path/'inputs';inputs.mkdir()
    (inputs/'selection_receipt.json').write_text(json.dumps(dict(passed=True,source_trace_sha256=SOURCE_TRACE_SHA,
        selected_control=251,controls_copied=251,physics_steps_copied=2510,learned_prefix_controls=1,
        outputs={'precontrol251.npz':'synthetic','actual_prefix251.npz':'synthetic'})))
    (tmp_path/'recorded_source_audit_v2.json').write_text('{}')
    events=[]
    class Data:
        def __init__(self,native):
            self.vector=np.zeros(291);self.qpos=self.vector[1:31];self.qvel=self.vector[31:60]
            self.warning=SimpleNamespace(number=np.zeros(8,np.int32),lastinfo=np.zeros(8,np.int32))
        @property
        def time(self):return self.vector[0]
    def setstate(model,data,value,spec):events.append('set');data.vector[:]=value
    def forward(model,data):events.append('forward');data.vector[60:]=-99
    class Seed:
        def __init__(self,*args,**kwargs):self.history=BFMHistory();self.previous_action=np.zeros(23,np.float32);self.recorded_controls=0;self.pushes=0
        def _terms(self,q,v,p):self.pushes+=1;return None,terms(q,v,p)
        def identity(self):return {'fake':True}
    class NP:
        __version__='1.26.4'
        def __getattr__(self,key):return getattr(np,key)
    def archive(path):
        return {'actual_prefix251.npz':prefix,'precontrol251.npz':snap,'original29.npz':{},'trace.npz':dict(target=np.zeros((1569,23)))}[path.name]
    env=dict(np=NP(),json=json,BASE=tmp_path,START=251,SWITCH=1269,LIFECYCLE=1569,EXTENSION=250,
        SOURCE_TRACE_SHA=SOURCE_TRACE_SHA,protocol=protocol,KEYS=tuple(named(np.zeros(300,np.float32))),
        BUNDLE=tmp_path,REFERENCE=tmp_path,TEACHER=tmp_path,ONNX=tmp_path,DEPS=tmp_path,RECORDED=tmp_path,
        frozen=lambda:None,archive=archive,sha256=lambda p:'synthetic',
        load_native_bundle=lambda *a:(object(),c,{},dict(phases=[dict(control_start=0,control_stop=250)]),{}),
        load_motion_override=lambda *a:({},{}),position_servo_copy=lambda *a:None,
        Native23Tracker=lambda *a,**k:SimpleNamespace(lo=np.full(23,-10.),hi=np.full(23,10.)),
        Native23BFMRolloutSeed=Seed,mujoco=SimpleNamespace(__version__='3.2.3',MjData=Data,mj_setState=setstate,mj_forward=forward),
        HOOKS=SimpleNamespace(register_live=lambda d:events.append('register'),instrument_seed=lambda s:events.append('instrument')),
        get_state=lambda model,data:data.vector.copy(),assess=lambda *a:([],{}),
        flat=lambda s:np.concatenate([s.history.data[k].reshape(-1) for k in sorted(s.history.data)]))
    functions(SOURCE/'run_width251_actual_oracle.py',['setup','assert_initial_actual_unchanged'],env)
    return env,snap,prefix,events


def test_injected_setup_full291_history_and_no_inference(tmp_path):
    env,snap,prefix,events=setup_fixture(tmp_path);s=env['setup']()
    env['assert_initial_actual_unchanged'](s)
    assert events==['register','instrument','set','forward','set']
    assert s['fresh'].pushes==251 and s['fresh'].recorded_controls==251
    exact(s['data'].vector,snap['integration'],'reapplied hidden integration')
    assert s['request']['MPC_controls']==1018 and s['request']['BFM_prefix_controls']==250
    assert s['request']['learned_prefix_controls']==1


def test_injected_setup_rejects_shifted_history(tmp_path):
    env,snap,prefix,events=setup_fixture(tmp_path)
    prefix['control_history_before'][250,0]+=1
    with pytest.raises(AssertionError):env['setup']()


def test_actual_control_history_updates_once_only_after_preview():
    events=[];seed=SimpleNamespace(previous_action=np.zeros(23,np.float32),history=BFMHistory())
    data=SimpleNamespace(qpos=np.zeros(30),qvel=np.zeros(29))
    trace={k:[] for k in CONTROL};trace.update(qpos=[data.qpos.copy()],qvel=[data.qvel.copy()],physics_torque=[],
        physics_time=[0.],physics_expected_time=[0.],physics_qpos=[data.qpos.copy()],physics_qvel=[data.qvel.copy()],
        range_excess=[],velocity_ratio=[],effort_ratio=[],clock_error=[])
    for k in seed.history.data:trace['control_history_'+k]=[]
    def record(control,q,v,target):events.append('history');seed.previous_action=target.astype(np.float32)
    seed.record_control=record
    feasible=[False]
    def preview(*args):events.append('preview');return {'feasible':feasible[0]},[None]*11,[None]*10
    def step(native,d,target,c,f,t,expected,pred,force):
        events.append('step');t['physics_torque'].append(np.zeros(23))
        for k in ('physics_time','physics_expected_time'):t[k].append(expected+.002)
        t['physics_qpos'].append(d.qpos.copy());t['physics_qvel'].append(d.qvel.copy())
        return expected+.002,None
    env=dict(np=np,preview_native_control=preview,get_state=lambda *a:np.zeros(291),
        flat=lambda s:np.zeros(300,np.float32),KEYS=tuple(seed.history.data),record_native_substep=step,
        assess=lambda *a:([],dict(range_excess=0,velocity_ratio=0,effort_ratio=0,clock_error=0)),
        mujoco=SimpleNamespace(mj_kinematics=lambda *a:events.append('kinematics')))
    functions(SOURCE/'run_actual_student_oracle.py',['apply_control'],env)
    s=dict(native=None,c={},fresh=seed,data=data,planner=SimpleNamespace(feasibility=object()),expected=0.,
        motion=dict(joint_pos=np.zeros((2000,23)),body_pos_w=np.zeros((2000,1,3))))
    target=np.ones(23)
    failure=env['apply_control'](s,trace,251,target)
    assert failure['kind']=='imminent_control_infeasible' and events==['preview']
    feasible[0]=True;events.clear();assert env['apply_control'](s,trace,251,target) is None
    assert events==['preview','history']+['step']*10+['kinematics']
    assert trace['global_control']==[251] and trace['source_frame']==[262]


def test_prefix_mode_not_counted_as_expert():
    controls=np.arange(254);modes=np.r_[np.zeros(250,int),np.ones(4,int)]
    assert np.sum((modes==1)&(controls>=START))==3
    tree=ast.parse((SOURCE/'run_width251_actual_oracle.py').read_text())
    run=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='run_branch')
    selected=next(v.value for v in ast.walk(run) if isinstance(v,ast.keyword) and v.arg=='actual_expert_controls')
    value=eval(compile(ast.Expression(selected),'<actual producer expression>','eval'),
               dict(np=np,START=START,trace=dict(controller_mode=modes,global_control=controls)))
    assert value==3


@pytest.mark.parametrize('change',['none','unlisted_role','wrong_selected_path','wrong_review_request','not_selected'])
def test_actual_admission_subject_membership(tmp_path,monkeypatch,change):
    import recovery_admission as admission
    subjects={};inputs={}
    for key,relative in [('trace','trace.npz'),('semantics','semantics.json'),('semantics_owner','owner.json'),
                         ('selected_snapshot','inputs/precontrol251.npz'),('selected_prefix','inputs/actual_prefix251.npz'),
                         ('input_selection','inputs/selection_receipt.json')]:
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(key)
        subjects[key]={'path':str(p),'sha256':admission.sha(p)};inputs[str(p)]=admission.sha(p)
    for key,symbol in [('trace','SOURCE_TRACE_SHA'),('semantics','SEMANTICS_SHA'),('semantics_owner','SEMANTICS_OWNER_SHA')]:
        monkeypatch.setattr(admission,symbol,subjects[key]['sha256'])
    if change=='unlisted_role':inputs.pop(subjects['selected_prefix']['path'])
    if change=='wrong_selected_path':subjects['selected_snapshot']=subjects['trace'].copy()
    code=tmp_path/'source_snapshot_v1/probe.py';code.parent.mkdir();code.write_text('# synthetic')
    frozen=tmp_path/'frozen_inputs.json';frozen.write_text(json.dumps({'source_sha256':{'probe.py':admission.sha(code)},'input_sha256':inputs}))
    request=tmp_path/'execution_request.json'
    request.write_text(json.dumps(dict(root_selected=change!='not_selected',protocol=protocol(),subjects=subjects)))
    review=tmp_path/'review.json';review.write_text(json.dumps(dict(passed=True,
        request_sha256='bad' if change=='wrong_review_request' else admission.sha(request),frozen_receipt_sha256=admission.sha(frozen))))
    clearance=dict(approved=True,request_sha256=admission.sha(request),frozen_receipt_sha256=admission.sha(frozen),
        review=dict(path=str(review),sha256=admission.sha(review),pass_field='passed'))
    (tmp_path/'execution_clearance.json').write_text(json.dumps(clearance))
    if change=='none':assert admission.admit(tmp_path)['root_selected'] is True
    else:
        with pytest.raises(ValueError):admission.admit(tmp_path)
