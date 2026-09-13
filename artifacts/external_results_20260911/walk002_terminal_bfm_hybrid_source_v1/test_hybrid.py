"""Pure-array and stubbed-controller tests; no MuJoCo stepping or ONNX inference."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

import hybrid_checks as c
import runner as r

def valid():
    q=np.zeros(30); q[2]=.8; q[3]=1.
    return dict(q=q,dq=np.zeros(29),force=np.zeros(23),warning_number=np.zeros(8,np.int32),
        warning_lastinfo=np.zeros(8,np.int32),time=0.,expected_time=0.,
        limits=np.tile([-1.,1.],(23,1)),speed=np.ones(23)*20,effort=np.ones(23)*100)

def test_valid_native(): assert c.first_issue(**valid()) is None

@pytest.mark.parametrize('field,index,value,reason',[
    ('q',7,1.00000101,'native_joint_range'),('dq',6,20.00001,'native_joint_speed'),
    ('force',0,100.00001,'native_actuator_effort'),('force',0,np.nan,'nonfinite_native_state_or_force'),
    ('q',0,np.nan,'nonfinite_native_state_or_force'),('dq',0,np.inf,'nonfinite_native_state_or_force'),
    ('q',2,.249,'fall'),('q',3,.99,'quaternion_norm'),('warning_number',2,1,'engine_warning'),
    ('warning_lastinfo',2,1,'engine_warning')])
def test_first_native_failure(field,index,value,reason):
    kw=valid();kw[field][index]=value
    assert c.first_issue(**kw)==reason

def test_clock_never_tolerance_or_resync():
    kw=valid();kw['time']=np.nextafter(0.,1.)
    assert c.first_issue(**kw)=='independent_repeated_clock_mismatch'

def test_range_inside_strict_tolerance():
    kw=valid();kw['q'][7]=1.00000099
    assert c.first_issue(**kw) is None

@pytest.mark.parametrize('bad',[np.array([-0.]),np.array([0.],np.float32),np.array([[0.]]),np.array([np.nextafter(0.,1.)])])
def test_bitexact_rejects_sign_dtype_shape_ulp(bad):
    with pytest.raises(ValueError):c.exact(bad,np.array([0.]),'sentinel')

def empty_trace():
    t={key:[] for key in c.SHAPES}
    for key in ('qpos','qvel','physics_qpos','physics_qvel','physics_time','physics_expected_time','physics_warning_number','physics_warning_lastinfo'):
        t[key].append(np.zeros(c.SHAPES[key],np.int32 if key in c.WARNINGS else np.float64))
    return t

def test_empty_trace_schema():
    t=empty_trace();a=c.arrays(t)
    for key,shape in c.SHAPES.items(): assert a[key].shape==(len(t[key]),)+shape
    assert a['physics_warning_number'].dtype==np.int32 and a['history'].dtype==np.float32

@pytest.mark.parametrize('key',['physics_qpos','physics_time','physics_warning_number','qpos','target','physics_substeps'])
def test_invalid_trace_lengths_rejected(key):
    t=empty_trace();t[key].append(np.zeros(c.SHAPES[key]))
    with pytest.raises(ValueError):c.arrays(t)

def test_quaternion_qualified_strict_threshold():
    kw=valid();kw['q'][3]=1.+2e-10
    assert c.first_issue(**kw)=='quaternion_norm'

@pytest.mark.parametrize('key',['physics_qpos','physics_qvel','physics_requested_torque','physics_torque','physics_actuator_force','physics_time','physics_expected_time','physics_warning_number','physics_warning_lastinfo'])
def test_prefix_compares_every_field(key):
    shape=c.SHAPES[key]
    dtype=np.int32 if key in c.WARNINGS else np.float64
    source={key:np.zeros((2,)+shape,dtype=dtype)}
    value=np.zeros(shape,dtype=dtype)
    c.prefix_sample(source,{key:value},1)
    value=value+np.asarray(1,dtype=dtype)
    with pytest.raises(ValueError):c.prefix_sample(source,{key:value},1)

def test_endpoint_all291_and_auxiliary():
    e=dict(completed_controls=np.asarray(1117),integration_state_spec=np.asarray(8191),final_integration=np.zeros(291),
        qpos=np.zeros(30),qvel=np.zeros(29),qacc_warmstart=np.zeros(29),ctrl=np.zeros(23),time=np.asarray(0.),
        warning_counts=np.zeros(8,np.int32),warning_lastinfo=np.zeros(8,np.int32))
    s={k:v.copy() for k,v in e.items() if k not in ('completed_controls','integration_state_spec','final_integration','warning_counts')}
    s.update(integration=e['final_integration'].copy(),warning_number=e['warning_counts'].copy())
    r.verify_endpoint(e,s,1117)
    for key in s:
        bad={k:v.copy() for k,v in s.items()};bad[key].flat[-1]=1
        with pytest.raises(ValueError):r.verify_endpoint(e,bad,1117)

def test_active_terminal_state_not_stale_seed():
    obj=object.__new__(r.Hybrid)
    obj.seed=SimpleNamespace(history='stale',previous_action='stale',recorded_controls=1117)
    obj.controller_history='active';obj.controller_previous='raw';obj.controller_count=1417
    assert obj.active()==('active','raw',1417)

def test_history_snapshot_copies_active_values():
    h=SimpleNamespace(data={'actions':np.ones((4,23),np.float32)})
    p=np.ones(23,np.float32)*6
    snap=c.history_snapshot(h,p,1417)
    h.data['actions'][0,0]=0;p[0]=0
    assert snap['history_actions'][0,0]==1 and snap['previous_action'][0]==6 and int(snap['recorded_controls'])==1417

def test_unchanged_terminal_math_source():
    old=Path(__file__).parent.parent.parent/'sonic23_teleop_six_hour_20260910/bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1/runner_snapshot.py'
    if not old.exists():old=Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910/bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1/runner_snapshot.py')
    oldtree=ast.parse(old.read_text())
    newtree=ast.parse(Path(r.__file__).read_text().replace('self.contract','contract').replace('self.kp','kp').replace('self.limits','limits'))
    for name in ('action','target'):
        def expressions(tree):
            return [ast.dump(n.value,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        wanted=[x for x in expressions(oldtree) if ('Mult' in x if name=='action' else "attr='clip'" in x)]
        assert len(wanted)==1 and wanted[0] in expressions(newtree)

def test_no_state_setter_or_private_solver():
    source=Path(r.__file__).read_text()
    for forbidden in ('mj_setState(','.propose(','.ilqr(','.window('):assert forbidden not in source
    assert source.count('mujoco.mj_forward(')==1
    assert source.index('verify_endpoint(self.endpoint')<source.index('goal = self.seed._goal(')
    assert source.index('prefix_sample(self.source')<source.index('self.prefix_steps += 1')

def test_quiet_does_not_pass_short_window():
    assert not c.quiet_metrics({'physics_torque':np.zeros((1499,23))},{})['pass_all']

def test_main_hold_guard_retains_counts():
    source=Path(r.__file__).read_text()
    assert "if main_report['strict_physical_limits_pass'] and main_report['completed_controls'] == COUNT:" in source
    assert 'engine.segment(COUNT, COUNT+EXTENSION' in source
    assert 'original_main_quiet_failure_preserved=True' in source

@pytest.mark.parametrize('fault',['actor_exception','actor_nonfinite','prefix_mismatch','endpoint_mismatch'])
def test_failed_switch_does_not_advance_active_history_or_step(fault,monkeypatch,tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).parent/'repo'))
    from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory
    obj=object.__new__(r.Hybrid)
    obj.args=SimpleNamespace(producer=tmp_path,boundary_endpoint=tmp_path/'endpoint.npz')
    (tmp_path/'trace.npz').write_bytes(b'identity-only')
    (tmp_path/'endpoint.npz').write_bytes(b'identity-only')
    calls=[]
    def actor(*args,**kwargs):
        calls.append('actor')
        if fault=='actor_exception':raise RuntimeError('actor sentinel')
        return [np.full((1,23),np.nan,np.float32)]
    def goal(*args):calls.append('goal');return np.zeros((1,256),np.float32)
    obj.seed=SimpleNamespace(history=BFMHistory(),previous_action=np.zeros(23,np.float32),recorded_controls=1117,
        _terms=lambda *args:(np.zeros(52,np.float32),{k:np.ones(v.shape[1],np.float32) for k,v in obj.seed.history.data.items()}),
        _goal=goal,sessions={'actor':SimpleNamespace(run=actor)})
    obj.controller_history=obj.controller_previous=obj.controller_count=None
    obj.history_type=BFMHistory
    obj.native=object()
    obj.data=SimpleNamespace(qpos=np.zeros(30),qvel=np.zeros(29),time=0.,warning=SimpleNamespace(number=np.zeros(8,np.int32),lastinfo=np.zeros(8,np.int32)))
    obj.expected_time=0.;obj.prefix_steps=11170;obj.history_checks=1117;obj.switch_verified=False;obj.initial_issue=None
    obj.endpoint={'time':np.asarray(0.)}
    obj.motion={'joint_pos':np.zeros((1428,23))}
    obj.source={'qpos':np.zeros((1118,30)),'qvel':np.zeros((1118,29)),
        'fresh_seed_previous_action':np.zeros((1118,23),np.float32),'fresh_seed_measured_history':np.zeros((1118,300),np.float32)}
    if fault=='prefix_mismatch':obj.source['fresh_seed_measured_history'][-1,0]=1
    obj.snapshot=lambda:dict(integration=np.zeros(291),**c.history_snapshot(*obj.active()))
    obj.save=lambda directory,trace,failure,initial,start,stop:(c.arrays(trace),failure)
    def endpoint(*args):
        if fault=='endpoint_mismatch':raise ValueError('endpoint sentinel')
    monkeypatch.setattr(r,'verify_endpoint',endpoint)
    monkeypatch.setattr(r,'integration',lambda *args:np.zeros(291))
    monkeypatch.setattr(r.mujoco,'mj_step',lambda *args:pytest.fail('physics forbidden in stub test'))
    arr,failure=obj.segment(1117,1118,tmp_path)
    assert failure['kind']=='precontrol_gate_or_inference_failure' and len(arr['target'])==0
    h,p,n=obj.active()
    assert n==1117 and np.count_nonzero(p)==0
    assert all(np.count_nonzero(v)==0 for v in h.data.values())
    assert (tmp_path/'failed_precontrol.npz').exists()
    assert not calls if fault in ('prefix_mismatch','endpoint_mismatch') else calls==['goal','actor']
