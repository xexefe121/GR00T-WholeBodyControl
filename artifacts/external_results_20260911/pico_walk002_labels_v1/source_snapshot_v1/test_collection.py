"""Focused source, observation convention, and partial-failure checks; no physics."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import collection_checks as checks
from collect_qualified_rows import CollectionState, save_rows, preserved_failure
import student_linear_runtime as unchanged


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_exact_rejects_signed_zero_change(dtype):
    with pytest.raises(ValueError, match='Byte mismatch'):
        checks.exact(np.array([-0.], dtype), np.array([0.], dtype), 'signed zero')


def test_exact_rejects_numeric_equal_dtype_change():
    with pytest.raises(ValueError):
        checks.exact(np.zeros(2, np.float32), np.zeros(2, np.float64), 'dtype')


def test_actual_MPC_prior_retains_values_outside_actor_range():
    c = dict(default_q=np.zeros(23), kp=np.ones(23), training_effort=np.ones(23))
    actual = checks.action_from_target(np.full(23, 2.), c)
    checks.exact(actual, np.full(23, 8., np.float32), 'unclipped prior')


@pytest.mark.parametrize('clip,total,stop,source', [('pico',6530,6230,5780), ('walk002',1417,1117,667)])
def test_actual_timeline_counts_and_boundaries(clip, total, stop, source):
    root = checks.local('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
    t = checks.read(root / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1' / clip / 'timeline.json')
    p, n, end = checks.moving_phases(t, clip)
    assert (n, end) == (total, stop)
    assert [sum(checks.phase_code(k,p)==i for k in range(250,stop)) for i in range(3)] == [100,source,100]
    for k in (249, stop, total):
        with pytest.raises(ValueError):
            checks.phase_code(k, p)
    changed = copy.deepcopy(t)
    changed['phases'][1]['control_start'] += 1
    with pytest.raises(AssertionError):
        checks.moving_phases(changed, clip)


def fixture_rows():
    dims = dict(actions=23, base_ang_vel=3, dof_pos=23, dof_vel=23, projected_gravity=3)
    result = []
    for control in (249, 250, 251, 252):
        result.append(dict(control=control, qpos=np.zeros(30), qvel=np.zeros(29), state=np.zeros(52,np.float32),
            previous_action=np.full(23,8.,np.float32), history=np.zeros(300,np.float32),
            named_history={k:np.zeros((4,n),np.float32) for k,n in dims.items()}))
    return result


def fixture_phases():
    return dict(acquisition_ramp=dict(control_start=250,control_stop=252),
                source_motion=dict(control_start=252,control_stop=300),
                return_ramp=dict(control_start=300,control_stop=310))


def test_only_selected_controls_inferred_with_exact_first_frame():
    state = CollectionState()
    calls = []
    def infer(row, frame):
        calls.append((row['control'],frame,row['previous_action'].copy()))
        return np.full(23,40.,np.float32), np.full(23,10.), row['state']
    def features(q,dq,frame,base,prior):
        assert np.all(base==10.) and np.all(prior==8.)
        return np.zeros(1069,np.float32)
    trace = {'target':np.full((253,23),.3)}
    result = state.collect(iter(fixture_rows()),trace,fixture_phases(),250,252,infer,features,lambda *a:None)
    assert [(c,f) for c,f,p in calls]==[(250,261),(251,262)]
    checks.exact(result['expert_target'],trace['target'][250:252],'actual targets')
    checks.exact(result['base_target'],np.full((2,23),10.),'unclipped baseline')
    checks.exact(result['residual_rad'],trace['target'][250:252]-10.,'no residual clipping')
    assert state.current['control']==252


@pytest.mark.parametrize('fail_at', [250,251])
def test_inference_failure_retains_only_completed_rows(tmp_path, fail_at):
    state = CollectionState()
    def infer(row,frame):
        if row['control']==fail_at:
            raise RuntimeError('stub inference failure')
        return np.ones(23,np.float32),np.ones(23),row['state']
    with pytest.raises(RuntimeError,match='stub inference failure'):
        state.collect(iter(fixture_rows()),{'target':np.zeros((253,23))},fixture_phases(),250,252,
            infer,lambda *a:np.zeros(1069,np.float32),lambda *a:None)
    assert len(state.rows['control'])==fail_at-250
    assert state.inference_attempts==fail_at-249
    assert state.current['control']==fail_at
    trace=dict(target=np.zeros((253,23)),qpos=np.zeros((254,30)),qvel=np.zeros((254,29)))
    snapshots=dict(control_integration_before=np.zeros((253,291)),integration_spec=np.asarray(8191),
        time=np.zeros(253),warning_counts=np.zeros((253,8),np.int32),warning_lastinfo=np.zeros((253,8),np.int32),
        original_trace_sha256=np.asarray('stub trace'))
    c=dict(joint_limits=np.tile([-2.,2.],(23,1)))
    save_rows(tmp_path/'partial.npz',state,trace,snapshots,c,False)
    preserved_failure(tmp_path/'context.npz',state,snapshots,'actual_state_baseline_collection',252)
    with np.load(tmp_path/'context.npz',allow_pickle=False) as data:
        checks.exact(data['control_integration_before'],snapshots['control_integration_before'][fail_at],'failure full291')
        assert data['selected_moving_control'].item() is True
        assert data['failure_phase'].item()=='actual_state_baseline_collection'
    with np.load(tmp_path/'partial.npz',allow_pickle=False) as data:
        n=fail_at-250
        assert data['features'].shape==(n,1069)
        assert data['state'].shape==(n,52)
        assert data['control_integration_before'].shape==(n,291)
        assert data['history_actions'].shape==(n,4,23)
        assert data['teacher_qpos'].shape==(n+1,30)
        assert data['complete'].item() is False


def test_baseline_helper_preserves_raw_actor_times_five_and_original_goal():
    seen={}
    raw=np.arange(23,dtype=np.float32)/2
    def run(none,inputs):
        seen.update(inputs)
        return [raw[None]]
    c=dict(default_q=np.zeros(23),kp=np.ones(23),training_effort=np.ones(23))
    seed=SimpleNamespace(contract=c,sessions={'actor':SimpleNamespace(run=run)},
        _terms=lambda q,dq,p:(np.arange(52,dtype=np.float32),None),
        _goal=lambda frame,q:np.array([[frame]],np.float32))
    prior=np.full(23,8.,np.float32); history=np.arange(300,dtype=np.float32)
    action,target,sensed=unchanged.infer_base(seed,np.zeros(30),np.zeros(29),prior,history,261)
    checks.exact(action,raw*5,'raw arithmetic')
    checks.exact(target,c['default_q']+raw*5*.25,'unclipped target arithmetic')
    checks.exact(seen['last_action'],prior[None],'actor prior')
    checks.exact(seen['history'],history[None],'actor history')
    assert seen['z'].item()==261 and target.max()>2


def test_linear_features_preserves_base_and_raw_prior(monkeypatch):
    seen={}
    class Goals:
        def __init__(self,*args):pass
        def __call__(self,q,dq,previous,frame):
            seen['previous_target']=previous.copy();seen['frame']=frame
            return np.zeros(1023,np.float32)
    monkeypatch.setattr(unchanged,'GoalFeatures',Goals)
    c=dict(default_q=np.zeros(23),joint_limits=np.tile([-1.,1.],(23,1)),kp=np.ones(23),training_effort=np.ones(23))
    builder=unchanged.LinearFeatures({}, {}, c)
    x=builder(np.zeros(30),np.zeros(29),261,np.full(23,10.),np.full(23,8.,np.float32))
    checks.exact(x[-46:-23],np.full(23,10.,np.float32),'unclipped base input')
    checks.exact(x[-23:],np.full(23,8.,np.float32),'raw prior input')
    assert np.all(seen['previous_target']==1.) and seen['frame']==261


def test_collector_no_physics_or_planner_calls():
    tree=ast.parse(Path(__file__).with_name('collect_qualified_rows.py').read_text())
    forbidden={'mj_step','mj_step1','mj_step2','propose','solve','fit','backward','mj_setState'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in forbidden


@pytest.mark.parametrize('completed,rows', [(False,6847),(True,6846)])
def test_compatibility_rejects_incomplete_producer_before_consuming_labels(tmp_path, completed, rows):
    from audit_collection_compatibility import run
    checks.write(tmp_path/'rows_complete.json',dict(complete=completed,completed_rows=rows,manifest_sha256='test'))
    with pytest.raises(AssertionError):
        run({},tmp_path,'test')
    assert not (tmp_path/'compatibility').exists()


@pytest.mark.parametrize('name', ['student_linear_runtime.py','terminal_yaw4_goal.py',
    'gear_sonic/utils/g1_true23_bfm_seed_observations.py','gear_sonic/utils/g1_true23_mjbatch_bfm_seed.py',
    'gear_sonic/utils/g1_true23_mpc_student.py','gear_sonic/utils/g1_true23_mjbatch_mpc.py'])
def test_helpers_byte_identical_to_original_collector(name):
    here=Path(__file__).parent
    original=here.parent.parent/'bfm_entry250_labels_v1/source_snapshot_v1'
    assert (here/name).read_bytes()==(original/name).read_bytes()
