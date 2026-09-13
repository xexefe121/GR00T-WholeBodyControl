"""Pure capture checks; no MuJoCo import or native steps."""
import ast
import json
from pathlib import Path
import sys
import numpy as np
import pytest
BASE=Path(__file__).parent;SOURCE=BASE/'source_snapshot_v1';sys.path.insert(0,str(SOURCE))
from capture_checks import exact,original_warning,compare_sample

def test_explicit_original_warning_serialization():
    native=np.zeros(8,np.int32);old=np.zeros(8,np.int64)
    serialized=original_warning(native,old,'warning')
    assert serialized.dtype==np.int64 and native.dtype==np.int32 and serialized is not native

def test_nonzero_warning_mismatch_is_not_hidden_by_cast():
    native=np.zeros(8,np.int32);native[2]=1
    with pytest.raises(ValueError,match='byte mismatch'):original_warning(native,np.zeros(8,np.int64),'warning')

@pytest.mark.parametrize('native,old',[(np.zeros(8,np.int64),np.zeros(8,np.int64)),(np.zeros(8,np.int32),np.zeros(8,np.int32)),(np.zeros(7,np.int32),np.zeros(8,np.int64))])
def test_unexpected_ledger_schema_rejected(native,old):
    with pytest.raises(ValueError,match='serialization'):original_warning(native,old,'warning')

def test_signed_zero_and_dtype_checked():
    with pytest.raises(ValueError):exact(np.asarray([-0.]),np.asarray([0.]),'zero')
    with pytest.raises(ValueError):exact(np.zeros(1,np.float64),np.zeros(1,np.float32),'dtype')

def sample_and_trace():
    sample=dict(qpos=np.zeros(30),qvel=np.zeros(29),ctrl=np.ones(23),force=np.ones(23),
        warning_counts=np.zeros(8,np.int32),warning_lastinfo=np.zeros(8,np.int32),time=.002)
    trace=dict(physics_qpos=np.zeros((2,30)),physics_qvel=np.zeros((2,29)),
        physics_torque=np.ones((1,23)),physics_actuator_torque=np.ones((1,23)),
        physics_warning_counts=np.zeros((2,8),np.int64),physics_warning_lastinfo=np.zeros((2,8),np.int64),
        physics_time=np.asarray([0.,.002]))
    return sample,trace

def test_native_sample_exact_including_independent_clock():
    sample,trace=sample_and_trace();compare_sample(sample,trace,1,.002)
    with pytest.raises(ValueError,match='clock'):compare_sample(sample,trace,1,0.)

@pytest.mark.parametrize('key',['ctrl','force','qpos','qvel'])
def test_first_sample_mismatch_rejected(key):
    sample,trace=sample_and_trace();sample[key][0]=np.nextafter(sample[key][0],np.inf)
    with pytest.raises(ValueError):compare_sample(sample,trace,1,.002)

def test_strict_oracle_ast_unchanged():
    origin=BASE.parent/'velocity_chord_student_evaluation_v1/source_snapshot_v1/evaluate_velocity_chord_student.py'
    nodes=[]
    for path in (origin,SOURCE/'strict_native.py'):
        nodes.append(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='assess'))
    assert ast.dump(nodes[0])==ast.dump(nodes[1])

def test_source_has_one_native_call_and_no_model_inference():
    source=(SOURCE/'capture_old_prefix.py').read_text();tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert sum(n.func.attr=='mj_step' for n in calls)==1
    assert not any(n.func.attr in ('InferenceSession','propose') for n in calls)
    assert "for current_control in range(1268)" in source and 'boundary(1268)' in source
    assert source.index("exact(state_vector,initial")<source.index('mujoco.mj_step')
    assert source.index('samples.append(sample())',source.index('mujoco.mj_step'))<source.index('compare_sample(samples[-1],trace,completed')

def test_saved_initial_fixture_is_exact_and_old_schema_supported():
    old=BASE.parent.parent/'sonic23_teleop_six_hour_20260910/bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1/trace.npz'
    with np.load(old) as trace,np.load(BASE.parent/'walk003_canonical_initial_fixture_v1/initial_integration_state.npz') as fixture:
        state=fixture['state_vector'];assert state.shape==(291,) and fixture['state_spec']==8191
        exact(state[1:31],trace['physics_qpos'][0],'fixture qpos')
        exact(state[31:60],trace['physics_qvel'][0],'fixture qvel')
        assert state[0]==trace['physics_time'][0]==0
        assert trace['physics_warning_counts'].dtype==trace['physics_warning_lastinfo'].dtype==np.int64
        assert not np.any(trace['physics_warning_counts']) and not np.any(trace['physics_warning_lastinfo'])
        assert 'physics_expected_time' not in trace and 'final_integration' not in trace
