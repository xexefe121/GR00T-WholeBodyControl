"""Synthetic accounting/launcher/finally checks. No native or model construction."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source_draft_v1'))
import runner
import prepare_stage as stage
from stage_verdict import validate_report

def report(name):
    n,s=(0,2) if name=='witness' else (21348,8)
    r=dict(passed=True,error=None,model_inference_calls=0,optimizer_updates=0,other_oracle_native_steps=0,
        plant_foundation_connected=False,real_wallclock_experiment=False,
        api_counters=dict(step_attempted=n,step_returned=n,serialization_attempted=s,serialization_returned=s,
            denied_step_calls=0,denied_serialization_calls=0,serialization_records=[dict(native_returned=True) for _ in range(s)]))
    for key,count,valid in (('expert',18190,18190),('direct',3158,3157)):
        c={k:count for k in ('attempted','returned','capture_attempts','captured','verification_attempts')};c['verified']=valid
        r[key]=dict(equivalence_passed=True,error=None,adapter_counters=c)
    r.update(original_direct_physics_pass=False,exact_expected_failure_reproduction=True)
    return r

@pytest.mark.parametrize('name',['witness','replay'])
def test_fixed_success_counters(name):assert validate_report(name,report(name))

@pytest.mark.parametrize('key',['step_attempted','step_returned','serialization_attempted','serialization_returned','denied_step_calls','denied_serialization_calls'])
def test_wrong_counts_fail(key):
    r=report('replay');r['api_counters'][key]+=1
    with pytest.raises(AssertionError):validate_report('replay',r)

@pytest.mark.parametrize('key',['error','capture_error'])
def test_failed_serialization_cannot_pass(key):
    r=report('witness');r['api_counters']['serialization_records'][0][key]='fault'
    with pytest.raises(AssertionError):validate_report('witness',r)

def test_expected_direct_failure_is_equivalence_only():
    r=report('replay');r['direct']['adapter_counters']['verified']=3158
    with pytest.raises(AssertionError):validate_report('replay',r)

@pytest.mark.parametrize('name',['witness','replay'])
def test_launcher_fixed_entry_command_no_native_invocation(name):
    args=stage.arguments(name);assert args[args.index('--cd')+1]=='/'
    assert args[args.index('--')+1]=='bash' and args[-4]=='--request' and args[-2]=='--clearance'
    assert args[-3].endswith(name+'_request.json')
    source=stage.durable_text(name)
    assert '-WindowStyle Hidden' in source and 'FileMode]::CreateNew' in source
    assert source.index('$nativeHandle = $taskChild.Handle')<source.index('$taskChild.WaitForExit()')
    assert 'request_subject' in source and 'launch_receipt_subject' in source
    assert "'prerun_hashes.json'" in source and "'postrun_hashes.json'" in source
    assert 'automatic_retry=$false' in source

def test_constructor_failure_always_attempts_exit_pair(monkeypatch,tmp_path):
    calls=[]
    class Broken:
        def __init__(self,*args):raise RuntimeError('scripted partial constructor')
    monkeypatch.setitem(sys.modules,'native_stepper',SimpleNamespace(NativeStepper=Broken))
    monkeypatch.setattr(runner,'mjb_pair',lambda *args:(calls.append('exit') or b'M',dict(passed=True)))
    api=SimpleNamespace(counters=lambda:dict(step_attempted=0),MjData=lambda model:object())
    r=runner.run_case(object(),{},api,b'M',[],{},tmp_path/'out')
    assert not r['equivalence_passed'] and calls==['exit']
    assert (tmp_path/'out/partial_captured_trace.npz').is_file()
    assert r['exit_model_identity']['passed'] is True

def test_initialized_replay_error_still_verifies_exit(monkeypatch,tmp_path):
    calls=[]
    class Broken:
        def __init__(self,*args):self.identity=object();self.failure=None
        def restore_initial(self,*args):raise RuntimeError('scripted restore failure')
        def verify_model_exit(self):calls.append('exit');return dict(passed=True)
        def counters(self):return dict(attempted=0,returned=0)
    monkeypatch.setitem(sys.modules,'native_stepper',SimpleNamespace(NativeStepper=Broken))
    monkeypatch.setattr(runner,'mjb_pair',lambda *args:pytest.fail('no extra fallback serialization'))
    api=SimpleNamespace(counters=lambda:dict(step_attempted=0),MjData=lambda model:object())
    first={'initial_integration':np.zeros(291,np.float64),'physics_warning_counts':np.zeros((1,8),np.int32),
           'physics_warning_lastinfo':np.zeros((1,8),np.int32)}
    fixture=dict(state_vector=np.zeros(291,np.float64),state_spec=np.asarray(8191,np.int64))
    r=runner.run_case(object(),{},api,b'M',[SimpleNamespace(arrays=first)],fixture,tmp_path/'out')
    assert not r['equivalence_passed'] and calls==['exit']
    assert (tmp_path/'out/partial_captured_trace.npz').is_file()
