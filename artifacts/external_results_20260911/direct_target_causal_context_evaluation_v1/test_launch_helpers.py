"""Synthetic JSON and generated-text checks, zero model/native calls."""
import json
from pathlib import Path
import pytest
from diagnostic_verdict import verdict
from prepare_bound_launcher import quote,wsl_arguments,run_text,durable_text,write_new
from freeze_final_package import role

def put(base,name,value):
    path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

def complete(base):
    segments={}
    for name,count in (('nominal',1569),('extension',250)):
        segments[name]=dict(full_segment_completed=True,completed_controls=count,physics_steps=10*count,failure=None,
            quiet_standing_diagnostic=dict(quiet_standing_diagnostic_pass=True),forbidden_inference_calls=0)
    put(base,'pilot_outcome.json',segments)
    for name in ('canonical_initial_full291_parity','canonical_prefix250_parity','actual_query250_input_parity','actual_query250_ownexport_output_parity'):
        put(base,name+'.json',dict(passed=True))
    return segments

def test_complete(tmp_path):
    complete(tmp_path);assert verdict(tmp_path,'evaluation',0)['passed']

@pytest.mark.parametrize('raw',[None,1,7,False,'0'])
def test_raw_not_success(tmp_path,raw):
    complete(tmp_path);r=verdict(tmp_path,'evaluation',raw)
    assert not r['passed'] and r['raw_python_exit_code']==raw and r['diagnostic_exit_code']==2

@pytest.mark.parametrize('field,value',[
    ('full_segment_completed',False),('completed_controls',261),('physics_steps',2617),
    ('failure',{'reasons':['native_joint_speed']}),('forbidden_inference_calls',1),
    ('quiet_standing_diagnostic',None),('quiet_standing_diagnostic',{'quiet_standing_diagnostic_pass':False})])
def test_stopped_native_raw_zero(tmp_path,field,value):
    data=complete(tmp_path);data['nominal'][field]=value;put(tmp_path,'pilot_outcome.json',data)
    r=verdict(tmp_path,'evaluation',0);assert not r['passed'] and r['raw_python_exit_code']==0

def test_unrun_hold(tmp_path):
    data=complete(tmp_path);data['extension']={'full_segment_completed':False,'not_run_reason':'main failed'}
    put(tmp_path,'pilot_outcome.json',data);assert not verdict(tmp_path,'evaluation',0)['passed']

def test_missing_evidence(tmp_path):assert not verdict(tmp_path,'evaluation',0)['passed']

def test_bad_parity(tmp_path):
    complete(tmp_path);put(tmp_path,'actual_query250_ownexport_output_parity.json',dict(passed=False))
    assert not verdict(tmp_path,'evaluation',0)['passed']

def test_witness(tmp_path):
    r=dict(pass_all=True,expected_head_calls=1,attempted_head_calls=1,returned_head_calls=1,BFM_inference_calls=0,physics_steps=0,fitting_launched=False)
    put(tmp_path,'head_witness/report.json',r);assert verdict(tmp_path,'witness',0)['passed']
    r['attempted_head_calls']=2;put(tmp_path,'head_witness/report.json',r);assert not verdict(tmp_path,'witness',0)['passed']

def test_role_requires_positive_receipt(tmp_path):
    p=tmp_path/'review.json';p.write_text(json.dumps(dict(passed=True,subject='a'*64)))
    assert role(dict(path=str(p),pass_field='passed'))
    p.write_text(json.dumps(dict(passed=False,subject='a'*64)))
    with pytest.raises(AssertionError):role(dict(path=str(p),pass_field='passed'))

def test_no_overwrite(tmp_path):
    p=tmp_path/'receipt';write_new(p,'first')
    with pytest.raises(FileExistsError):write_new(p,'second')
    assert p.read_text()=='first'

def test_command_path():
    base=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_student_evaluation_v1')
    a=wsl_arguments(base,'evaluation');assert a[-1].endswith('/source_draft_v1/evaluate_direct_target_student.py')
    assert a[3]=='/' and 'OMP_NUM_THREADS=1' in a
    assert quote("x'y")=="'x''y'"

def test_generated_failure_and_handle_order(tmp_path):
    text=run_text(tmp_path,tmp_path/'evaluation_process','evaluation','a'*64)
    assert text.index('raw_exit.json')<text.index('diagnostic_verdict.py')
    assert '$rawExit = $LASTEXITCODE' in text
    d=durable_text(tmp_path)
    assert d.index('$nativeHandle = $taskChild.Handle')<d.index('$taskChild.WaitForExit()')
    assert 'CreateNew' in d and '-WindowStyle Hidden' in d and 'launch_clearance.json' in d
    assert '\x0b' not in d and 'postrun_hashes.json' in d and 'automatic_retry=$false' in d
