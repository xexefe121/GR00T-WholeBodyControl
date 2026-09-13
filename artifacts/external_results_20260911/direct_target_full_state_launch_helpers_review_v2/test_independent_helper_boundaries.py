"""Independent synthetic saved-owner receipts; no task model or native calls."""
from pathlib import Path
import json
import sys
from unittest.mock import patch
import pytest

SOURCE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_full_state_evaluation_v1')
sys.path.insert(0,str(SOURCE))
import verify_completed_stage as owner
from prepare_bound_launcher import require_review_subjects

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path

def fixture(base):
    folder=base/'witness_process';folder.mkdir()
    (base/'verify_completed_stage.py').write_text('# synthetic declared owner file\n')
    data=base/'input.bin';data.write_bytes(b'fixed synthetic input')
    receipt=put(folder/'launch_receipt.json',{'input_hashes':{data.as_posix():owner.sha(data)}})
    review=put(base/'source_review.json',{'passed':True})
    clearance=put(folder/'launch_clearance.json',{'launch_receipt_sha256':owner.sha(receipt),'review':{'path':review.as_posix(),'sha256':owner.sha(review)}})
    put(folder/'start.json',{'wrapper_pid':10408,'receipt_sha256':owner.sha(receipt),'clearance_sha256':owner.sha(clearance),'review_sha256':owner.sha(review)})
    put(folder/'child.json',{'child_pid':23164,'wrapper_pid':10408})
    put(folder/'postrun_hashes.json',{data.as_posix():owner.sha(data)})
    put(folder/'raw_exit.json',{'raw_python_exit_code':0})
    put(folder/'exit.json',{'error':None,'all_postrun_hashes_exact':True,'raw_child_exit_code':0,'exit_code':0})
    put(folder/'diagnostic_verdict.json',{'raw_python_exit_code':0,'diagnostic_exit_code':0,'passed':True,'reasons':[]})
    for name in ('stdout.log','stderr.log'):(folder/name).write_text('')
    put(base/'head_witness/report.json',{'synthetic':True})
    check=put(base/'process_check.json',{'wrapper_absent':True,'child_absent':True,'wrapper_pid':10408,'child_pid':23164})
    return check

def run(base,check):
    with (patch.object(owner,'__file__',str(base/'verify_completed_stage.py')),
          patch.object(sys,'argv',['verify','--mode','witness','--process-check',str(check)])):
        owner.main()

def test_owner_complete_with_exact_process_ids(tmp_path):
    check=fixture(tmp_path);run(tmp_path,check)
    result=json.loads((tmp_path/'witness_completion_verification.json').read_text())
    assert result['owner_completion_accounting_passed'] is True
    assert result['diagnostic_passed'] is True and result['process_absence']['child_pid']==23164
    with pytest.raises(FileExistsError):run(tmp_path,check)

@pytest.mark.parametrize('field,value',[('wrapper_pid',1),('child_pid',1),('wrapper_absent',False),('child_absent',False)])
def test_owner_rejects_wrong_or_live_process_proof(tmp_path,field,value):
    check=fixture(tmp_path);proof=json.loads(check.read_text());proof[field]=value;put(check,proof)
    with pytest.raises(AssertionError):run(tmp_path,check)
    assert not (tmp_path/'witness_completion_verification.json').exists()

@pytest.mark.parametrize('case',['changed_input','changed_receipt','changed_review','changed_raw','changed_exit','omitted_pin'])
def test_owner_rejects_identity_and_exit_mismatches(tmp_path,case):
    check=fixture(tmp_path);folder=tmp_path/'witness_process'
    if case=='changed_input':(tmp_path/'input.bin').write_bytes(b'changed')
    elif case=='changed_receipt':put(folder/'launch_receipt.json',{'input_hashes':{},'changed':True})
    elif case=='changed_review':put(tmp_path/'source_review.json',{'passed':False})
    elif case=='changed_raw':put(folder/'raw_exit.json',{'raw_python_exit_code':None})
    elif case=='changed_exit':put(folder/'exit.json',{'error':None,'all_postrun_hashes_exact':True,'raw_child_exit_code':1,'exit_code':1})
    elif case=='omitted_pin':put(folder/'postrun_hashes.json',{})
    with pytest.raises(AssertionError):run(tmp_path,check)
    assert not (tmp_path/'witness_completion_verification.json').exists()

def test_same_hash_other_receipt_path_not_accepted(tmp_path):
    binding=put(tmp_path/'binding.json',{});receipt=put(tmp_path/'receipt.json',{});other=put(tmp_path/'other.json',{})
    review={'binding_subject':{'path':str(binding),'sha256':owner.sha(binding)},
            'launch_receipt_subject':{'path':str(other),'sha256':owner.sha(receipt)}}
    with pytest.raises(AssertionError):require_review_subjects(review,binding,receipt)
