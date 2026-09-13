"""Synthetic configuration/launcher contracts; no actual task release reads."""
import json,subprocess
from pathlib import Path
from unittest.mock import patch
import pytest
from prepare_bound_launcher import require_review_subjects,durable_text,sha,wsl_arguments
from prepare_review_configuration import configuration
from freeze_final_package import require_helper_review


def test_no_selection_never_reads_actual_fit():
    with patch('prepare_review_configuration.actual_paths',side_effect=AssertionError('must not read actual inputs')):
        with pytest.raises(ValueError):configuration(root_selected=False,condition=None,release={},root_audit={},fit_owner={},source_review={},helper_review={})


def test_literal_review_subjects(tmp_path):
    binding=tmp_path/'binding.json';receipt=tmp_path/'receipt.json';binding.write_text('{}');receipt.write_text('{}')
    good={name:{'path':str(path),'sha256':sha(path)} for name,path in [('binding_subject',binding),('launch_receipt_subject',receipt)]}
    require_review_subjects(good,binding,receipt)
    bad=json.loads(json.dumps(good));bad['binding_subject']['path']=str(tmp_path/'other.json')
    with pytest.raises(AssertionError):require_review_subjects(bad,binding,receipt)
    bad=json.loads(json.dumps(good));bad['launch_receipt_subject']['sha256']='0'*64
    with pytest.raises(AssertionError):require_review_subjects(bad,binding,receipt)


def test_streamed_hash_and_known_exit_preserved(tmp_path):
    text=durable_text(tmp_path)
    assert 'ComputeHash($stream)' in text and 'ReadAllBytes' not in text and 'FileShare]::Delete' in text
    assert 'binding_subject' in text and 'launch_receipt_subject' in text and '.Contains(' not in text
    assert text.index('$nativeHandle = $taskChild.Handle')<text.index('$taskChild.WaitForExit()')
    assert 'Child exit status unavailable.' in text and 'automatic_retry=$false' in text


def test_fixed_original_stage_scope_and_frozen_python():
    base=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_causal_context_evaluation_v1')
    for mode,filename in [('witness','head_activation_witness.py'),('evaluation','evaluate_direct_target_student.py')]:
        args=wsl_arguments(base,mode)
        assert args[3]=='/' and args[-1].endswith('/source_draft_v1/'+filename)
        assert args[-2]=='-B' and 'PYTHONDONTWRITEBYTECODE=1' in args


def test_helper_review_cannot_omit_source(tmp_path):
    code=tmp_path/'helper.py';code.write_text('x=1')
    prep=tmp_path/'launch_helper_preparation_v1.json';prep.write_text(json.dumps({'helper_sha256':{'helper.py':sha(code)}}))
    review=tmp_path/'review.json';review.write_text(json.dumps({'passed':True,'helper_sha256':{}}))
    with patch('freeze_final_package.BASE',tmp_path):
        with pytest.raises(AssertionError):require_helper_review({'path':str(review),'pass_field':'passed'})
        review.write_text(json.dumps({'passed':True,'helper_sha256':{'helper.py':sha(code)}}))
        require_helper_review({'path':str(review),'pass_field':'passed'})


def test_actual_powershell_generated_path_normalization(tmp_path):
    text=durable_text(tmp_path)
    marker='$receiptPath.Replace('
    expression=marker+text.split(marker,1)[1].split(')',1)[0]+')'
    command="$receiptPath='E:\\synthetic\\evaluation_process\\launch_receipt.json'; $actual="+expression+"; if ($actual -cne 'E:/synthetic/evaluation_process/launch_receipt.json') { throw 'normalization failed' }; Write-Output $actual"
    result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
    assert result.stdout.strip()=='E:/synthetic/evaluation_process/launch_receipt.json'


@pytest.mark.parametrize('condition',[None,'','both','best','blinded','CAUSAL',0,False])
def test_explicit_condition_required_before_any_fit_read(condition):
    with patch('prepare_review_configuration.actual_paths',side_effect=AssertionError('must not read actual inputs')):
        with pytest.raises(ValueError):configuration(root_selected=True,condition=condition,release={},root_audit={},fit_owner={},source_review={},helper_review={})


@pytest.mark.parametrize('condition',['causal'])
def test_exact_condition_path_and_all16_roles(tmp_path,condition):
    from freeze_final_package import actual_paths,SUBJECTS
    request={'subjects':{key:{'path':str(tmp_path/key)} for key in ('checkpoint','coefficient_source','full_state_root_audit','full_state_root_owner','energy_source')},
        'full_state_paths':{'request':str(tmp_path/'generation_request'),'report':str(tmp_path/'generation_report')}}
    with patch('freeze_final_package.FIT',tmp_path),patch('freeze_final_package.read',return_value=request):
        paths=actual_paths(condition)
    assert len(SUBJECTS)==16 and set(SUBJECTS)<=set(paths)
    assert paths['head']==tmp_path/'fit/student_head.onnx'
    assert paths['normalization']==tmp_path/'fit/shared/normalization.npz'
    assert paths['context_alignment']==tmp_path/'fit/shared/context_alignment.json'
    assert paths['coefficient']==tmp_path/'coefficient_source'
    assert paths['fit_report']==tmp_path/'fit/report.json'
    assert paths['energy_source']==tmp_path/'energy_source'
    assert not any(name in paths for name in ('paired_report','blinded_fit_report','causal_fit_report'))


@pytest.mark.parametrize('condition',['causal'])
def test_context_binding_keeps_scope_and_no_condition_default(condition):
    from freeze_final_package import base_binding
    config={'root_selected':True,'context_condition':condition,'selected_witness_calls':1,'selected_main_controls':1569,
        'conditional_hold_controls':250,'reviews':{'release':{},'source':{}},'root_training_audit':{},'fit_owner_completion':{}}
    with patch('freeze_final_package.role',side_effect=lambda x:x):b=base_binding(config,{}, {'onnx_dependencies':[]})
    assert b['context_condition']==condition and b['architecture']==[1323,256,256,23] and b['ordinary_final_step']==71000
    assert b['requested_main_controls']==1569 and b['conditional_hold_controls']==250 and b['expected_head_calls']==1
    assert b['learned_BFM_calls']==0 and b['clock_foundation_connected'] is False and b['hardware_authorized'] is False


@pytest.mark.parametrize('mode',['witness','evaluation'])
@pytest.mark.parametrize('kind',['run','durable'])
def test_actual_powershell_parser_for_each_template(tmp_path,mode,kind):
    from prepare_bound_launcher import run_text
    text=run_text(tmp_path,tmp_path/(mode+'_process'),mode,'a'*64) if kind=='run' else durable_text(tmp_path/(mode+'_process'))
    path=tmp_path/'parse_only.txt';path.write_text(text)
    command="$tokens=$null;$errors=$null;[void][System.Management.Automation.Language.Parser]::ParseFile('"+str(path).replace("'","''")+"',[ref]$tokens,[ref]$errors);if ($errors.Count -ne 0) { throw ($errors | Out-String) }; Write-Output 'PARSE_PASS'"
    result=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
    assert result.stdout.strip()=='PARSE_PASS'
