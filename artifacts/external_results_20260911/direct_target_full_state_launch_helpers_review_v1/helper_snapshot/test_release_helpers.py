"""Synthetic configuration/launcher contracts; no actual task release reads."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from prepare_bound_launcher import require_review_subjects,durable_text,sha,wsl_arguments
from prepare_review_configuration import configuration
from freeze_final_package import require_helper_review


def test_no_selection_never_reads_actual_fit():
    with patch('prepare_review_configuration.actual_paths',side_effect=AssertionError('must not read actual inputs')):
        with pytest.raises(ValueError):configuration(root_selected=False,release={},root_audit={},fit_owner={},source_review={},helper_review={})


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
    base=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_full_state_evaluation_v1')
    for mode,filename in [('witness','head_activation_witness.py'),('evaluation','evaluate_direct_target_student.py')]:
        args=wsl_arguments(base,mode)
        assert args[3]=='/' and args[-1].endswith('/source_draft_v1/'+filename)
        assert args[-2]=='-B' and 'PYTHONDONTWRITEBYTECODE=1' in args


def test_helper_review_cannot_omit_source(tmp_path):
    code=tmp_path/'helper.py';code.write_text('x=1')
    prep=tmp_path/'launch_helper_preparation.json';prep.write_text(json.dumps({'helper_sha256':{'helper.py':sha(code)}}))
    review=tmp_path/'review.json';review.write_text(json.dumps({'passed':True,'helper_sha256':{}}))
    with patch('freeze_final_package.BASE',tmp_path):
        with pytest.raises(AssertionError):require_helper_review({'path':str(review),'pass_field':'passed'})
        review.write_text(json.dumps({'passed':True,'helper_sha256':{'helper.py':sha(code)}}))
        require_helper_review({'path':str(review),'pass_field':'passed'})
