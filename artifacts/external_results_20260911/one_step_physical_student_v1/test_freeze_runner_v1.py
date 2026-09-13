"""Synthetic gate tests only; never invoke actual freezer, trainer or graphs."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('freeze_draft',BASE/'freeze_fit_v1.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

def put(path,data):
    path.write_text(json.dumps(data),encoding='utf-8');return path

def selection():
    return dict(kind='one_physical_fit_selection',model_fitting_authorized=True,
        additional_updates=5000,ordinary_final_step=75000,exact_conflict_groups=0,expected_valid_rows=3054)

def test_all_fixed_budgets():
    for valid in range(1,3055):
        b=f.budget(valid)
        assert b['training_head_rows']==5000*(3057+1152+valid)
        assert b['head_onnx_calls']==1126+2*((valid+255)//256)
        assert b['head_onnx_calls']<=1150 and b['training_head_rows']<=36315000
        assert b['BFM_calls']==b['native_steps']==0

@pytest.mark.parametrize('bad',[0,3055,-1,True,1.5])
def test_bad_budget_rejected(bad):
    with pytest.raises(AssertionError):f.budget(bad)

def test_actual_selection_hash_and_scope(tmp_path):
    path=put(tmp_path/'synthetic_selection.json',selection())
    assert f.selected(path,f.sha(path))['ordinary_final_step']==75000
    with pytest.raises(AssertionError):f.selected(path,'0'*64)
    for field,value in [('model_fitting_authorized',False),('ordinary_final_step',75001),
                        ('additional_updates',4999),('expected_valid_rows',True),('exact_conflict_groups',1)]:
        data=selection();data[field]=value;put(path,data)
        with pytest.raises(AssertionError):f.selected(path,f.sha(path))

def test_direct_subject_review_not_transitive(tmp_path):
    subject=put(tmp_path/'subject.json',{'subject':1});digest=f.sha(subject)
    linked=put(tmp_path/'linked.json',{'subject_hash':digest})
    review=put(tmp_path/'review.json',{'passed':True,'request':str(linked),'request_sha256':f.sha(linked)})
    role=dict(path=str(review),sha256=f.sha(review),pass_field='passed')
    with pytest.raises(AssertionError):f.review_role(role,{str(subject):digest},{})
    put(review,dict(passed=True,direct_subjects=[digest]));role['sha256']=f.sha(review)
    pins={};result=f.review_role(role,{str(subject):digest},pins)
    assert result['subjects'][f.pathstr(subject)]==digest
    assert pins[f.pathstr(review)]==f.sha(review)
    put(subject,{'changed':True})
    with pytest.raises(AssertionError):f.review_role(role,{str(subject):digest},{})

def test_failed_review_blocks(tmp_path):
    subject=put(tmp_path/'subject.json',{'subject':1})
    review=put(tmp_path/'review.json',{'passed':False,'subject':f.sha(subject)})
    with pytest.raises(AssertionError):
        f.review_role(dict(path=str(review),sha256=f.sha(review),pass_field='passed'),{str(subject):f.sha(subject)},{})

def conflict():
    return dict(passed=True,rows_checked=3054,valid_labels=3054,failed_branches=0,
        compatibility_with_unchanged_3057_nominal_centers=dict(total_rows=6111,unique_feature_inputs=6111,
            duplicate_inputs=0,exact_target_conflicts=0,float32_residual_conflicts=0,signed_zero_canonicalized=True,
            rows_removed=0,targets_averaged=0))

def test_actual_conflict_schema():
    assert f.validate_conflicts(conflict(),3054)['unique_feature_inputs']==6111

@pytest.mark.parametrize('field,value',[('exact_target_conflicts',1),('float32_residual_conflicts',1),
    ('rows_removed',1),('targets_averaged',1),('signed_zero_canonicalized',False),('unique_feature_inputs',6110)])
def test_conflict_no_hidden_removal(field,value):
    report=conflict();report['compatibility_with_unchanged_3057_nominal_centers'][field]=value
    with pytest.raises(AssertionError):f.validate_conflicts(report,3054)

def test_create_new_preserves_previous_bytes(tmp_path,monkeypatch):
    p=tmp_path/'attempt.json';f.write_new(p,dict(first=True));before=p.read_bytes()
    with pytest.raises(FileExistsError):f.write_new(p,dict(second=True))
    assert p.read_bytes()==before
    monkeypatch.setattr(f,'BASE',tmp_path)
    with pytest.raises(AssertionError):f.assert_absent(('attempt.json',))

def test_finalize_requires_direct_review_and_pins(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'BASE',tmp_path)
    selection_path=put(tmp_path/'synthetic_selection.json',selection())
    request=put(tmp_path/'training_request.json',dict(reviews=dict(branch_data=dict(path=str(tmp_path/'data_review.json')),
        source=dict(path=str(tmp_path/'source_review.json'),sha256='1'*64))))
    for name in ('training_frozen_inputs.json','freeze_report.json','data_review.json','source_review.json'):
        put(tmp_path/name,dict(synthetic=True,name=name))
    (tmp_path/'run_fit_durable_v1.ps1').write_text('# synthetic never executed',encoding='utf-8')
    plan=put(tmp_path/'fit_launch_plan.json',dict(selection_path=str(selection_path),selection_sha256=f.sha(selection_path),
        input_sha256={str(request):f.sha(request)},command=dict(synthetic_no_execution=True),budgets=f.budget(3054)))
    review=put(tmp_path/'final_review.json',dict(passed=True))
    args=SimpleNamespace(final_review=review,final_review_sha256=f.sha(review))
    with pytest.raises(AssertionError):f.finalize(args)
    assert not (tmp_path/'training_clearance.json').exists()
    subjects=[plan,tmp_path/'training_frozen_inputs.json',request,selection_path,tmp_path/'data_review.json']
    put(review,dict(passed=True,subjects=[f.sha(p) for p in subjects]));args.final_review_sha256=f.sha(review)
    f.finalize(args)
    launch=f.read(tmp_path/'fit_launch_receipt.json')
    assert launch['automatic_resume'] is False and launch['budgets']['head_onnx_calls']==1150
    assert not (tmp_path/'fit').exists() and not (tmp_path/'fit_process').exists()
    assert str(tmp_path/'freeze_report.json').replace('\\','/') in launch['input_sha256']
    with pytest.raises(AssertionError):f.finalize(args)

def test_freezer_has_no_model_execution_imports():
    tree=ast.parse((BASE/'freeze_fit_v1.py').read_text(encoding='utf-8'))
    banned={'torch','onnxruntime','mujoco','subprocess','multiprocessing'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):assert not {a.name.split('.')[0] for a in node.names}&banned
        if isinstance(node,ast.ImportFrom):assert (node.module or '').split('.')[0] not in banned

def test_runtime_source_still_matches_review():
    prep=f.read(BASE/'preparation_report_v2.json')
    assert len(prep['source_sha256'])==36
    for name,digest in prep['source_sha256'].items():assert f.sha(BASE/'source_draft_v2'/name)==digest

def test_actual_fit_artifacts_absent():
    for name in ('training_request.json','training_frozen_inputs.json','training_clearance.json',
                 'source_snapshot_v1','fit','fit_process'):
        assert not (BASE/name).exists(),name

def test_powershell_preservation_contract():
    source=(BASE/'run_fit_durable_v1.ps1').read_text(encoding='utf-8')
    assert '-WindowStyle Hidden' in source and '[IO.FileMode]::CreateNew' in source
    assert source.index('$capturedHandle=$child.Handle')<source.index('$child.WaitForExit()')<source.index('$rawExit=$child.ExitCode')
    assert '[IO.FileShare]::Delete' in source and 'Get-FileHash' not in source and 'Get-Content' not in source
    assert 'no automatic rerun' in source and "'prerun_pins.json','postrun_pins.json'" in source

def test_powershell_test_receipt():
    result=f.read(BASE/'launcher_tests_v2/powershell_checks.json')
    assert result['passed'] is True and result['child_exit_code']==7
    assert result['atomic_replace_while_reader_open'] is True and result['no_model_or_native_calls'] is True
