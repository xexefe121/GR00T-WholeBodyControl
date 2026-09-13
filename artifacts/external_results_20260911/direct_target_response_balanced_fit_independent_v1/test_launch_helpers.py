"""Synthetic file bindings only. Never launch an auditor or a child process."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('freeze_fixture',BASE/'freeze_launch.py')
freeze=importlib.util.module_from_spec(spec);spec.loader.exec_module(freeze)

def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(freeze,'BASE',tmp_path)
    for name in ('run_audit_durable.ps1','prepare_audit_request.py','source_preparation.json','verify_completion.py','freeze_launch.py','runtime.exe','owner.json','source.py'):
        (tmp_path/name).write_text('{}')
    helpers={name:freeze.sha(tmp_path/name) for name in ('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py')}
    review=dict(source_review_pass=True,helper_sha256=helpers)
    rp=tmp_path/'review.json';rp.write_text(json.dumps(review))
    request=dict(kind='saved_response_balanced_warm_only',subjects={'owner_completion':dict(path=str(tmp_path/'owner.json'),sha256=freeze.sha(tmp_path/'owner.json'))},
        source_sha256={str(tmp_path/'source.py'):freeze.sha(tmp_path/'source.py')},source_review=dict(path=str(rp),sha256=freeze.sha(rp),pass_field='source_review_pass'),python_path=str(tmp_path/'runtime.exe'))
    return request,review

def save_request(tmp_path,request):
    p=tmp_path/'audit_request.json';p.write_text(json.dumps(request));return SimpleNamespace(request_sha256=freeze.sha(p))

def test_exact_reviewed_freeze_and_no_overwrite(tmp_path,monkeypatch):
    request,_=fixture(tmp_path,monkeypatch);args=save_request(tmp_path,request);freeze.main(args)
    receipt=json.loads((tmp_path/'launch_receipt.json').read_text())
    assert receipt['kind']=='saved_response_balanced_warm_only' and receipt['automatic_retry'] is False
    assert receipt['request_sha256']==args.request_sha256 and receipt['source_review_sha256']==request['source_review']['sha256']
    before=(tmp_path/'launch_receipt.json').read_bytes()
    with pytest.raises(FileExistsError):freeze.main(args)
    assert (tmp_path/'launch_receipt.json').read_bytes()==before

@pytest.mark.parametrize('name',['prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py'])
def test_each_unreviewed_helper_rejected(tmp_path,monkeypatch,name):
    request,review=fixture(tmp_path,monkeypatch);review['helper_sha256'][name]='wrong'
    rp=tmp_path/'review.json';rp.write_text(json.dumps(review));request['source_review']['sha256']=freeze.sha(rp)
    with pytest.raises(AssertionError):freeze.main(save_request(tmp_path,request))
    assert not (tmp_path/'launch_receipt.json').exists()

def test_failed_review_rejected(tmp_path,monkeypatch):
    request,review=fixture(tmp_path,monkeypatch);review['source_review_pass']=False
    rp=tmp_path/'review.json';rp.write_text(json.dumps(review));request['source_review']['sha256']=freeze.sha(rp)
    with pytest.raises(AssertionError):freeze.main(save_request(tmp_path,request))
    assert not (tmp_path/'launch_receipt.json').exists()

def test_wrong_audit_kind_rejected(tmp_path,monkeypatch):
    request,_=fixture(tmp_path,monkeypatch);request['kind']='saved_matched_context_pair_only'
    with pytest.raises(AssertionError):freeze.main(save_request(tmp_path,request))
    assert not (tmp_path/'launch_receipt.json').exists()

def test_changed_actual_owner_rejected(tmp_path,monkeypatch):
    request,_=fixture(tmp_path,monkeypatch);(tmp_path/'owner.json').write_text('changed')
    with pytest.raises(AssertionError):freeze.main(save_request(tmp_path,request))
    assert not (tmp_path/'launch_receipt.json').exists()
