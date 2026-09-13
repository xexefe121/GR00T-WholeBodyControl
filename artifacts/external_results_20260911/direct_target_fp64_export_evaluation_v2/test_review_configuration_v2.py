"""Pure synthetic release-audit subject checks; no actual model or physics."""
import pytest
from prepare_review_configuration_v2 import require_export_audit
def data():return dict(passed=True,evidence_audit_passed=True,export_qualified=True,canonical_evaluation_cleared=False,
    direct_subject_sha256={'head':'a'*64,'export_manifest':'b'*64,'root_training_audit':'c'*64})
def test_positive_named_export_evidence():require_export_audit(data(),data()['direct_subject_sha256'])
@pytest.mark.parametrize('name',['passed','evidence_audit_passed','export_qualified'])
def test_no_failed_audit_pass(name):
    r=data();r[name]=False
    with pytest.raises(AssertionError):require_export_audit(r,data()['direct_subject_sha256'])
def test_actual_audit_does_not_clear_controller():
    r=data();r['canonical_evaluation_cleared']=True
    with pytest.raises(AssertionError):require_export_audit(r,data()['direct_subject_sha256'])
@pytest.mark.parametrize('name',['head','export_manifest','root_training_audit'])
def test_wrong_actual_subject(name):
    r=data();r['direct_subject_sha256'][name]='d'*64
    with pytest.raises(AssertionError):require_export_audit(r,data()['direct_subject_sha256'])
def test_unrelated_metadata_not_direct_subject():
    r=data();r['unrelated_head']='a'*64;r['direct_subject_sha256']['head']='d'*64
    with pytest.raises(AssertionError):require_export_audit(r,data()['direct_subject_sha256'])
