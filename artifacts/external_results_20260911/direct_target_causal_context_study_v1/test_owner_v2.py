import hashlib
import pytest
from verify_completed_v2 import canonical_pins,validate_process_pin_report
def fixture(tmp_path):
    p=tmp_path/'pin';p.write_bytes(b'fixed');digest=hashlib.sha256(b'fixed').hexdigest()
    pins={str(p):digest};record=dict(all_exact=True,count=1,files={str(p):dict(expected=digest,actual=digest,matched=True)})
    return pins,record
def test_exact_pin_membership_passes(tmp_path):
    pins,record=fixture(tmp_path);validate_process_pin_report(record,pins)
def test_omitted_pin_fails_even_all_exact(tmp_path):
    pins,record=fixture(tmp_path);record['files']={};record['count']=0
    with pytest.raises(ValueError):validate_process_pin_report(record,pins)
def test_extra_pin_fails(tmp_path):
    pins,record=fixture(tmp_path);record['files'][str(tmp_path/'extra')]=next(iter(record['files'].values()))
    with pytest.raises(ValueError):validate_process_pin_report(record,pins)
def test_wrong_expected_digest_fails(tmp_path):
    pins,record=fixture(tmp_path);next(iter(record['files'].values()))['expected']='0'*64
    with pytest.raises(ValueError):validate_process_pin_report(record,pins)
