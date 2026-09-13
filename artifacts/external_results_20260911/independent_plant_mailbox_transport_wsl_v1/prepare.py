"""Freeze the unchanged five-file Windows transport suite for one WSL run."""
from pathlib import Path
import hashlib
import json
import shutil

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
OLD=NEW/'independent_plant_mailbox_transport_v1'
REVIEW=NEW/'independent_plant_mailbox_transport_root_review_v1/review.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

if __name__=='__main__':
    review=json.loads(REVIEW.read_text())
    assert review['source_review_passed'] is True
    pins={}
    for path,digest in review['input_sha256'].items():
        assert sha(path)==digest
        pins[str(Path(path))]=digest
    dest=BASE/'source_snapshot_v1';dest.mkdir()
    for name in ['clock_core.py','history.py','mailbox.py','shared_mailbox.py','test_transport.py']:
        original=OLD/'source_snapshot_v1'/name
        shutil.copyfile(original,dest/name)
        assert sha(dest/name)==sha(original)
        pins[str(dest/name)]=sha(dest/name)
    for path in [REVIEW,OLD/'source_test_receipt.json',Path(__file__),BASE/'run_tests.py',BASE/'run_and_account.py']:
        pins[str(path)]=sha(path)
    request=dict(kind='one_selected_unchanged_mailbox_transport_WSL_synthetic_suite',expected_tests=11,
        expected_python_major_minor=[3,11],expected_numpy='1.26.4',source_review_sha256=sha(REVIEW),
        input_sha256=pins,model_calls=0,native_steps=0,optimizer_updates=0,plant_ticks=0,
        scope='Unchanged synthetic spawn/mailbox and pure admission tests only; no actual stepper or plant-clock loop.')
    with (BASE/'request.json').open('x') as f:f.write(json.dumps(request,indent=2)+'\n')
    print(json.dumps(dict(request_sha256=sha(BASE/'request.json'),review_sha256=sha(REVIEW),pins=len(pins))))
