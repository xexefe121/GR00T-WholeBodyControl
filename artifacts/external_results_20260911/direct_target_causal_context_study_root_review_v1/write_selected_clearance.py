"""Record root's single selected matched-pair launch clearance."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
TASK=BASE.parent/'direct_target_causal_context_study_v1'
review=BASE/'review.json'
assert hashlib.sha256(review.read_bytes()).hexdigest()=='3a15c0b9b896e4bbc1c2a661205ee4e595af007f57eb5392c6f211167747f819'
r=json.loads(review.read_text());assert r['prelaunch_review_pass'] is True
value=json.loads((TASK/'training_clearance_draft.json').read_text())
assert value['approved'] is False
assert value['request_sha256']==r['training_request_sha256'] and value['frozen_receipt_sha256']==r['frozen_receipt_sha256']
assert value['launcher_sha256']==r['launcher_sha256']
value.update(approved=True,review_path=review.as_posix(),review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(),
             selected_execution='one fixed matched pair; blinded then causal; no automatic retry')
path=TASK/'training_clearance.json'
with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
