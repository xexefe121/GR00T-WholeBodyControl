"""Select the single corrected launch after preserving the no-child attempt."""
import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
TASK=HERE.parent/'direct_target_causal_context_study_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
review=HERE/'review.json'
assert sha(review)=='822395eb3f20588e7299fea0a4e8deaac4a82aa86f4f7cbd539ac5a21807d3a8'
r=json.loads(review.read_text());assert r['prelaunch_review_pass'] is True
old=TASK/'training_clearance.json'
assert sha(old)=='d4995998f3ab79e998518dab38839d2864e7c29cbaa3b30869153edaa6e103aa'
assert old.read_bytes()==(TASK/'failed_launch_attempt_v1/training_clearance.json').read_bytes()
assert not (TASK/'fit_process_v2').exists() and not (TASK/'fit').exists()
value=json.loads((TASK/'training_clearance_draft.json').read_text())
assert value['approved'] is False and value['attempt']==2
assert value['request_sha256']==r['training_request_sha256']
assert value['frozen_receipt_sha256']==r['frozen_receipt_sha256'] and value['launcher_sha256']==r['launcher_sha256']
value.update(approved=True,review_path=review.as_posix(),review_sha256=sha(review),
    selected_execution='one corrected launcher dispatch; fixed blinded then causal matched pair; no automatic retry',
    preserved_prior_clearance_sha256=sha(old))
temporary=TASK/'training_clearance_attempt2.new.json'
with temporary.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
os.replace(temporary,old)
print(json.dumps({'path':str(old),'sha256':sha(old),'approved':True}))
