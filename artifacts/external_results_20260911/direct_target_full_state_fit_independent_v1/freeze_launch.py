"""Pin the already reviewed actual audit invocation; no task computations."""
import json,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
request=json.loads((BASE/'audit_request.json').read_text())
assert sha(BASE/'audit_request.json')=='4257b57c529aea3ca2bfa5ab0230e10a089ba13e6b3004a28398eaf6afdf6103'
owner=BASE.parent/'direct_target_full_state_student_v1/owner_completion_verification.json'
assert sha(owner)=='8bacc308af12b0c888313848a8b20fdce04e11e252cb62cc850a3102efb5fce5'
pins=dict(request['source_sha256'])
pins.update({v['path']:v['sha256'] for v in request['subjects'].values()})
pins[request['source_review']['path']]=request['source_review']['sha256']
for p in [BASE/'audit_request.json',BASE/'run_audit_durable.ps1',BASE/'prepare_audit_request.py',BASE/'source_preparation.json',owner,Path(request['python_path']),Path(__file__)]:pins[p.resolve().as_posix()]=sha(p)
for p,d in pins.items():assert sha(p)==d
receipt=dict(selected_single_saved_audit=True,request_sha256=sha(BASE/'audit_request.json'),source_review_sha256=request['source_review']['sha256'],
    python_path=request['python_path'],input_sha256=pins,fit_owner_sha256=sha(owner),automatic_retry=False,task_model_calls=0,native_steps=0)
with (BASE/'launch_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(sha(BASE/'launch_receipt.json'))
