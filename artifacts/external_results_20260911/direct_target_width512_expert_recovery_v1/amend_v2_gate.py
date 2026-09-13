"""Separate new transport clearance from unchanged original task admission."""
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def subject(p):return dict(path=str(p).replace('\\','/'),sha256=sha(p))
old=BASE/'v2_transport_before_gate_split'
for name in ('run_recovery_durable_v2.ps1','verify_recovery_completed_v2.py'):
    prior=(old/name).read_text();assert (BASE/name).read_text()==prior
    updated=prior.replace("'execution_clearance.json'","'execution_clearance_v2.json'")
    assert updated!=prior
    with (BASE/name).open('w',newline='\n') as f:f.write(updated)
receipt=json.loads((BASE/'launch_receipt_v2.json').read_text())
for name in ('run_recovery_durable_v2.ps1','verify_recovery_completed_v2.py'):
    receipt['input_sha256'][str(BASE/name)]=sha(BASE/name)
clearance=BASE/'execution_clearance.json';root_review=Path(json.loads(clearance.read_text())['review']['path'])
assert sha(clearance)=='29d674c341ae0264a44a2189a15740a88b3a5e218423173e4fa59743167d6260'
for p in [clearance,root_review,Path(__file__),*sorted(old.iterdir())]:receipt['input_sha256'][str(p)]=sha(p)
receipt['launcher_sha256']=sha(BASE/'run_recovery_durable_v2.ps1')
receipt['clearance_layers']=dict(task_admission=dict(clearance=subject(clearance),review=subject(root_review),
    unchanged_request_and_frozen=True,previous_task_entry=False),
    new_transport_clearance_path=str(BASE/'execution_clearance_v2.json').replace('\\','/'),
    root_must_select_corrected_transport=True)
with (BASE/'launch_receipt_v2.json').open('w',newline='\n') as f:json.dump(receipt,f,indent=2);f.write('\n')
revision=json.loads((BASE/'launcher_revision_v2.json').read_text())
revision.update(launch_receipt=subject(BASE/'launch_receipt_v2.json'),launcher=subject(BASE/'run_recovery_durable_v2.ps1'),
    owner=subject(BASE/'verify_recovery_completed_v2.py'),launch_pin_count=len(receipt['input_sha256']),
    clearance_layers=receipt['clearance_layers'],previous_revision=subject(old/'launcher_revision_v2.json'))
with (BASE/'launcher_revision_v3.json').open('x') as f:json.dump(revision,f,indent=2)
print(json.dumps(revision))
