"""Preserved owner-only completion checks; no task work or launch."""
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def subject(p):return dict(path=str(p).replace('\\','/'),sha256=sha(p))
old=BASE/'helpers_preserved_v1'
assert (BASE/'launch_receipt.json').read_bytes()==(old/'launch_receipt.json').read_bytes()
receipt=json.loads((BASE/'launch_receipt.json').read_text())
for name in ('verify_recovery_completed.py','test_execution_helpers.py'):
    receipt['input_sha256'][str(BASE/name)]=sha(BASE/name)
for p in [BASE/'execution_helper_tests_v3.xml',Path(__file__),*sorted(old.iterdir())]:receipt['input_sha256'][str(p)]=sha(p)
receipt['owner_metadata_amendment']=dict(previous_launch_receipt=subject(old/'launch_receipt.json'),
    previous_owner=subject(old/'verify_recovery_completed.py'),current_owner=subject(BASE/'verify_recovery_completed.py'),
    scope='child/handle/arguments/identity/driver/raw consistency only; request/frozen/native source/launcher unchanged',
    synthetic_tests=subject(BASE/'execution_helper_tests_v3.xml'),actual_recovery_dispatched=False)
with (BASE/'launch_receipt.json').open('w',newline='\n') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
prepared=json.loads((BASE/'concrete_preparation.json').read_text())
prepared.update(launch_receipt=subject(BASE/'launch_receipt.json'),launch_pins=len(receipt['input_sha256']),
                amendment=receipt['owner_metadata_amendment'])
with (BASE/'concrete_preparation_v2.json').open('x') as f:json.dump(prepared,f,indent=2)
print(json.dumps(dict(launch_receipt=prepared['launch_receipt'],owner=subject(BASE/'verify_recovery_completed.py'),launch_pins=prepared['launch_pins'])))
