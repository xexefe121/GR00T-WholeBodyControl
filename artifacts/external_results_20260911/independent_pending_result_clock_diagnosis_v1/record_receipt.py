"""Bind completed saved diagnosis and input identities without rerunning it."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
report=json.loads((BASE/'report.json').read_text())
assert report['saved_diagnosis_passed'] is True and report['behavioral_qualification'] is False
assert sha(BASE/'report.json')=='c17f2e2b9d9b7330bb97f9b15974cf339a94539318e8dc4fa7813f380269632f'
for p,h in report['input_sha256'].items():assert sha(p)==h
receipt=dict(saved_diagnosis_passed=True,independent_saved_audit_pending=True,
    source_sha256=sha(BASE/'diagnose_saved.py'),report_sha256=sha(BASE/'report.json'),note_sha256=sha(BASE/'NOTE.md'),
    evidence_sha256={p.name:sha(p) for p in (BASE/'diagnosis.log',Path(__file__))},
    input_sha256=report['input_sha256'],all_inputs_unchanged=True,native_steps=0,model_calls=0,optimizer_updates=0,
    new_worker_processes=0,behavioral_qualification=False,actual_next_run_selected=False)
with (BASE/'diagnostic_receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'diagnostic_receipt_sha256':sha(BASE/'diagnostic_receipt.json')}))
