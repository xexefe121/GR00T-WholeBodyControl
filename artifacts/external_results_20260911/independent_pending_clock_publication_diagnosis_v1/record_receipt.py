"""Bind completed saved-only diagnosis without another task audit."""
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((B/'report.json').read_text())
assert r['saved_diagnosis_passed'] and r['new_native_steps']==r['model_calls']==r['optimizer_updates']==0
files=['report.json','NOTE.md','diagnose_completed.py','inspect_saved.py','inspection.json','record_receipt.py']
receipt={'passed':True,'saved_data_only':True,'files':{name:sha(B/name) for name in files},
 'recovered_job_BUSY':[136,192,213,614],'unrecovered_result_activation':656,'result_BUSY_deadline_remaining_ns':17413713,
 'physical_qualified':False,'timing_qualified':False,'new_native_steps':0,'model_calls':0,'optimizer_updates':0}
with (B/'diagnostic_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(sha(B/'diagnostic_receipt.json'))
