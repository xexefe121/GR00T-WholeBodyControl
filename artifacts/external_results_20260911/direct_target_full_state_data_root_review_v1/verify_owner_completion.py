"""Saved identity/PID receipt repair; no array reconstruction or model/native calls."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
B=Path(__file__).resolve().parent
def local(p):
    s=str(p).replace('\\','/')
    if s.startswith('/mnt/') and s[6:7]=='/':s=s[5].upper()+':'+s[6:]
    return Path(s)
def read(p):return json.loads(local(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with local(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
report=read(B/'results_v1/report.json')
exit_record=read(B/'process_v1/exit.json')
start=read(B/'process_v1/start.json')
child=read(B/'process_v1/child.json')
absent=read(B/'process_absence.json')
assert report['data_review_pass'] and report['rows_checked']==354612
assert report['unique_numerical_feature_rows']==367570 and report['duplicate_rows']==0
assert exit_record['raw_python_exit_code']==exit_record['exit_code']==0 and exit_record['all_postrun_pins_exact']
assert start['wrapper_pid']==child['wrapper_pid']==exit_record['wrapper_pid']==absent['wrapper_pid']==20280
assert child['child_pid']==exit_record['child_pid']==absent['child_pid']==27728 and absent['processes_absent'] is True
for p,digest in report['input_sha256'].items():assert sha(p)==digest,p
assert len(report['input_sha256'])==50
files=['process_v1/start.json','process_v1/child.json','process_v1/exit.json','results_v1/report.json',
       'results_v1/duplicate_conflicts.json','dispatch_v1.json','process_absence.json','verify_owner_completion.py']
output={str(B/f):sha(B/f) for f in files}
invalid=B/'owner_completion_v1_invalid.json'
assert sha(invalid)=='c6a931d17a8425bf70faea467c27d9a7128b959c0f63a5c515997b057da2a015'
result=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),wrapper_pid=20280,child_pid=27728,
    processes_absent=True,raw_exit_known=True,raw_python_exit_code=0,exit_code=0,
    all_current_input_pins_exact=True,input_pin_count=50,output_sha256=output,
    data_review_pass=True,checks=report['checks'],rows_checked=354612,old_overlap_rows=140622,
    new_features_and_maps=213990,training_rows=367570,unique_training_inputs=367570,conflicting_labels=0,
    model_calls=0,native_steps=0,saved_data_audit_repeated=False,
    invalid_prior_receipt=dict(path=str(invalid),sha256=sha(invalid),valid=False,
        reason='Initial PowerShell completion checker failed to map /mnt paths and continued after nonterminating hash errors. Its current-pin claim is invalid. This separate verifier maps paths and raises on every failure. Original actual data audit and its completed postrun checks remain unchanged.'))
dest=B/'owner_completion_v2.json'
assert not dest.exists()
dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=True,input_pins=50,report_sha256=sha(B/'results_v1/report.json'),owner_sha256=sha(dest))))
