"""Concrete source/receipt review with fake tests only; never starts the runner."""
import hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
B=Path(__file__).resolve().parent
TASK=B.parent/'independent_plant_process_clock_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
request_path=TASK/'clock_request.json';receipt_path=TASK/'clock_process/launch_receipt.json'
assert sha(request_path)=='d6a6c3332039046c77ff9c93eaedd0694f0ac3c8487a474760c25646e75ba91a'
assert sha(receipt_path)=='4ecd884ea49e2b307febb63967c222946c9ef6fdce73a8b52c59571b9a383c80'
r=read(request_path);receipt=read(receipt_path)
assert receipt['request_sha256']==sha(request_path)
for p,digest in receipt['input_hashes'].items():assert sha(p)==digest,p
expected=dict(requested_controls=1819,main_controls=1569,hold_controls=250,native_step_budget=18190,
    serialization_budget=4,model_inference_calls=0,optimizer_updates=0,other_oracle_native_steps=0,epoch_lead_ns=200000000)
for key,value in expected.items():assert type(r[key]) is int and r[key]==value,key
assert r['expected_model_sha256']=='717cc6f01a61be2fd1e79bff25710b30de4d01d5f14511f0f76ae45168194ee4'
assert r['command_table_sha256']=='417418e216fdfd35bc6dc690921778b7001c92a3e9b5e5ae9829bfee40729721'
assert r['epoch_rebase_allowed'] is False and r['hardware_authorized'] is False
assert r['outer_process_timeout_seconds']==120
source=TASK/'source_runner_v1'
test=subprocess.run([sys.executable,'-m','pytest','-q','--rootdir=.','--confcutdir=.',
    'test_scaffold.py','test_runner_stubs.py'],cwd=source,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
(B/'tests_stdout.txt').write_text(test.stdout)
(B/'tests_stderr.txt').write_text(test.stderr)
assert test.returncode==0 and '30 passed' in test.stdout,test.stdout+test.stderr
for p,digest in receipt['input_hashes'].items():assert sha(p)==digest,p
assert not (TASK/'run').exists() and not (TASK/'clock_process/started.lock').exists()
result=dict(prelaunch_review_pass=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),
    request_subject=dict(path=request_path.as_posix(),sha256=sha(request_path)),
    launch_receipt_subject=dict(path=receipt_path.as_posix(),sha256=sha(receipt_path)),
    inputs_verified=len(receipt['input_hashes']),input_sha256=receipt['input_hashes'],
    source_sha256={p.name:sha(p) for p in source.glob('*.py')},
    review_source_sha256=sha(__file__),synthetic_tests_passed=30,test_stdout_sha256=sha(B/'tests_stdout.txt'),
    source_findings=[],reviewed_contracts=[
        'Recorded expert full1569 plus continuous250 scope and four prior-witness model serializations.',
        'Original native loader/PD/oracle/full291/373capture/history/mailbox/fixed deadline components byte-preserved.',
        'One future epoch chosen after readiness; unchanged Session allocation must complete before that same epoch.',
        'Worker raw exit and worker report failure both required; missing/held input cannot pass preliminary qualification.',
        'Actual returned capture/overflow and first failure retained; cleanup and model-exit failures remain distinct.',
        'Exact postrun owned arrays/raw capsules/issued jobs/worker ledgers and per-stage process receipts.',
        'Hidden .NET hash supervisor and bounded120second Linux process-group timeout with explicit unrecoverable-memory limitation.'
    ],actual_run_selected=False,
    selection_condition='A later explicit clearance may select one run after current training/diagnostics and heavy audits finish; no overlap with those compute jobs.',
    model_calls=0,native_steps=0,worker_processes=0,process_clock_runs=0,
    qualification='Source/request ready only. Actual fixed-clock behavior and independent saved trace audit remain unqualified.')
dest=B/'review.json';assert not dest.exists()
dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=True,tests=30,input_pins=len(receipt['input_hashes']),review_sha256=sha(dest))))
