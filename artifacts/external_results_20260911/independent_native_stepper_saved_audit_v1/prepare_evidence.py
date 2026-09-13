"""Capture synthetic-only audit tests and immutable source preparation receipt."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=('audit_saved.py','saved_math.py','prepare_request.py','test_saved_audit.py','prepare_evidence.py','README.md')
sources={name:sha(BASE/name) for name in names}
for name in names:
    if name.endswith('.py'):ast.parse((BASE/name).read_text())
for name in ('preparation_report.json','synthetic_tests.json','synthetic_stdout.log','synthetic_stderr.log'):
    if (BASE/name).exists():raise RuntimeError('Preserve existing preparation artifact '+name)
command=[sys.executable,'-m','unittest','-v','test_saved_audit']
completed=subprocess.run(command,cwd=BASE,capture_output=True,text=True)
for name,text in [('synthetic_stdout.log',completed.stdout),('synthetic_stderr.log',completed.stderr)]:
    with (BASE/name).open('x',encoding='utf-8') as f:f.write(text)
tests=dict(passed=completed.returncode==0,exit_code=completed.returncode,command=command,tests=18,
    source_sha256=sources,stdout_sha256=sha(BASE/'synthetic_stdout.log'),stderr_sha256=sha(BASE/'synthetic_stderr.log'),
    synthetic_only=True,task_model_calls=0,optimizer_updates=0,native_steps=0)
with (BASE/'synthetic_tests.json').open('x',encoding='utf-8') as f:json.dump(tests,f,indent=2);f.write('\n')
if completed.returncode or 'Ran 18 tests' not in completed.stderr:raise RuntimeError('Synthetic tests incomplete: '+completed.stderr)
observed=BASE.parent/'independent_native_stepper_equivalence_v1/source_draft_v1'
schema={str(observed/name):sha(observed/name) for name in ('runner.py','replay_core.py','counted_api.py','capture_schema.py','native_stepper.py','clock_core.py','model_identity.py')}
receipt=dict(kind='independent_native_stepper_saved_audit_source_preparation',created_utc=datetime.now(timezone.utc).isoformat(),
    preparation_passed=True,source_only=True,actual_audit_run=False,actual_audit_request_created=False,
    source_sha256=sources,observed_producer_schema_sha256=schema,
    tests={'path':str(BASE/'synthetic_tests.json'),'sha256':sha(BASE/'synthetic_tests.json'),'passed':True,'count':18},
    expected=dict(expert_samples=18190,direct_samples=3158,total_samples=21348,expert_verified=18190,direct_verified=3157,
        packed_dtype='uint8[N,2984]',packed_float64_components=373,state_spec=8191,integration_size=291,
        witness_MJB_serializations=2,replay_MJB_serializations=8,actual_native_replays=0),
    model_calls=0,native_steps=0,optimizer_updates=0,
    pending=['Final native producer/owner schema and actual completed inputs must be aligned before freezing a real audit request.',
             'Root source review and completed owner/root stage gates remain required before the pure audit runs.'])
with (BASE/'preparation_report.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'preparation_passed':True,'tests':18,'report_sha256':sha(BASE/'preparation_report.json')}))
