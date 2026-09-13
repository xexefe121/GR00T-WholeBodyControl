"""Independent source pins and synthetic regression review; no runtime work."""
import hashlib,json,sys,xml.etree.ElementTree as ET
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'independent_plant_timing_saved_audit_v1'
SOURCE=BASE/'source_draft_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
prep=BASE/'source_preparation_v2.json'
assert sha(prep)=='78d7855373d409fcd6c3a44edef433d6be520faaaf956a739483dad2b8dd6ae8'
r=read(prep)
assert r['passed'] is True and r['source_preparation_only'] is True
assert len(r['source_sha256'])==25 and len(r['original_source_sha256'])==20
for role,folder in [('source_sha256','source_draft_v2'),('prior_source_sha256','source_draft_v1'),('original_source_sha256','source_original_v2')]:
 for name,digest in r[role].items():assert sha(BASE/folder/name)==digest,name
assert len(r['unchanged_from_v1'])==21
assert set(r['changed_from_v1'])=={'timing_saved_math.py','timing_sidecar_io.py','test_timing_saved_math.py','test_timing_sidecar_io.py'}
unchanged=[n for n,h in r['original_source_sha256'].items() if r['source_sha256'][n]==h]
assert len(unchanged)==17
for n in r['unchanged_from_v1']:assert r['source_sha256'][n]==r['prior_source_sha256'][n]
for role in ('evidence_sha256','antecedent_sha256'):
 for name,digest in r[role].items():
  p=Path(name)
  if not p.is_absolute():p=BASE/p
  assert sha(p)==digest,name
tests=OUT/'root_tests_v2.xml'
suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert suites and all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
assert sum(int(s.get('tests',0)) for s in suites)==219
sys.path.insert(0,str(SOURCE))
from test_timing_saved_math import fixture,check
from test_timing_sidecar_io import reserved
from timing_sidecar_io import reserved_capture
results={}
for name in ('sibling_wall','child_thread_cpu','child_process_cpu'):
 f=fixture();p={row['phase']:row for row in f['rows']}
 if name=='sibling_wall':
  p[2]['end_wall_before']=p[3]['start_wall_before']+5
  p[2]['end_wall_after']=p[3]['start_wall_before']+6
 else:
  key='end_'+name.removeprefix('child_')
  p[11][key]=p[10][key]+1
 try:check(f)
 except AssertionError:results[name+'_rejected']=True
 else:raise AssertionError(name+' accepted')
a=reserved();a[2]['session']['stepper']['verification_attempts']=a[2]['session']['stepper']['returned']
v=reserved_capture(*a)
assert v['actual_native_verification_attempted'] is True
assert v['actual_native_verification_returned'] is None and v['new_verification_credit']==0
results['verification_attempt_does_not_establish_return']=True
for field in ('actual_clock_calls','actual_gc_callbacks_installed','worker_processes','native_steps','model_calls','actual_task_arrays_loaded'):assert r[field]==0
for field in ('actual_request_created','actual_saved_audit_executed','actual_clock_selected'):assert r[field] is False
result=dict(passed=True,source_review_pass=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),
 source_preparation_sha256=sha(prep),source_sha256=r['source_sha256'],unchanged_original_modules=17,
 unchanged_v1_modules=21,root_test_receipt=dict(path=tests.as_posix(),sha256=sha(tests),junit_cases=219),
 pytest_passed=163,pytest_subtests_passed=56,independent_regressions=results,
 preserved_v1_gap_proof_sha256=sha(OUT/'v1_gap_proof.json'),
 reviewed_properties=['All original physical/control/protocol/epoch/MJB/stage gates retained',
 'Bounded fatal counter gaps require independent saved evidence; no new native or verification credit',
 'Wall/thread/process clocks constrained by explicit nesting and sequential scopes',
 'Foreign GC overlap and missing/incomplete spans remain uncertainty, not causal attribution',
 'Verification attempts do not imply returned verification',
 'Producer and all retained sidecars bound by actual hashes and lengths'],
 actual_clock_selected=False,actual_saved_audit_selected=False,actual_task_arrays_loaded=0,
 task_model_calls=0,native_steps=0,actual_gc_callbacks_installed=0,
 limitations=['Source review only; no simulation or timing qualification.',
 'Instrumentation overhead remains unmeasured; nested scopes cannot be summed as disjoint time.'],writer_sha256=sha(__file__))
out=OUT/'review.json'
with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,sha256=sha(out),junit_cases=219,independent_regressions=results)))
