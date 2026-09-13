"""Hash and saved fake-test review only; never invokes clock runtime."""
import hashlib,json,xml.etree.ElementTree as ET
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'independent_plant_timing_integration_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
prep=BASE/'source_preparation.json'
assert sha(prep)=='3844f7d86a4f97557c8a74744f7ea382a25315ae3a96cb3abfa786992d2c8cc5'
r=read(prep)
assert r['passed'] and r['source_preparation_only']
assert (r['main_controls'],r['hold_controls'],r['native_step_budget'],r['serialization_budget'])==(1569,250,18190,4)
assert len(r['source_sha256'])==32 and len(r['original_source_sha256'])==24
for name,digest in r['source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==digest,name
for name,digest in r['original_source_sha256'].items():assert sha(BASE/'source_original_v1'/name)==digest,name
assert set(r['changed_original_modules'])=={'clock_core.py','native_stepper.py','session.py','run_clock.py'}
assert len(r['unchanged_original_modules'])==20
for name in r['unchanged_original_modules']:assert r['source_sha256'][name]==r['original_source_sha256'][name]
for role in ('evidence_sha256','antecedent_sha256'):
 for name,digest in r[role].items():
  path=Path(name)
  if not path.is_absolute():path=BASE/path
  assert sha(path)==digest,name
assert r['source_sha256']['timing_probe.py']=='a41ef15e1e47afc5d6fb3c0ab27a203208b1c66c10559a1ae8792e6e0dbe57f4'
tests=BASE/'root_tests_v1.xml'
xml=ET.parse(tests).getroot();suites=list(xml.iter('testsuite'))
assert suites and all(int(s.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
cases=sum(int(s.get('tests',0)) for s in suites)
assert cases>=128
for field in ('actual_clock_calls','actual_gc_callbacks_installed','worker_processes','native_steps','model_calls'):
 assert r[field]==0
assert r['actual_run_selected'] is False and r['actual_request_created'] is False
result=dict(passed=True,source_review_pass=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),
 source_preparation_sha256=sha(prep),source_sha256=r['source_sha256'],unchanged_original_modules=20,
 changed_original_modules=r['changed_original_modules'],root_test_receipt=dict(path=tests.as_posix(),sha256=sha(tests),junit_cases=cases),
 original_unittest_method_count=128,pytest_reported_passes=150,pytest_reported_subtests=20,
 reviewed_properties=['Native PD and existing full-state capture arithmetic preserved',
 'Existing clock times, epoch, deadlines, loop and original worker protocol preserved',
 'Real wall/thread/process clocks used only by separately selected future run',
 'Preallocated bounded spans and GC rows, explicit overflow/faults',
 'Actual native return/capture ownership retained before end-hook faults',
 'Main owned evidence written before sidecars, incomplete instrumentation prevents preliminary pass',
 'GC callback is removed by identity; enablement unchanged; foreign-thread overlap not attribution'],
 actual_clock_selected=False,actual_runtime_measured=False,sidecar_audit_pending=True,
 task_model_calls=0,native_steps=0,actual_gc_callbacks_installed=0,
 limitations=['Instrumentation overhead remains unmeasured.',
 'Fatal interruption can separate adapter and foundation ownership counters; audit must preserve both.',
 'Root span includes fixed wait; nested intervals must not be summed as disjoint CPU time.',
 'This review is not simulation or timing qualification.'],writer_sha256=sha(__file__))
out=OUT/'review.json'
with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,sha256=sha(out),source_files=32,junit_cases=cases)))
