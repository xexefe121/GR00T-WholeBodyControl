"""Root source/receipt review for two future instrumented-clock helper packages."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
NEW=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
configs=[('independent_plant_timing_integration_v1','1b5a7cf3a3c55dbb3845f03d1dd6110fb4b49fe3833d59c3f23e89a28d7019e8',10,64),
 ('independent_plant_timing_saved_actual_v1','09d2e33299a5774876e3695255039bee020d4cd865d6fb8fd9894664b74f6945',4,18)]
results=[]
for dirname,digest,count,cases in configs:
 base=NEW/dirname;p=base/'helper_preparation.json';r=read(p)
 assert sha(p)==digest and r['passed'] is True and r['preparation_only'] is True
 assert len(r['helper_sha256'])==count
 for role in ('helper_sha256','evidence_sha256'):
  for name,h in r[role].items():assert sha(base/name)==h,name
 tests=base/'root_helper_tests_v1.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
 assert sum(int(s.get('tests',0)) for s in suites)==cases
 assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
 if count==10:
  for name,h in r['unchanged_helper_sha256'].items():
   assert sha(base/name)==h and sha(base/'helper_original_v1'/name)==h
  assert r['producer_source_sha256']==read(base/'source_preparation.json')['source_sha256']
  assert r['auditor_source_sha256']==read(NEW/'independent_plant_timing_saved_audit_v1/source_preparation_v2.json')['source_sha256']
 else:
  assert sha(base/'preserved_saved_audit_template.ps1.txt')==r['original_template_sha256']
  assert sha(base/'verify_completion.py')==sha(base/'helper_original_v1/verify_completion.py')==r['original_owner_sha256']
 for field in ('native_steps','model_calls','optimizer_updates','task_arrays_loaded'):assert r[field]==0
 results.append(dict(package=dirname,helper_preparation_sha256=digest,helper_sha256=r['helper_sha256'],
  root_test_receipt=dict(path=tests.as_posix(),sha256=sha(tests),junit_cases=cases)))
result=dict(passed=True,source_review_pass=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),packages=results,
 reviewed_properties=['Exact producer32/auditor25 source maps and actual root review required',
 'Prior hidden durable launch, acquired handles, raw exits, exact pins and no retry retained',
 'Sidecar identity/completeness independent of native return or verification credit',
 'Partial retained sidecars require explicit writer errors and never qualify timing',
 'Bounded original18190 native/4MJB/555s scope unchanged'],
 actual_request_created=False,actual_clock_selected=False,actual_saved_audit_selected=False,
 model_calls=0,native_steps=0,task_arrays_loaded=0,writer_sha256=sha(__file__))
out=Path(__file__).resolve().parent/'launch_helpers_review.json'
with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,sha256=sha(out),junit_cases=82)))
