"""Independent root source/fake-test review, with no actual trajectory reads."""
import hashlib,json,xml.etree.ElementTree as ET
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;BASE=OUT.parent/'direct_target_width251_collection_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
p=BASE/'source_preparation.json'
assert sha(p)=='6704baa14dd451db12269ab58449e1575ee78c31bc31f4b1ed3da416b1d4cd7e'
r=read(p);source=Path(r['source_directory'])
assert r['passed'] and r['source_preparation_only'] and not r['actual_collection_selected'] and not r['actual_collection_executed']
assert len(r['source_sha256'])==9 and len(r['unchanged_pure_modules'])==4
assert {p.name for p in source.glob('*.py')}==set(r['source_sha256'])
for name,digest in r['source_sha256'].items():assert sha(source/name)==digest,name
for name,sub in r['unchanged_pure_modules'].items():assert sha(sub['path'])==r['source_sha256'][name]==sub['sha256']
for path,digest in r['source_references'].items():assert sha(path)==digest,path
for role in ('design','output_schema'):
 sub=r[role];assert sha(sub['path'])==sub['sha256']
sub=r['tests']['receipt'];assert sha(sub['path'])==sub['sha256'] and r['tests']['passed'] and r['tests']['count']==28
tests=BASE/'root_tests_v1.xml';xml=ET.parse(tests).getroot();suites=list(xml.iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==28
assert all(int(s.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
assert r['required_actual_scope']==dict(control_start=251,control_stop_exclusive=1269,rows=1018,phase_rows=[99,819,100],
 fresh_student_state_queries=1,connected_expert_rows_after_first=1017,committed_plans=204,required_main_controls=1569,
 required_continuous_hold_controls=250,required_independent_reports=4)
for field in ('task_arrays_opened','task_model_calls','bfm_actor_calls','bfm_backward_calls','native_steps','replans','optimizer_updates'):assert r[field]==0
result=dict(passed=True,source_review_pass=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),
 source_preparation_sha256=sha(p),source_sha256=r['source_sha256'],source_directory=source.as_posix(),
 root_tests=dict(path=tests.as_posix(),sha256=sha(tests),passed=28),unchanged_pure_modules=4,
 scope=r['required_actual_scope'],reviewed_properties=['Exact qualified recovery request and completed owner required',
 'Original full lifecycle and continuous hold require four positive independent reports',
 'Actual controls251..1268 only, excluding learned250 and all terminal commands',
 'All204 committed maps reconstruct actual targets without altered reduction or clipping',
 'Full291/prior/named and flat history chronology and main-to-hold continuity checked',
 'Unchanged1000 direct features plus raw323 causal context; no normalization refit',
 'One student-state query is distinguished from1017 connected expert states'],
 actual_collection_selected=False,actual_collection_executed=False,model_fitting_authorized=False,
 task_arrays_opened=0,task_model_calls=0,native_steps=0,writer_sha256=sha(__file__))
out=OUT/'review.json'
with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,source_files=9,tests=28,sha256=sha(out))))
