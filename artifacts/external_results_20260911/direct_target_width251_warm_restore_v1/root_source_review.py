"""Root source and independent synthetic receipt review; no checkpoint loads."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
p=BASE/'source_preparation.json';assert sha(p)=='2dfd48854ee875fcc259bac8dff954e0b2e6b6272e1a4816ded6faadd6e66b1c'
r=read(p);assert r['source_preparation_pass'] is True and len(r['source_sha256'])==5
for name,h in r['source_sha256'].items():assert sha(BASE/'source_prepared_v1'/name)==h
for name,item in r['unchanged_sources'].items():assert sha(item['path'])==item['sha256']==r['source_sha256'][name]
assert len(r['unchanged_sources'])==3
for name,h in r['source_references'].items():assert sha(name)==h
assert sha(r['tests']['path'])==r['tests']['sha256']
assert sha(BASE/'DESIGN.md')==r['design_sha256']
assert sha(BASE/'prepare_source.py')==r['preparation_source_sha256']
tests=BASE/'root_tests_v1.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==43
assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
for field in ('actual_checkpoint_reads','actual_task_data_reads','model_forward_calls','gradient_calls','optimizer_updates','native_steps'):assert r[field]==0
result=dict(passed=True,source_review_pass=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),
 source_preparation_sha256=sha(p),source_sha256=r['source_sha256'],
 root_test_receipt=dict(path=tests.as_posix(),sha256=sha(tests),junit_cases=43),scope=r['scope'],
 reviewed_properties=['Strict source ordinary81000/full512 architecture and original forward identity',
 'All six learned actor tensors, both moments, scalar16000 steps and entire AdamW group preserved',
 'Deep-copy loading prevents source-state aliasing; source RNG and normalization verified',
 'Owned progress separates attempted/returned/verified restoration and preserves partial failures',
 'No expansion, new zero moments, normalization refit or selected LR/update/forward'],
 actual_checkpoint_reads=0,actual_task_data_reads=0,model_forward_calls=0,native_steps=0,
 actual_fit_selected=False,writer_sha256=sha(__file__))
out=BASE/'root_review.json'
with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,sha256=sha(out),junit_cases=43)))
