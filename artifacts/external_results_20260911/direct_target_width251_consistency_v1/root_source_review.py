"""Independent source/receipt review; synthetic data only."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
p=BASE/'source_preparation.json';assert sha(p)=='96f4dd494985f27d7e87b93ef98fcdbe4d4135c82e2a1d790ee850953ec72f86'
r=read(p);assert r['source_preparation_pass'] is True and len(r['source_sha256'])==8
for name,h in r['source_sha256'].items():
 assert sha(BASE/'source_prepared_v1'/name)==h
 assert sha(BASE/'source_draft_v1'/name)==h,'Root-tested draft differs'
for name,item in r['unchanged_sources'].items():assert sha(item['path'])==item['sha256']==r['source_sha256'][name]
assert len(r['unchanged_sources'])==2
for name,h in r['source_references'].items():assert sha(name)==h
for name,h in r['documentation_sha256'].items():assert sha(BASE/name)==h
for role in ('tests','metadata_schema_proof'):
 item=r[role];assert sha(item['path'])==item['sha256']
assert sha(BASE/'prepare_source.py')==r['preparation_source_sha256']
proof=read(r['metadata_schema_proof']['path'])
for name,h in proof['metadata_read_sha256'].items():assert sha(name)==h
tests=BASE/'root_tests_v1.xml';suites=list(ET.parse(tests).getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==52
assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
for field in ('actual_numeric_files_read','actual_checkpoint_reads','task_model_calls','gradient_calls','native_steps','optimizer_updates'):assert r[field]==0
result=dict(passed=True,source_review_pass=True,reviewed_utc=datetime.now(timezone.utc).isoformat(),
 source_preparation_sha256=sha(p),source_sha256=r['source_sha256'],
 root_test_receipt=dict(path=tests.as_posix(),sha256=sha(tests),junit_cases=52),
 reviewed_properties=['13976 exact old nominal/physical and new recovery inputs; targets retained individually',
 'Raw1323 and current1000 bit/signed-zero groups and exact promoted normalization groups separated',
 'Digest matches checked by bytes; no averaging, dropping or target repair',
 '72 first24-per-phase queries with256 candidate blocks and fixed earliest-row ties',
 'Old fitted data lineage and actual completed collection/request/source/qualification hashes required',
 'No model, checkpoint, gradient, native simulation or controller execution'],
 scope=r['scope'],actual_saved_diagnosis_selected=False,actual_collection_admitted=False,
 actual_numeric_files_read=0,model_calls=0,native_steps=0,optimizer_updates=0,
 limitations=['Source review only; actual data consistency remains pending.',
 'Normalized aliases describe prelayer source algebra, not executed ORT kernels.',
 '72 proximity queries do not prove global coverage or controller stability.'],writer_sha256=sha(__file__))
out=BASE/'root_review.json'
with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,sha256=sha(out),junit_cases=52)))
