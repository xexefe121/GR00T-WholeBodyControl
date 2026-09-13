"""Root review of the independent durable-stage saved-evidence extension."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
prep=BASE/'source_preparation_audit_v3.json'
assert sha(prep)=='0a013cf050d4d68599d59b6f6031a275063eab55f6b4a2b8c56018f3363438fd'
p=read(prep);prior=BASE/'root_source_review_v2.json'
assert sha(prior)=='7c943a119a6dff1d621447f43ba2fa84418e706a2295f5450ad1d893eb49c84a'
assert p['source_preparation_passed'] is True and len(p['source_sha256'])==14
for name,digest in p['source_sha256'].items():assert sha(BASE/'source_audit_v3'/name)==digest,name
for name in p['byte_identical_prior_sources']:
    assert (BASE/'source_audit_v3'/name).read_bytes()==(BASE/'source_audit_v2'/name).read_bytes(),name
assert len(p['byte_identical_prior_sources'])==10
for role in ('prior_root_source_review','timeout_source_review','timeout_preparation','exact_delta'):
    item=p[role];assert sha(item['path'])==item['sha256'],role
tests=p['synthetic_tests'];assert sha(tests['path'])==tests['sha256']
suites=ET.parse(tests['path']).getroot().findall('testsuite')
assert sum(int(s.get('tests','0')) for s in suites)==66
assert all(int(s.get(k,'0'))==0 for s in suites for k in ('failures','errors','skipped'))
v=dict(passed=True,source_review_pass=True,source_sha256=p['source_sha256'],
    source_preparation_subject=dict(path=str(prep),sha256=sha(prep)),prior_root_source_review=dict(path=str(prior),sha256=sha(prior)),
    reviewed=['Root read full independent clock_stage_math and all23 new fixture/corruption tests.',
        'Root reviewed complete audit_clock and prepare_request deltas; remaining10 source files byte-identical to reviewedv2.',
        'Exact source identities and unchanged hot-loop AST, actual source namespace and launched timeout verified.',
        'Stage paths bound by owner and audit; ordered fixed deadlines, epoch, full trace/native/API counters reconstructed.',
        'Raw nonzero/unknown exit, incomplete native scope or missing preservation prevents stage/timing qualification.',
        'Original physics/control/protocol/strict predicate math remains unchanged.'],
    tests=tests,actual_audit_selected=False,native_steps=0,model_calls=0,optimizer_updates=0,
    limitations=['Source/synthetic review only; completed actual clock and owner packet required for saved audit.',
        'Hard-kill prefixes lacking final report/owner require preservation accounting and cannot qualify.'])
out=BASE/'root_source_review_v3.json'
with out.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'sha256':sha(out),'sources':14,'tests':66}))
