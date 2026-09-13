"""Root source review of the saved-only pair auditor; no actual pair audit."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'direct_target_context_pair_fit_independent_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prep=json.loads((BASE/'source_preparation_v2.json').read_text())
assert prep['source_preparation_passed'] is True
sources={p.name:sha(p) for p in (BASE/'source_draft_v2').glob('*.py')}
assert sources==prep['source_sha256'] and len(sources)==7
for name,digest in prep['helper_sha256'].items():assert sha(BASE/name)==digest,name
old=(BASE/'source_draft_v1/audit_saved_pair.py').read_text()
assert (BASE/'source_draft_v2/audit_saved_pair.py').read_text()==old.replace('fit_process/','fit_process_v2/').replace("base/'fit_process'","base/'fit_process_v2'").replace('owner_completion_verification_v2.json','owner_completion_verification_v3.json')
for name in sources:
    if name!='audit_saved_pair.py':assert (BASE/'source_draft_v2'/name).read_bytes()==(BASE/'source_draft_v1'/name).read_bytes()
tests=BASE/'root_source_tests_v2.xml'
suites=ET.parse(tests).getroot().findall('testsuite')
assert sum(int(x.get('tests','0')) for x in suites)==16
assert all(int(x.get(k,'0'))==0 for x in suites for k in ('failures','errors','skipped'))
value=dict(source_review_pass=True,source_sha256=sources,helper_sha256=prep['helper_sha256'],
    source_preparation={'path':str(BASE/'source_preparation_v2.json'),'sha256':sha(BASE/'source_preparation_v2.json')},
    root_tests={'path':str(tests),'sha256':sha(tests),'passed':16},
    reviewed=['Root read complete saved numerical/context/graph audit and qualified preserved helpers.',
        'Root read request, launch freezer, durable wrapper and completion verifier; reviewed all process/owner-v3 metadata deltas.',
        'Independent context chronology/applied-prior/moments/schedule/graph corruption tests passed.',
        'Complete pair prerequisite retained; existing incomplete producer cannot run this auditor or receive release.'],
    actual_pair_audit_selected=False,actual_model_calls=0,actual_native_steps=0,optimizer_updates=0)
out=HERE/'review.json'
with out.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'source_review_pass':True,'sha256':sha(out),'source_preparation_sha256':sha(BASE/'source_preparation_v2.json')}))
