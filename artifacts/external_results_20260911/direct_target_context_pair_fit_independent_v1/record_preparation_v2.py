"""Record the process/owner metadata-only saved-audit alignment."""
from pathlib import Path
import hashlib,json,ast,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_draft_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=json.loads((BASE/'source_preparation.json').read_text())
assert sha(BASE/'source_preparation.json')=='c5c2bc9809b9e60255e159b84ee56d3ff5c88960e3c380296cb89d160ffc2662'
for name,digest in prior['source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==digest
for name,digest in prior['helper_sha256'].items():assert sha(BASE/name)==digest
sources={p.name:sha(p) for p in sorted(SOURCE.glob('*.py'))}
for p in SOURCE.glob('*.py'):ast.parse(p.read_text())
unchanged=[n for n in sources if n!='audit_saved_pair.py']
assert all(sources[n]==prior['source_sha256'][n] for n in unchanged)
old=(BASE/'source_draft_v1/audit_saved_pair.py').read_text()
assert (SOURCE/'audit_saved_pair.py').read_text()==old.replace('fit_process/','fit_process_v2/').replace("base/'fit_process'","base/'fit_process_v2'").replace('owner_completion_verification_v2.json','owner_completion_verification_v3.json')
suites=ET.parse(BASE/'tests_v3.xml').getroot().findall('testsuite')
assert sum(int(s.get('tests','0')) for s in suites)==16 and all(int(s.get(k,'0'))==0 for s in suites for k in ('errors','failures','skipped'))
names=('prepare_audit_request_v2.py','freeze_launch_v2.py','run_audit_durable_v2.ps1','verify_completion_v2.py','record_preparation_v2.py','prepare_v2.py')
helpers={n:sha(BASE/n) for n in names}
result=dict(prior,source_directory=SOURCE.as_posix(),source_sha256=sources,helper_sha256=helpers,
 source_preparation_version=2,preserved_v1_preparation_sha256=sha(BASE/'source_preparation.json'),
 unchanged_v1_sources=unchanged,metadata_only_delta=dict(path=(BASE/'source_delta_v2.patch').as_posix(),sha256=sha(BASE/'source_delta_v2.patch'),
  description='Producer process_v2/owner_v3 identities and source_draft_v2/helper filenames only. All numerical/context/graph/test code byte-identical.'),
 tests=dict(path=(BASE/'tests_v3.xml').as_posix(),sha256=sha(BASE/'tests_v3.xml')),
 producer_process_directory='fit_process_v2',producer_owner='owner_completion_verification_v3.json',
 actual_audit_request_present=False,actual_audit_run=False)
assert not (BASE/'audit_request.json').exists()
with (BASE/'source_preparation_v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(source_preparation_sha256=sha(BASE/'source_preparation_v2.json'),sources=len(sources),helpers=len(helpers),tests=16,source_delta_sha256=sha(BASE/'source_delta_v2.patch'))))
