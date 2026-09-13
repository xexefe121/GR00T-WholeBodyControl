"""Record split-study path and declared execution metadata alignment only."""
from pathlib import Path
import hashlib, json, ast, xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'source_draft_v3'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
prior = json.loads((BASE / 'source_preparation_v2.json').read_text())
assert sha(BASE / 'source_preparation_v2.json') == '5846e1e2201600693f0511dbbe143f13c5595e466ac637daa711a37dfce08078'
for name, digest in prior['source_sha256'].items(): assert sha(BASE / 'source_draft_v2' / name) == digest
for name, digest in prior['helper_sha256'].items(): assert sha(BASE / name) == digest
sources = {p.name: sha(p) for p in sorted(SOURCE.glob('*.py'))}
for path in SOURCE.glob('*.py'): ast.parse(path.read_text())
unchanged = [name for name in sources if name != 'audit_saved_pair.py']
assert all(sources[name] == prior['source_sha256'][name] for name in unchanged)
suites = ET.parse(BASE / 'tests_v4.xml').getroot().findall('testsuite')
assert sum(int(s.get('tests', '0')) for s in suites) == 16
assert all(int(s.get(k, '0')) == 0 for s in suites for k in ('errors', 'failures', 'skipped'))
names = ('prepare_audit_request_v3.py', 'freeze_launch_v3.py', 'run_audit_durable_v3.ps1', 'verify_completion_v3.py', 'record_preparation_v3.py', 'prepare_v3.py')
helpers = {name: sha(BASE / name) for name in names}
n = BASE.parent
subjects = {
    'prior_root_source_review': n / 'direct_target_context_pair_audit_root_review_v2/review.json',
    'split_source_preparation': n / 'direct_target_causal_context_study_v2/source_preparation.json',
    'split_source_review': n / 'direct_target_context_split_source_review_v1/review.json',
}
result = dict(prior, source_directory=SOURCE.as_posix(), source_sha256=sources, helper_sha256=helpers,
    source_preparation_version=3, preserved_v2_preparation_sha256=sha(BASE / 'source_preparation_v2.json'),
    unchanged_v2_sources=unchanged,
    metadata_only_delta=dict(path=(BASE / 'source_delta_v3.patch').as_posix(), sha256=sha(BASE / 'source_delta_v3.patch'),
        description='Corrected study_v2 path, source/helper v3 names, explicit split first-layer restoration/report fields only. Numerical/context/graph/test code unchanged.'),
    tests=dict(path=(BASE / 'tests_v4.xml').as_posix(), sha256=sha(BASE / 'tests_v4.xml')),
    subjects={key: dict(path=path.as_posix(), sha256=sha(path)) for key, path in subjects.items()},
    producer_experiment='direct_target_causal_context_study_v2', producer_process_directory='fit_process_v2',
    producer_owner='owner_completion_verification_v3.json', actual_audit_request_present=False, actual_audit_run=False)
assert not (BASE / 'audit_request.json').exists()
with (BASE / 'source_preparation_v3.json').open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
print(json.dumps(dict(source_preparation_sha256=sha(BASE / 'source_preparation_v3.json'), sources=len(sources), helpers=len(helpers), tests=16, source_delta_sha256=sha(BASE / 'source_delta_v3.patch'))))
