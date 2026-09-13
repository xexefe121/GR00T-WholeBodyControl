"""Remove one invalid CPU reduction-order check; preserve all other audit code."""
from pathlib import Path
import difflib,hashlib,json
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'direct_target_response_balanced_fit_independent_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
target=BASE/'source_prepared_v1';target.mkdir(exist_ok=False)
prep=json.loads((OLD/'source_preparation.json').read_text());unchanged={}
line="            close('nominal_f32_mean',progress[:,0],wanted['nominal'],nominal=True)\n"
for p in sorted((OLD/'source_prepared_v1').glob('*.py')):
    assert sha(p)==prep['source_sha256'][p.name]
    if p.name=='audit_saved_warm.py':
        old=p.read_text();assert old.count(line)==1;new=old.replace(line,'')
        (target/p.name).write_text(new)
        with (BASE/'source_delta.patch').open('x') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='preserved_v2/'+p.name,tofile='source_prepared_v1/'+p.name)))
    else:(target/p.name).write_bytes(p.read_bytes());unchanged[p.name]=sha(p)
for name in ('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py'):(BASE/name).write_bytes((OLD/name).read_bytes())
with (BASE/'derivation.json').open('x') as f:json.dump(dict(unchanged_source_sha256=unchanged,
    unchanged_helper_sha256={n:sha(BASE/n) for n in ('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py')},
    removed_line=line.strip(),other_production_source_bytes_unchanged=True,source_v2_preparation_sha256=sha(OLD/'source_preparation.json'),
    preserved_v2_failed_report_sha256=sha(OLD/'results_v1/report.json'),preserved_v2_owner_sha256=sha(OLD/'owner_completion.json'),actual_audit_executed=False),f,indent=2)
print('One-line correction prepared; no request or audit.')
