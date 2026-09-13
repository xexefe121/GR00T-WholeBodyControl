"""Prepare a fresh metadata-only auditor correction; no saved numerical audit."""
from pathlib import Path
import ast,difflib,hashlib,json
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'direct_target_response_balanced_fit_independent_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
oldprep=json.loads((OLD/'source_preparation.json').read_text())
oldsource=OLD/'source_prepared_v1';target=BASE/'source_prepared_v1';target.mkdir(exist_ok=False)
delta=[];unchanged={}
for p in sorted(oldsource.glob('*.py')):
    assert sha(p)==oldprep['source_sha256'][p.name]
    before=p.read_text();after=before
    if p.name=='audit_balanced_math.py':
        after=after.replace("GROUPS=('root_position'", "ENERGY_PROVENANCE_RULE='Emean/Eg'\nPRODUCER_WEIGHT_RULE='mean_six_teacher_group_energies_over_group_energy'\n\nGROUPS=('root_position'")
        assert after.count("energy['rule']=='Emean/Eg'")==1
        after=after.replace("energy['rule']=='Emean/Eg'","energy['rule']==ENERGY_PROVENANCE_RULE")
        assert after.count("result['response_weight_rule']='Emean/Eg'")==1
        after=after.replace("result['response_weight_rule']='Emean/Eg'","result['response_weight_rule']=PRODUCER_WEIGHT_RULE")
    elif p.name=='audit_saved_warm.py':
        after=after.replace('energy_weights,balanced_metrics,warm_initial_errors,ledger_expectations','energy_weights,balanced_metrics,warm_initial_errors,ledger_expectations,PRODUCER_WEIGHT_RULE')
        for obj in ('checkpoint','report'):
            old=obj+"['response_weight_rule']=='Emean/Eg'";assert after.count(old)==1
            after=after.replace(old,obj+"['response_weight_rule']==PRODUCER_WEIGHT_RULE")
    ast.parse(after)
    if after==before:(target/p.name).write_bytes(p.read_bytes());unchanged[p.name]=sha(p)
    else:(target/p.name).write_text(after)
    delta.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='preserved_v1/'+p.name,tofile='source_prepared_v1/'+p.name))
for name in ('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py'):
    (BASE/name).write_bytes((OLD/name).read_bytes())
with (BASE/'source_delta.patch').open('x') as f:f.write(''.join(delta))
with (BASE/'derivation.json').open('x') as f:json.dump(dict(old_preparation_sha256=sha(OLD/'source_preparation.json'),
    unchanged_source_sha256=unchanged,unchanged_helper_sha256={n:sha(BASE/n) for n in ('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py')},
    old_failed_report_sha256=sha(OLD/'results_v1/report.json'),old_owner_sha256=sha(OLD/'owner_completion.json'),
    all_numerical_arithmetic_unchanged=True,actual_audit_executed=False),f,indent=2)
print('Fresh source prepared; no request or audit run.')
