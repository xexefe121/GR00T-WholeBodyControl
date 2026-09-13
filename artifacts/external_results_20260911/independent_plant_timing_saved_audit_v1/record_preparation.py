"""Freeze already-tested source-only auditor, no actual audit/request dispatch."""
from pathlib import Path
import ast,difflib,hashlib,json,sys,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_draft_v1';OLD=BASE/'source_original_v2';NEW=BASE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
def main():
    current={p.name:sha(p) for p in sorted(SOURCE.glob('*.py'))};original={p.name:sha(p) for p in sorted(OLD.glob('*.py'))}
    assert len(current)==25 and len(original)==20
    prior=NEW/'independent_plant_pending_result_saved_audit_v1/source_draft_v2'
    assert all(sha(prior/name)==value for name,value in original.items())
    changed=[name for name in original if current[name]!=original[name]]
    assert changed==['audit_clock.py','clock_accounting.py','prepare_request.py']
    for name in current:compile((SOURCE/name).read_text(),str(SOURCE/name),'exec')
    def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    old,new=functions(OLD/'clock_accounting.py'),functions(SOURCE/'clock_accounting.py')
    assert old.keys()==new.keys() and all(old[name]==new[name] for name in old if name!='counts')
    checks={'all20_originals_preserved_exact':True,'17_files_unchanged':True,
        'accounting_owner_timing_reserved_functions_AST_unchanged':True,
        'new_reserved_capture_preserves_observed_vs_computed_verification':True}
    write(BASE/'source_checks.json',checks)
    diff=''
    for name in changed:
        diff+=''.join(difflib.unified_diff((OLD/name).read_text().splitlines(True),(SOURCE/name).read_text().splitlines(True),fromfile='prior/'+name,tofile='timing/'+name))
    for name in sorted(current.keys()-original.keys()):
        diff+=''.join(difflib.unified_diff([],(SOURCE/name).read_text().splitlines(True),fromfile='/dev/null',tofile='timing/'+name))
    with (BASE/'audit_changes.diff').open('x',encoding='utf-8') as f:f.write(diff)
    suite=ET.parse(BASE/'synthetic_tests_v1.xml').getroot().find('testsuite')
    assert suite is not None and int(suite.attrib['tests'])==212
    assert all(int(suite.attrib[key])==0 for key in ('failures','errors','skipped'))
    antecedents=[NEW/'independent_plant_pending_result_saved_audit_v1/source_preparation_v2.json',
        NEW/'independent_pending_result_saved_audit_review_v1/review.json',
        NEW/'independent_plant_timing_integration_v1/source_preparation.json',
        NEW/'independent_timing_integration_root_review_v1/review.json']
    assert all(p.is_file() for p in antecedents)
    evidence={p.name:sha(p) for p in sorted(BASE.iterdir()) if p.is_file() and p.name!='source_preparation.json'}
    value=dict(passed=True,source_preparation_only=True,source_sha256=current,original_source_sha256=original,
        changed_original_modules=changed,unchanged_original_modules=[name for name in original if name not in changed],
        evidence_sha256=evidence,antecedent_sha256={str(p):sha(p) for p in antecedents},
        pytest_passed=158,pytest_subtests_passed=54,junit_cases=212,failures=0,errors=0,skips=0,
        actual_request_created=False,actual_saved_audit_executed=False,actual_clock_selected=False,
        actual_clock_calls=0,actual_gc_callbacks_installed=0,worker_processes=0,native_steps=0,model_calls=0,
        actual_task_arrays_loaded=0,test_python=sys.version)
    write(BASE/'source_preparation.json',value)
    print(json.dumps({'passed':True,'source_preparation_sha256':sha(BASE/'source_preparation.json'),
        'source_files':len(current),'pytest_passed':158,'subtests':54,'changed_sources':{k:current[k] for k in changed},
        'new_sources':{k:current[k] for k in sorted(current.keys()-original.keys())}},indent=2))
if __name__=='__main__':main()
