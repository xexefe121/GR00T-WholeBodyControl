"""Freeze source-only four-finding correction; preserve the original preparation."""
from pathlib import Path
import ast,difflib,hashlib,json,sys,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_draft_v2';OLD=BASE/'source_draft_v1';NEW=BASE.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')

def main():
    prior=read(BASE/'source_preparation.json')
    assert sha(BASE/'source_preparation.json')=='77710e582160f9ab345268792fd3d1a776a8afc48045d410a451b4cbaa379fc9'
    current={p.name:sha(p) for p in sorted(SOURCE.glob('*.py'))}
    original={p.name:sha(p) for p in sorted(OLD.glob('*.py'))}
    assert original==prior['source_sha256'] and len(current)==25 and current.keys()==original.keys()
    changed=[name for name in current if current[name]!=original[name]]
    assert changed==['test_timing_saved_math.py','test_timing_sidecar_io.py','timing_saved_math.py','timing_sidecar_io.py']
    for name in current:compile((SOURCE/name).read_text(),str(SOURCE/name),'exec')
    baseline=BASE/'source_original_v2'
    assert all(sha(baseline/name)==value for name,value in prior['original_source_sha256'].items())
    def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(OLD/'timing_saved_math.py'),functions(SOURCE/'timing_saved_math.py')
    assert a.keys()==b.keys() and all(a[k]==b[k] for k in a if k!='audit')
    a,b=functions(OLD/'timing_sidecar_io.py'),functions(SOURCE/'timing_sidecar_io.py')
    assert a.keys()==b.keys() and all(a[k]==b[k] for k in a if k!='reserved_capture')
    diff=''.join(''.join(difflib.unified_diff((OLD/name).read_text().splitlines(True),(SOURCE/name).read_text().splitlines(True),fromfile='preserved_v1/'+name,tofile='corrected_v2/'+name)) for name in changed)
    with (BASE/'audit_v1_to_v2.diff').open('x',encoding='utf-8') as f:f.write(diff)
    suite=ET.parse(BASE/'synthetic_tests_v2.xml').getroot().find('testsuite')
    assert suite is not None and int(suite.attrib['tests'])==219
    assert all(int(suite.attrib[key])==0 for key in ('failures','errors','skipped'))
    antecedents=[BASE/'source_preparation.json',
        NEW/'independent_timing_saved_audit_root_review_v1/prove_v1_gaps.py',
        NEW/'independent_timing_saved_audit_root_review_v1/v1_gap_proof.json',
        NEW/'independent_plant_timing_integration_v1/source_preparation.json',
        NEW/'independent_timing_integration_root_review_v1/review.json']
    assert read(antecedents[2])['proved_on_frozen_source_v1'] is True
    evidence=[BASE/'record_preparation_v2.py',BASE/'V2_CORRECTIONS.md',BASE/'synthetic_tests_v2.xml',BASE/'audit_v1_to_v2.diff']
    value=dict(passed=True,source_preparation_only=True,source_version='source_draft_v2',
        source_sha256=current,prior_source_sha256=original,
        original_source_sha256=prior['original_source_sha256'],
        changed_from_v1=changed,unchanged_from_v1=sorted(current.keys()-set(changed)),
        original_17_modules_unchanged=True,other_module_functions_AST_unchanged=True,
        evidence_sha256={p.name:sha(p) for p in evidence},antecedent_sha256={p.as_posix():sha(p) for p in antecedents},
        pytest_passed=163,pytest_subtests_passed=56,junit_cases=219,failures=0,errors=0,skips=0,
        findings_corrected=['verification attempt is not a return','sequential sibling wall overlap',
            'child thread CPU outside parent','child process CPU outside parent'],
        extra_regression='sequential sibling thread/process CPU chronology',
        actual_request_created=False,actual_saved_audit_executed=False,actual_clock_selected=False,
        actual_clock_calls=0,actual_gc_callbacks_installed=0,worker_processes=0,native_steps=0,model_calls=0,
        actual_task_arrays_loaded=0,test_python=sys.version)
    write(BASE/'source_preparation_v2.json',value)
    print(json.dumps(dict(passed=True,source_preparation_sha256=sha(BASE/'source_preparation_v2.json'),
        source_files=25,changed_source_sha256={k:current[k] for k in changed},junit_cases=219),indent=2))

if __name__=='__main__':main()
