"""Source/fake-test receipt only. Never execute a benchmark or load task arrays."""
import ast, difflib, hashlib, json
from pathlib import Path
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
SOURCE=BASE/'source_draft_v1'
OLD=BASE.parent/'independent_plant_clock_timeout_correction_v1/source_draft_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    previous={p.name:sha(p) for p in OLD.glob('*.py')}
    current={p.name:sha(p) for p in SOURCE.glob('*.py')}
    assert set(current)==set(previous)|{'test_pending_publication.py'}
    assert all(current[n]==v for n,v in previous.items() if n!='clock_core.py')
    old=(OLD/'clock_core.py').read_text();new=(SOURCE/'clock_core.py').read_text()
    def methods(text):
        cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='PlantFoundation')
        return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
    a,b=methods(old),methods(new)
    untouched=[n for n in a if n not in ('__init__','_boundary','tick','summary')]
    assert all(a[n]==b[n] for n in untouched)
    tests=BASE/'synthetic_tests_final.xml';suites=list(ET.parse(tests).getroot())
    assert sum(int(v.attrib['tests']) for v in suites)==72  # 67 tests plus five unittest subtests.
    assert all(int(v.attrib.get(k,0))==0 for v in suites for k in ('errors','failures','skipped'))
    diff=BASE/'clock_core_final.diff'
    with diff.open('x',encoding='utf-8') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original/clock_core.py',tofile='source_draft_v1/clock_core.py')))
    pins={p.resolve().as_posix():sha(p) for p in list(SOURCE.glob('*.py'))+list(OLD.glob('*.py'))}
    for p in [tests,BASE/'derive_source.py',Path(__file__),BASE/'derivation.json',diff,
              BASE.parent/'independent_clock_publication_diagnosis_v1/diagnostic_receipt.json',
              BASE.parent/'independent_plant_pending_publication_design_v1/assessment.json',
              BASE.parent/'independent_plant_clock_saved_actual_v1/results_v1/report.json',
              BASE.parent/'independent_plant_clock_saved_actual_v1/owner_completion.json']:
        pins[p.resolve().as_posix()]=sha(p)
    result=dict(preparation_passed=True,preparation_only=True,source_sha256=current,
        original_source_sha256=previous,unchanged_original_files=[n for n in previous if n!='clock_core.py'],
        changed_original_files=['clock_core.py'],new_source_files=['test_pending_publication.py'],
        unchanged_foundation_method_AST=untouched,tests=dict(passed=67,failed=0,skipped=0,subtests=5,path=tests.as_posix(),sha256=sha(tests)),
        original_attempt_plus_retry_limit=10,original_step_ns=2000000,original_control_steps=10,
        retry_guard_timestamp_retained=True,transport_start_end_retained=True,
        deadline_check_is_not_atomic_with_transport_start=True,original_admission_deadline_unchanged=True,
        expired_guard_timestamp_retained=True,fixed_transport_record_bound=54560,fixed_foundation_event_bound=58196,
        original_capacity=80000,source_only_protocol_audit_adaptation_pending=True,
        metadata_preflight_correction='Original XML check expected67; pytest XML includes five subtests, total72. Failed source preserved; no model/native calls.',
        actual_run_selected=False,actual_binding_created=False,native_steps=0,model_calls=0,optimizer_updates=0,
        spawned_workers=0,task_arrays_loaded=0,input_sha256=pins,
        limitations=['Saved v3 protocol audit cannot audit the new repeated publication/expiry ledger.',
                     'This source cannot establish whether actual BUSY retries succeed or meet 2 ms deadlines.',
                     'Worker-result BUSY remains outside scope. Original run failures stay immutable.',
                     'The unchanged executable is copied for lineage only; no request or launcher is prepared.'])
    assert all(sha(p)==v for p,v in pins.items())
    with (BASE/'source_preparation.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(source_files=len(current),tests=67,sha256=sha(BASE/'source_preparation.json'),core_sha256=current['clock_core.py'])))

if __name__=='__main__':main()
