"""Freeze source-only independent retry ledger auditing and synthetic evidence."""
import ast,difflib,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent
SOURCE=BASE/'source_draft_v1'
OLD=BASE.parent/'independent_plant_clock_saved_root_review_v1/source_audit_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}

def main():
    original={p.name:sha(p) for p in OLD.glob('*.py')};current={p.name:sha(p) for p in SOURCE.glob('*.py')}
    changed=['audit_clock.py','clock_protocol_math.py','clock_stage_math.py','prepare_request.py','test_clock_protocol_math.py','test_clock_stage_math.py']
    added=['retry_publication_math.py','test_retry_publication_math.py']
    assert set(current)==set(original)|set(added) and len(current)==16
    unchanged=[n for n in original if n not in changed]
    assert len(unchanged)==8 and all(current[n]==original[n] for n in unchanged)
    methods={}
    for name,excepted in [('audit_clock.py',{'run_audit'}),('clock_protocol_math.py',{'protocol'}),('clock_stage_math.py',{'source_contract','request_contract'})]:
        old,new=functions(OLD/name),functions(SOURCE/name)
        keep=[k for k in old if k not in excepted]
        assert all(old[k]==new[k] for k in keep)
        methods[name]=keep
    test=BASE/'synthetic_tests_final.xml';suites=list(ET.parse(test).getroot())
    assert sum(int(s.attrib['tests']) for s in suites)==90
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
    diff=BASE/'source_changes.diff'
    with diff.open('x',encoding='utf-8') as f:
        for name in changed:
            f.write(''.join(difflib.unified_diff((OLD/name).read_text().splitlines(True),(SOURCE/name).read_text().splitlines(True),fromfile='source_audit_v3/'+name,tofile='source_draft_v1/'+name)))
    pins={p.resolve().as_posix():sha(p) for p in list(OLD.glob('*.py'))+list(SOURCE.glob('*.py'))}
    for p in [test,diff,Path(__file__),BASE/'derive_protocol.py',
              BASE.parent/'independent_plant_clock_saved_root_review_v1/root_source_review_v3.json',
              BASE.parent/'independent_plant_pending_publication_v1/source_preparation.json',
              BASE.parent/'independent_plant_pending_publication_v1/source_draft_v1/clock_core.py',
              BASE.parent/'independent_pending_publication_root_review_v1/review.json']:
        pins[p.resolve().as_posix()]=sha(p)
    result=dict(preparation_passed=True,preparation_only=True,source_sha256=current,original_source_sha256=original,
        changed_original_files=changed,unchanged_original_files=unchanged,new_files=added,
        unchanged_function_AST=methods,tests=dict(passed=90,failed=0,skipped=0,path=test.as_posix(),sha256=sha(test)),
        pending_core_sha256='c67d970962a1fe67258154d13beb82c04a69f70c25426e77736def8c790770d1',
        required_clock_request_contract='pending_BUSY_same_job_max10_before_original_activation',
        audited_new_evidence=['same immutable job across at most ten attempts','once per predecessor physics index',
                             'saved predeadline check separately from actual transport start/end',
                             'BUSY-only retry and terminal status/expiry','returned versus recorded and ambiguous final attempt',
                             'literal pending/last descriptor and counters','unchanged admission ordering and strict original deadline',
                             'unchanged physical/PD/full291/history/timing/stage/fault/process/MJB checks'],
        actual_task_arrays_read=0,model_calls=0,native_steps=0,optimizer_updates=0,worker_processes=0,
        actual_audit_request_created=False,actual_clock_request_created=False,execution_selected=False,
        input_sha256=pins,
        limitations=['Old source_audit_v3 must not audit retry ledgers; this separate source map requires a fresh review/request.',
                     'A saved publication after deadline is evidence, never timely admission or command qualification.',
                     'Missing or ambiguous publication outcomes retain failure status; no inferred retry or success.',
                     'No actual runtime, model, task arrays or worker processes were used by preparation.'])
    assert all(sha(p)==v for p,v in pins.items())
    with (BASE/'source_preparation.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(source_files=16,tests=90,preparation_sha256=sha(BASE/'source_preparation.json'),source_sha256=current)))

if __name__=='__main__':main()
