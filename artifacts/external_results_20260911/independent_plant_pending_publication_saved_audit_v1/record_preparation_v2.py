"""Bounded source-only author review packet; root review remains separate."""
import ast,difflib,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent
OLD=BASE/'source_draft_v1'
SOURCE=BASE/'source_draft_v2'
REVIEW=BASE.parent/'pending_retry_saved_audit_adversarial_review_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
    old_prep=BASE/'source_preparation.json'
    assert sha(old_prep)=='bc29292ea4b50501737a2831cb8c975c82168d75c1bf6d8c3f63312cafb4b917'
    original=read(old_prep)['source_sha256']
    assert {p.name:sha(p) for p in OLD.glob('*.py')}==original
    current={p.name:sha(p) for p in SOURCE.glob('*.py')}
    changed=['audit_clock.py','clock_protocol_math.py','retry_publication_math.py','test_clock_protocol_math.py','test_retry_publication_math.py']
    new=['test_adversarial_terminal_bounds.py']
    assert set(current)==set(original)|set(new) and len(current)==17
    unchanged=[name for name in original if name not in changed]
    assert len(unchanged)==11 and all(current[name]==original[name] for name in unchanged)
    tests=BASE/'synthetic_tests_v2_final2.xml';suites=list(ET.parse(tests).getroot())
    assert sum(int(s.attrib['tests']) for s in suites)==106
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
    counters=read(REVIEW/'v1_counterexamples.json')
    assert all(v['incorrectly_accepted'] for v in counters['synthetic_counterexamples'].values())
    diff=BASE/'v1_to_v2.diff'
    with diff.open('x',encoding='utf-8') as f:
        for name in changed:
            f.write(''.join(difflib.unified_diff((OLD/name).read_text().splitlines(True),(SOURCE/name).read_text().splitlines(True),fromfile='preserved_source_v1/'+name,tofile='source_draft_v2/'+name)))
    pins={p.resolve().as_posix():sha(p) for p in list(OLD.glob('*.py'))+list(SOURCE.glob('*.py'))}
    for p in [old_prep,tests,diff,Path(__file__),REVIEW/'probe_v1.py',REVIEW/'v1_counterexamples.json',
              BASE.parent/'independent_plant_pending_publication_v1/source_preparation.json',
              BASE.parent/'independent_pending_publication_root_review_v1/review.json']:
        pins[p.resolve().as_posix()]=sha(p)
    findings=[
        dict(id='terminal_tick_bound',original='Failed status allowed expiry beyond saved terminal tick.',fix='Every publication/expiry index is bounded by captured terminal tick; returned count permits only one reserved native return.'),
        dict(id='eligible_retry_coverage',original='Expiry could skip prior eligible ticks; pending tail could remain stale after later native steps.',fix='Expiry occurs on immediately next tick; pending final attempt index must be returned-1 or returned.'),
        dict(id='unknown_return_terminal',original='At most one unknown observer return did not prove it was the actual last pre-native action.',fix='Unknown observer return must match returned/saved terminal index, with no later native/foundation/transport/admission work.'),
        dict(id='outer_interrupt_preservation',original='Core failure alone omitted watchdog BaseException carried by report.first_error.',fix='Pass the already-bound outer interruption through protocol; retain returned versus committed-event distinction for a terminal publish.')]
    result=dict(preparation_passed=True,preparation_only=True,author_self_review=True,root_source_review_pending=True,
        original_preparation_sha256=sha(old_prep),source_sha256=current,original_source_sha256=original,
        changed_original_files=changed,unchanged_original_files=unchanged,new_files=new,
        tests=dict(passed=106,failed=0,skipped=0,path=tests.as_posix(),sha256=sha(tests)),
        adversarial_counterexamples=counters['synthetic_counterexamples'],findings_fixed=findings,
        native_oracle_timing_history_stage_math_unchanged=True,producer_source_unchanged=True,
        source_diff_sha256=sha(diff),input_sha256=pins,
        actual_task_arrays_read=0,model_calls=0,native_steps=0,optimizer_updates=0,worker_processes=0,
        actual_clock_request_created=False,actual_audit_request_created=False,actual_dispatch=False,
        limitations=['This is an author self-review with synthetic counterexamples, not independent root clearance.',
                     'Exact partial-state integrity still requires matching observed counters, owned descriptors and bound stage/report evidence.',
                     'No incomplete or ambiguous publication acquires returned/admission or physical/timing qualification credit.'])
    assert all(sha(p)==v for p,v in pins.items())
    target=BASE/'source_preparation_v2.json'
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    review=dict(author_self_review=True,preparation_passed=True,independent_root_review_pending=True,
        source_preparation={'path':target.as_posix(),'sha256':sha(target)},source_sha256=current,
        findings_fixed=findings,synthetic_tests=106,native_steps=0,model_calls=0,actual_task_arrays=0,
        old_source_and_counterexamples_preserved=True,source_diff={'path':diff.as_posix(),'sha256':sha(diff)})
    with (REVIEW/'review_v2.json').open('x',encoding='utf-8') as f:json.dump(review,f,indent=2);f.write('\n')
    print(json.dumps(dict(preparation_sha256=sha(target),author_review_sha256=sha(REVIEW/'review_v2.json'),source_files=17,tests=106,pins=len(pins))))

if __name__=='__main__':main()
