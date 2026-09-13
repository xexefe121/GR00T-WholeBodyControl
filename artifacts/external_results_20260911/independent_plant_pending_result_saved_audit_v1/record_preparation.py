"""Source/hash/AST freeze only; no task data or runtime evaluation."""
import ast,difflib,hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;S=B/'source_draft_v1'
O=B.parent/'independent_plant_pending_publication_saved_audit_v1/source_draft_v2'
P=B.parent/'independent_plant_pending_result_v1/source_draft_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
old=json.loads((B/'original_source_hashes.json').read_text());assert old=={p.name:sha(p) for p in O.glob('*.py')}
new={p.name:sha(p) for p in sorted(S.glob('*.py'))}
changed=[n for n in old if old[n]!=new[n]];unchanged=[n for n in old if old[n]==new[n]]
assert set(changed)=={'audit_clock.py','clock_protocol_math.py','clock_stage_math.py','prepare_request.py','test_clock_protocol_math.py','test_clock_stage_math.py'}
assert set(new)-set(old)=={'worker_result_math.py','test_worker_result_math.py'}
allowed={'audit_clock.py':{'run_audit'},'clock_protocol_math.py':{'job_expected','admission_reason','protocol'},
 'clock_stage_math.py':{'source_contract','request_contract'},'prepare_request.py':{'prepare'}}
ast_checks={}
for name,changed_names in allowed.items():
    a,b=funcs(O/name),funcs(S/name);actual={k for k in a if a[k]!=b.get(k)};assert actual==changed_names,(name,actual)
    ast_checks[name]={'changed_functions':sorted(actual),'unchanged_functions':sorted(set(a)-actual)}
diff=''.join(''.join(difflib.unified_diff((O/n).read_text().splitlines(True),(S/n).read_text().splitlines(True),fromfile='prior/'+n,tofile='source_draft_v1/'+n)) for n in changed)
with (B/'source_changes.diff').open('x') as f:f.write(diff)
files=['README.md','derive_source.py','adapt_tests.py','record_preparation.py','original_source_hashes.json','source_changes.diff',
       'all_pytest_v1.log','synthetic_tests_v1.xml','worker_tests_v2.log']
assert '126 passed, 26 subtests passed' in (B/'all_pytest_v1.log').read_text(encoding='utf-8-sig')
log=(B/'worker_tests_v2.log').read_text(encoding='utf-8-sig');assert 'Ran 26 tests' in log and log.rstrip().endswith('OK')
external={
 'prior_saved_audit_source_review':B.parent/'independent_pending_publication_saved_root_review_v1/review.json',
 'producer_preparation':B.parent/'independent_plant_pending_result_v1/source_preparation.json',
 'producer_source_review':B.parent/'independent_pending_result_source_review_v1/review.json',
 'producer_schema':B.parent/'independent_plant_pending_result_v1/OUTPUT_SCHEMA_DELTA.md'}
report=dict(source_preparation_passed=True,source_frozen_for_independent_review=True,preparation_only=True,execution_selected=False,
 source_sha256=new,original_source_sha256=old,unchanged_files=unchanged,changed_files=changed,new_files=sorted(set(new)-set(old)),
 ast_checks=ast_checks,producer_source_sha256={p.name:sha(p) for p in sorted(P.glob('*.py'))},
 evidence_sha256={n:sha(B/n) for n in files},external_subjects={k:{'path':str(p),'sha256':sha(p)} for k,p in external.items()},
 tests={'earlier_full_suite':{'passed':126,'subtests_passed':26,'log':'all_pytest_v1.log','source_before_final_worker_delta':True},
        'final_worker_suite':{'passed':26,'log':'worker_tests_v2.log'},'final_combined_suite_executed':False,
        'final_combined_deferred_reason':'Separate selected canonical controller collecting actual timings'},
 actual_clock_calls=0,actual_audit_requests=0,worker_processes=0,native_steps=0,model_calls=0,optimizer_updates=0)
with (B/'source_preparation.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({'source_preparation_sha256':sha(B/'source_preparation.json'),'source_count':len(new),
 'unchanged':len(unchanged),'changed':changed,'new_files':report['new_files'],'worker_math_sha256':new['worker_result_math.py']}))
