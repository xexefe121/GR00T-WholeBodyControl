"""Hash/AST-only source freeze; no task inputs or runtime execution."""
import ast,difflib,hashlib,json,sys
from pathlib import Path
B=Path(__file__).resolve().parent;S=B/'source_draft_v1'
O=B.parent/'independent_plant_pending_publication_v1/source_draft_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def functions(path):
    tree=ast.parse(path.read_text());out={}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):out[node.name]=ast.dump(node,include_attributes=False)
        if isinstance(node,ast.ClassDef):
            for method in node.body:
                if isinstance(method,(ast.FunctionDef,ast.AsyncFunctionDef)):out[node.name+'.'+method.name]=ast.dump(method,include_attributes=False)
    return out
original=json.loads((B/'original_source_hashes.json').read_text())
assert original=={p.name:sha(p) for p in sorted(O.glob('*.py'))}
current={p.name:sha(p) for p in sorted(S.glob('*.py'))}
unchanged=[n for n in original if original[n]==current[n]]
changed=[n for n in original if original[n]!=current[n]]
assert set(changed)=={'clock_core.py','recorded_protocol.py','dummy_worker.py','run_clock.py','test_scaffold.py','test_pending_publication.py'}
assert set(current)-set(original)=={'pending_result.py','test_pending_result.py'}
allowed={'clock_core.py':{'Job.identity','PlantFoundation._admit','PlantFoundation._boundary'},
 'recorded_protocol.py':{'decode_job'},'dummy_worker.py':{'worker_once','process_main'},
 'run_clock.py':{'imported_source_paths'}}
ast_checks={}
for name,exceptions in allowed.items():
    old,new=functions(O/name),functions(S/name)
    actual={key for key in old if old[key]!=new.get(key)}
    assert actual==exceptions,(name,actual,exceptions)
    ast_checks[name]={'changed_prior_functions':sorted(actual),'unchanged_prior_functions':sorted(set(old)-actual),
                     'new_functions':sorted(set(new)-set(old))}
diff=''.join(''.join(difflib.unified_diff((O/n).read_text().splitlines(True),(S/n).read_text().splitlines(True),fromfile='original/'+n,tofile='source_draft_v1/'+n)) for n in changed)
with (B/'source_changes.diff').open('x') as f:f.write(diff)
log=(B/'all_synthetic_tests_final.log').read_text(encoding='utf-8-sig')
assert 'Ran 67 tests' in log and log.rstrip().endswith('OK')
evidence_names=['derive_source.py','adapt_stub_tests.py','original_source_hashes.json','source_changes.diff',
 'DESIGN.md','OUTPUT_SCHEMA_DELTA.md','all_synthetic_tests_final.log','record_preparation.py']
external={
 'prior_producer_source_review':B.parent/'independent_pending_publication_root_review_v1/review.json',
 'completed_saved_audit_report':B.parent/'independent_plant_pending_publication_saved_actual_v1/results_v1/report.json',
 'completed_saved_audit_owner':B.parent/'independent_plant_pending_publication_saved_actual_v1/owner_completion_dispatch_v3.json',
 'owner_correction_review':B.parent/'independent_pending_clock_owner_correction_root_review_v1/review_v3.json',
 'diagnosis_report':B.parent/'independent_pending_clock_publication_diagnosis_v1/report.json',
 'diagnosis_receipt':B.parent/'independent_pending_clock_publication_diagnosis_v1/diagnostic_receipt.json'}
report=dict(source_preparation_passed=True,preparation_only=True,execution_selected=False,
 source_sha256=current,original_source_sha256=original,unchanged_files=unchanged,changed_files=changed,
 new_files=sorted(set(current)-set(original)),ast_checks=ast_checks,
 evidence_sha256={n:sha(B/n) for n in evidence_names},
 external_subjects={role:{'path':str(p),'sha256':sha(p)} for role,p in external.items()},
 tests={'passed':67,'failures':0,'errors':0,'runtime':sys.version,'synthetic_only':True},
 worker_retry={'max_total_attempts':20,'max_publish_per_worker_iteration':1,'existing_loop_sleep_seconds':0.001,
               'only_retry_status':'BUSY','deadline_source':'literal_Job_deadline_from_plant_epoch_activation',
               'pending_blocks_new_polls_and_computation':True,'max_buffered_taken_jobs':2},
 native_steps_executed=0,model_calls=0,optimizer_updates=0,actual_process_calls=0,
 actual_timing_qualified=False,physical_qualified=False,matching_saved_audit_preparation_required=True)
with (B/'source_preparation.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({'source_preparation_sha256':sha(B/'source_preparation.json'),'source_count':len(current),
                 'unchanged':len(unchanged),'changed':changed,'new':report['new_files'],'source_sha256':current}))
