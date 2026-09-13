"""Root source review; synthetic tests only, no task arrays/models/native calls."""
import ast
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path
from datetime import datetime, timezone

OUT=Path(__file__).resolve().parent
NEW=OUT.parent
BASE=NEW/'direct_target_width512_saved_semantics_review_v1'
OLD=NEW/'direct_target_response_saved_semantics_review_v1'
RUN=NEW/'direct_target_causal_width512_evaluation_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def functions(p):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(p).read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

prep_path=BASE/'source_preparation.json'
assert sha(prep_path)=='cff165de5b610c35ef7e973db7702944045fd615782bfaa6537aab14306eaaf3'
prep=read(prep_path)
assert prep['source_preparation_pass'] is True and prep['tests_run']==61
assert prep['failures']==prep['errors']==prep['skips']==0
assert len(prep['source_sha256'])==16
for n,h in prep['source_sha256'].items():assert sha(BASE/n)==h,n
for n,h in prep['original_source_sha256'].items():assert sha(OLD/n)==h,n
for n,h in prep['artifacts'].items():assert sha(BASE/n)==h,n
for n in ('audit_saved.py','context_math.py','fixed_maps.py','test_saved.py','verify_completion.py'):
    assert sha(BASE/n)==sha(OLD/n),n
assert functions(BASE/'saved_common.py')==functions(OLD/'saved_common.py')
assert functions(BASE/'prepare_audit_stage.py')==functions(OLD/'prepare_audit_stage.py')
runtime_review=NEW/'direct_target_width512_evaluation_independent_review_v2/review.json'
assert sha(runtime_review)=='db2a997d44634cb59d7d4bd92408fcf361323d143c6109024b457e5bd4a458de'
assert read(runtime_review)['source_sha256']==prep['actual_producer_source_sha256']
for n,h in prep['actual_producer_source_sha256'].items():assert sha(RUN/'source_draft_v1'/n)==h,n
assert sha(prep['actual_producer_source_preparation']['path'])==prep['actual_producer_source_preparation']['sha256']
stage=BASE/'stage_source_preparation.json'
assert sha(stage)=='883a0da493f3f0f5463f7d9a14c581adb840d5358e946b5db2b1b8962298d340'
stage_data=read(stage)
assert stage_data['passed'] is True and stage_data['audit_source_sha256']==prep['source_sha256']
for n,h in stage_data['helper_sha256'].items():assert sha(BASE/n)==h,n
assert sha(BASE/'preserved_run_audit_template.ps1.txt')==sha(OLD/'preserved_run_audit_template.ps1.txt')
assert not (BASE/'request.json').exists() and not (BASE/'results_v1').exists()
sys.path.insert(0,str(BASE))
suite=unittest.defaultTestLoader.discover(str(BASE),pattern='test_*.py')
stream=io.StringIO()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
with (OUT/'root_synthetic_tests.log').open('x',encoding='utf-8') as f:f.write(stream.getvalue())
assert result.wasSuccessful() and result.testsRun==61 and not result.skipped,stream.getvalue()
for n,h in prep['source_sha256'].items():assert sha(BASE/n)==h,n
report=dict(passed=True,source_review_pass=True,helper_review_pass=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 source_preparation_subject=dict(path=prep_path.as_posix(),sha256=sha(prep_path)),source_sha256=prep['source_sha256'],
 helper_preparation_subject=dict(path=stage.as_posix(),sha256=sha(stage)),helper_sha256=stage_data['helper_sha256'],
 original_sources_exact=True,common_math_AST_identical=True,actual_runtime_sources_exact=38,
 segment_context_fixed_map_files_byte_identical=True,synthetic_tests=61,failures=0,errors=0,skips=0,
 test_log_sha256=sha(OUT/'root_synthetic_tests.log'),writer_sha256=sha(__file__),
 reviewed_semantics=['Unchanged complete trace/PD/strict failure and incoming1323 context reconstruction, prefix250/witness and applied target feedback.',
 'Sixteen actual81000 width512 release roles; source71000, optimizer6000 to16000,10000 updates, seed20260912, split training/dense FP64 export, unchanged coefficient/context and distinct producer/energy rule labels.',
 'Separate runtime and helper reviews; all helper/source bytes and independent fit audit owner bind actual completed release.',
 'Root physical replay exact through every saved step; root intent same trace and same physics report, preserving negative verdicts.',
 'Only saved same-clock unique fixed maps; missing coverage explicit; no model calls, native steps or expert replans.',
 'Actual request and launch remain absent pending concrete review; durable wrapper template and completion owner unchanged.'],
 task_arrays_loaded=0,model_calls=0,ORT_calls=0,BFM_calls=0,native_steps=0,replans=0,
 actual_request_created=False,actual_audit_executed=False,hardware_authorized=False)
path=OUT/'review.json'
with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(path=path.as_posix(),sha256=sha(path),tests=61,passed=True)))
