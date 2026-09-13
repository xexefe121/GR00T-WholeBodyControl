"""Independent source review and synthetic tests only; never task model/native calls."""
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
BASE=NEW/'direct_target_context_saved_semantics_review_v1'
OLD=NEW/'direct_target_full_state_saved_semantics_review_v1'
RUN=NEW/'direct_target_causal_context_evaluation_v2'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def functions(p):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(p).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

prep_path=BASE/'source_preparation.json'
assert sha(prep_path)=='4d9124b58361eb4cf3f93e2dd2201f17ea1ba1c6739b7dbf43b6d42c85283c1a'
prep=read(prep_path)
assert prep['source_preparation_pass'] is True and prep['tests_run']==36
assert prep['failures']==prep['errors']==prep['skips']==0
assert len(prep['source_sha256'])==10
for n,h in prep['source_sha256'].items():assert sha(BASE/n)==h,n
for n,h in prep['original_source_sha256'].items():assert sha(OLD/n)==h,n
for n,h in prep['artifacts'].items():assert sha(BASE/n)==h,n
assert sha(BASE/'fixed_maps.py')==sha(OLD/'fixed_maps.py')
for item, expected in prep['unchanged_function_AST'].items():
    n,fn=item.split(':')
    assert expected is True and functions(BASE/n)[fn]==functions(OLD/n)[fn]
runtime_review=NEW/'direct_target_context_namespace_review_v2/review.json'
assert sha(runtime_review)=='949b5124b0f452a917c87660a79f294e5276a8b20bc3da4c2a516d6d88dd3a29'
assert read(runtime_review)['source_sha256']==prep['actual_producer_source_sha256']
for n,h in prep['actual_producer_source_sha256'].items():assert sha(RUN/'source_draft_v1'/n)==h,n
assert sha(prep['actual_producer_source_preparation']['path'])==prep['actual_producer_source_preparation']['sha256']
assert not (BASE/'request.json').exists() and not (BASE/'results_v1').exists()
sys.path.insert(0,str(BASE))
suite=unittest.defaultTestLoader.discover(str(BASE),pattern='test_*.py')
stream=io.StringIO()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
with (OUT/'root_synthetic_tests.log').open('x',encoding='utf-8') as f:f.write(stream.getvalue())
assert result.wasSuccessful() and result.testsRun==36 and not result.skipped
for n,h in prep['source_sha256'].items():assert sha(BASE/n)==h,n
report=dict(passed=True,source_review_pass=True,reviewer='root',reviewed_utc=datetime.now(timezone.utc).isoformat(),
 source_preparation_subject=dict(path=prep_path.as_posix(),sha256=sha(prep_path)),source_sha256=prep['source_sha256'],
 original_sources_exact=True,unchanged_function_AST=prep['unchanged_function_AST'],actual_runtime_sources_exact=36,
 fixed_map_file_byte_identical=True,synthetic_tests=36,failures=0,errors=0,skips=0,
 test_log_sha256=sha(OUT/'root_synthetic_tests.log'),writer_sha256=sha(__file__),
 reviewed_semantics=['original current1000 reconstruction plus incoming prior23 and named pre-update history300',
 'causal condition and18 actual release subjects with saved independent training and completion evidence',
 'applied learned target inverse action, original BFM raw feedback and next history propagation',
 '1323 BFM prefix, query250 and one WSL witness equality; terminal zeros with continuing history',
 'issued final command may return zero steps; previous issued controls must have ten; failure cannot qualify',
 'rejected proposal retains prior/history/native state; staged capsule history independently derived',
 'only unique existing same-clock full58 maps; missing coverage explicit and unchanged map clips',
 'full1569 and conditional continuous250 unchanged; exact independent native evidence consumed'],
 task_arrays_loaded=0,model_calls=0,ORT_calls=0,BFM_calls=0,native_steps=0,replans=0,
 actual_request_created=False,actual_audit_executed=False,hardware_authorized=False)
path=OUT/'review.json'
with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(path=path.as_posix(),sha256=sha(path),tests=36,passed=True)))
