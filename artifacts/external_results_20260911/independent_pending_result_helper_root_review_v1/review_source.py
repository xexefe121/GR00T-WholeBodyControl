"""Root helper source review and synthetic metadata tests; no task execution."""
import hashlib,io,json,sys,unittest
from pathlib import Path
OUT=Path(__file__).resolve().parent;NEW=OUT.parent
BASE=NEW/'independent_plant_pending_result_v1';OLD=NEW/'independent_plant_pending_publication_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
prep_path=BASE/'helper_preparation.json'
assert sha(prep_path)=='e59ccb854256266a2b20ee2eda5547c7bcd5652aa60525aca0d6fb4979e5eed2'
p=read(prep_path);assert p['passed'] is True and p['tests']==15 and p['failures']==p['errors']==p['skips']==0
assert len(p['helper_sha256'])==8 and len(p['producer_source_sha256'])==24 and len(p['auditor_source_sha256'])==20
for name,digest in p['helper_sha256'].items():assert sha(BASE/name)==digest
for name,digest in p['evidence_sha256'].items():assert sha(BASE/name)==digest
for name,digest in p['unchanged_helper_sha256'].items():assert sha(BASE/name)==sha(OLD/name)==digest
producer=read(NEW/'independent_pending_result_source_review_v1/review.json')
auditor=read(NEW/'independent_pending_result_saved_audit_review_v1/review.json')
assert producer['passed'] is producer['source_review_pass'] is True
assert auditor['passed'] is auditor['source_review_pass'] is True
assert sha(NEW/'independent_pending_result_source_review_v1/review.json')==p['producer_review_sha256']
assert sha(NEW/'independent_pending_result_saved_audit_review_v1/review.json')==p['auditor_review_sha256']
assert producer['source_sha256']==p['producer_source_sha256']
assert auditor['source_sha256']==p['auditor_source_sha256']
for name,digest in p['producer_source_sha256'].items():assert sha(BASE/'source_draft_v1'/name)==digest
for name,digest in p['auditor_source_sha256'].items():assert sha(NEW/'independent_plant_pending_result_saved_audit_v1/source_draft_v2'/name)==digest
ante=read(BASE/'antecedent_preflight.json');assert ante['passed'] is ante['all_declared_paths_exist'] is True
assert len(ante['input_sha256'])==29
for path,digest in ante['input_sha256'].items():assert sha(path)==digest
assert read(BASE/'template_parse.json')['passed'] is True
for name in ('clock_request.json','clock_process','run','clock_launch_clearance.json'):assert not (BASE/name).exists()
sys.path.insert(0,str(BASE));suite=unittest.defaultTestLoader.discover(str(BASE),pattern='test_launch_helpers.py')
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
with (OUT/'root_tests.log').open('x',encoding='utf-8') as f:f.write(stream.getvalue())
assert result.wasSuccessful() and result.testsRun==15 and not result.skipped,stream.getvalue()
for name,digest in p['helper_sha256'].items():assert sha(BASE/name)==digest
report=dict(passed=True,helper_review_pass=True,reviewer='root',helper_preparation_subject=dict(path=prep_path.as_posix(),sha256=sha(prep_path)),helper_sha256=p['helper_sha256'],producer_source_sha256=p['producer_source_sha256'],auditor_source_sha256=p['auditor_source_sha256'],producer_review_sha256=p['producer_review_sha256'],auditor_review_sha256=p['auditor_review_sha256'],unchanged_helpers=5,antecedents_verified=29,root_synthetic_tests=15,test_log_sha256=sha(OUT/'root_tests.log'),reviewed_semantics=[
'Original recorded-command/native/runtime roles, full1819controls/18190steps, four serializations, one epoch and240/120/180/555second watchdogs retained.',
'New literal original deadline and immutable-result BUSY-only max20, once per existing1ms worker iteration; existing job max10 once per physics tick. No deadline rebasing, ambiguous retry or reply recomputation.',
'Exact dedicated producer/auditor preparation paths and hashes plus full24/20 source maps are required; actual request still absent.',
'Five established files byte-identical including durable template, stage, verdict and completion. Generated PS parse/path normalization and acquired-handle/known-exit/posthash/failure behavior preserved.',
'Missing diagnosis receipt name corrected before helper freeze; all29 declared existing antecedents verified. Helper review is not a timed-run selection.'
],request_created=False,actual_dispatch=False,task_arrays_loaded=0,model_calls=0,native_steps=0,worker_processes_started=0,writer_sha256=sha(__file__))
with (OUT/'review.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(passed=True,sha256=sha(OUT/'review.json'),tests=15)))
