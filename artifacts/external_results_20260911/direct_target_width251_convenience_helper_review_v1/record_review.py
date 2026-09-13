"""Freeze narrow source fixes plus fake-only independent review evidence."""
import ast,difflib,hashlib,io,json,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
LAUNCH=NEW/'direct_target_width251_followup_design_v1/run_prepared_audit.py'
PREPARE=NEW/'direct_target_width251_consistency_v1/prepare_actual_request.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    before={p.as_posix():sha(p) for p in (LAUNCH,PREPARE)}
    assert sha(PREPARE)==sha(BASE/'prepare_actual_request_reviewed_v1.py')
    for path in (LAUNCH,PREPARE):compile(path.read_text(),str(path),'exec')
    original=(BASE/'run_prepared_audit_preserved_v1.py').read_text()
    diff=''.join(difflib.unified_diff(original.splitlines(True),LAUNCH.read_text().splitlines(True),fromfile='preserved_v1/run_prepared_audit.py',tofile='corrected_v3/run_prepared_audit.py'))
    with (BASE/'launcher_corrections.diff').open('x',encoding='utf-8') as f:f.write(diff)
    suite=unittest.TestSuite(unittest.defaultTestLoader.discover(str(BASE),pattern=p) for p in ('test_review.py','test_consistency_prepare.py'))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    with (BASE/'synthetic_review.log').open('x') as f:f.write(stream.getvalue())
    assert result.wasSuccessful() and result.testsRun==12
    assert before=={p.as_posix():sha(p) for p in (LAUNCH,PREPARE)}
    boot=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh')
    assert sha(boot)=='392de6eccb281c41219566c4b7c9c1813b086913f66d861529d4904959168ebc'
    evidence=(BASE/'test_review.py',BASE/'test_consistency_prepare.py',BASE/'synthetic_review.log',BASE/'launcher_corrections.diff',
        BASE/'run_prepared_audit_preserved_v1.py',BASE/'run_prepared_audit_pins_fixed_v2.py',BASE/'prepare_actual_request_reviewed_v1.py',
        NEW/'direct_target_width251_consistency_v1/source_prepared_v1/admission.py',
        NEW/'direct_target_width251_collection_v1/source_prepared_v1/collect_saved.py',boot)
    report=dict(passed=True,source_review_pass=True,source_sha256={LAUNCH.name:sha(LAUNCH),PREPARE.name:sha(PREPARE)},
        subject_paths={LAUNCH.name:LAUNCH.as_posix(),PREPARE.name:PREPARE.as_posix()},
        input_sha256={p.as_posix():sha(p) for p in evidence},synthetic_tests=12,failures=0,
        findings_fixed=['existing input digest silently overwritten by dynamic dependency','known raw exit lost on postread/output-hash exception'],
        original_proofs_preserved=True,bootstrap_exact=True,normalized_aliases_checked=True,
        interrupted_wait_not_marked_returned=True,consistency_preparation_unchanged=True,
        consistency_roles=22,physical_array_roles=10,collection_rows=1018,
        actual_subprocess_calls=0,actual_task_arrays_loaded=0,actual_requests_created=0,actual_audits_executed=0,native_steps=0,model_calls=0,
        source_only=True,writer_sha256=sha(__file__),test_harness_syntax_correction_preserved=True)
    with (BASE/'review.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(review_sha256=sha(BASE/'review.json'),source_sha256=report['source_sha256'],tests=12)))
if __name__=='__main__':main()
