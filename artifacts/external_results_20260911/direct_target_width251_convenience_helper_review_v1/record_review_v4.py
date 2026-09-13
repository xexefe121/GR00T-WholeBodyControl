"""Freeze the WSL-child exit naming correction; no actual process invocation."""
import difflib,hashlib,io,json,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
LAUNCH=NEW/'direct_target_width251_followup_design_v1/run_prepared_audit.py'
PREPARE=NEW/'direct_target_width251_consistency_v1/prepare_actual_request.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    before=(BASE/'run_prepared_audit_preservation_v3.py').read_text();after=LAUNCH.read_text()
    assert after==before.replace('raw_python_exit_code=code','raw_child_exit_code=code')
    prior=json.loads((BASE/'review.json').read_text());assert sha(PREPARE)==prior['source_sha256'][PREPARE.name]
    diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='preserved_v3/run_prepared_audit.py',tofile='corrected_v4/run_prepared_audit.py'))
    with (BASE/'exit_label_v4.diff').open('x') as f:f.write(diff)
    suite=unittest.TestSuite(unittest.defaultTestLoader.discover(str(BASE),pattern=p) for p in ('test_review.py','test_consistency_prepare.py'))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    with (BASE/'synthetic_review_v4.log').open('x') as f:f.write(stream.getvalue())
    assert result.wasSuccessful() and result.testsRun==12
    report=dict(passed=True,source_review_pass=True,source_sha256={LAUNCH.name:sha(LAUNCH),PREPARE.name:sha(PREPARE)},
        source_paths={LAUNCH.name:LAUNCH.as_posix(),PREPARE.name:PREPARE.as_posix()},
        preserved_review_sha256=sha(BASE/'review.json'),preserved_source_sha256=sha(BASE/'run_prepared_audit_preservation_v3.py'),
        exact_single_field_exit_label_change=True,raw_child_is_WSL_exit=True,python_exit_not_inferred=True,
        tests=12,failures=0,synthetic_receipt_sha256=sha(BASE/'synthetic_review_v4.log'),
        test_source_sha256={name:sha(BASE/name) for name in ('test_review.py','test_consistency_prepare.py')},
        actual_subprocess_calls=0,actual_task_arrays_loaded=0,actual_requests_created=0,native_steps=0,model_calls=0,
        source_only=True,writer_sha256=sha(__file__),diff_sha256=sha(BASE/'exit_label_v4.diff'))
    with (BASE/'review_v4.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(review_sha256=sha(BASE/'review_v4.json'),source_sha256=report['source_sha256'],tests=12)))
if __name__=='__main__':main()
