"""Read-only failure-classification source review; no real fit audit calls."""
import ast
import hashlib
import json
from pathlib import Path
NEW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
BASE=NEW/'direct_target_continuation_failure_audit_v1';OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def subject(p):return dict(path=Path(p).as_posix(),sha256=sha(p))
prep=BASE/'source_preparation_v2.json'
assert sha(prep)=='09e4d2e33b6faec8e603e9c3d89d413e74ba74206c918fc6bd75255c2048e2aa'
metadata=read(prep)
for path,digest in metadata['source_sha256'].items():assert sha(path)==digest
assert sha(BASE/'source_v2/audit_saved_fit.py')=='77ad5ba95f261b7c94ce91feb86a5fa2a95c31253d980d689b23dd813663039d'
assert sha(BASE/'audit_saved_fit.py')=='9caec15cf18bcc7eb06201f91615c11d668df82a47b51e9655322c54a3642ba6'
added="        check('saved_parity_maximum_and_tolerance_identity',maximum==numerical['maximum_preclamp_rad'] and numerical['tolerance_rad']==1e-5)\n        check('saved_parity_failure_classification',numerical['passed'] is False and math.isfinite(maximum) and maximum>1e-5 and numerical['nonfinite_comparisons']==[])\n"
current=(BASE/'source_v2/audit_saved_fit.py').read_text()
assert current.count(added)==1 and current.replace(added,'')==(BASE/'audit_saved_fit.py').read_text()
for name in ('audit_math.py','audit_restoration.py'):
    assert (BASE/'source_v2'/name).read_bytes()==(NEW/'direct_target_continuation_root_audit_v1'/name).read_bytes()
tree=ast.parse(current)
check_function=next(v for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='run' for v in ast.walk(node) if isinstance(v,ast.FunctionDef) and v.name=='check')
assert 'name not in expected_failed_release_gates' in ast.unparse(check_function)
assert "expected_failed_release_gates={'ordinary_final_completed','report_export_pass','all_export_parity_within_selected_tolerance'}" in current
assert "len(failed_release_gates)==3" in current
assert 'passed=False,evidence_audit_passed=True,optimization_evidence_verified=True,export_qualified=False,canonical_cleared=False' in current
subjects={k:subject(p) for k,p in dict(preparation=prep,auditor=BASE/'source_v2/audit_saved_fit.py',math=BASE/'source_v2/audit_math.py',
    restoration=BASE/'source_v2/audit_restoration.py',first_delta=BASE/'source_delta.patch',second_delta=BASE/'source_delta_v2.patch',
    prior_source_review=NEW/'direct_target_continuation_root_audit_source_review_v1/review.json',
    failed_fit_report=NEW/'direct_target_continuation_v1/fit/report.json',
    independent_nine_comparisons=NEW/'direct_target_continuation_failed_export_review_v1/report.json',review_source=Path(__file__)).items()}
review=dict(source_review_pass=True,passed=True,verdict='CLEAR',scope='One independent saved-evidence audit of the completed optimization and failed original export; no release clearance.',
    subjects=subjects,input_sha256={v['path']:v['sha256'] for v in subjects.values()},
    checks=dict(exact_v2_two_check_delta=True,math_and_restoration_byte_unchanged=True,
        exactly_three_original_release_failures_retained=True,all_other_checks_still_mandatory=True,
        saved_maximum_identity_and_original_tolerance_always_required=True,
        finite_over_tolerance_failure_mandatory=True,optimization_and_diagnostic_completion_mandatory=True,
        passed_false_evidence_pass_true_and_export_canonical_false_distinguished=True),
    model_calls=0,native_steps=0,optimizer_updates=0,actual_root_saved_fit_audit_run=False,
    export_qualified=False,canonical_cleared=False)
with (OUT/'review.json').open('x',encoding='utf-8') as f:json.dump(review,f,indent=2);f.write('\n')
print(json.dumps({'source_verdict':'CLEAR','review_sha256':sha(OUT/'review.json')}))
