"""Preserve success-only auditor and derive a truthful failed-release evidence audit."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / 'direct_target_continuation_root_audit_v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OLD / 'audit_saved_fit.py') == 'd4055bfea1fd7d7a0813954cc8f6135f43466ed021160e781a6ef5d6b8c829a5'
source = (OLD / 'audit_saved_fit.py').read_text()
original = source
replacements = [
    ('    checks=[]\n', "    checks=[]\n    expected_failed_release_gates={'ordinary_final_completed','report_export_pass','all_export_parity_within_selected_tolerance'}\n"),
    ('        if not condition:raise AssertionError(name)', '        if not condition and name not in expected_failed_release_gates:raise AssertionError(name)'),
    ("        check('ordinary_final_completed',", "        check('completed_optimization_and_diagnostics_preserved',report['optimization_completed'] is True and report['final_export_diagnostics_completed'] is True and report['ordinary_final_step']==55000 and report['additional_updates']==50000)\n        check('failed_release_status_preserved',report['completed'] is False and report['numerical_gate_passed'] is False and report['export_parity_passed'] is False)\n        check('ordinary_final_completed',"),
    ('        result=dict(passed=True,checks=len(checks),initial_metrics=', "        failed_release_gates=[c['name'] for c in checks if not c['passed']]\n        check('exact_expected_failed_release_gates',set(failed_release_gates)==expected_failed_release_gates and len(failed_release_gates)==3)\n        result=dict(passed=False,evidence_audit_passed=True,optimization_evidence_verified=True,export_qualified=False,canonical_cleared=False,failed_release_gates=failed_release_gates,checks=len(checks),initial_metrics="),
    ("    print(json.dumps({'passed':True,'checks':len(checks),'report_sha256':sha(output/'report.json')}))", "    print(json.dumps({'evidence_audit_passed':True,'export_qualified':False,'failed_release_gates':failed_release_gates,'checks':len(checks),'report_sha256':sha(output/'report.json')}))"),
]
for before, after in replacements:
    assert source.count(before) == 1, before
    source = source.replace(before, after)
with (BASE / 'audit_saved_fit.py').open('x') as f: f.write(source)
for name in ('audit_math.py', 'audit_restoration.py'):
    assert not (BASE / name).exists()
    shutil.copyfile(OLD / name, BASE / name)
delta = ''.join(difflib.unified_diff(original.splitlines(True), source.splitlines(True), fromfile='success_only/audit_saved_fit.py', tofile='failure_evidence/audit_saved_fit.py'))
(BASE / 'source_delta.patch').write_text(delta)
result = dict(source_preparation_only=True, actual_audit_run=False,
    original_auditor=dict(path=str(OLD / 'audit_saved_fit.py'), sha256=sha(OLD / 'audit_saved_fit.py')),
    source_sha256={name:sha(BASE / name) for name in ('audit_saved_fit.py','audit_math.py','audit_restoration.py','source_delta.patch')},
    changes='Exactly three failed original release gates retained as failures; all other original checks unchanged and mandatory. Added positive completed-optimization/diagnostic/step55000 and negative release-status checks. Evidence-pass is distinct from export qualification.',
    task_model_calls=0, ORT_calls=0, optimizer_updates=0, native_steps=0)
(BASE / 'source_preparation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(source_preparation_sha256=sha(BASE / 'source_preparation.json'), auditor_sha256=sha(BASE / 'audit_saved_fit.py'))))
