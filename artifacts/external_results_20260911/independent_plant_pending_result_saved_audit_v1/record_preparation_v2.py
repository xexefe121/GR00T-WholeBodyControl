"""Freeze preserved v2 source for independent final combined synthetic review."""
import ast,hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;S=B/'source_draft_v2';O=B/'source_draft_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prior=json.loads((B/'source_preparation.json').read_text())
old={p.name:sha(p) for p in O.glob('*.py')};assert old==prior['source_sha256']
new={p.name:sha(p) for p in sorted(S.glob('*.py'))}
assert [n for n in old if old[n]!=new[n]]==['worker_result_math.py']
assert set(new)-set(old)=={'test_cross_job_iteration.py'}
for p in S.glob('*.py'):ast.parse(p.read_text())
text=(S/'worker_result_math.py').read_text()
addition="    chronological_iterations=[row['iteration'] for k in ordered for _,row,_ in pubs.get(k,[])]\n    if chronological_iterations!=sorted(chronological_iterations):raise AssertionError('Publication iterations moved backwards across jobs')\n"
assert text.count(addition)==1 and text.replace(addition,'')==(O/'worker_result_math.py').read_text()
report=dict(source_preparation_passed=True,source_frozen_for_independent_review=True,preparation_only=True,execution_selected=False,
 source_sha256=new,prior_v1_source_sha256=old,unchanged_from_v1=[n for n in old if old[n]==new[n]],
 changed_from_v1=['worker_result_math.py'],new_files=['test_cross_job_iteration.py'],exact_two_line_delta_verified=True,
 prior_preparation={'path':str(B/'source_preparation.json'),'sha256':sha(B/'source_preparation.json')},
 evidence_sha256={n:sha(B/n) for n in ['v1_to_v2.diff','v2_derivation.json','derive_v2.py','record_preparation_v2.py',
    'correct_test_import.py','test_import_correction_v2.json','test_cross_job_iteration_preserved_invalid_import.py.txt']},
 inherited_evidence=prior['external_subjects'],producer_source_sha256=prior['producer_source_sha256'],
 tests=prior['tests'],final_combined_synthetic_review_pending=True,
 actual_clock_calls=0,actual_audit_requests=0,worker_processes=0,native_steps=0,model_calls=0,optimizer_updates=0)
with (B/'source_preparation_v2.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({'source_preparation_v2_sha256':sha(B/'source_preparation_v2.json'),'source_count':len(new),
 'worker_math_sha256':new['worker_result_math.py']}))
