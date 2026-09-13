"""Preserve v1, add only the reviewed cross-job iteration-order requirement."""
import difflib,hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent;O=B/'source_draft_v1';S=B/'source_draft_v2'
S.mkdir(exist_ok=False)
for p in O.glob('*.py'):(S/p.name).write_bytes(p.read_bytes())
p=S/'worker_result_math.py';before=p.read_text()
needle="    ordered=sorted(rows,key=lambda k:rows[k]['take_index'])\n"
assert before.count(needle)==1
after=before.replace(needle,needle+"    chronological_iterations=[row['iteration'] for k in ordered for _,row,_ in pubs.get(k,[])]\n    if chronological_iterations!=sorted(chronological_iterations):raise AssertionError('Publication iterations moved backwards across jobs')\n")
p.write_text(after,newline='\n')
original_test=B.parent/'independent_pending_result_saved_audit_review_v1/test_cross_job_iteration.py'
text=original_test.read_text();lines=text.splitlines(True)
text=''.join(line for line in lines if not line.startswith('SOURCE = ') and not line.startswith('sys.path.insert('))
text=text.replace("        if 'key' in row:\n            row['key'] = 2", "        if 'key' in row:\n            row['key'] = 2\n        if row.get('direction') == 'worker-jobs':\n            row['slot'] = 0")
text=text.replace("descriptor['source'].update(key=2,", "descriptor['source'].update(slot=0,key=2,")
(S/'test_cross_job_iteration.py').write_text(text,newline='\n')
with (B/'v1_to_v2.diff').open('x') as f:
    f.write(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='source_draft_v1/worker_result_math.py',tofile='source_draft_v2/worker_result_math.py')))
with (B/'v2_derivation.json').open('x') as f:
    json.dump({'preserved_v1_preparation_sha256':hashlib.sha256((B/'source_preparation.json').read_bytes()).hexdigest(),
        'expert_regression_source':str(original_test),'expert_regression_sha256':hashlib.sha256(original_test.read_bytes()).hexdigest(),
        'test_copy_changes':['remove hardcoded v1 import override','use physical slot0 for activation2'],
        'actual_tests_run_by_this_derivation':0,'native_steps':0,'model_calls':0},f,indent=2)
print(json.dumps({'worker_math_v2_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_only':True}))
