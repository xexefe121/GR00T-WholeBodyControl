"""Record reviewed owner-schema delta and already returned synthetic test result."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent
SOURCE=BASE/'source_v2'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
old=json.loads((BASE/'preparation_report.json').read_text())
for name,digest in old['source_sha256'].items():assert sha(BASE/name)==digest
names=['audit_saved.py','saved_math.py','prepare_request.py','test_saved_audit.py','README.md']
for name in names:
    if name.endswith('.py'):ast.parse((SOURCE/name).read_text())
assert (SOURCE/'saved_math.py').read_bytes()==(BASE/'saved_math.py').read_bytes()
differences=[]
for name in names:
    differences.extend(difflib.unified_diff((BASE/name).read_text().splitlines(True),(SOURCE/name).read_text().splitlines(True),fromfile='v1/'+name,tofile='source_v2/'+name))
with (BASE/'source_delta_v2.patch').open('x',encoding='utf-8') as f:f.write(''.join(differences))
source_hashes={str(SOURCE/name):sha(SOURCE/name) for name in names}
tests=dict(passed=True,tests=19,exit_code=0,synthetic_only=True,
    command=['C:/Users/camer/AppData/Local/Programs/Python/Python310/python.exe','-m','unittest','-v','test_saved_audit'],
    working_directory=str(SOURCE),reported_unittest_elapsed_seconds=.023,
    evidence='Successful exec tool result333dbf: Ran19tests,OK,exit0. This script records that returned result without repeating tests.',
    source_sha256=source_hashes,model_calls=0,native_steps=0,optimizer_updates=0)
with (BASE/'synthetic_tests_v2.json').open('x',encoding='utf-8') as f:json.dump(tests,f,indent=2);f.write('\n')
owner=BASE.parent/'independent_native_stepper_equivalence_v1'
observed={str(owner/name):sha(owner/name) for name in ['verify_completed_stage.py','stage_verdict.py','prepare_stage.py','source_preparation.json']}
result=dict(kind='independent_native_stepper_saved_audit_owner_schema_preparation_v2',created_utc=datetime.now(timezone.utc).isoformat(),
    preparation_passed=True,source_only=True,actual_audit_run=False,actual_audit_request_created=False,
    source_directory=str(SOURCE),source_sha256=source_hashes,observed_final_owner_schema_sha256=observed,
    prior_preparation={'path':str(BASE/'preparation_report.json'),'sha256':sha(BASE/'preparation_report.json')},
    prior_root_review={'path':str(BASE.parent/'independent_native_stepper_saved_audit_root_review_v1/review.json'),
                       'sha256':sha(BASE.parent/'independent_native_stepper_saved_audit_root_review_v1/review.json')},
    delta={'path':str(BASE/'source_delta_v2.patch'),'sha256':sha(BASE/'source_delta_v2.patch')},
    tests={'path':str(BASE/'synthetic_tests_v2.json'),'sha256':sha(BASE/'synthetic_tests_v2.json'),'count':19,'passed':True},
    changes=['PIDs checked against actual start/child receipts because actual exit schema has no PIDs.',
        'Raw/diagnostic/exit known-success and captured-handle fields checked explicitly.',
        'Owner input map must exactly equal actual launch map; pre/post cover that full map.',
        'Start record binds actual clearance, launch and review; consumed process records must appear in owner output hashes.',
        'prepare_request resolves unchanged selected input root from nested source_v2.'],
    unchanged=['Binary/typed fault decoding, PD, full291 boundaries, all native sample comparisons and repeated clock arithmetic.','All unrun v1 sources/receipt retained byte-exact.'],
    budgets={'model_calls':0,'native_steps':0,'optimizer_updates':0},
    pending=['Root delta source CLEAR, then actual completed witness/replay owner receipts before one pure audit request/run.'])
with (BASE/'preparation_report_v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'preparation_passed':True,'tests':19,'report_sha256':sha(BASE/'preparation_report_v2.json'),'audit_source_sha256':sha(SOURCE/'audit_saved.py')}))
