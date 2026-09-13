import difflib
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
OUT = BASE / 'source_v2'
OUT.mkdir(exist_ok=False)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
before = (BASE / 'audit_saved_fit.py').read_text()
needle = "        check('all_export_parity_within_selected_tolerance',"
assert before.count(needle) == 1
addition = "        check('saved_parity_maximum_and_tolerance_identity',maximum==numerical['maximum_preclamp_rad'] and numerical['tolerance_rad']==1e-5)\n        check('saved_parity_failure_classification',numerical['passed'] is False and math.isfinite(maximum) and maximum>1e-5 and numerical['nonfinite_comparisons']==[])\n"
after = before.replace(needle, addition + needle)
(OUT / 'audit_saved_fit.py').write_text(after)
for name in ('audit_math.py', 'audit_restoration.py'):
    shutil.copyfile(BASE / name, OUT / name)
(BASE / 'source_delta_v2.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='failure_v1/audit_saved_fit.py',tofile='failure_v2/audit_saved_fit.py')))
result = dict(source_preparation_only=True, actual_audit_run=False,
    prior_preparation=dict(path=str(BASE/'source_preparation.json'),sha256=sha(BASE/'source_preparation.json')),
    source_sha256={str(p):sha(p) for p in sorted(OUT.glob('*.py'))},
    delta_sha256=sha(BASE/'source_delta_v2.patch'),
    mandatory_identity_checks_added=['independent maximum equals saved maximum','selected tolerance remains1e-5','finite reconstructed maximum exceeds1e-5 and saved passed=false with no nonfinite comparisons'],
    all_other_checks_unchanged=True, task_model_calls=0, ORT_calls=0, native_steps=0, optimizer_updates=0)
(BASE / 'source_preparation_v2.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(source_preparation_sha256=sha(BASE/'source_preparation_v2.json'),auditor_sha256=sha(OUT/'audit_saved_fit.py'))))
