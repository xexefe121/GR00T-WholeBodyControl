import difflib
import hashlib
import json
from pathlib import Path
import shutil

BASE=Path(__file__).resolve().parent
OUT=BASE/'source_v2'
OUT.mkdir(exist_ok=False)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
before=(BASE/'audit_saved_export.py').read_text()
old="export_manifest=sha(export/'manifest.json'),root_training_audit="
new="export_manifest=sha(frozen_path),output_manifest=sha(export/'manifest.json'),root_training_audit="
assert before.count(old)==1
after=before.replace(old,new)
(OUT/'audit_saved_export.py').write_text(after)
for name in ('audit_graph.py','audit_arrays.py','audit_math.py'):
    shutil.copyfile(BASE/name,OUT/name)
delta=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='v1/audit_saved_export.py',tofile='source_v2/audit_saved_export.py'))
(BASE/'source_delta_v2.patch').write_text(delta)
result=dict(source_preparation_only=True,actual_audit_run=False,
    prior_preparation_sha256=sha(BASE/'source_preparation.json'),
    source_sha256={str(p):sha(p) for p in OUT.glob('*.py')},delta_sha256=sha(BASE/'source_delta_v2.patch'),
    change='Release role export_manifest explicitly binds export_frozen_inputs.json; separate output_manifest binds export/manifest.json. Both underlying identities remain verified.',
    unchanged_graph_corruption_tests=dict(passed=5,xml_sha256=sha(BASE/'graph_tests.xml')),
    task_model_calls=0,ORT_calls=0,optimizer_updates=0,native_steps=0)
(BASE/'source_preparation_v2.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(preparation_sha256=sha(BASE/'source_preparation_v2.json'),auditor_sha256=sha(OUT/'audit_saved_export.py'))))
