"""Record source-only durable helper checks; creates no launch request."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
xml=ET.parse(BASE/'launch_helper_tests.xml').getroot()
assert len(xml.findall('.//testcase'))==8 and not any(xml.findall('.//'+k) for k in ('failure','error','skipped'))
result=dict(source_preparation_passed=True,preparation_only=True,source_preparation_sha256=sha(BASE/'source_preparation.json'),
    helper_sha256={n:sha(BASE/n) for n in ('prepare_audit_request.py','freeze_launch.py','run_audit_durable.ps1','verify_completion.py')},
    evidence_sha256={n:sha(BASE/n) for n in ('launch_helper_delta.patch','launch_helper_derivation.json','derive_launch_helpers.py','test_launch_helpers.py','launch_helper_tests.xml','record_launch_preparation.py')},
    synthetic_tests_passed=8,powershell_AST_passed=True,actual_request_created=False,actual_launch_created=False,
    task_model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,processes_launched=0,
    review_requirement='Root source review must directly contain all four helper hashes before freeze_launch.py permits a future receipt.')
with (BASE/'launch_helper_preparation.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(sha(BASE/'launch_helper_preparation.json'))
