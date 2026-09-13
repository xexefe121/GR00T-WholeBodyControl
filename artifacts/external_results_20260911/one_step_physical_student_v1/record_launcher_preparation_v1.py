"""Record source/stub checks; this script never freezes or launches a fit."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
tests=ET.parse(ROOT/'launcher_tests_v2/pytest_final.xml').getroot().find('testsuite')
assert tests is not None and int(tests.attrib['tests'])==23
assert all(int(tests.attrib[key])==0 for key in ('errors','failures','skipped'))
io=read(ROOT/'launcher_tests_v2/powershell_checks.json');assert io['passed'] is True
prep=read(ROOT/'preparation_report_v2.json')
source={str(ROOT/'source_draft_v2'/n).replace('\\','/'):h for n,h in prep['source_sha256'].items()}
for path,digest in source.items():assert sha(Path(path))==digest
for name in ('freeze_fit_v1.py','run_fit_durable_v1.ps1','read_fit_progress_v1.ps1'):
    source[(ROOT/name).as_posix()]=sha(ROOT/name)
evidence={str(ROOT/name).replace('\\','/'):sha(ROOT/name) for name in (
    'preparation_report_v2.json','preparation_tests_v2_final.xml','test_freeze_runner_v1.py',
    'test_runner_io_v1.ps1','test_runner_io_v2.ps1','launcher_tests_v2/powershell_checks.json',
    'launcher_tests_v2/pytest.xml','launcher_tests_v2/pytest_final.xml','record_launcher_preparation_v1.py')}
absent=['training_request.json','training_frozen_inputs.json','training_clearance.json','source_snapshot_v1',
    'fit_launch_plan.json','fit_launch_receipt.json','fit','fit_process']
assert all(not (ROOT/n).exists() for n in absent)
write(ROOT/'launcher_preparation_v1.json',dict(passed=True,source_preparation_only=True,
    tests=23,errors=0,failures=0,skips=0,powershell_parse_pass=True,
    shared_delete_atomic_replace_pass=True,child_handle_exit7_pass=True,source_sha256=source,evidence_sha256=evidence,
    prior_attempts_preserved=dict(io_v1='PowerShell null backup path bound as empty: harmless Replace test failed before child',
      pytest_initial='Incorrect cwd scanned unrelated E:/WpSystem; no collected tests; corrected cwd in final run'),
    actual_artifacts_absent=absent,model_evaluations=0,optimizer_updates=0,ORT_calls=0,native_steps=0,
    selected_scope=dict(updates=5000,ordinary_final_step=75000,valid_rows_max=3054,
        training_head_rows_max=36315000,head_ONNX_calls_max=1150),launch_performed=False))
print(json.dumps(dict(report_sha256=sha(ROOT/'launcher_preparation_v1.json'),
    helpers={n:sha(ROOT/n) for n in ('freeze_fit_v1.py','run_fit_durable_v1.ps1','read_fit_progress_v1.ps1')})))
