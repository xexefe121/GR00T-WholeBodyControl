"""Bind the one proved transport correction; no recovery dispatch."""
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def subject(p):return dict(path=str(p).replace('\\','/'),sha256=sha(p))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,value):
    with p.open('x',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
assert sha(BASE/'execution_request.json')=='c3e20cd213091bd9ec0db5a99871404ae609712de5be3e664b6bd844f62edea8'
assert sha(BASE/'frozen_inputs.json')=='954ac6190406c575964db094fdd69d9fb387482a5ba2f19ed289b2194f698ce1'
original=read(BASE/'launch_receipt.json');assert sha(BASE/'launch_receipt.json')=='7012ee94d666085f2c48256949a94f34392ccac6388e2ee928248c383ad536c0'
for p,h in original['input_sha256'].items():assert sha(Path(p))==h,p
probe=read(BASE/'argv_preflight_v2/report.json')
assert probe['passed'] and probe['raw_exit_code']==0 and probe['windows_child_absent'] and probe['powershell_version'].startswith('5.1.')
assert probe['wsl_arguments'][:-3]==original['wsl_arguments'][:-1]
assert all(v['rejected'] is True for v in probe['guard_tests']) and len(probe['guard_tests'])==3
assert probe['task_python_calls']==probe['model_calls']==probe['native_steps']==0
assert probe['linux_output']['cwd']=='/' and probe['linux_output']['argv']==['FIXED_ARG_ONE','FIXED_ARG_TWO']
fixed=(BASE/'run_recovery_durable_v2.ps1').read_text()
assert probe['actual_launcher_assignment'] in fixed
assert (BASE/'run_recovery_v2.sh').read_text()==(BASE/'run_recovery.sh').read_text().replace('recovery_process_v1','recovery_process_v2')
expected=(BASE/'verify_recovery_completed.py').read_text().replace("'recovery_process_v1'","'recovery_process_v2'").replace("'launch_receipt.json'","'launch_receipt_v2.json'").replace("'run_recovery_durable.ps1'","'run_recovery_durable_v2.ps1'").replace("'owner_completion.json'","'owner_completion_v2.json'")
assert expected==(BASE/'verify_recovery_completed_v2.py').read_text()
receipt=dict(original)
receipt['wsl_arguments']=list(original['wsl_arguments'])
receipt['wsl_arguments'][-1]='/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_width512_expert_recovery_v1/run_recovery_v2.sh'
receipt['launcher_sha256']=sha(BASE/'run_recovery_durable_v2.ps1')
add=['run_recovery_durable_v2.ps1','run_recovery_v2.sh','verify_recovery_completed_v2.py','prepare_launcher_revision_v2.py',
     'argv_probe_v2.sh','run_argv_probe_v2.ps1','argv_preflight_v2/report.json','argv_preflight_v2/stdout.log','argv_preflight_v2/stderr.log',
     'owner_completion.json','launch_preserved_v1/execution_clearance.json','launch_preserved_v1/owner_completion.json',
     'recovery_process_v1/dispatch.json','recovery_process_v1/start.json','recovery_process_v1/child.json','recovery_process_v1/exit.json',
     'recovery_process_v1/prerun_pins.json','recovery_process_v1/postrun_pins.json','recovery_process_v1/stderr.log',
     'recovery_process_v1/process_absence.json','recovery_process_v1/linux_absence_scan.json','freeze_launcher_revision_v2.py']
for name in add:receipt['input_sha256'][str(BASE/name)]=sha(BASE/name)
receipt['prebody_transport_revision']=dict(original_launch_receipt=subject(BASE/'launch_receipt.json'),
    original_exit=subject(BASE/'recovery_process_v1/exit.json'),original_owner=subject(BASE/'owner_completion.json'),
    actual_argv_preflight=subject(BASE/'argv_preflight_v2/report.json'),
    process_directory='recovery_process_v2',owner_filename='owner_completion_v2.json',
    main_sources_request_frozen_inputs_unchanged=True,task_work_repeated=False,actual_recovery_dispatched=False)
write(BASE/'launch_receipt_v2.json',receipt)
write(BASE/'launcher_revision_v2.json',dict(passed=True,source_only_transport_correction=True,
    launch_receipt=subject(BASE/'launch_receipt_v2.json'),launcher=subject(BASE/'run_recovery_durable_v2.ps1'),
    owner=subject(BASE/'verify_recovery_completed_v2.py'),shell=subject(BASE/'run_recovery_v2.sh'),
    request=subject(BASE/'execution_request.json'),frozen=subject(BASE/'frozen_inputs.json'),
    argv_preflight=subject(BASE/'argv_preflight_v2/report.json'),launch_pin_count=len(receipt['input_sha256']),
    no_actual_recovery_dispatched=True))
print(json.dumps(read(BASE/'launcher_revision_v2.json')))
