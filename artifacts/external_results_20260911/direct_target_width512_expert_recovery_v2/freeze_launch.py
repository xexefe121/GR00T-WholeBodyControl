"""Bind completed non-numerical path proof and fresh corrected supervisor."""
from prepare_metadata import BASE,OLD,sha,read,write,subject,norm

request=read(BASE/'execution_request.json')
frozen=read(BASE/'frozen_inputs.json')
proof=read(BASE/'path_preflight/report.json')
assert proof['passed'] and proof['actual_inherited_frozen_function_ast']
assert proof['request_sha256']==sha(BASE/'execution_request.json')
assert proof['frozen_receipt_sha256']==sha(BASE/'frozen_inputs.json')
assert proof['checked_inputs']==44 and proof['checked_sources']==20
assert read(BASE/'path_preflight/exit.json')['raw_exit_code']==0
for name in ('launcher_parse.json','bash_parse.json'):assert read(BASE/name)['passed']
old=read(OLD/'launch_receipt_v2.json')
args=[s.replace(OLD.name,BASE.name).replace('/run_recovery_v2.sh','/run_recovery.sh') for s in old['wsl_arguments']]
assert all(not any(c.isspace() for c in s) and '"' not in s and "'" not in s for s in args)
probe=read(OLD/'argv_preflight_v2/report.json')
assert probe['passed'] and probe['actual_launcher_assignment'] in (BASE/'run_recovery_durable.ps1').read_text()
pins=dict(frozen['input_sha256'])
for name,digest in frozen['source_sha256'].items():pins[norm(BASE/'source_snapshot_v1'/name)]=digest
paths=[BASE/name for name in ('execution_request.json','frozen_inputs.json','prepare_metadata.py','metadata_derivation.json',
    'check_inherited_paths.py','path_preflight/report.json','path_preflight/exit.json','path_preflight/stderr.log',
    'run_recovery_durable.ps1','verify_recovery_completed.py','run_recovery.sh','launcher_parse.json','bash_parse.json',
    'bash_parse_stderr.log','freeze_launch.py')]
paths += [OLD/'argv_preflight_v2/report.json',OLD/'run_recovery_durable_v2.ps1',OLD/'verify_recovery_completed_v2.py',
    OLD/'run_recovery_v2.sh',OLD/'launch_receipt_v2.json',
    OLD.parent/'direct_target_width512_expert_recovery_review_v1/transport_v2_review.json']
for p in paths:
    key=norm(p);digest=sha(p)
    if key in pins:assert pins[key]==digest
    pins[key]=digest
for name,digest in pins.items():assert '\\' not in name and sha(name)==digest,name
receipt=dict(kind='same_fixed_pre251_recovery_normalized_path_metadata',request_sha256=sha(BASE/'execution_request.json'),
    frozen_receipt_sha256=sha(BASE/'frozen_inputs.json'),launcher_sha256=sha(BASE/'run_recovery_durable.ps1'),
    wsl_arguments=args,input_sha256=pins,source_review_sha256=request['subjects']['source_review']['sha256'],
    dispatch_authorized=False,root_concrete_clearance_required=True,original_task20_sources_unchanged=True,
    input_extraction_repeated=False,previous_task_work_all_zero=True,
    actual_inherited_path_preflight=subject(BASE/'path_preflight/report.json'),
    previous_attempt_owner=subject(OLD/'owner_completion_v2.json'),
    process_directory='recovery_process_v1',owner_filename='owner_completion.json',clearance_filename='execution_clearance.json')
write(BASE/'launch_receipt.json',receipt)
result=dict(passed=True,request=subject(BASE/'execution_request.json'),frozen=subject(BASE/'frozen_inputs.json'),
    launch_receipt=subject(BASE/'launch_receipt.json'),launcher=subject(BASE/'run_recovery_durable.ps1'),
    owner=subject(BASE/'verify_recovery_completed.py'),shell=subject(BASE/'run_recovery.sh'),
    actual_inherited_path_preflight=subject(BASE/'path_preflight/report.json'),
    sources=20,frozen_inputs=44,launch_pins=len(pins),actual_task_dispatched=False)
write(BASE/'concrete_preparation.json',result)
print(__import__('json').dumps(result))
