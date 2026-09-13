"""Preserve zero-task-call failed launcher; prepare one separately selected repair."""
import json
import shutil
from prepare_execution import BASE,sha,read,write
def prepare():
    failure=read(BASE/'fit_process/exit.json')
    if failure['child_started'] is not False or failure['child_pid'] is not None or failure['raw_python_exit_code'] is not None or failure['error']!='Unexpected request.':raise ValueError('Unexpected first attempt state.')
    if (BASE/'fit').exists():raise ValueError('Task outputs exist; cannot claim zero execution.')
    original=BASE/'run_fit_durable_v1.ps1';text=original.read_text(encoding='utf-8')
    changes=[("$processRoot=Join-Path $runRoot 'fit_process'","$processRoot=Join-Path $runRoot 'fit_process_v2'"),
        ('$request.coefficient -ne 1.8188207859141674)', '$request.coefficient -ne 1.8188207859141674d)')]
    for old,new in changes:
        if text.count(old)!=1:raise ValueError('Launcher source differs.')
        text=text.replace(old,new)
    with (BASE/'run_fit_durable_v2.ps1').open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    owner=(BASE/'verify_completed_v2.py').read_text(encoding='utf-8')
    if owner.count("process=BASE/'fit_process'")!=1:raise ValueError('Owner process source differs.')
    owner=owner.replace("process=BASE/'fit_process'","process=BASE/'fit_process_v2'").replace("owner_completion_verification_v2.json","owner_completion_verification_v3.json")
    with (BASE/'verify_completed_v3.py').open('x',encoding='utf-8',newline='\n') as f:f.write(owner)
    monitor=(BASE/'read_fit_progress.ps1').read_text(encoding='utf-8').replace('fit_process/','fit_process_v2/')
    with (BASE/'read_fit_progress_v2.ps1').open('x',encoding='utf-8',newline='\n') as f:f.write(monitor)
    write(BASE/'launcher_v2_derivation.json',dict(original_launcher_sha256=sha(original),launcher_sha256=sha(BASE/'run_fit_durable_v2.ps1'),
        substitutions=changes,original_owner_sha256=sha(BASE/'verify_completed_v2.py'),owner_sha256=sha(BASE/'verify_completed_v3.py'),
        owner_only_changes=['fit_process -> fit_process_v2','owner_completion_verification_v2.json -> owner_completion_verification_v3.json'],
        original_exit_sha256=sha(BASE/'fit_process/exit.json'),first_attempt_task_model_calls=0,first_attempt_optimizer_updates=0,
        request_and_trainer_unchanged=True,automatic_retry=False,actual_relaunch_selected=False))
def freeze():
    evidence=read(BASE/'launcher_v2_test.json')
    if evidence['passed'] is not True:raise ValueError('Actual PS5.1 check failed.')
    preserved=BASE/'failed_launch_attempt_v1';preserved.mkdir(exist_ok=False)
    for name in ('training_frozen_inputs.json','training_clearance_draft.json','training_clearance.json','dispatch.json','dispatch_process.json'):
        shutil.copyfile(BASE/name,preserved/name)
    receipt=read(BASE/'training_frozen_inputs.json')
    for name in ('run_fit_durable_v2.ps1','verify_completed_v3.py','read_fit_progress_v2.ps1','prepare_launcher_v2.py','launcher_v2_derivation.json','launcher_v2_test.json','test_launcher_v2.ps1','fit_process/exit.json'):
        receipt['input_sha256'][(BASE/name).as_posix()]=sha(BASE/name)
    receipt.update(owner_checker='verify_completed_v3.py',process_directory='fit_process_v2',preserved_failed_launch_receipt_sha256=sha(preserved/'training_frozen_inputs.json'))
    for p,digest in receipt['input_sha256'].items():
        if sha(p)!=digest:raise ValueError('Changed pin: '+p)
    with (BASE/'training_frozen_inputs.json').open('w',encoding='utf-8') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    clear=read(BASE/'training_clearance_draft.json')
    clear.update(frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),launcher_path=(BASE/'run_fit_durable_v2.ps1').as_posix(),launcher_sha256=sha(BASE/'run_fit_durable_v2.ps1'),process_directory='fit_process_v2',attempt=2,prior_task_model_calls=0)
    with (BASE/'training_clearance_draft.json').open('w',encoding='utf-8') as f:json.dump(clear,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),launcher_sha256=sha(BASE/'run_fit_durable_v2.ps1'),inputs=len(receipt['input_sha256']),sources=len(receipt['source_sha256']))))
if __name__=='__main__':
    import sys
    if sys.argv[1:]==['prepare']:prepare()
    elif sys.argv[1:]==['freeze']:freeze()
    else:raise SystemExit('Use prepare or freeze; no task execution.')
