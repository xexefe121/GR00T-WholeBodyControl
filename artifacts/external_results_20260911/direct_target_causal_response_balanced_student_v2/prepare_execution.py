"""Concrete metadata for one-line repaired namespace; never launches a fit."""
from pathlib import Path
import json,hashlib,shutil
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_response_balanced_student_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,value):
    with Path(p).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def subject(p,field):return dict(path=Path(p).as_posix(),sha256=sha(p),pass_field=field)

def prepare():
    for name in ('run_fit_durable_v1.ps1','read_fit_progress_v1.ps1','verify_completed.py','test_execution_helpers.py',
                 'execution_helper_tests.xml','test_launcher.ps1','launcher_tests.json'):
        destination=BASE/name
        if destination.exists():raise ValueError('Existing helper is immutable.')
        shutil.copyfile(OLD/name,destination)
    request=read(BASE/'training_request_proposal.json')
    request.update(root_selected=True,source_preparation_sha256=sha(BASE/'source_preparation.json'),
        pending_before_actual_fit=['root concrete one-line repair namespace request/launcher clearance'])
    request['subjects'].update(source_preparation=subject(BASE/'source_preparation.json','source_preparation_passed'),
        inherited_warm_source_review=subject(BASE.parent/'direct_target_response_warm_source_review_v1/review.json','warm_training_source_review_pass'),
        inherited_root_data_export_review=subject(BASE.parent/'direct_target_response_root_source_review_v1/review.json','data_export_source_review_pass'),
        root_zero_forward_failure_audit=subject(BASE.parent/'direct_target_response_initial_failure_root_v1/report.json','passed'))
    for key,item in request['subjects'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError('Changed request subject: '+key)
        if 'pass_field' in item:
            record=read(item['path'])
            if record[item['pass_field']] is not True:raise ValueError('Qualification failed: '+key)
            for name,value in item.get('required_fields',{}).items():
                if record.get(name)!=value:raise ValueError('Qualification binding differs: '+key+'/'+name)
    write(BASE/'training_request.json',request)

def freeze():
    if (BASE/'fit').exists() or (BASE/'training_clearance.json').exists():raise ValueError('Already executed/cleared.')
    prep=read(BASE/'source_preparation.json');request=read(BASE/'training_request.json')
    pins=dict(read(OLD/'training_frozen_inputs.json')['input_sha256'])
    for item in list(request['subjects'].values())+list(prep['subjects'].values()):pins[Path(item['path']).as_posix()]=item['sha256']
    helper_names=('run_fit_durable_v1.ps1','read_fit_progress_v1.ps1','verify_completed.py','test_execution_helpers.py',
                  'execution_helper_tests.xml','test_launcher.ps1','launcher_tests.json')
    for name in helper_names:
        if sha(BASE/name)!=sha(OLD/name):raise ValueError('Reviewed execution helper changed: '+name)
        pins[(BASE/name).as_posix()]=sha(BASE/name)
    pins[(BASE/'prepare_execution.py').as_posix()]=sha(__file__)
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Changed frozen input: '+path)
    snapshot=BASE/'source_snapshot_v1';snapshot.mkdir(exist_ok=False)
    for name,digest in prep['source_sha256'].items():
        source=Path(prep['source_directory'])/name
        if sha(source)!=digest:raise ValueError('Changed prepared source: '+name)
        shutil.copyfile(source,snapshot/name)
        if sha(snapshot/name)!=digest:raise ValueError('Source snapshot mismatch.')
    receipt=dict(source_directory=snapshot.as_posix(),source_sha256=prep['source_sha256'],input_sha256=pins,
        training_request_sha256=sha(BASE/'training_request.json'),all_inputs_rehashed=True,
        owner_checker='verify_completed.py',process_directory='fit_process_v1',prior_attempt_root=OLD.as_posix(),
        prior_forward_calls=0,prior_optimizer_updates=0,task_model_calls=0,optimizer_updates=0,native_steps=0)
    write(BASE/'training_frozen_inputs.json',receipt)
    clearance=dict(approved=False,request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        launcher_path=(BASE/'run_fit_durable_v1.ps1').as_posix(),launcher_sha256=sha(BASE/'run_fit_durable_v1.ps1'),
        condition='causal',updates=3000,ordinary_start_step=68000,ordinary_final_step=71000,
        optimizer_start_step=3000,optimizer_final_step=6000,automatic_retry=False,
        process_directory='fit_process_v1',review_pass_field='prelaunch_review_pass')
    write(BASE/'training_clearance_draft.json',clearance)
    print(json.dumps(dict(request_sha256=clearance['request_sha256'],frozen_receipt_sha256=clearance['frozen_receipt_sha256'],
        launcher_sha256=clearance['launcher_sha256'],inputs=len(pins),sources=len(prep['source_sha256']))))
if __name__=='__main__':
    import sys
    if sys.argv[1:]==['prepare']:prepare()
    elif sys.argv[1:]==['freeze']:freeze()
    else:raise SystemExit('Use prepare or freeze; no model/dispatch.')
