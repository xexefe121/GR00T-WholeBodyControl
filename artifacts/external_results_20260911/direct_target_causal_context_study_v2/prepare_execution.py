"""Concrete metadata only for selected split study; no model or dispatch."""
import hashlib,json,shutil
from pathlib import Path
BASE=Path(__file__).resolve().parent;OLD=BASE.parent/'direct_target_causal_context_study_v1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def item(path,**extra):return dict(path=Path(path).as_posix(),sha256=sha(path),**extra)
def prepare():
    for name in ('run_fit_durable_v2.ps1','verify_completed_v3.py','read_fit_progress_v2.ps1','test_launcher_v2.ps1'):
        destination=BASE/name
        if destination.exists():raise ValueError('Existing helper.')
        shutil.copyfile(OLD/name,destination)
    proposal=read(BASE/'training_request_proposal.json');proposal.update(root_selected=True,
        source_preparation_sha256=sha(BASE/'source_preparation.json'),
        pending_before_actual_fit=['root concrete split-study request/launcher clearance'])
    proposal['subjects'].update(
        context_preparation=item(BASE/'source_preparation.json',pass_field='source_preparation_passed'),
        split_algebra_review=item(BASE.parent/'direct_target_context_split_independent_review_v1/review.json',pass_field='preparation_review_passed'),
        failed_initial_root_audit=item(BASE.parent/'direct_target_context_initial_failure_root_v1/report.json'),
        failed_initial_owner=item(OLD/'owner_completion_verification_v3.json',pass_field='accounting_passed',required_fields=dict(paired_completion_passed=False)))
    write(BASE/'training_request.json',proposal)
def freeze():
    if (BASE/'fit').exists() or (BASE/'training_clearance.json').exists():raise ValueError('Already executed/cleared.')
    test=read(BASE/'launcher_v2_test.json')
    if test['passed'] is not True or test['actual_corrected_guard_passed'] is not True:raise ValueError('Pinned PS5.1 guard test failed.')
    prep=read(BASE/'source_preparation.json');request=read(BASE/'training_request.json')
    pins=dict(read(OLD/'training_frozen_inputs.json')['input_sha256'])
    for subject in list(request['subjects'].values())+list(prep['subjects'].values()):pins[subject['path']]=subject['sha256']
    for name in ('run_fit_durable_v2.ps1','verify_completed_v3.py','read_fit_progress_v2.ps1','test_launcher_v2.ps1','launcher_v2_test.json','prepare_execution.py'):
        pins[(BASE/name).as_posix()]=sha(BASE/name)
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Changed pin: '+path)
    source=BASE/'source_snapshot_v1';source.mkdir(exist_ok=False)
    for name,digest in prep['source_sha256'].items():
        original=Path(prep['source_directory'])/name
        if sha(original)!=digest:raise ValueError('Changed prepared source.')
        shutil.copyfile(original,source/name)
        if sha(source/name)!=digest:raise ValueError('Source copy differs.')
    receipt=dict(source_directory=source.as_posix(),source_sha256=prep['source_sha256'],input_sha256=pins,
        training_request_sha256=sha(BASE/'training_request.json'),all_inputs_rehashed=True,owner_checker='verify_completed_v3.py',
        process_directory='fit_process_v2',prior_attempt_root=OLD.as_posix(),prior_initial_model_calls=1437,prior_initial_model_rows=367570,
        prior_optimizer_updates=0,task_model_calls=0,optimizer_updates=0,native_steps=0)
    write(BASE/'training_frozen_inputs.json',receipt)
    clear=dict(approved=False,request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        launcher_path=(BASE/'run_fit_durable_v2.ps1').as_posix(),launcher_sha256=sha(BASE/'run_fit_durable_v2.ps1'),
        conditions=['blinded','causal'],updates_per_condition=3000,ordinary_final_step=68000,automatic_retry=False,
        process_directory='fit_process_v2',review_pass_field='prelaunch_review_pass')
    write(BASE/'training_clearance_draft.json',clear)
    print(json.dumps(dict(request_sha256=clear['request_sha256'],frozen_receipt_sha256=clear['frozen_receipt_sha256'],launcher_sha256=clear['launcher_sha256'],inputs=len(pins),sources=len(prep['source_sha256']))))
if __name__=='__main__':
    import sys
    if sys.argv[1:]==['prepare']:prepare()
    elif sys.argv[1:]==['freeze']:freeze()
    else:raise SystemExit('Use prepare or freeze; no model/dispatch.')
