"""Prepare immutable request, source snapshot and launch pins; never dispatch."""
from pathlib import Path
import hashlib,json,shutil,xml.etree.ElementTree as ET
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_context_study_v2'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def prepare():
    prep=read(BASE/'source_preparation.json')
    review_path=BASE.parent/'direct_target_response_warm_source_review_v1/review.json'
    review=read(review_path)
    if review['source_review_pass'] is not True or review['warm_training_source_review_pass'] is not True:
        raise ValueError('Independent source-only review absent.')
    proposal=read(BASE/'training_request_proposal.json')
    proposal.update(root_selected=True,source_preparation_sha256=sha(BASE/'source_preparation.json'),
        pending_before_actual_fit=['root data/export/concrete request and launcher clearance'])
    proposal['subjects']['warm_source_review']=dict(path=review_path.as_posix(),sha256=sha(review_path),
        pass_field='warm_training_source_review_pass',required_fields={'source_review_pass':True})
    proposal['subjects']['source_preparation']=dict(path=(BASE/'source_preparation.json').as_posix(),
        sha256=sha(BASE/'source_preparation.json'),pass_field='source_preparation_passed')
    root_review=BASE.parent/'direct_target_response_root_source_review_v1/review.json'
    proposal['subjects']['root_source_data_export_review']=dict(path=root_review.as_posix(),sha256=sha(root_review),
        pass_field='passed')
    write(BASE/'training_request.json',proposal)

def freeze():
    if (BASE/'fit').exists() or (BASE/'training_clearance.json').exists():raise ValueError('Existing fit/clearance must remain untouched.')
    if read(BASE/'launcher_tests.json')['passed'] is not True:raise ValueError('Actual PowerShell guard test failed.')
    tests=ET.parse(BASE/'execution_helper_tests.xml').getroot()
    suites=[tests] if tests.tag=='testsuite' else list(tests.iter('testsuite'))
    if sum(int(s.get('tests',0)) for s in suites)!=8 or any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped')):
        raise ValueError('Execution-helper tests failed.')
    prep=read(BASE/'source_preparation.json');request=read(BASE/'training_request.json')
    pins={Path(path).as_posix():digest for path,digest in read(OLD/'training_frozen_inputs.json')['input_sha256'].items()}
    for item in list(request['subjects'].values())+list(prep['subjects'].values()):pins[Path(item['path']).as_posix()]=item['sha256']
    for group in ('paths','full_state_paths','restoration_predictions','context_paths'):
        for path in request[group].values():
            canonical=Path(path).as_posix()
            if canonical not in pins:pins[canonical]=sha(path)
    for name in ('run_fit_durable_v1.ps1','read_fit_progress_v1.ps1','verify_completed.py','test_execution_helpers.py',
                 'execution_helper_tests.xml','test_launcher.ps1','launcher_tests.json','derive_execution.py','prepare_execution.py','EXECUTION_SCHEMA.md'):
        pins[(BASE/name).as_posix()]=sha(BASE/name)
    # The old complete input map covers unchanged corpora/runtime; every current value is rechecked.
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Frozen input changed: '+path)
    snapshot=BASE/'source_snapshot_v1';snapshot.mkdir(exist_ok=False)
    for name,digest in prep['source_sha256'].items():
        source=Path(prep['source_directory'])/name
        if sha(source)!=digest:raise ValueError('Reviewed prepared source changed.')
        shutil.copyfile(source,snapshot/name)
        if sha(snapshot/name)!=digest:raise ValueError('Source snapshot copy differs.')
    receipt=dict(source_directory=snapshot.as_posix(),source_sha256=prep['source_sha256'],input_sha256=pins,
        training_request_sha256=sha(BASE/'training_request.json'),all_inputs_rehashed=True,
        owner_checker='verify_completed.py',process_directory='fit_process_v1',task_model_calls=0,optimizer_updates=0,native_steps=0)
    write(BASE/'training_frozen_inputs.json',receipt)
    clearance=dict(approved=False,request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        launcher_path=(BASE/'run_fit_durable_v1.ps1').as_posix(),launcher_sha256=sha(BASE/'run_fit_durable_v1.ps1'),
        condition='causal',updates=3000,ordinary_start_step=68000,ordinary_final_step=71000,
        optimizer_start_step=3000,optimizer_final_step=6000,automatic_retry=False,
        process_directory='fit_process_v1',review_pass_field='prelaunch_review_pass')
    write(BASE/'training_clearance_draft.json',clearance)
    print(json.dumps(dict(request_sha256=clearance['request_sha256'],frozen_receipt_sha256=clearance['frozen_receipt_sha256'],
        launcher_sha256=clearance['launcher_sha256'],input_count=len(pins),source_count=len(prep['source_sha256']))))

if __name__=='__main__':
    import sys
    if sys.argv[1:]==['prepare']:prepare()
    elif sys.argv[1:]==['freeze']:freeze()
    else:raise SystemExit('Use prepare or freeze. No task model call or dispatch.')
