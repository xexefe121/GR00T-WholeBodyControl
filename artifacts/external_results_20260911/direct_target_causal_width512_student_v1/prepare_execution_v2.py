"""Selected concrete width512 request/freeze only; never starts a model or fit."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
OLD=NEW/'direct_target_causal_response_balanced_student_v2'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as stream:json.dump(v,stream,indent=2,allow_nan=False);stream.write('\n')
def subject(p,field=None):
    result=dict(path=Path(p).as_posix(),sha256=sha(p))
    if field:result['pass_field']=field
    return result

def prepare(review_path):
    review_path=Path(review_path)
    prep=read(BASE/'source_preparation.json');review=read(review_path)
    if review['source_review_pass'] is not True:raise ValueError('Independent width source review absent.')
    request=read(BASE/'training_request_proposal.json')
    request.update(root_selected=True,source_preparation_sha256=sha(BASE/'source_preparation.json'),
        pending_before_actual_fit=['root concrete selected width512 request/source/runtime/launcher clearance'])
    subjects=request['subjects']
    subjects['inherited_semantics_review']=subjects['semantics_review']
    for name,path,field in [
        ('source_preparation',BASE/'source_preparation.json','source_preparation_passed'),
        ('source_review',review_path,'source_review_pass'),
        ('width_export_source_review',NEW/'direct_target_width512_export_root_review_v1/review.json','source_review_pass'),
        ('width_export_preparation',NEW/'direct_target_width512_export_preparation_v1/source_preparation.json','source_preparation_passed'),
        ('saved_schedule_proof',BASE/'saved_schedule_proof.json','passed'),
        ('semantics_review',NEW/'direct_target_response_saved_semantics_review_v1/results_v1/report.json','passed'),
        ('semantics_owner',NEW/'direct_target_response_saved_semantics_review_v1/owner_completion.json','completion_accounting_passed')]:
        subjects[name]=subject(path,field)
    subjects['semantics_review']['required_fields']=dict(context_condition='causal',features=1323,moving_controls=38)
    for key in ('previous_zero_forward_attempt','previous_forward_calls','previous_optimizer_updates','metadata_repair'):
        if key in request:request['inherited_'+key]=request.pop(key)
    sys.path.insert(0,prep['source_directory'])
    # Protocol module imports Torch through width512, but constructs no model and
    # initializes no CUDA. Use AST-compiled pure check_protocol with constants
    # supplied directly to avoid even that import in metadata preparation.
    import ast
    tree=ast.parse((Path(prep['source_directory'])/'response_contract.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_protocol')
    namespace=dict(UPDATES=10000,START_STEP=71000,FINAL_STEP=81000,OPTIMIZER_START=6000,OPTIMIZER_FINAL=16000,
        COEFFICIENT=1.8188207859141674,BUDGETS=request['budgets'],
        CHECKPOINT_SHA256='395aabf879c0a40b5187a4265ae850de49cfb62583e144145c1986d65402143d')
    exec(compile(ast.Module(body=[function],type_ignores=[]),'frozen_protocol','exec'),namespace)
    namespace['check_protocol'](request)
    for key,item in subjects.items():
        if sha(item['path'])!=item['sha256']:raise ValueError('Changed request subject: '+key)
        if 'pass_field' in item:
            record=read(item['path'])
            if record[item['pass_field']] is not True:raise ValueError('Qualification failed: '+key)
            for name,value in item.get('required_fields',{}).items():
                if record.get(name)!=value:raise ValueError('Qualification binding differs: '+key+'/'+name)
    write(BASE/'training_request.json',request)
    print(json.dumps(dict(training_request_sha256=sha(BASE/'training_request.json'))))

def freeze():
    if (BASE/'fit').exists() or (BASE/'training_clearance.json').exists():raise ValueError('Already executed/cleared.')
    prep=read(BASE/'source_preparation.json');request=read(BASE/'training_request.json')
    pins=dict(read(OLD/'training_frozen_inputs.json')['input_sha256'])
    for item in list(request['subjects'].values())+list(prep['subjects'].values()):pins[Path(item['path']).as_posix()]=item['sha256']
    for group in ('paths','full_state_paths','restoration_predictions','context_paths','schedule_paths'):
        for path in request[group].values():pins[Path(path).as_posix()]=sha(path)
    helper_names=('run_fit_durable_v1.ps1','read_fit_progress_v1.ps1','verify_completed.py','test_execution_helpers.py',
                  'execution_helper_tests.xml','test_launcher.ps1','launcher_tests.json','prepare_execution_v2.py')
    for name in helper_names:pins[(BASE/name).as_posix()]=sha(BASE/name)
    for path,digest in read(BASE/'saved_schedule_proof.json')['input_sha256'].items():pins[path]=digest
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
        owner_checker='verify_completed.py',process_directory='fit_process_v1',warm_source_root=OLD.as_posix(),
        task_model_calls=0,optimizer_updates=0,native_steps=0)
    write(BASE/'training_frozen_inputs.json',receipt)
    clearance=dict(approved=False,request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        launcher_path=(BASE/'run_fit_durable_v1.ps1').as_posix(),launcher_sha256=sha(BASE/'run_fit_durable_v1.ps1'),
        condition='causal',updates=10000,ordinary_start_step=71000,ordinary_final_step=81000,
        optimizer_start_step=6000,optimizer_final_step=16000,automatic_retry=False,
        hidden_width=512,expansion_seed=20260912,process_directory='fit_process_v1',review_pass_field='prelaunch_review_pass')
    write(BASE/'training_clearance_draft.json',clearance)
    print(json.dumps(dict(request_sha256=clearance['request_sha256'],frozen_receipt_sha256=clearance['frozen_receipt_sha256'],
        launcher_sha256=clearance['launcher_sha256'],inputs=len(pins),sources=len(prep['source_sha256']))))

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='prepare':prepare(sys.argv[2])
    elif sys.argv[1:]==['freeze']:freeze()
    else:raise SystemExit('Use prepare <independent-review> or freeze; no dispatch.')
