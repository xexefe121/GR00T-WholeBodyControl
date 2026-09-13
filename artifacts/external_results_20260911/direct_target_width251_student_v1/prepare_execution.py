"""JSON-only admission proof and one immutable fit packet. Never dispatches."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
BASE=Path(__file__).resolve().parent
NEW=BASE.parent
OLD=NEW/'direct_target_causal_width512_student_v1'
COLLECT=NEW/'direct_target_width251_collection_v1/actual_v1'
CONSISTENCY=NEW/'direct_target_width251_consistency_v1/actual_v1/results_v1/report.json'
SOURCE_REVIEW='15731f7dedf2cf3e05b1c331b519745c756d1e82a70ec727af4c75f8a024beec'
SELECTED_PROTOCOL='1cade1a88df2324511ca2b9599b69261e9a9b7a72ba34c540b09b8ac10896128'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as stream:json.dump(v,stream,indent=2,allow_nan=False);stream.write('\n')
def pathstr(p):return Path(p).as_posix()
def merge_pin(pins,p,digest):
    p=pathstr(p);key=p.casefold()
    for old,value in pins.items():
        if old.casefold()==key:
            if value!=digest:raise ValueError('Conflicting physical pin: '+p)
            return
    pins[p]=digest
def json_subject(p,field=None):
    if Path(p).suffix.lower()!='.json':raise ValueError('JSON subject required here')
    item=dict(path=pathstr(p),sha256=sha(p))
    if field:
        if read(p).get(field) is not True:raise ValueError('Positive JSON field required: '+str(p)+'/'+field)
        item['pass_field']=field
    return item
def declared_subject(p,digest):return dict(path=pathstr(p),sha256=digest)
def protocol_module():
    source=BASE/'source_prepared_v1'
    sys.path.insert(0,str(source))
    try:
        import recovery_protocol
        return recovery_protocol
    finally:sys.path.pop(0)
def member(record,item):
    matches=[v for k,v in record.items() if pathstr(k).casefold()==item['path'].casefold()]
    if not matches or any(v!=item['sha256'] for v in matches):raise ValueError('Missing exact literal JSON input member: '+item['path'])

def build_request(protocol_path,review_path):
    if sha(protocol_path)!=SELECTED_PROTOCOL or sha(review_path)!=SOURCE_REVIEW:raise ValueError('Use exact root-selected protocol/source review')
    config=read(protocol_path);review=read(review_path);prep=read(BASE/'source_preparation.json')
    if config['configuration_selected'] is not True or config['actual_fit_execution_selected'] is not False:raise ValueError('Configuration-only selection required')
    if review['source_review_pass'] is not True or review['source_preparation_sha256']!=sha(BASE/'source_preparation.json') or review['source_sha256']!=prep['source_sha256']:raise ValueError('Prepared source map not cleared')
    old=read(OLD/'training_request.json');source_report=read(OLD/'fit/report.json');manifest=read(OLD/'fit/shared/output_manifest.json')
    fit_manifest=read(OLD/'fit/output_manifest.json');collection=read(COLLECT/'results_v1/report.json')
    request={k:copy.deepcopy(old[k]) for k in ('paths','full_state_paths','runtime')}
    inherit=('coefficient_source','context_proof','energy_source','balance_source_review','full_state_root_audit','full_state_root_owner','full_state_data_review')
    subjects={k:copy.deepcopy(old['subjects'][k]) for k in inherit}
    for item in subjects.values():item['path']=pathstr(item['path'])
    for key,path,field in [
        ('fit_report',OLD/'fit/report.json','completed'),('fit_owner',OLD/'owner_completion_verification.json','completion_passed'),
        ('fit_audit',NEW/'direct_target_width512_fit_independent_v1/results_v1/report.json','evidence_audit_passed'),
        ('context_manifest',OLD/'fit/shared/output_manifest.json',None),('context_alignment',OLD/'fit/shared/context_alignment.json',None),
        ('semantics_review',NEW/'direct_target_width512_saved_semantics_review_v1/results_v1/report.json','passed'),
        ('semantics_owner',NEW/'direct_target_width512_saved_semantics_review_v1/owner_completion.json','completion_accounting_passed'),
        ('collection_report',COLLECT/'results_v1/report.json','passed'),('collection_request',COLLECT/'request.json',None),
        ('collection_qualification',COLLECT/'qualification.json',None),
        ('collection_source_review',NEW/'direct_target_width251_collection_root_review_v1/review.json','source_review_pass'),
        ('consistency_report',CONSISTENCY,'passed'),
        ('warm_restore_review',NEW/'direct_target_width251_warm_restore_v1/root_review.json','source_review_pass'),
        ('source_preparation',BASE/'source_preparation.json','source_preparation_pass'),('source_review',review_path,'source_review_pass'),
        ('independent_source_review',NEW/'direct_target_width251_fit_independent_review_v1/review.json','source_review_pass'),
        ('selected_protocol',protocol_path,'configuration_selected')]:subjects[key]=json_subject(path,field)
    subjects['checkpoint']=declared_subject(OLD/'fit/student_head.pt',source_report['checkpoint_sha256'])
    subjects['normalization']=declared_subject(OLD/'fit/shared/normalization.npz',manifest['files']['normalization.npz'])
    subjects['recovery_rows']=declared_subject(COLLECT/'results_v1/expert_rows.npz',collection['outputs']['expert_rows.npz'])
    request['subjects']=subjects
    shared=OLD/'fit/shared'
    request['context_paths']={k:pathstr(shared/name) for k,name in [
        ('normalization','normalization.npz'),('manifest','output_manifest.json'),('nominal_context','nominal_context.npy'),
        ('center_context','center_context.npy'),('physical_context','physical_context.npy'),('context_alignment','context_alignment.json'),
        ('data_identities','data_identities.npz'),('schedule_centers','source_schedule_centers.npy'),('schedule_axes','source_schedule_axes.npy')]}
    request['schedule_paths']={k:pathstr(shared/(k+'.npy')) for k in ('schedule_centers','schedule_axes')}
    request['restoration_predictions']={k:pathstr(OLD/'fit'/('final_GPU32_'+k+'.npy')) for k in ('nominal','full_state','physical')}
    for group in ('paths','full_state_paths'):
        request[group]={k:pathstr(v) for k,v in request[group].items()}
    module=protocol_module()
    request.update(updates=config['updates'],learning_rate_values=config['learning_rate_values'],recovery_coefficient=config['recovery_coefficient'])
    p=module.protocol(request)
    request.update(kind='qualified_width251_recovery_warm512_fit',root_selected=True,execution_requires_concrete_clearance=True,
        condition='causal',ordinary_start_step=81000,ordinary_final_step=p.final_step,optimizer_start_step=16000,optimizer_final_step=p.optimizer_final,
        fresh_optimizer=False,coefficient=module.COEFFICIENT,coefficient_recalibration=False,weight_decay=1e-5,gradient_clip=10.,
        features=1323,architecture=[1323,512,512,23],context_order='previous_action23_then_incoming_history300',
        first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323',
        initial_parity_tolerance_rad=1e-5,initial_byte_gate_required=False,parity_tolerance_rad=1e-5,
        automatic_retry=False,no_checkpoint_selection=True,context_and_normalization_reused=True,expansion_performed=False,
        recovery_rows=1018,recovery_phase_counts=[99,819,100],recovery_objective='equal_three_phase_normalized_MSE',
        response_schedule='unchanged10000_prefix_no_wrap',group_weights=list(module.GROUP_WEIGHTS),consistency_evidence_reviewed=True,
        budgets=p.budgets,source_preparation_sha256=sha(BASE/'source_preparation.json'))
    module.check_protocol(request)
    return request

def prove(request):
    """Only JSON and small source reads; never hashes/opens task binary arrays."""
    checks=[];s=request['subjects'];shared=read(s['context_manifest']['path'])
    report=read(s['fit_report']['path']);owner=read(s['fit_owner']['path']);audit=read(s['fit_audit']['path'])
    def check(name,value):
        if not value:raise ValueError('JSON lineage: '+name)
        checks.append(name)
    check('source81000_step16000',report['completed'] is True and report['ordinary_final_step']==81000 and report['optimizer_step']==16000)
    check('source_checkpoint',report['checkpoint_sha256']==s['checkpoint']['sha256']==protocol_module().SOURCE_CHECKPOINT)
    check('owner_numerical_accounting',all(owner[k] is True for k in ('owner_verification_passed','accounting_passed','completion_passed','numerical_completion_passed','processes_absent','all_postrun_pins_exact','raw_exit_known')) and owner['raw_python_exit_code']==owner['exit_code']==0)
    check('audit_qualified',audit['evidence_audit_passed'] is True and audit['export_qualified'] is True)
    for key in ('fit_report','checkpoint','normalization','context_manifest','context_alignment'):
        member(audit['input_sha256'],s[key]);checks.append('audit_literal_'+key)
    for key,role in [('checkpoint','checkpoint'),('fit_report','fit_report'),('normalization','normalization'),('context_manifest','shared_manifest'),('context_alignment','context_alignment')]:
        check('owner_'+role,owner['direct_subject_sha256'][role]==s[key]['sha256'])
    for key,path in request['context_paths'].items():
        if key!='manifest':check('shared_member_'+key,Path(path).name in shared['files'])
    for key,path in request['schedule_paths'].items():check('full_schedule_member_'+key,Path(path).name in shared['files'])
    old_manifest=read(OLD/'fit/output_manifest.json')
    for key,path in request['restoration_predictions'].items():check('prediction_manifest_'+key,Path(path).name in old_manifest['files'])
    consistency=read(s['consistency_report']['path']);collection=read(s['collection_report']['path']);crequest=read(s['collection_request']['path'])
    check('consistency_scope',consistency['passed'] is True and consistency['evidence_diagnosis_completed'] is True and consistency['old_rows']==12958 and consistency['new_rows']==1018 and consistency['rows']==13976)
    for role in ('collection_report','collection_request','collection_qualification','collection_source_review','normalization'):
        member(consistency['input_sha256'],s[role]);checks.append('consistency_literal_'+role)
    check('collection_scope',collection['passed'] is True and collection['collection_completed'] is True and collection['rows']==1018 and collection['control_start']==251 and collection['control_stop_exclusive']==1269 and collection['model_fitting_authorized'] is False)
    check('collection_request_binding',collection['request_sha256']==s['collection_request']['sha256'])
    for role,key in [('collection_qualification','qualification'),('collection_source_review','source_review')]:
        item=crequest['subjects'][key];check('collection_role_'+role,pathstr(item['path']).casefold()==s[role]['path'].casefold() and item['sha256']==s[role]['sha256'])
    check('collection_rows',collection['outputs']['expert_rows.npz']==s['recovery_rows']['sha256'])
    check('collection_norm',collection['outputs']['normalization.npz']==s['normalization']['sha256'])
    check('own_norm_path',request['context_paths']['normalization']==s['normalization']['path']==pathstr(OLD/'fit/shared/normalization.npz'))
    for role,item in s.items():
        if Path(item['path']).suffix=='.json':
            check('json_sha_'+role,sha(item['path'])==item['sha256'])
            if 'pass_field' in item:check('json_pass_'+role,read(item['path'])[item['pass_field']] is True)
    return dict(passed=True,checks=checks,check_count=len(checks),task_checkpoint_loads=0,task_array_loads=0,model_calls=0,native_steps=0,
        request_candidate=request,source_sha256=read(BASE/'source_preparation.json')['source_sha256'])

def freeze(request):
    for name in ('training_request.json','training_frozen_inputs.json','launch_receipt.json','source_snapshot_v1','fit','training_clearance.json','fit_process_v1'):
        if (BASE/name).exists():raise ValueError('Preserve existing packet/attempt: '+name)
    preparation=read(BASE/'execution_helper_preparation.json')
    if preparation['source_preparation_pass'] is not True:raise ValueError('Helper source evidence absent')
    for name,digest in preparation['helper_sha256'].items():
        if sha(BASE/name)!=digest:raise ValueError('Helper changed: '+name)
    proof=prove(request)
    pins={}
    for path,digest in read(OLD/'training_frozen_inputs.json')['input_sha256'].items():merge_pin(pins,path,digest)
    for item in request['subjects'].values():merge_pin(pins,item['path'],item['sha256'])
    shared=read(OLD/'fit/shared/output_manifest.json');outputs=read(OLD/'fit/output_manifest.json')
    for group in ('paths','full_state_paths','restoration_predictions','context_paths','schedule_paths'):
        for value in request[group].values():merge_pin(pins,value,sha(value))
    prep=read(BASE/'source_preparation.json')
    for item in prep.get('subjects',{}).values():merge_pin(pins,item['path'],item['sha256'])
    for name in list(preparation['helper_sha256'])+['execution_helper_preparation.json']:
        merge_pin(pins,BASE/name,sha(BASE/name))
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Changed actual input: '+path)
    snapshot=BASE/'source_snapshot_v1';snapshot.mkdir()
    for name,digest in prep['source_sha256'].items():
        source=Path(prep['source_directory'])/name
        if sha(source)!=digest:raise ValueError('Prepared source changed: '+name)
        shutil.copyfile(source,snapshot/name)
        if sha(snapshot/name)!=digest:raise ValueError('Snapshot differs: '+name)
    write(BASE/'training_request.json',request)
    receipt=dict(source_directory=pathstr(snapshot),source_sha256=prep['source_sha256'],input_sha256=pins,
        training_request_sha256=sha(BASE/'training_request.json'),all_inputs_rehashed=True,owner_checker='verify_completed.py',
        process_directory='fit_process_v1',warm_source_root=pathstr(OLD),model_calls=0,optimizer_updates=0,native_steps=0)
    write(BASE/'training_frozen_inputs.json',receipt)
    launchpins=dict(pins)
    for name,digest in prep['source_sha256'].items():merge_pin(launchpins,snapshot/name,digest)
    for name in ('training_request.json','training_frozen_inputs.json'):merge_pin(launchpins,BASE/name,sha(BASE/name))
    launch=dict(request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        launcher_path=pathstr(BASE/'run_fit_durable_v1.ps1'),launcher_sha256=sha(BASE/'run_fit_durable_v1.ps1'),
        input_sha256=launchpins,pin_count=len(launchpins),process_directory='fit_process_v1',automatic_retry=False,
        mandatory_arguments=['ClearanceSha256','LaunchReceiptSha256'],model_calls=0,native_steps=0)
    write(BASE/'launch_receipt.json',launch)
    write(BASE/'concrete_preparation.json',dict(passed=True,request_sha256=launch['request_sha256'],frozen_receipt_sha256=launch['frozen_receipt_sha256'],
        launch_receipt_sha256=sha(BASE/'launch_receipt.json'),launcher_sha256=launch['launcher_sha256'],inputs=len(pins),sources=len(prep['source_sha256']),launch_pin_count=len(launchpins),
        final_process_pin_count=len(launchpins)+3,configuration_selected=True,actual_dispatch=False,model_calls=0,native_steps=0,metadata_proof=proof))
    print(json.dumps(read(BASE/'concrete_preparation.json')|{'metadata_proof':'saved in receipt'}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prove','freeze']);parser.add_argument('--protocol',required=True);parser.add_argument('--review',required=True);args=parser.parse_args()
    request=build_request(args.protocol,args.review)
    if args.mode=='prove':
        result=prove(request);write(BASE/'metadata_schema_proof.json',result);print(json.dumps(dict(passed=True,checks=result['check_count'])))
    else:freeze(request)
