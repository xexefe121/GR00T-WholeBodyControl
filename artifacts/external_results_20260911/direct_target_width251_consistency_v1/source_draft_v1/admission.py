"""Exact saved-data provenance gate. No numerical arrays are opened here."""
import hashlib
import json
import sys
from pathlib import Path

PHYSICAL_KEYS=('dataset','start_control','successor_control','endpoint_features','label_fixed_map_target',
    'policy_actual_normalized_action','advanced_history','policy_applied_target','teacher_feedback_clipped','teacher_native_clipped')
ROLES=('old_centers','old_pico','old_walk002','physical_manifest','old_nominal_context','old_physical_context',
    'normalization','contract','old_shared_manifest','old_context_alignment','old_fit_report','old_fit_owner',
    'old_training_request','old_training_manifest','old_fit_audit','new_rows','collection_report','collection_request',
    'collection_qualification','collection_source_review','source_review','export_source')

def local(path):
    value=str(path).replace('\\','/')
    if sys.platform!='win32' and len(value)>2 and value[1]==':':value='/mnt/'+value[0].lower()+value[2:]
    return Path(value)

def key(path):
    value=str(path).replace('\\','/')
    if value.startswith('/mnt/') and value[6:7]=='/':value=value[5]+':'+value[6:]
    return value.casefold()

def read(path):return json.loads(local(path).read_text(encoding='utf-8-sig'))

def sha(path):
    value=hashlib.sha256()
    with local(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):value.update(part)
    return value.hexdigest()

def member(mapping,subject):
    result={}
    for path,digest in mapping.items():
        identity=key(path)
        if identity in result and result[identity]!=digest:raise ValueError('Conflicting path aliases')
        result[identity]=digest
    if result.get(key(subject['path']))!=subject['sha256']:raise ValueError('Exact consumed subject missing: '+subject['path'])

def report_gate(subjects,reports,physical):
    owner=reports['old_fit_owner'];fit=reports['old_fit_report'];audit=reports['old_fit_audit']
    for field in ('owner_verification_passed','accounting_passed','completion_passed','numerical_completion_passed','processes_absent','all_postrun_pins_exact'):
        if owner[field] is not True:raise ValueError('Old fit owner incomplete')
    if owner['raw_exit_known'] is not True or any(type(owner[k]) is not int or owner[k]!=0 for k in ('raw_python_exit_code','exit_code')):
        raise ValueError('Old fit known zero exits required')
    if fit['completed'] is not True or fit['ordinary_final_step']!=81000 or fit['architecture']!=[1323,512,512,23]:
        raise ValueError('Exact completed width81000 fit required')
    if audit['evidence_audit_passed'] is not True or audit['export_qualified'] is not True:raise ValueError('Old saved-fit audit incomplete')
    mapping={'old_fit_report':'fit_report','old_training_request':'training_request','old_training_manifest':'training_manifest',
             'old_shared_manifest':'shared_manifest','normalization':'normalization','old_context_alignment':'context_alignment'}
    for role,field in mapping.items():
        if owner['direct_subject_sha256'][field]!=subjects[role]['sha256']:raise ValueError('Old owner subject differs: '+role)
    for role in ('old_fit_report','old_training_request','old_training_manifest','old_shared_manifest','normalization','old_context_alignment'):
        member(audit['input_sha256'],subjects[role])
    old_request=reports['old_training_request'];frozen=reports['old_training_manifest']
    if frozen['training_request_sha256']!=subjects['old_training_request']['sha256']:raise ValueError('Actual old request/frozen identity')
    if frozen['source_sha256']['width512_promoted.py']!=subjects['export_source']['sha256']:raise ValueError('Exact qualified exporter source required')
    for role,path_key in [('old_centers','centers'),('old_pico','pico'),('old_walk002','walk002'),('physical_manifest','physical_manifest'),('contract','contract')]:
        if key(old_request['paths'][path_key])!=key(subjects[role]['path']):raise ValueError('Actual training input path differs')
        member(frozen['input_sha256'],subjects[role])
    shared=reports['old_shared_manifest']
    shared_root=local(subjects['old_shared_manifest']['path']).parent
    for role,name in [('old_nominal_context','nominal_context.npy'),('old_physical_context','physical_context.npy'),
                      ('normalization','normalization.npz'),('old_context_alignment','context_alignment.json')]:
        if local(subjects[role]['path']).resolve()!=(shared_root/name).resolve():raise ValueError('Actual shared output path differs')
        if shared['files'][name]!=subjects[role]['sha256']:raise ValueError('Actual shared output digest differs')
    alignment=reports['old_context_alignment']
    if (alignment['nominal_chronological_history_and_prior_pairs']!=9899 or alignment['physical_actual_prior_exact'] is not True
        or alignment['physical_history_once_shifted_exact'] is not True):raise ValueError('Qualified original context alignment required')
    manifest=reports['physical_manifest']
    if manifest['complete'] is not True or set(physical)!=set(PHYSICAL_KEYS):raise ValueError('Complete physical manifest and exact consumed array roles')
    parent=local(subjects['physical_manifest']['path']).parent.resolve()
    for role,item in physical.items():
        spec=manifest['arrays'][role];path=(parent/spec['path']).resolve()
        if path.parent!=parent or local(item['path']).resolve()!=path or item['sha256']!=spec['sha256']:raise ValueError('Physical array identity differs')
        member(frozen['input_sha256'],item)
    collection=reports['collection_report'];collection_request=reports['collection_request']
    if collection['passed'] is not True or collection['collection_completed'] is not True or collection['model_fitting_authorized'] is not False:
        raise ValueError('Completed qualified collection only')
    if (collection['rows'],collection['control_start'],collection['control_stop_exclusive'],collection['fresh_student_state_queries'])!=(1018,251,1269,1):
        raise ValueError('Wrong new trajectory scope')
    if collection['outputs']['expert_rows.npz']!=subjects['new_rows']['sha256'] or collection['outputs']['normalization.npz']!=subjects['normalization']['sha256']:
        raise ValueError('Collection row/norm output identity')
    if local(subjects['new_rows']['path']).resolve()!=(local(subjects['collection_report']['path']).parent/'expert_rows.npz').resolve():
        raise ValueError('Actual collector row output path differs')
    if collection['request_sha256']!=subjects['collection_request']['sha256']:raise ValueError('Collection request identity')
    if collection_request['root_selected_collection'] is not True or collection_request['model_fitting_authorized'] is not False:
        raise ValueError('Actual selected collection request required')
    for role,source_role in [('collection_qualification','qualification'),('collection_source_review','source_review')]:
        if collection_request['subjects'][source_role]!=subjects[role]:raise ValueError('Collection upstream subject differs')
        member(collection['input_sha256'],subjects[role])
    qualification=reports['collection_qualification'];source_review=reports['collection_source_review']
    if qualification['root_authorized_extraction'] is not True or qualification['model_fitting_authorized'] is not False:
        raise ValueError('Actual full recovery collection qualification required')
    if (qualification['control_start'],qualification['control_stop_exclusive'],qualification['rows'])!=(251,1269,1018):
        raise ValueError('Actual qualified collection scope differs')
    if source_review['source_review_pass'] is not True or source_review['source_sha256']!=collection['source_sha256'] or source_review['source_sha256']!=collection_request['source_sha256']:
        raise ValueError('Collection exact source identity differs')

def admit(request_path,clearance_path,clearance_sha,source_dir):
    request=read(request_path)
    if request.get('root_selected_saved_diagnosis') is not True:raise ValueError('Saved consistency diagnosis unselected')
    if request.get('model_calls')!=0 or request.get('native_steps')!=0 or request.get('optimizer_updates')!=0:raise ValueError('Pure saved-only scope required')
    if sha(clearance_path)!=clearance_sha:raise ValueError('Exact clearance argument required')
    clearance=read(clearance_path)
    if clearance['approved'] is not True or clearance['request_sha256']!=sha(request_path):raise ValueError('Concrete root selection required')
    subject=clearance['review']
    if sha(subject['path'])!=subject['sha256']:raise ValueError('Concrete review changed')
    review=read(subject['path'])
    if review['passed'] is not True or review['request_sha256']!=sha(request_path):raise ValueError('Concrete review does not cover request')
    subjects=request['subjects']
    if set(subjects)!=set(ROLES):raise ValueError('Exact saved diagnosis roles required')
    for role,item in subjects.items():
        if sha(item['path'])!=item['sha256']:raise ValueError('Changed saved input: '+role)
    for item in request['physical_arrays'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Changed physical array')
    names={p.name for p in source_dir.glob('*.py')}
    if set(request['source_sha256'])!=names:raise ValueError('Complete source map required')
    for name,digest in request['source_sha256'].items():
        if sha(source_dir/name)!=digest:raise ValueError('Changed diagnosis source')
    report_roles=set(ROLES)-{'old_centers','old_pico','old_walk002','old_nominal_context','old_physical_context',
                           'normalization','new_rows','export_source','contract'}
    reports={role:read(subjects[role]['path']) for role in report_roles}
    if reports['source_review']['source_review_pass'] is not True or reports['source_review']['source_sha256']!=request['source_sha256']:
        raise ValueError('Source review does not cover actual source')
    report_gate(subjects,reports,request['physical_arrays'])
    return request
