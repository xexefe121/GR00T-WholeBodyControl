"""Literal completed-report admission, independently of collection arithmetic."""
import json
import hashlib
import sys
from pathlib import Path

REPORT_ROLES=('main_physics','main_intent','hold_physics','hold_intent')
QUALIFIED_ROLES=('main_trace','main_report','hold_trace','hold_report','owner','recovery_request',*REPORT_ROLES,
                 'boundary_snapshot','boundary_review','selection_receipt','plan_records','motion','original29','contract',
                 'normalization','core','source_review')

def key(path):
    value=str(path).replace('\\','/')
    if value.startswith('/mnt/') and value[6:7]=='/':value=value[5]+':'+value[6:]
    return value.casefold()

def local(path):
    value=str(path).replace('\\','/')
    if sys.platform!='win32' and len(value)>2 and value[1]==':':value='/mnt/'+value[0].lower()+value[2:]
    return Path(value)

def read(path):return json.loads(local(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h=hashlib.sha256()
    with local(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def member(mapping,subject):
    result={}
    for path,digest in mapping.items():
        canonical=key(path)
        if canonical in result and result[canonical]!=digest:raise ValueError('Conflicting literal subject aliases')
        result[canonical]=digest
    if result.get(key(subject['path']))!=subject['sha256']:raise ValueError('Missing exact consumed subject: '+subject['path'])

def validate_reports(subjects,reports):
    owner=reports['owner']
    for field in ('completion_accounting_passed','requested_recovery_completed','raw_exit_known','all_postrun_pins_exact','processes_absent'):
        if owner[field] is not True:raise ValueError('Recovery owner is not completed: '+field)
    if type(owner['raw_python_exit_code']) is not int or owner['raw_python_exit_code']!=0 or type(owner['exit_code']) is not int or owner['exit_code']!=0:
        raise ValueError('Completed known zero exits required')
    if owner['accounting_uncertainty']!=[]:raise ValueError('Uncertain recovery work')
    for role in ('main_trace','main_report','hold_trace','hold_report','recovery_request'):
        member(owner['output_sha256'],subjects[role])
    recovery=reports['recovery_request']
    if recovery['root_selected'] is not True or recovery['kind']!='fixed_width81000_pre251_actual_expert_recovery':
        raise ValueError('Wrong selected recovery')
    expected_protocol={'initial_global_control':251,'prefix_controls':251,'learned_prefix_controls':1,
        'requested_branch_controls':1318,'MPC_controls':1018,'conditional_hold_controls':250,
        'original_source_controls':819,'requested_combined_main_controls':1569,'prefix_native_steps':2510}
    for field,value in expected_protocol.items():
        if recovery['protocol'][field]!=value:raise ValueError('Changed recovery scope: '+field)
    for role,source_role in [('boundary_snapshot','selected_snapshot'),('selection_receipt','input_selection')]:
        if recovery['subjects'][source_role]!=subjects[role]:raise ValueError('Actual recovery input subject mismatch')
    for part,controls,start in [('main',1569,0),('hold',250,1569)]:
        producer=reports[part+'_report'];physics=reports[part+'_physics'];intent=reports[part+'_intent']
        if producer['full_segment_completed'] is not True or producer['requested_controls']!=controls:raise ValueError('Partial producer segment')
        if producer['trace_sha256']!=subjects[part+'_trace']['sha256']:raise ValueError('Producer trace identity')
        member(physics['input_hashes'],subjects[part+'_trace'])
        for field in ('independent_segment_pass','recorded_trace_reproduced_through_last_sample','requested_segment_completed'):
            if physics[field] is not True:raise ValueError('Independent physics incomplete: '+field)
        if physics['intended_segment_controls']!=controls or physics['physics_steps']!=10*controls or physics['recorded_partial_substeps']!=0:
            raise ValueError('Wrong independent physical scope')
        comparison=physics['original_trace_comparison']
        expected={'all_qpos_bitexact','all_qvel_bitexact','all_command_torque_bitexact','physics_actuator_force_bitexact',
                  'all_physics_time_bitexact','physics_warning_number_bitexact','physics_warning_lastinfo_bitexact'}
        if set(comparison)!=expected or not all(x is True for x in comparison.values()):raise ValueError('Seven exact physical fields required')
        member(intent['hashes'],subjects[part+'_trace']);member(intent['hashes'],subjects[part+'_physics'])
        for field in ('independent_physical_pass','intended_segment_completed','requested_segment_quiet_pass'):
            if intent[field] is not True:raise ValueError('Independent intent incomplete: '+field)
        if intent['requested_controls']!=controls or intent['global_start']!=start:raise ValueError('Wrong independent intent scope')
        if part=='main' and (intent['full_lifecycle_source_intent_pass'] is not True or intent['source_metrics']['source_controls']!=819):
            raise ValueError('Original full source did not qualify')
    member(reports['hold_intent']['hashes'],subjects['main_trace'])
    qualification=reports['qualification']
    if qualification['root_authorized_extraction'] is not True or qualification['model_fitting_authorized'] is not False:
        raise ValueError('Separate collection-only root selection required')
    if qualification['control_start']!=251 or qualification['control_stop_exclusive']!=1269 or qualification['rows']!=1018:
        raise ValueError('Only1018 newly executed expert controls eligible')
    for role in QUALIFIED_ROLES:
        if qualification['subjects'][role]!=subjects[role]:raise ValueError('Qualification subject mismatch: '+role)
    if reports['boundary_review']['boundary_review_pass'] is not True or reports['selection_receipt']['passed'] is not True:
        raise ValueError('Qualified saved251 boundary missing')
    if reports['boundary_review']['snapshot_subject']['sha256']!=subjects['boundary_snapshot']['sha256']:
        raise ValueError('Reviewed snapshot bytes differ')
    if reports['boundary_review']['selection_subject']['sha256']!=subjects['selection_receipt']['sha256']:
        raise ValueError('Reviewed extraction receipt differs')
    if reports['selection_receipt']['outputs']['precontrol251.npz']!=subjects['boundary_snapshot']['sha256']:
        raise ValueError('Extraction snapshot digest differs')
    if reports['source_review']['source_review_pass'] is not True:raise ValueError('Collector source review missing')

def admit(request,source_directory):
    if request.get('root_selected_collection') is not True:raise ValueError('Actual collection unselected')
    if request.get('model_fitting_authorized') is not False:raise ValueError('No fitting selection in collection')
    subjects=request['subjects']
    if set(subjects)!=set(QUALIFIED_ROLES)|{'qualification'}:raise ValueError('Exact collection subject roles required')
    for role,entry in subjects.items():
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Changed subject: '+role)
    source_names={p.name for p in source_directory.glob('*.py')}
    if set(request['source_sha256'])!=source_names:raise ValueError('Complete collector source pins required')
    for name,digest in request['source_sha256'].items():
        if sha(source_directory/name)!=digest:raise ValueError('Changed collector source: '+name)
    reports={role:read(subjects[role]['path']) for role in (*REPORT_ROLES,'owner','recovery_request','main_report','hold_report',
                'qualification','source_review','boundary_review','selection_receipt')}
    validate_reports(subjects,reports)
    if reports['source_review']['source_sha256']!=request['source_sha256']:raise ValueError('Source review map mismatch')
    expected=list(range(251,1269,5))
    if [entry['control'] for entry in request['plans']]!=expected:raise ValueError('Exact204 committed plan subjects required')
    for entry in request['plans']:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Changed committed plan')
    if reports['qualification']['plans']!=request['plans']:raise ValueError('Qualified plan pins differ')
    return reports
