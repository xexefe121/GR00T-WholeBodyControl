"""Fail-closed concrete release gate; source preparation cannot authorize a fit."""
from pathlib import Path
import sys
from direct_data import read,sha
from training_support import frozen_check,final_identity
from recovery_protocol import check_protocol,COEFFICIENT
from recovery_data import check_recovery_subjects
from balance_contract import GROUP_WEIGHTS,ZERO_RESPONSE_ENERGIES,MEAN_ZERO_RESPONSE_ENERGY

REQUIRED_SUBJECTS=('checkpoint','normalization','fit_report','fit_owner','fit_audit',
    'coefficient_source','context_proof','context_alignment','context_manifest',
    'energy_source','balance_source_review','semantics_review','full_state_root_audit',
    'full_state_root_owner','full_state_data_review')

def gate(base,source):
    files={k:base/v for k,v in [('request','training_request.json'),('frozen','training_frozen_inputs.json'),('clearance','training_clearance.json')]}
    identity={k:sha(p) for k,p in files.items()}
    request=read(files['request']);receipt=read(files['frozen']);clearance=read(files['clearance'])
    if Path(receipt['source_directory']).resolve()!=Path(source).parent.resolve() or receipt['training_request_sha256']!=identity['request']:
        raise ValueError('Frozen source/request identity.')
    frozen_check(receipt);check_protocol(request)
    if request['root_selected'] is not True:raise ValueError('Actual continuation not selected.')
    if clearance['approved'] is not True or clearance['request_sha256']!=identity['request'] or clearance['frozen_receipt_sha256']!=identity['frozen']:
        raise ValueError('Concrete root clearance absent.')
    if sha(clearance['review_path'])!=clearance['review_sha256'] or sha(clearance['launcher_path'])!=clearance['launcher_sha256']:
        raise ValueError('Final review/launcher identity differs.')
    review=read(clearance['review_path'])
    if review['prelaunch_review_pass'] is not True or review['training_request_sha256']!=identity['request'] or review['frozen_receipt_sha256']!=identity['frozen']:
        raise ValueError('Concrete review binding differs.')
    identity['review_path']=clearance['review_path'];identity['review_sha256']=clearance['review_sha256']
    identity['launcher_path']=clearance['launcher_path'];identity['launcher_sha256']=clearance['launcher_sha256']
    if not set(REQUIRED_SUBJECTS).issubset(request['subjects']):raise ValueError('Required qualified subjects absent.')
    for key,subject in request['subjects'].items():
        path=Path(subject['path']).as_posix()
        if receipt['input_sha256'].get(path)!=subject['sha256'] or sha(path)!=subject['sha256']:
            raise ValueError('Unfrozen consumed subject: '+key)
        if 'pass_field' in subject:
            record=read(path)
            if record[subject['pass_field']] is not True:raise ValueError('Qualification failed: '+key)
            for field,value in subject.get('required_fields',{}).items():
                if record.get(field)!=value:raise ValueError('Qualification binding differs: '+key+'/'+field)
    for key in ('semantics_review','balance_source_review','fit_audit','fit_owner','context_proof','full_state_data_review'):
        if 'pass_field' not in request['subjects'][key]:raise ValueError('Literal positive qualification required: '+key)
    for group in ('paths','full_state_paths','restoration_predictions','context_paths','schedule_paths'):
        for key,value in request[group].items():
            path=Path(value).as_posix()
            if receipt['input_sha256'].get(path)!=sha(path):raise ValueError('Unfrozen path: '+group+'/'+key)
    for key,subject_key in [('normalization','normalization'),('context_alignment','context_alignment'),('manifest','context_manifest')]:
        if Path(request['context_paths'][key]).resolve()!=Path(request['subjects'][subject_key]['path']).resolve():
            raise ValueError('Consumed context subject path differs: '+key)
    source_report=read(request['subjects']['fit_report']['path'])
    if (source_report['completed'] is not True or source_report['ordinary_final_step']!=81000
        or source_report['condition']!='causal' or source_report['optimizer_step']!=16000
        or source_report['checkpoint_sha256']!=request['subjects']['checkpoint']['sha256']):
        raise ValueError('Exact completed causal81000 fit required.')
    if read(request['subjects']['coefficient_source']['path'])['coefficient']!=COEFFICIENT:
        raise ValueError('Original fixed coefficient differs.')
    energy=read(request['subjects']['energy_source']['path'])
    if (energy['group_weights']!=list(GROUP_WEIGHTS) or energy['zero_response_energies']!=list(ZERO_RESPONSE_ENERGIES)
        or energy['mean_zero_response_energy']!=MEAN_ZERO_RESPONSE_ENERGY):
        raise ValueError('Fixed teacher-energy weights differ.')
    runtime=request['runtime'];path=Path(runtime['verification_path']).as_posix()
    if Path(sys.executable).resolve()!=Path(runtime['python_path']).resolve():raise ValueError('Wrong isolated Python.')
    if receipt['input_sha256'].get(path)!=runtime['verification_sha256'] or sha(path)!=runtime['verification_sha256'] or read(path)[runtime['pass_field']] is not True:
        raise ValueError('Runtime qualification differs.')
    check_recovery_subjects(request,receipt["input_sha256"])
    return receipt,request,identity

def final_response_identity(base,identity,receipt):
    final_identity(base,identity,receipt)
    for name in ('review','launcher'):
        if sha(identity[name+'_path'])!=identity[name+'_sha256']:
            raise ValueError('Final '+name+' changed.')

