"""Concrete future execution gates; preparation has no permissive fallback."""
from pathlib import Path
import sys
from direct_data import read,sha
from context_contract import CONDITIONS,UPDATES,COEFFICIENT,BUDGETS
from training_support import frozen_check

def gate(base,source):
    files={k:base/v for k,v in [('request','training_request.json'),('frozen','training_frozen_inputs.json'),('clearance','training_clearance.json')]}
    identity={k:sha(p) for k,p in files.items()}
    request=read(files['request']);receipt=read(files['frozen']);clearance=read(files['clearance'])
    if Path(receipt['source_directory']).resolve()!=Path(source).parent.resolve() or receipt['training_request_sha256']!=identity['request']:raise ValueError('Frozen source/request identity.')
    frozen_check(receipt)
    if clearance['approved'] is not True or clearance['request_sha256']!=identity['request'] or clearance['frozen_receipt_sha256']!=identity['frozen']:raise ValueError('Actual selected pair clearance absent.')
    if sha(clearance['review_path'])!=clearance['review_sha256'] or sha(clearance['launcher_path'])!=clearance['launcher_sha256']:raise ValueError('Final review/launcher changed.')
    review=read(clearance['review_path'])
    if review['prelaunch_review_pass'] is not True or review['training_request_sha256']!=identity['request'] or review['frozen_receipt_sha256']!=identity['frozen']:raise ValueError('Concrete paired review binding differs.')
    if request['kind']!='matched_causal_context_utility_study' or request['root_selected'] is not True or tuple(request['conditions'])!=CONDITIONS or request['updates_per_condition']!=UPDATES or request['ordinary_final_step']!=68000 or request['coefficient']!=COEFFICIENT or request['budgets']!=BUDGETS:raise ValueError('Fixed paired protocol differs.')
    if request['learning_rate']!=[1e-5,1e-6] or request['context_std_floor']!=.05 or request['features']!=1323 or request['context_order']!='previous_action23_then_incoming_history300' or request['coefficient_recalibration'] is not False:raise ValueError('Context or optimizer metadata differs.')
    for key,subject in request['subjects'].items():
        path=Path(subject['path']).as_posix()
        if receipt['input_sha256'].get(path)!=subject['sha256'] or sha(path)!=subject['sha256']:raise ValueError('Unfrozen consumed subject: '+key)
        if 'pass_field' in subject:
            record=read(path)
            if record[subject['pass_field']] is not True:raise ValueError('Required qualification failed: '+key)
            for field,value in subject.get('required_fields',{}).items():
                if record.get(field)!=value:raise ValueError('Qualification subject mismatch: '+key+'/'+field)
    for group in ('paths','full_state_paths','restoration_predictions','schedule_paths'):
        for key,value in request[group].items():
            path=Path(value).as_posix()
            if receipt['input_sha256'].get(path)!=sha(path):raise ValueError('Unfrozen consumed path: '+group+'/'+key)
    runtime=request['runtime'];path=Path(runtime['verification_path']).as_posix()
    if Path(sys.executable).resolve()!=Path(runtime['python_path']).resolve():raise ValueError('Wrong isolated training Python.')
    if receipt['input_sha256'].get(path)!=runtime['verification_sha256'] or sha(path)!=runtime['verification_sha256'] or read(path)[runtime['pass_field']] is not True:raise ValueError('Runtime qualification missing.')
    source_report=read(request['subjects']['fit_report']['path'])
    if source_report['completed'] is not True or source_report['ordinary_final_step']!=65000 or source_report['checkpoint_sha256']!=request['subjects']['checkpoint']['sha256']:raise ValueError('Exact qualified65000 source fit required.')
    if read(request['subjects']['coefficient_source']['path'])['coefficient']!=COEFFICIENT:raise ValueError('Original coefficient receipt differs.')
    return receipt,request,identity
