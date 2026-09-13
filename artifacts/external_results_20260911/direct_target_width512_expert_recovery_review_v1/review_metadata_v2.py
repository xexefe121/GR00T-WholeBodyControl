"""Narrow saved-byte review of fresh namespace/path metadata; no task dispatch."""
import copy
import json
from pathlib import Path
from review_concrete import BASE, RUN as OLD, read, sha, subject, canonical, canonical_map
RUN=OLD.with_name('direct_target_width512_expert_recovery_v2')
EXPECTED={
 'execution_request.json':'68ee9617562fa8b73ee4abd55d1703ca2f279887c09a5a04b851b814cd34be44',
 'frozen_inputs.json':'c6531b52a65e057602ed560c00448d7ebc2573a501ab9204d651b3af8955367e',
 'launch_receipt.json':'20c30bbfc99454c434006492e393bdfea7d0cc408b191c0868a38ef328593fd1',
 'run_recovery_durable.ps1':'63c139fde9ece01db4ebbd4d419847620c2aaf099535711558a95538afc6363b',
 'verify_recovery_completed.py':'f06fb5e730cdb3416c058719f91d27ad17ccc07784538371cd6f72c500e263b8',
 'path_preflight/report.json':'9ce1667c5998215b5db53b503dfbfb05c430a73054509da2e2112d9855934345',
}
REPLACEMENTS=[(OLD.name,RUN.name),('recovery_process_v2','recovery_process_v1'),
 ('execution_clearance_v2.json','execution_clearance.json'),('launch_receipt_v2.json','launch_receipt.json'),
 ('run_recovery_durable_v2.ps1','run_recovery_durable.ps1'),('owner_completion_v2.json','owner_completion.json')]

def normalized(path):return Path(path).resolve().as_posix()
def linux(path):return '/mnt/'+path[0].lower()+path[2:] if len(path)>2 and path[1]==':' else path

def main():
    for name,digest in EXPECTED.items():assert sha(RUN/name)==digest,name
    oldreq=read(OLD/'execution_request.json');request=read(RUN/'execution_request.json')
    oldfrozen=read(OLD/'frozen_inputs.json');frozen=read(RUN/'frozen_inputs.json')
    launch=read(RUN/'launch_receipt.json');proof=read(RUN/'path_preflight/report.json')
    prior_owner=read(OLD/'owner_completion_v2.json')
    assert prior_owner['completion_accounting_passed'] is True and prior_owner['requested_recovery_completed'] is False
    assert prior_owner['raw_python_exit_code']==1 and prior_owner['processes_absent'] is True
    assert len(prior_owner['actual_work_counters'])==21
    assert all(value==0 for row in prior_owner['actual_work_counters'].values() for value in row.values())
    expected=copy.deepcopy(oldreq)
    expected['environment']['PYTHONPATH']=linux(RUN.as_posix())+'/source_snapshot_v1'
    for role,item in expected['subjects'].items():
        item['path']=normalized(item['path'])
        if role in ('selected_snapshot','selected_prefix','input_selection'):item['path']=(RUN/'inputs'/Path(item['path']).name).as_posix()
    new_roles={'prior_failed_request':'execution_request.json','prior_failed_frozen':'frozen_inputs.json',
        'prior_failed_owner':'owner_completion_v2.json','prior_failed_exit':'recovery_process_v2/exit.json',
        'prior_failed_work':'work_counters.json','prior_failed_failure':'failure.json'}
    for role,name in new_roles.items():expected['subjects'][role]=subject(OLD/name)
    expected['metadata_repair']=dict(input_keys_forward_slash=True,task_source_changes=0,previous_task_work_all_zero=True,
        previous_attempt_preserved=True,input_arrays_copied_byte_exact=True,input_extraction_repeated=False)
    assert expected==request,'Unexpected request derivative'
    expected_pins={}
    for path,digest in oldfrozen['input_sha256'].items():
        path=normalized(path)
        if Path(path).parent==OLD/'inputs':path=(RUN/'inputs'/Path(path).name).as_posix()
        assert path not in expected_pins or expected_pins[path]==digest
        expected_pins[path]=digest
    expected_pins.update({x['path']:x['sha256'] for x in request['subjects'].values()})
    expected_frozen=dict(oldfrozen,input_sha256=expected_pins,request_sha256=EXPECTED['execution_request.json'])
    assert frozen==expected_frozen and len(expected_pins)==44
    assert all('\\' not in path and path[1:3]==':/' for path in expected_pins)
    assert frozen['source_sha256']==oldfrozen['source_sha256'] and len(frozen['source_sha256'])==20
    for name,digest in frozen['source_sha256'].items():assert sha(RUN/'source_snapshot_v1'/name)==sha(OLD/'source_snapshot_v1'/name)==digest
    for name in ('precontrol251.npz','actual_prefix251.npz','selection_receipt.json'):assert sha(RUN/'inputs'/name)==sha(OLD/'inputs'/name)
    helpers=[('run_recovery_durable_v2.ps1','run_recovery_durable.ps1'),
             ('verify_recovery_completed_v2.py','verify_recovery_completed.py'),('run_recovery_v2.sh','run_recovery.sh')]
    for old,new in helpers:
        text=(OLD/old).read_text()
        for a,b in REPLACEMENTS:text=text.replace(a,b)
        assert text==(RUN/new).read_text(),new
    # Owner returns byte-for-byte to the previously25-tested canonical helper.
    assert sha(RUN/'verify_recovery_completed.py')==sha(OLD/'verify_recovery_completed.py')
    oldlaunch=read(OLD/'launch_receipt_v2.json')
    expected_args=[v.replace(OLD.name,RUN.name).replace('/run_recovery_v2.sh','/run_recovery.sh') for v in oldlaunch['wsl_arguments']]
    assert launch['wsl_arguments']==expected_args
    argv=read(OLD/'argv_preflight_v2/report.json')
    assert argv['actual_launcher_assignment'] in (RUN/'run_recovery_durable.ps1').read_text() and argv['passed'] is True
    assert launch['request_sha256']==EXPECTED['execution_request.json'] and launch['frozen_receipt_sha256']==EXPECTED['frozen_inputs.json']
    assert launch['launcher_sha256']==EXPECTED['run_recovery_durable.ps1'] and launch['dispatch_authorized'] is False
    pins=canonical_map(launch['input_sha256']);assert len(pins)==85
    for path,digest in pins.values():assert '\\' not in path and sha(path)==digest,path
    for path,digest in expected_pins.items():assert pins[canonical(path)][1]==digest
    expected_checked={linux(p):d for p,d in expected_pins.items()}
    expected_checked.update({linux((RUN/'source_snapshot_v1'/n).as_posix()):d for n,d in frozen['source_sha256'].items()})
    assert proof['checked_sha256']==expected_checked and len(expected_checked)==64
    assert proof['checked_sources']==20 and proof['checked_inputs']==44
    assert proof['passed'] is True and proof['actual_inherited_frozen_function_ast'] is True and proof['task_modules_imported'] is False
    assert proof['source_sha256']==frozen['source_sha256']['run_actual_student_oracle.py']
    assert proof['request_sha256']==EXPECTED['execution_request.json'] and proof['frozen_receipt_sha256']==EXPECTED['frozen_inputs.json']
    exited=read(RUN/'path_preflight/exit.json');assert exited['raw_exit_code']==0 and exited['handle_acquired'] is True
    assert exited['arguments'][-1]==linux(RUN.as_posix())+'/check_inherited_paths.py'
    stderr=(RUN/'path_preflight/stderr.log').read_text().strip()
    assert stderr=="wsl: Failed to translate 'E:\\windsurf\\Windsurf\\bin'"
    for name in ('launcher_parse.json','bash_parse.json'):assert read(RUN/name)['passed'] is True
    absent=['execution_clearance.json','recovery_process_v1','owner_completion.json','ATTEMPT_STARTED','initial_seed',
        'nominal','post_lifecycle_hold_5s','initial_restore_preflight.json','outcome.json','failure.json','work_counters.json']
    for name in absent:assert not (RUN/name).exists(),name
    report=dict(passed=True,concrete_review_pass=True,metadata_review_pass=True,
        request_sha256=EXPECTED['execution_request.json'],frozen_receipt_sha256=EXPECTED['frozen_inputs.json'],
        launch_receipt_sha256=EXPECTED['launch_receipt.json'],launcher_sha256=EXPECTED['run_recovery_durable.ps1'],
        request_subject=subject(RUN/'execution_request.json'),frozen_subject=subject(RUN/'frozen_inputs.json'),
        launch_receipt_subject=subject(RUN/'launch_receipt.json'),source_sha256=frozen['source_sha256'],
        helper_sha256={new:sha(RUN/new) for _,new in helpers},input_sha256=launch['input_sha256'],
        launch_pin_count=85,frozen_input_count=44,source_count=20,input_copies_byte_exact=3,
        actual_path_proof=subject(RUN/'path_preflight/report.json'),actual_path_proof_exit=subject(RUN/'path_preflight/exit.json'),
        actual_path_proof_source=subject(RUN/'check_inherited_paths.py'),actual_argv_proof=subject(OLD/'argv_preflight_v2/report.json'),
        path_proof_stderr=stderr,path_proof_warning_interpretation='WSL PATH translation warning; selected pure checker returned0 and verified all64 exact paths.',
        prior_transport_review=subject(BASE/'transport_v2_review.json'),prior_source_review=subject(BASE/'source_review.json'),
        prior_boundary_review=subject(BASE/'boundary_review.json'),prior_failed_owner=subject(OLD/'owner_completion_v2.json'),
        inherited_tests=subject(BASE/'concrete_helper_tests_final.xml'),
        changes=['Forward-slash Windows pin and role paths; three input paths point to identical fresh copies.',
                 'Fresh namespace with canonical single execution_clearance.json shared by task admission, supervisor and owner.',
                 'Six exact prior-failure subjects added; all21 prior task counters zero and failed attempt preserved.'],
        source_math_changes=0,protocol_changes=0,source_or_input_extraction_repeated=False,
        wsl_arguments=expected_args,required_clearance_argument='-ClearanceSha256 actual new execution_clearance.json SHA256',
        processes_and_outputs_absent=absent,actual_recovery_dispatched=False,actual_recovery_execution_cleared=False,
        task_model_calls=0,native_calls=0,replans=0,optimizer_calls=0,labels_admissible=False,
        limitations=['Root concrete selection remains required; expert is sole dispatcher.',
                    'Pure WSL path proof validates actual inherited frozen() conversion/hashes, not subsequent native/expert behavior.',
                    'No completed recovery, physical/intent qualification or collection is claimed.',
                    'First metadata reviewer invocation assumed empty stderr; that review-only assumption was corrected to preserve the exact nonfatal WSL PATH warning. No task or path proof was rerun.'],review_source=subject(Path(__file__)))
    out=BASE/'metadata_v2_review.json'
    with out.open('x',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(review=subject(out),passed=True,pins=85,checked_linux_paths=64)))

if __name__=='__main__':main()
