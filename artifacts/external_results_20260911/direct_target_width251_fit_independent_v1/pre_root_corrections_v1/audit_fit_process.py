"""Independent saved fit dispatch/absence/owner chain; never inspects live processes."""
from pathlib import Path
from audit_release import release_paths

def validate_process_chain(base,request,clearance,launch_sha,start,child,end,owner,owner_expected,read,bind,sha,check,compare,canonical):
    process=base/'fit_process_v1';dispatch=read(bind(process/'dispatch.json'));absence=read(bind(process/'process_absence.json'))
    pids=[end['wrapper_pid'],end['child_pid']]
    check('process_integer_identity',all(type(p) is int and p>0 for p in pids) and pids[0]!=pids[1])
    check('process_integer_exit',type(end['raw_python_exit_code']) is int and type(end['exit_code']) is int)
    check('dispatch_handle',dispatch['captured_handle_nonzero'] is True and type(dispatch['wrapper_pid']) is int and dispatch['wrapper_pid']==pids[0])
    normalize=lambda args:[str(x).replace('\\','/') for x in args]
    clear_sha=sha(base/'training_clearance.json')
    wanted=['-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(base/'run_fit_durable_v1.ps1'),'-ClearanceSha256',clear_sha,'-LaunchReceiptSha256',launch_sha]
    compare('dispatch_arguments',normalize(dispatch['arguments']),normalize(wanted))
    check('dispatch_PS51',str(dispatch['executable']).replace('\\','/').casefold()=='c:/windows/system32/windowspowershell/v1.0/powershell.exe')
    for name,item in [('start',start),('exit',end),('dispatch',dispatch),('owner',owner)]:
        check(name+'_launch_sha',item['launch_receipt_sha256']==launch_sha)
        check(name+'_clearance_sha',item['clearance_sha256']==clear_sha)
    for key in ('ordinary_start_step','ordinary_final_step','optimizer_start_step','optimizer_final_step','condition','updates','budgets','automatic_retry'):
        compare('start_scope.'+key,start[key],request[key])
    command=start['command'];snapshot=base/'source_snapshot_v1'
    check('actual_python',canonical(command['python'])==canonical(request['runtime']['python_path']))
    compare('actual_driver_arguments',normalize(command['arguments']),['-u',(snapshot/'train_recovery.py').as_posix()])
    check('actual_working_directory',canonical(command['working_directory'])==canonical(snapshot))
    compare('actual_CUBLAS',command['CUBLAS_WORKSPACE_CONFIG'],':4096:8');compare('actual_threads',command['threads'],1)
    expected_absence=dict(wrapper_pid=pids[0],child_pid=pids[1],wrapper_absent=True,child_absent=True,
        dispatch_sha256=sha(process/'dispatch.json'),start_sha256=sha(process/'start.json'),child_sha256=sha(process/'child.json'),exit_sha256=sha(process/'exit.json'))
    compare('process_absence_receipt',absence,expected_absence);compare('owner_absence_body',owner['process_absence'],absence)
    check('owner_absence_sha',owner['process_absence_sha256']==sha(process/'process_absence.json'))
    # Direct paths add a literal path check to the existing exact hash-only map.
    required=set(owner_expected)-{'failure','setup_failure'}
    check('owner_direct_path_membership',set(owner['direct_subjects'])==required)
    paths=release_paths(base,request)
    paths.update(process_exit=process/'exit.json',source_fit_report=Path(request['subjects']['fit_report']['path']),
        balance_source_review=Path(request['subjects']['balance_source_review']['path']))
    for role,subject in owner['direct_subjects'].items():
        check('owner_literal_path:'+role,canonical(subject['path'])==canonical(paths[role]))
        check('owner_literal_hash:'+role,subject['sha256']==owner_expected[role]==sha(bind(subject['path'])))
    records={canonical(p):digest for p,digest in owner['output_sha256'].items()}
    check('owner_output_no_alias',len(records)==len(owner['output_sha256']))
    expected_paths={canonical(p) for p in paths.values()}|{canonical(process/name) for name in ('dispatch.json','start.json','child.json','exit.json','prerun_pins.json','postrun_pins.json','process_absence.json')}
    check('owner_output_membership',set(records)==expected_paths)
    for name in ('dispatch.json','start.json','child.json','exit.json','prerun_pins.json','postrun_pins.json','process_absence.json'):
        p=process/name;check('owner_process_output:'+name,records.get(canonical(p))==sha(bind(p)))
    for subject in owner['direct_subjects'].values():
        check('owner_direct_output:'+subject['path'],records.get(canonical(subject['path']))==subject['sha256'])
