"""Saved completion accounting only; no task-array/model/native imports."""
import argparse
import hashlib
import json
from pathlib import Path

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def canonical(p):return str(Path(p).resolve()).replace('\\','/').casefold()
def union(*maps):
    result={}
    for values in maps:
        for p,digest in values.items():
            key=canonical(p)
            if key in result:raise ValueError('Duplicate launch subject: '+p)
            result[key]=(p,digest)
    return result

def validate_counters(counts,budgets):
    assert counts['limits']==budgets and set(counts['counts'])==set(budgets)
    uncertainty=[]
    for key,value in counts['counts'].items():
        assert set(value)=={'attempted_calls','returned_calls','attempted_units','returned_units'}
        assert all(type(v) is int for v in value.values())
        limits=budgets[key]
        assert 0<=value['returned_calls']<=value['attempted_calls']<=limits['attempted_calls_max']
        assert 0<=value['returned_units']<=value['attempted_units']<=limits['attempted_units_max']
        if key in ('batch_fd_step','batch_line_step') and value['attempted_calls']!=value['returned_calls']:
            uncertainty.append(key+': interrupted Batch call; internal completed lane-steps unknown.')
        if key.startswith('native_') and key.endswith('_step') and value['attempted_calls']!=value['returned_calls']:
            uncertainty.append(key+': native step raised; exact internal progress unknown.')
    for kind in ('attempted_calls','returned_calls','attempted_units','returned_units'):
        expected=sum(counts['counts']['native_'+name+'_step'][kind] for name in ('BFM','initial_certificate','restoration_certificate','preview'))
        assert counts['counts']['native_private_step'][kind]==expected
    return uncertainty

def validate_process_metadata(start,exit_report,child,linux,receipt,final,driver,complete):
    assert start['arguments']==receipt['wsl_arguments']
    if child:
        assert child['wrapper_pid']==start['wrapper_pid'] and child['handle_acquired'] is True
    assert type(exit_report['final_work_counters_present']) is bool
    assert type(exit_report['requested_branch_completed']) is bool
    assert exit_report['final_work_counters_present']==final
    assert exit_report['requested_branch_completed']==bool(driver and driver['requested_branch_completed'])
    uncertainty=[]
    if (exit_report['raw_exit_known'] or final) and (child is None or linux is None):
        uncertainty.append('Known child exit/final work ledger lacks actual child or Linux identity receipt.')
    if exit_report['raw_python_exit_code']==0:
        assert exit_report['exit_code']==0 and not uncertainty
        assert final and complete and driver and driver['driver_returned'] is True and driver['requested_branch_completed'] is True
    return uncertainty

def verify(base):
    process=base/'recovery_process_v2'
    receipt=read(base/'launch_receipt_v2.json');clearance=read(base/'execution_clearance_v2.json');request=read(base/'execution_request.json')
    start=read(process/'start.json');exit_report=read(process/'exit.json');absence=read(process/'process_absence.json')
    assert clearance['approved'] is True
    assert start['request_sha256']==clearance['request_sha256']==receipt['request_sha256']==sha(base/'execution_request.json')
    assert start['frozen_receipt_sha256']==clearance['frozen_receipt_sha256']==receipt['frozen_receipt_sha256']==sha(base/'frozen_inputs.json')
    assert start['launch_receipt_sha256']==clearance['launch_receipt_sha256']==sha(base/'launch_receipt_v2.json')
    assert start['clearance_sha256']==sha(base/'execution_clearance_v2.json')
    review=read(clearance['review']['path'])
    assert sha(clearance['review']['path'])==clearance['review']['sha256'] and review[clearance['review']['pass_field']] is True
    for key in ('request_sha256','frozen_receipt_sha256','launch_receipt_sha256'):assert review[key]==clearance[key]
    child=read(process/'child.json') if (process/'child.json').exists() else None
    linux=read(process/'linux_process.json') if (process/'linux_process.json').exists() else None
    dispatch=read(process/'dispatch.json')
    assert dispatch['wrapper_pid']==start['wrapper_pid'] and dispatch['clearance_sha256']==start['clearance_sha256']
    assert dispatch['launcher_sha256']==receipt['launcher_sha256']==sha(base/'run_recovery_durable_v2.ps1')
    assert dispatch['arguments']==['-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',
        str(base/'run_recovery_durable_v2.ps1').replace('\\','/'),'-ClearanceSha256',sha(base/'execution_clearance_v2.json')]
    extra={str(base/'launch_receipt_v2.json'):sha(base/'launch_receipt_v2.json'),
           str(base/'execution_clearance_v2.json'):sha(base/'execution_clearance_v2.json'),
           clearance['review']['path']:clearance['review']['sha256']}
    pins=union(receipt['input_sha256'],extra)
    for filename in ('prerun_pins.json','postrun_pins.json'):
        report=read(process/filename)
        actual=union({p:v['expected'] for p,v in report['files'].items()})
        assert actual.keys()==pins.keys() and report['all_exact'] is True
        for key,(p,digest) in actual.items():
            entry=report['files'][p]
            assert digest==pins[key][1] and entry['actual']==digest and entry['matched'] is True
    for p,digest in pins.values():assert sha(p)==digest,p
    windows=[start['wrapper_pid']]+([] if child is None else [child['child_pid']])
    linux_ids=[] if linux is None else [linux['linux_pid']]
    assert all(type(v) is int and v>0 for v in windows+linux_ids)
    assert type(exit_report['exit_code']) is int and type(exit_report['raw_exit_known']) is bool
    if exit_report['raw_exit_known']:assert type(exit_report['raw_python_exit_code']) is int
    assert absence['windows_expected_pids']==windows and absence['linux_expected_pids']==linux_ids
    assert absence['windows_all_absent'] is True and absence['linux_all_absent'] is True
    assert exit_report['wrapper_pid']==start['wrapper_pid']
    assert exit_report['child_pid']==(None if child is None else child['child_pid'])
    counts=read(base/'work_counters.json') if (base/'work_counters.json').exists() else None
    final=bool(counts and counts['phase']=='task_ended' and counts['active']==[])
    uncertainty=[]
    if not final:uncertainty.append('No final Python work ledger; work after last durable checkpoint is unknown.')
    if exit_report['raw_exit_known'] is not True:uncertainty.append('No known child/WSL raw exit.')
    if counts:uncertainty.extend(validate_counters(counts,request['protocol']['budgets']))
    outcome=read(base/'outcome.json') if (base/'outcome.json').exists() else {}
    complete=bool(outcome.get('nominal',{}).get('full_segment_completed') and outcome.get('extension',{}).get('full_segment_completed'))
    driver=read(base/'driver_completion.json') if (base/'driver_completion.json').exists() else None
    uncertainty.extend(validate_process_metadata(start,exit_report,child,linux,receipt,final,driver,complete))
    if exit_report['raw_python_exit_code']==0:
        assert final and complete and driver and driver['driver_returned'] is True and driver['requested_branch_completed'] is True
        assert all(counts['counts']['native_live_step'][name]==15680 for name in ('attempted_calls','returned_calls','attempted_units','returned_units'))
        assert counts['counts']['ordinary_ilqr']['returned_calls']==204
        assert not uncertainty
    elif complete:uncertainty.append('Saved full branch disagrees with failed raw process; inspect finalization failure.')
    files=[base/'execution_request.json',base/'frozen_inputs.json',base/'launch_receipt_v2.json',base/'execution_clearance_v2.json',process/'exit.json']
    files.extend(process/name for name in ('dispatch.json','start.json','process_absence.json','prerun_pins.json','postrun_pins.json'))
    for name in ('child.json','linux_process.json'):
        if (process/name).exists():files.append(process/name)
    for name in ('work_counters.json','work_snapshots.jsonl','driver_completion.json','failure.json','outcome.json','last_actual_state.npz','last_actual_state_capture.json',
                 'initial_restore_preflight.json','initial_seed/report.json','nominal/trace.npz','nominal/report.json','nominal/first_expert_target.npz',
                 'post_lifecycle_hold_5s/trace.npz','post_lifecycle_hold_5s/report.json'):
        if (base/name).exists():files.append(base/name)
    return dict(kind='saved_recovery_owner_accounting',completion_accounting_passed=not uncertainty,requested_recovery_completed=complete,
        raw_exit_known=exit_report['raw_exit_known'],raw_python_exit_code=exit_report['raw_python_exit_code'],exit_code=exit_report['exit_code'],
        all_postrun_pins_exact=True,processes_absent=True,accounting_uncertainty=uncertainty,
        high_level_expected_seed_rejection_may_have_attempted_returned_difference=True,
        counter_first_failure=None if counts is None else counts['first_failure'],
        physical_and_intent_qualification_pending=True,labels_admissible=False,
        input_pin_count=len(pins),input_sha256={p:h for p,h in pins.values()},output_sha256={str(p):sha(p) for p in files},
        windows_expected_pids=windows,linux_expected_pids=linux_ids,owner_task_model_calls=0,owner_native_steps=0,
        actual_work_counters=None if counts is None else counts['counts'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);args=parser.parse_args()
    result=verify(args.base)
    with (args.base/'owner_completion_v2.json').open('x') as f:json.dump(result,f,indent=2)
    return 0 if result['completion_accounting_passed'] else 1
if __name__=='__main__':raise SystemExit(main())
