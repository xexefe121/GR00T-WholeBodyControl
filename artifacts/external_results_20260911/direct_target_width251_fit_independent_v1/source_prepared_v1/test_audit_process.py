"""Synthetic metadata chains, no actual process/file hash inspection."""
from pathlib import Path
import copy
import pytest
import audit_fit_process as process

def fixture(tmp_path,monkeypatch):
    base=tmp_path;folder=base/'fit_process_v1';canonical=lambda p:Path(p).resolve().as_posix();sha=lambda p:canonical(p)
    roles={'source_checkpoint':base/'input.pt'}
    monkeypatch.setattr(process,'release_paths',lambda *a:dict(roles))
    request=dict(subjects={k:dict(path=str(base/(k+'.json'))) for k in ('fit_report','balance_source_review')},runtime=dict(python_path=str(base/'python.exe')),
        ordinary_start_step=81000,ordinary_final_step=91000,optimizer_start_step=16000,optimizer_final_step=26000,condition='causal',updates=10000,budgets={'training':1},automatic_retry=False)
    clear_sha=sha(base/'training_clearance.json');launch_sha='launch'
    start=dict(wrapper_pid=11,clearance_sha256=clear_sha,launch_receipt_sha256=launch_sha,**{k:request[k] for k in ('ordinary_start_step','ordinary_final_step','optimizer_start_step','optimizer_final_step','condition','updates','budgets','automatic_retry')},
        command=dict(python=request['runtime']['python_path'],arguments=['-u',(base/'source_snapshot_v1/train_recovery.py').as_posix()],working_directory=str(base/'source_snapshot_v1'),CUBLAS_WORKSPACE_CONFIG=':4096:8',threads=1))
    end=dict(wrapper_pid=11,child_pid=12,raw_python_exit_code=0,exit_code=0,clearance_sha256=clear_sha,launch_receipt_sha256=launch_sha)
    dispatch=dict(dispatcher_pid=10,automatic_retry=False,wrapper_pid=11,captured_handle_nonzero=True,executable='C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',clearance_sha256=clear_sha,launch_receipt_sha256=launch_sha,
        arguments=['-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(base/'run_fit_durable_v1.ps1'),'-ClearanceSha256',clear_sha,'-LaunchReceiptSha256',launch_sha])
    absence=dict(wrapper_pid=11,child_pid=12,wrapper_absent=True,child_absent=True,**{n+'_sha256':sha(folder/(n+'.json')) for n in ('dispatch','start','child','exit')})
    roles.update(process_exit=folder/'exit.json',source_fit_report=Path(request['subjects']['fit_report']['path']),balance_source_review=Path(request['subjects']['balance_source_review']['path']))
    direct={k:dict(path=str(p),sha256=sha(p)) for k,p in roles.items()}
    output={str(p):sha(p) for p in list(roles.values())+[folder/(n+'.json') for n in ('dispatch_attempt','dispatch','start','child','exit','prerun_pins','postrun_pins','process_absence')]+[folder/n for n in ('stdout.log','stderr.log','wrapper_stdout.log','wrapper_stderr.log')]}
    owner=dict(clearance_sha256=clear_sha,launch_receipt_sha256=launch_sha,process_absence=copy.deepcopy(absence),process_absence_sha256=sha(folder/'process_absence.json'),direct_subjects=direct,output_sha256=output)
    records={canonical(folder/'dispatch.json'):dispatch,canonical(folder/'dispatch_attempt.json'):copy.deepcopy(dispatch),canonical(folder/'process_absence.json'):absence}
    def check(name,passed):assert passed,name
    def compare(name,a,b):assert type(a)==type(b) and a==b,name
    args=[base,request,{},launch_sha,start,{},end,owner,{k:v['sha256'] for k,v in direct.items()},lambda p:records[canonical(p)],lambda p:p,sha,check,compare,canonical]
    return args,dispatch,absence,owner

@pytest.mark.parametrize('change',[None,'argument','wrapper','handle','raw_bool','start_scope','absence','owner_path','owner_hash','output_omission','log_omission','dispatch_attempt'])
def test_saved_process_chain_rejects_mismatches(tmp_path,monkeypatch,change):
    args,dispatch,absence,owner=fixture(tmp_path,monkeypatch)
    if change=='argument':dispatch['arguments'].remove('-NonInteractive')
    if change=='wrapper':dispatch['wrapper_pid']=13
    if change=='handle':dispatch['captured_handle_nonzero']=False
    if change=='raw_bool':args[6]['raw_python_exit_code']=False
    if change=='start_scope':args[4]['updates']=3
    if change=='absence':absence['child_absent']=False
    if change=='owner_path':owner['direct_subjects']['source_checkpoint']['path']=str(tmp_path/'wrong.pt')
    if change=='owner_hash':owner['direct_subjects']['source_checkpoint']['sha256']='bad'
    if change=='output_omission':owner['output_sha256'].pop(str(tmp_path/'fit_process_v1/start.json'))
    if change=='log_omission':owner['output_sha256'].pop(str(tmp_path/'fit_process_v1/stdout.log'))
    if change=='dispatch_attempt':dispatch['dispatcher_pid']=999
    if change is None:process.validate_process_chain(*args)
    else:
        with pytest.raises(AssertionError):process.validate_process_chain(*args)
