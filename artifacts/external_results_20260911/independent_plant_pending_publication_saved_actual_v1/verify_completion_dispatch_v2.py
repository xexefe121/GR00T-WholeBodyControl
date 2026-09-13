"""Saved audit completion/PID/hash verification only; no task numerical work."""
import ctypes
from ctypes import wintypes
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def absent(pid):
    assert type(pid) is int and pid>0
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD);api.OpenProcess.restype=wintypes.HANDLE
    api.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD));api.GetExitCodeProcess.restype=wintypes.BOOL
    api.CloseHandle.argtypes=(wintypes.HANDLE,);api.CloseHandle.restype=wintypes.BOOL
    handle=api.OpenProcess(0x1000,False,pid)
    if not handle:
        if ctypes.get_last_error()==87:return True
        raise OSError(ctypes.get_last_error(),'Process state unavailable')
    try:
        code=wintypes.DWORD()
        if not api.GetExitCodeProcess(handle,ctypes.byref(code)):raise OSError(ctypes.get_last_error(),'Process exit unavailable')
        return code.value!=259
    finally:api.CloseHandle(handle)

def normalized_pins(mapping):
    result={}
    for path,digest in mapping.items():
        key=str(path).replace(chr(92),'/')
        if key.startswith('/mnt/') and key[6:7]=='/':key=key[5]+':'+key[6:]
        key=key.casefold()
        assert key not in result or result[key]==digest,'Conflicting normalized path'
        result[key]=digest
    return result


def outcome(end,report,failure,request_sha):
    raw=end['raw_python_exit_code']
    assert type(end['raw_exit_known']) is bool and end['raw_exit_known']==(raw is not None)
    assert raw is None or type(raw) is int
    assert type(end['exit_code']) is int and report is not None and failure is None
    assert report['request_sha256']==request_sha
    assert report['model_calls']==report['native_steps_executed']==report['optimizer_updates']==0
    positive=report['evidence_integrity_passed']
    assert type(positive) is bool
    if positive:
        assert end['raw_exit_known'] is True and raw==end['exit_code']==0 and end['error'] is None
        assert report['online_policy_qualified'] is False and report['realtime_teleoperation_qualified'] is False
    else:
        assert report['component_qualified'] is False and report.get('error')
        assert raw!=0 and end['exit_code']!=0
    return positive


def validate_dispatch_correction(dispatch):
    original_path=BASE/'dispatch.json';stop_path=BASE/'preflight_dispatch_failure_v1/after_stop.json'
    original_sha='380077859e7ffc6b3c9ce779cb1d425be0a02aa2e354bb682fe8970ffdc4c750'
    stop_sha='90b305d2fc6b80f2a6f1b258bafacb25ae00c089f3beb65ad6a7cb1f8477c490'
    assert sha(original_path)==original_sha==dispatch['original_dispatch_sha256']
    assert sha(stop_path)==stop_sha==dispatch['preflight_after_stop_sha256']
    original=read(original_path);stop=read(stop_path)
    assert original['wrapper_pid']==27700 and original['handle_acquired'] is True
    assert stop['stopped_only_confirmed_owned_wrapper']==27700 and stop['handle_acquired'] is True
    assert stop['exit_code_known'] is True and stop['observed_wrapper_exit_code']==-1
    assert stop['observed_present_pids']==[] and stop['process_directory_absent'] is True and stop['results_directory_absent'] is True
    assert stop['actual_saved_audit_calls']==0 and stop['automatic_retry'] is False
    assert dispatch['corrected_argument_only_selection'] is True and dispatch['prior_actual_audit_calls']==0 and dispatch['automatic_retry'] is False
    assert dispatch['wrapper_pid']!=original['wrapper_pid'] and dispatch['handle_acquired'] is True
    args=dispatch['exact_arguments']
    assert len(args)==8 and args[:5]==['-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File']
    assert Path(args[5]).resolve()==(BASE/'run_audit_durable.ps1').resolve()
    assert args[6:] == ['-LaunchReceiptSha256',sha(BASE/'launch_receipt.json')]
    assert original['launch_receipt_sha256']==dispatch['launch_receipt_sha256']==sha(BASE/'launch_receipt.json')
    assert original['request_sha256']==dispatch['request_sha256']==sha(BASE/'request.json')
    assert original['clearance_sha256']==dispatch['clearance_sha256']==sha(BASE/'launch_clearance.json')
    assert sha(BASE/'preflight_dispatch_failure_v1/dispatch.json')==original_sha
    assert datetime.fromisoformat(dispatch['utc'].replace('Z','+00:00'))>datetime.fromisoformat(stop['utc'].replace('Z','+00:00'))
    return dict(original_dispatch_sha256=original_sha,preflight_after_stop_sha256=stop_sha,
        corrected_dispatch_sha256=sha(BASE/'dispatch_v2.json'),prior_actual_audit_calls=0,
        original_wrapper_forcibly_stopped_before_script_body=True,argument_only_dispatch_correction=True)

def main():
    receipt=read(BASE/'launch_receipt.json');start=read(BASE/'process_v1/start.json');child=read(BASE/'process_v1/child.json');end=read(BASE/'process_v1/exit.json');dispatch=read(BASE/'dispatch_v2.json')
    correction=validate_dispatch_correction(dispatch)
    assert start['wrapper_pid']==child['wrapper_pid']==end['wrapper_pid']==dispatch['wrapper_pid']
    assert child['child_pid']==end['child_pid'] and child['handle_acquired'] is True
    assert absent(end['wrapper_pid']) and absent(end['child_pid'])
    clearance_path=BASE/'launch_clearance.json';clearance=read(clearance_path)
    assert start['clearance_sha256']==end['clearance_sha256']==sha(clearance_path)
    assert start['concrete_review_sha256']==end['concrete_review_sha256']==clearance['review']['sha256']==sha(clearance['review']['path'])
    concrete=read(clearance['review']['path']);assert concrete['passed'] is True
    for role,path in [('request_subject',BASE/'request.json'),('launch_receipt_subject',BASE/'launch_receipt.json')]:
        assert Path(concrete[role]['path']).resolve()==path.resolve() and concrete[role]['sha256']==sha(path)
    assert start['request_sha256']==receipt['request_sha256']==sha(BASE/'request.json')
    assert start['launch_receipt_sha256']==dispatch['launch_receipt_sha256']==end['launch_receipt_sha256']==sha(BASE/'launch_receipt.json')
    for name in ('prerun_pins.json','postrun_pins.json'):
        record=read(BASE/'process_v1'/name);assert record['all_exact'] is True
        assert set(record['files'])==set(receipt['input_sha256'])
        for path,digest in receipt['input_sha256'].items():
            entry=record['files'][path];assert entry['expected']==entry['actual']==digest and entry['matched'] is True
    for path,digest in receipt['input_sha256'].items():assert sha(path)==digest
    report_path=BASE/'results_v1/report.json';failure_path=BASE/'results_v1/failure.json'
    report=read(report_path) if report_path.exists() else None
    failure=read(failure_path) if failure_path.exists() else None
    positive=outcome(end,report,failure,sha(BASE/'request.json'))
    if positive:
        assert normalized_pins(report['input_sha256'])==normalized_pins(read(BASE/'request.json')['input_sha256'])
    assert report['comparisons_sha256']==sha(BASE/'results_v1/comparisons.jsonl')
    assert end['all_postrun_pins_exact'] is True
    output_paths=[BASE/'request.json',BASE/'launch_receipt.json',BASE/'dispatch_v2.json',BASE/'helper_preparation.json',clearance_path,Path(clearance['review']['path']),Path(__file__)]
    output_paths += [BASE/'dispatch.json',BASE/'verify_completion.py',BASE/'derive_owner_dispatch_v2.py',BASE/'owner_dispatch_v2.diff'] + list((BASE/'preflight_dispatch_failure_v1').glob('*'))
    output_paths+=list((BASE/'process_v1').glob('*.json'))+list((BASE/'process_v1').glob('*.log'))+list((BASE/'results_v1').glob('*'))
    result=dict(dispatch_correction=correction,completion_accounting_passed=True,evidence_integrity_passed=positive,behavioral_qualification=False,
        completed_utc=datetime.now(timezone.utc).isoformat(),request_sha256=sha(BASE/'request.json'),report_sha256=sha(report_path) if report is not None else None,
        failure_sha256=sha(failure_path) if failure is not None else None,clearance_sha256=sha(clearance_path),concrete_review_sha256=sha(clearance['review']['path']),
        source_review_sha256=receipt['source_review_sha256'],clock_owner_sha256=sha(read(BASE/'request.json')['roles']['owner']),
        launch_receipt_sha256=sha(BASE/'launch_receipt.json'),process_exit_sha256=sha(BASE/'process_v1/exit.json'),
        wrapper_pid=end['wrapper_pid'],child_pid=end['child_pid'],processes_absent=True,raw_exit_known=end['raw_exit_known'],
        raw_python_exit_code=end['raw_python_exit_code'],exit_code=end['exit_code'],all_pre_post_current_launch_pins_exact=True,
        launch_pin_count=len(receipt['input_sha256']),output_sha256={p.resolve().as_posix():sha(p) for p in output_paths},
        task_model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,actual_audit_repeated=False,canonical_evaluation_cleared=False)
    with (BASE/'owner_completion_dispatch_v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(completion_accounting_passed=True,evidence_integrity_passed=positive,behavioral_qualification=False,owner_sha256=sha(BASE/'owner_completion_dispatch_v2.json'),report_sha256=result['report_sha256'])))
if __name__=='__main__':main()
