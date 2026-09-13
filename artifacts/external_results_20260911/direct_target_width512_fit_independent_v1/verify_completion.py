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
def main():
    receipt=read(BASE/'launch_receipt.json');start=read(BASE/'process_v1/start.json');child=read(BASE/'process_v1/child.json');end=read(BASE/'process_v1/exit.json');dispatch=read(BASE/'dispatch.json')
    assert start['wrapper_pid']==child['wrapper_pid']==end['wrapper_pid']==dispatch['wrapper_pid']
    assert child['child_pid']==end['child_pid'] and child['handle_acquired'] is True
    assert absent(end['wrapper_pid']) and absent(end['child_pid'])
    assert end['raw_exit_known'] is True and end['raw_python_exit_code'] is not None
    assert start['request_sha256']==receipt['request_sha256']==sha(BASE/'audit_request.json')
    assert start['launch_receipt_sha256']==dispatch['launch_receipt_sha256']==end['launch_receipt_sha256']==sha(BASE/'launch_receipt.json')
    for name in ('prerun_pins.json','postrun_pins.json'):
        record=read(BASE/'process_v1'/name);assert record['all_exact'] is True
        assert set(record['files'])==set(receipt['input_sha256'])
        for path,digest in receipt['input_sha256'].items():
            entry=record['files'][path];assert entry['expected']==entry['actual']==digest and entry['matched'] is True
    for path,digest in receipt['input_sha256'].items():assert sha(path)==digest
    report=read(BASE/'results_v1/report.json')
    positive=report['evidence_audit_passed'] is True
    assert (end['raw_python_exit_code']==0 and end['exit_code']==0 and end['error'] is None)==positive
    assert end['all_postrun_pins_exact'] is True
    output_paths=[BASE/'audit_request.json',BASE/'launch_receipt.json',BASE/'dispatch.json',BASE/'source_preparation.json',Path(__file__)]
    output_paths+=list((BASE/'process_v1').glob('*.json'))+list((BASE/'process_v1').glob('*.log'))+list((BASE/'results_v1').glob('*.json'))
    result=dict(completion_accounting_passed=True,evidence_audit_passed=positive,export_qualified=report['export_qualified'],
        completed_utc=datetime.now(timezone.utc).isoformat(),request_sha256=sha(BASE/'audit_request.json'),report_sha256=sha(BASE/'results_v1/report.json'),
        source_review_sha256=receipt['source_review_sha256'],fit_owner_sha256=receipt['fit_owner_sha256'],
        launch_receipt_sha256=sha(BASE/'launch_receipt.json'),process_exit_sha256=sha(BASE/'process_v1/exit.json'),
        wrapper_pid=end['wrapper_pid'],child_pid=end['child_pid'],processes_absent=True,raw_exit_known=True,
        raw_python_exit_code=end['raw_python_exit_code'],exit_code=end['exit_code'],all_pre_post_current_launch_pins_exact=True,
        launch_pin_count=len(receipt['input_sha256']),output_sha256={p.resolve().as_posix():sha(p) for p in output_paths},
        task_model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,actual_audit_repeated=False,canonical_evaluation_cleared=False)
    with (BASE/'owner_completion.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(completion_accounting_passed=True,evidence_audit_passed=positive,export_qualified=result['export_qualified'],owner_sha256=sha(BASE/'owner_completion.json'),report_sha256=result['report_sha256'])))
if __name__=='__main__':main()
