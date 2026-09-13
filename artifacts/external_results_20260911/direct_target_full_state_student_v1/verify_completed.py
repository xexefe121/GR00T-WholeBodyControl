"""Saved-only completion verification; never loads a model or launches training."""
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def absent(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD);kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD));kernel.GetExitCodeProcess.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=(wintypes.HANDLE,);kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:
        if ctypes.get_last_error()==87:return True
        raise OSError(ctypes.get_last_error(),'Cannot prove process absence')
    try:
        code=wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise OSError(ctypes.get_last_error(),'Cannot inspect process state')
        return code.value!=259
    finally:kernel.CloseHandle(handle)

def expected(calls,rows):
    return dict(calls_attempted=calls,calls_returned=calls,calls_synchronized=calls,calls_verified=calls,
                rows_attempted=rows,rows_returned=rows,rows_verified=rows)

def main():
    fit=BASE/'fit';report=read(fit/'report.json');exit_report=read(BASE/'fit_process/exit.json')
    for name in ('completed','optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed','all_frozen_inputs_unchanged'):
        if report[name] is not True:raise ValueError('Completion field failed: '+name)
    if exit_report['raw_python_exit_code']!=0 or exit_report['exit_code']!=0 or exit_report['exit_known'] is not True or exit_report['error'] is not None:raise ValueError('Durable exit incomplete/unknown.')
    if report['ordinary_final_step']!=65000 or report['additional_updates']!=10000 or report['optimizer_step']!=10000 or report['fresh_optimizer'] is not True:raise ValueError('Wrong final optimizer/global scope.')
    if report['execution_dtype']!='float64' or report['public_input_dtype']!='float32' or report['public_output_dtype']!='float32' or report['FP32_ONNX_release'] is not False:raise ValueError('Wrong export semantics.')
    if report['parity_tolerance_rad']!=1e-5 or report['max_preclip_error_rad']>1e-5:raise ValueError('Fixed numerical gate failed.')
    counts=report['counters']
    if counts['calibration']!=expected(3,14686) or counts['training']!=expected(30000,146860000) or counts['gradients']!=dict(attempted=3,returned=3,synchronized=3,verified=3):raise ValueError('Fixed training/calibration counters differ.')
    for label in ('initial_GPU32','final_GPU32','CPU64','GPU64','ORT64'):
        if counts['diagnostics'][label]!=expected(1437,367570):raise ValueError('Fixed backend counters differ.')
    if any(counts[key]!=0 for key in ('native_calls','BFM_calls','manual_export_trace_calls')):raise ValueError('Unexpected calls.')
    if sha(fit/'student_head.pt')!=report['checkpoint_sha256'] or sha(fit/'student_head.onnx')!=report['onnx_sha256'] or sha(fit/'normalization.npz')!=report['normalization_sha256']:raise ValueError('Final output identity differs.')
    if read(fit/'restoration.json')['all_restoration_checks_passed'] is not True:raise ValueError('Restoration gate failed.')
    receipt=read(BASE/'training_frozen_inputs.json');request=read(BASE/'training_request.json');clearance=read(BASE/'training_clearance.json')
    if sha(BASE/'training_request.json')!=report['training_request_sha256'] or sha(BASE/'training_frozen_inputs.json')!=report['frozen_receipt_sha256']:raise ValueError('Final request/frozen identity differs.')
    pins=dict(receipt['input_sha256']);pins.update({(Path(receipt['source_directory'])/key).as_posix():value for key,value in receipt['source_sha256'].items()})
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Current source/input pin changed: '+path)
    if read(BASE/'fit_process/postrun_pins.json')['all_exact'] is not True:raise ValueError('Postrun launcher pins failed.')
    if sha(clearance['review_path'])!=clearance['review_sha256'] or sha(clearance['launcher_path'])!=clearance['launcher_sha256']:raise ValueError('Clearance subjects changed.')
    output_manifest=read(fit/'output_manifest.json')
    if sha(fit/'output_manifest.json')!=report['output_manifest_sha256']:raise ValueError('Output manifest differs.')
    for name,digest in output_manifest['files'].items():
        path=(fit/name).resolve()
        if path.parent!=fit.resolve() or sha(path)!=digest:raise ValueError('Output file changed: '+name)
    pids=[exit_report['wrapper_pid'],exit_report['child_pid']]
    if any(type(pid)!=int or pid<=0 for pid in pids) or not all(absent(pid) for pid in pids):raise ValueError('Actual wrapper/child processes still present or unknown.')
    direct_subject_sha256=dict(fit_report=sha(fit/'report.json'),checkpoint=sha(fit/'student_head.pt'),head=sha(fit/'student_head.onnx'),
        normalization=sha(fit/'normalization.npz'),training_request=sha(BASE/'training_request.json'),training_frozen_inputs=sha(BASE/'training_frozen_inputs.json'),
        coefficient=sha(fit/'coefficient.json'),output_manifest=sha(fit/'output_manifest.json'),process_exit=sha(BASE/'fit_process/exit.json'))
    result=dict(owner_verification_passed=True,report_sha256=direct_subject_sha256['fit_report'],checkpoint_sha256=report['checkpoint_sha256'],onnx_sha256=report['onnx_sha256'],
        direct_subject_sha256=direct_subject_sha256,counters=counts,metrics=report['metrics'],parity=report['export_parity'],
        expected_pids=pids,processes_absent=True,raw_exit_known=True,raw_python_exit_code=0,exit_code=0,postpins_exact=True,
        current_pin_count=len(pins),task_model_calls=0,optimizer_updates=0,native_steps=0,behavioral_qualification=False)
    with (BASE/'owner_completion_verification.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(owner_verification_passed=True,sha256=sha(BASE/'owner_completion_verification.json'))))

if __name__=='__main__':main()
